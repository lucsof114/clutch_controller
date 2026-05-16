#!/usr/bin/env python3
"""
Extrinsic calibration pipeline.

Calibrates camera-to-world poses for a multi-camera rig given pre-computed
intrinsics.  Runs detection, BFS initialisation, and joint optimisation in
one call.  Results are saved as {cam_id: TorchCam.to_dict()} so each camera
can be reconstructed directly with TorchCam.from_dict(results[cam_id]).

Usage:
    python calibration/extrinsic_calibration.py <recording_dir> <intrinsics_id> [options]

    recording_dir   Path to the extrinsic recording
                    (e.g. .../clutch_db/recordings/20260515_224247)
    intrinsics_id   Recording ID whose intrinsics to use
                    (e.g. 20260515_224145)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cameras.camera import TorchCam
from calibration.intrinsic_calibration import rodrigues_to_matrix, _load_board

PNP_MIN_PTS = 6


# ── Detection ──────────────────────────────────────────────────────────────────

def _marker_corners_3d(board) -> dict:
    obj_pts = board.getObjPoints()
    board_ids = board.getIds().flatten()
    return {int(board_ids[i]): np.array(obj_pts[i], dtype=np.float64)
            for i in range(len(obj_pts))}


def _detect_frame(img_path: Path, detector, mc3d: dict) -> Optional[dict]:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img)
    if charuco_ids is None or len(charuco_ids) < 4:
        return None
    result = {
        "charuco_corners": charuco_corners.reshape(-1, 2),
        "charuco_ids":     charuco_ids.flatten(),
    }
    if marker_ids is not None and len(marker_ids) > 0:
        quads_2d, quads_3d = [], []
        for corners, mid in zip(marker_corners, marker_ids.flatten()):
            mid = int(mid)
            if mid in mc3d:
                quads_2d.append(corners.reshape(4, 2))
                quads_3d.append(mc3d[mid])
        if quads_2d:
            result["aruco_corners_2d"] = np.concatenate(quads_2d).astype(np.float64)
            result["aruco_corners_3d"] = np.concatenate(quads_3d).astype(np.float64)
    return result


def _build_detections(recording_dir: Path, camera_serials: List[str],
                      cache_path: Path, max_workers: int = 40) -> dict:
    """
    Detect charuco + aruco in all frames.  Results are cached at cache_path.

    Returns dict[(cam_id, frame_id)] -> detection dict.
    """
    if cache_path.exists():
        print(f"Loading detection cache from {cache_path}...")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"  {len(cache)} cam-frame detections loaded")
        return cache

    marker_detector = _load_board()
    mc3d = _marker_corners_3d(marker_detector.board)

    tasks = []
    for serial in camera_serials:
        cam_dir = recording_dir / serial
        if not cam_dir.exists():
            continue
        for frame_dir in sorted(cam_dir.iterdir()):
            if not frame_dir.is_dir():
                continue
            png = frame_dir / "frame.png"
            if png.exists():
                tasks.append((serial, int(frame_dir.name), png))

    print(f"Detecting in {len(tasks)} images across {len(camera_serials)} cameras...")
    cache = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_detect_frame, path, marker_detector.detector, mc3d): (cam, fid)
                   for cam, fid, path in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            cam, fid = futures[fut]
            result = fut.result()
            if result is not None:
                cache[(cam, fid)] = result
            if i % 500 == 0:
                print(f"  [{i}/{len(tasks)}] detected: {len(cache)}")

    print(f"  Done: {len(cache)} detections")
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  Cached to {cache_path}")
    return cache


# ── PnP + frame_data ──────────────────────────────────────────────────────────

def _pnp(pts_3d: np.ndarray, pts_2d: np.ndarray,
         cam_matrix: np.ndarray, dist: np.ndarray,
         threshold: float) -> Optional[Tuple]:
    if len(pts_3d) < PNP_MIN_PTS:
        return None
    ok, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, cam_matrix, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, cam_matrix, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - pts_2d, axis=1).mean()
    if err > threshold:
        return None
    return rvec.flatten(), tvec.flatten(), err


def _build_frame_data(detections: dict, cameras: Dict[str, TorchCam],
                      board, pnp_threshold: float) -> dict:
    """
    Run PnP per cam-frame.  Returns frame_data[frame_id][cam_id] for frames
    where at least 2 cameras have successful PnP.
    """
    frame_data: dict = defaultdict(dict)
    for (cam_id, fid), det in detections.items():
        if cam_id not in cameras:
            continue
        intr = cameras[cam_id].to_dict()["intrinsics"]
        cam_matrix = np.array([[intr["fx"], 0, intr["cx"]],
                                [0, intr["fy"], intr["cy"]],
                                [0, 0, 1]], dtype=np.float64)
        dist = np.array(intr["dist_coeffs"][:5], dtype=np.float64)

        ids    = det["charuco_ids"]
        pts_3d = board.getChessboardCorners()[ids].astype(np.float64)
        pts_2d = det["charuco_corners"].astype(np.float64)

        result = _pnp(pts_3d, pts_2d, cam_matrix, dist, pnp_threshold)
        if result is None:
            continue
        rvec, tvec, err = result
        entry: dict = {
            "charuco_pts_3d": pts_3d,
            "charuco_pts_2d": pts_2d,
            "rvec": rvec,
            "tvec": tvec,
            "pnp_err": err,
        }
        if "aruco_corners_2d" in det:
            entry["aruco_pts_2d"] = det["aruco_corners_2d"]
            entry["aruco_pts_3d"] = det["aruco_corners_3d"]
        frame_data[fid][cam_id] = entry

    return {fid: cams for fid, cams in frame_data.items() if len(cams) >= 2}


# ── BFS initialisation ────────────────────────────────────────────────────────

def _rvec_tvec_to_4x4(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3]  = tvec.flatten()
    return T


def _inv4x4(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti


def _bfs_init(frame_data: dict, camera_serials: List[str], root_cam: str) -> dict:
    """
    BFS from root_cam through the camera visibility graph.

    For each camera pair, picks the 5 lowest-PnP-error shared frames and
    averages the resulting relative transforms for a robust initialisation.

    Returns cam_to_world[cam_id] -> (4,4) numpy array.
    """
    pair_frames: dict = defaultdict(list)
    for fid, cams in frame_data.items():
        for ca, cb in combinations(sorted(cams.keys()), 2):
            pair_frames[(ca, cb)].append(fid)

    adj: dict = defaultdict(set)
    for ca, cb in pair_frames:
        adj[ca].add(cb)
        adj[cb].add(ca)

    cam_to_world = {root_cam: np.eye(4, dtype=np.float64)}
    visited = {root_cam}
    queue = deque([root_cam])

    while queue:
        cur = queue.popleft()
        for nbr in adj[cur]:
            if nbr in visited:
                continue
            key = tuple(sorted([cur, nbr]))
            candidates = sorted(
                (frame_data[fid][cur]["pnp_err"] + frame_data[fid][nbr]["pnp_err"], fid)
                for fid in pair_frames[key]
                if cur in frame_data[fid] and nbr in frame_data[fid]
            )[:5]
            if not candidates:
                continue
            translations, rotations = [], []
            for _, fid in candidates:
                T_b_cur = _rvec_tvec_to_4x4(frame_data[fid][cur]["rvec"], frame_data[fid][cur]["tvec"])
                T_b_nbr = _rvec_tvec_to_4x4(frame_data[fid][nbr]["rvec"], frame_data[fid][nbr]["tvec"])
                T = cam_to_world[cur] @ T_b_cur @ _inv4x4(T_b_nbr)
                translations.append(T[:3, 3])
                rotations.append(T[:3, :3])
            t_avg = np.mean(translations, axis=0)
            U, _, Vt = np.linalg.svd(np.mean(rotations, axis=0))
            if np.linalg.det(U @ Vt) < 0:
                U[:, -1] *= -1
            T_avg = np.eye(4, dtype=np.float64)
            T_avg[:3, :3] = U @ Vt
            T_avg[:3, 3]  = t_avg
            cam_to_world[nbr] = T_avg
            visited.add(nbr)
            queue.append(nbr)
            print(f"  {cur} -> {nbr} ({len(candidates)} frames)")

    missing = set(camera_serials) - set(cam_to_world)
    if missing:
        raise RuntimeError(f"BFS could not reach: {missing}")
    return cam_to_world


# ── Optimisation ──────────────────────────────────────────────────────────────

def _optimize(cameras: Dict[str, TorchCam],
              frame_data: dict,
              cam_to_world_np: dict,
              root_cam: str,
              num_epochs: int,
              lr_cams: float,
              lr_boards: float,
              huber_charuco_delta: float,
              huber_aruco_delta: float,
              aruco_weight: float,
              max_frames: int,
              device: str,
              log_dir: str) -> Tuple[dict, float]:
    dev = torch.device(device)

    for cam in cameras.values():
        cam.to(dev)
        for p in cam.parameters():
            p.requires_grad_(False)

    # Select best frames by avg PnP error
    scored = sorted(
        (np.mean([d["pnp_err"] for d in cams.values()]), fid)
        for fid, cams in frame_data.items()
    )[:max_frames]
    frame_data = {fid: frame_data[fid] for _, fid in scored}
    print(f"Selected {len(frame_data)} frames "
          f"(PnP err: {scored[0][0]:.2f} – {scored[-1][0]:.2f} px)")

    non_root = [c for c in cameras if c != root_cam]

    # Camera poses (non-root only)
    cam_rvecs: Dict[str, nn.Parameter] = {}
    cam_tvecs: Dict[str, nn.Parameter] = {}
    for cam_id in non_root:
        T = cam_to_world_np[cam_id]
        rv, _ = cv2.Rodrigues(T[:3, :3])
        cam_rvecs[cam_id] = nn.Parameter(torch.tensor(rv.flatten(), dtype=torch.float64, device=dev))
        cam_tvecs[cam_id] = nn.Parameter(torch.tensor(T[:3, 3],    dtype=torch.float64, device=dev))

    # Board poses (per frame), initialised from root cam or best cam
    frame_ids = sorted(frame_data)
    fid_to_idx = {fid: i for i, fid in enumerate(frame_ids)}
    board_rvecs_list, board_tvecs_list = [], []
    for fid in frame_ids:
        cams = frame_data[fid]
        src = root_cam if root_cam in cams else min(cams, key=lambda c: cams[c]["pnp_err"])
        T_board_world = cam_to_world_np[src] @ _rvec_tvec_to_4x4(cams[src]["rvec"], cams[src]["tvec"])
        rv, _ = cv2.Rodrigues(T_board_world[:3, :3])
        board_rvecs_list.append(rv.flatten())
        board_tvecs_list.append(T_board_world[:3, 3])

    board_rvecs = nn.Parameter(torch.tensor(np.stack(board_rvecs_list), dtype=torch.float64, device=dev))
    board_tvecs = nn.Parameter(torch.tensor(np.stack(board_tvecs_list), dtype=torch.float64, device=dev))

    # Tensorize observations
    def _tensorize(key_3d: str, key_2d: str) -> dict:
        tensors = {}
        for cam_id in cameras:
            fidxs, pts3, pts2 = [], [], []
            for fid in frame_ids:
                d = frame_data[fid].get(cam_id)
                if d is None or key_3d not in d:
                    continue
                n = len(d[key_3d])
                fidxs.extend([fid_to_idx[fid]] * n)
                pts3.append(d[key_3d])
                pts2.append(d[key_2d])
            if fidxs:
                tensors[cam_id] = (
                    torch.tensor(fidxs, dtype=torch.long, device=dev),
                    torch.tensor(np.concatenate(pts3), dtype=torch.float64, device=dev),
                    torch.tensor(np.concatenate(pts2), dtype=torch.float64, device=dev),
                )
        return tensors

    charuco_tensors = _tensorize("charuco_pts_3d", "charuco_pts_2d")
    aruco_tensors   = _tensorize("aruco_pts_3d",   "aruco_pts_2d")
    n_charuco = sum(len(t[0]) for t in charuco_tensors.values())
    n_aruco   = sum(len(t[0]) for t in aruco_tensors.values())
    print(f"Observations: {n_charuco} charuco, {n_aruco} aruco across {len(frame_ids)} frames")

    cam_params = [p for cam_id in non_root for p in [cam_rvecs[cam_id], cam_tvecs[cam_id]]]
    optimizer = torch.optim.Adam([
        {"params": cam_params,                    "lr": lr_cams},
        {"params": [board_rvecs, board_tvecs],    "lr": lr_boards},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    huber_c = nn.HuberLoss(reduction="sum", delta=huber_charuco_delta)
    huber_a = nn.HuberLoss(reduction="sum", delta=huber_aruco_delta)
    writer   = SummaryWriter(log_dir=log_dir)

    def _reproject_loss(tensors: dict, huber_fn, weight: float = 1.0):
        loss = torch.tensor(0.0, dtype=torch.float64, device=dev)
        sq_err = 0.0
        n = 0
        for cam_id, (fidxs, pts_3d, pts_2d) in tensors.items():
            b_R    = rodrigues_to_matrix(board_rvecs[fidxs])
            p_world = (b_R @ pts_3d.unsqueeze(-1)).squeeze(-1) + board_tvecs[fidxs]
            if cam_id == root_cam:
                p_cam = p_world
            else:
                c_R   = rodrigues_to_matrix(cam_rvecs[cam_id])
                p_cam = (c_R.T @ (p_world - cam_tvecs[cam_id]).unsqueeze(-1)).squeeze(-1)
            proj = cameras[cam_id].project_camera_points(p_cam)
            loss = loss + weight * huber_fn(proj, pts_2d)
            with torch.no_grad():
                sq_err += ((proj - pts_2d) ** 2).sum().item()
                n      += len(fidxs)
        return loss, sq_err, n

    best_rmse    = float("inf")
    best_cam_rv  = best_cam_tv = None
    best_board_rv = best_board_tv = None

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss_c, sq_err, n_pts = _reproject_loss(charuco_tensors, huber_c)
        loss_a, _, _          = _reproject_loss(aruco_tensors,   huber_a, aruco_weight)
        (loss_c + loss_a).backward()
        optimizer.step()
        scheduler.step()

        rmse = np.sqrt(sq_err / n_pts) if n_pts > 0 else float("inf")
        if rmse < best_rmse:
            best_rmse    = rmse
            best_cam_rv  = {c: cam_rvecs[c].data.clone() for c in non_root}
            best_cam_tv  = {c: cam_tvecs[c].data.clone() for c in non_root}
            best_board_rv = board_rvecs.data.clone()
            best_board_tv = board_tvecs.data.clone()

        writer.add_scalar("reproj/rmse",      rmse,      epoch)
        writer.add_scalar("reproj/best_rmse", best_rmse, epoch)
        for cam_id in non_root:
            pos = cam_tvecs[cam_id].detach().cpu()
            writer.add_scalar(f"cam_pos/{cam_id}_x", pos[0].item(), epoch)
            writer.add_scalar(f"cam_pos/{cam_id}_y", pos[1].item(), epoch)
            writer.add_scalar(f"cam_pos/{cam_id}_z", pos[2].item(), epoch)

        if epoch % 200 == 0 or epoch == num_epochs - 1:
            print(f"  epoch {epoch:>5d}  rmse={rmse:.4f} px  best={best_rmse:.4f} px")

    writer.close()

    # Restore best checkpoint
    for cam_id in non_root:
        cam_rvecs[cam_id].data.copy_(best_cam_rv[cam_id])
        cam_tvecs[cam_id].data.copy_(best_cam_tv[cam_id])

    final_cam_to_world = {root_cam: np.eye(4, dtype=np.float64)}
    for cam_id in non_root:
        rv = cam_rvecs[cam_id].detach().cpu().numpy()
        tv = cam_tvecs[cam_id].detach().cpu().numpy()
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3]  = tv
        final_cam_to_world[cam_id] = T

    print(f"\nBest RMSE: {best_rmse:.4f} px")
    return final_cam_to_world, best_rmse


# ── Public entry point ────────────────────────────────────────────────────────

def run_extrinsic_calibration(
    recording_dir: str,
    intrinsics_id: str,
    root_cam: Optional[str] = None,
    num_epochs: int = 10000,
    max_frames: int = 50,
    max_workers: int = 40,
    pnp_threshold: float = 3.0,
    lr_cams: float = 0.001,
    lr_boards: float = 0.001,
    huber_charuco_delta: float = 3.0,
    huber_aruco_delta: float = 3.0,
    aruco_weight: float = 0.1,
    device: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> dict:
    """
    Run full extrinsic calibration pipeline.

    Detects charuco/aruco corners (cached), runs BFS initialisation, then
    jointly optimises camera poses and per-frame board poses.

    Saves results to:
        {db}/calibration/extrinsics/{recording_id}/results.json

    Each camera entry in the JSON is a TorchCam.to_dict() — intrinsics and
    extrinsics together — so cameras can be loaded with:
        TorchCam.from_dict(results[cam_id])

    Returns the results dict.
    """
    recording_dir = Path(recording_dir).resolve()
    db_dir        = recording_dir.parent.parent  # .../clutch_db
    recording_id  = recording_dir.name

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if log_dir is None:
        log_dir = f"/tmp/clutch_tb/extrinsics/{recording_id}"

    # Auto-detect camera serials from subdirectories
    camera_serials = sorted(
        d.name for d in recording_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and (d / "000000" / "frame.png").exists()
    )
    if not camera_serials:
        raise RuntimeError(f"No decoded camera directories found in {recording_dir}")
    print(f"Cameras: {camera_serials}")

    if root_cam is None:
        root_cam = camera_serials[0]
    elif root_cam not in camera_serials:
        raise ValueError(f"root_cam {root_cam!r} not found in {camera_serials}")
    print(f"Root camera: {root_cam}")

    # Load intrinsics
    cameras: Dict[str, TorchCam] = {}
    for serial in camera_serials:
        path = db_dir / "calibration" / "intrinsics" / serial / intrinsics_id / "results.json"
        with open(path) as f:
            cameras[serial] = TorchCam.from_dict(json.load(f))
    print(f"Loaded intrinsics for: {list(cameras)}")

    # Detect charuco + aruco (cached per recording)
    cache_path = recording_dir / "detections_cache.pkl"
    detections = _build_detections(recording_dir, camera_serials, cache_path, max_workers)

    # PnP per cam-frame, filter to multi-camera frames
    marker_detector = _load_board()
    frame_data = _build_frame_data(detections, cameras, marker_detector.board, pnp_threshold)
    print(f"Frames with >=2 cameras after PnP: {len(frame_data)}")

    # BFS initialisation
    print("\nBFS initialisation:")
    cam_to_world_np = _bfs_init(frame_data, camera_serials, root_cam)
    for cam_id, T in cam_to_world_np.items():
        pos = T[:3, 3]
        print(f"  {cam_id}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

    # Joint optimisation
    print("\nOptimising extrinsics...")
    final_cam_to_world, best_rmse = _optimize(
        cameras, frame_data, cam_to_world_np, root_cam,
        num_epochs=num_epochs, lr_cams=lr_cams, lr_boards=lr_boards,
        huber_charuco_delta=huber_charuco_delta, huber_aruco_delta=huber_aruco_delta,
        aruco_weight=aruco_weight, max_frames=max_frames,
        device=device, log_dir=log_dir,
    )

    # Update each camera's pose and serialise
    results: dict = {}
    for cam_id, cam in cameras.items():
        T = torch.tensor(final_cam_to_world[cam_id], dtype=torch.float64)
        cam.set_pose(T)
        results[cam_id] = cam.to_dict()

    # Save
    out_dir  = db_dir / "calibration" / "extrinsics" / recording_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrinsic camera calibration")
    parser.add_argument("recording_dir",  help="Path to extrinsic recording directory")
    parser.add_argument("intrinsics_id",  help="Intrinsics recording ID (e.g. 20260515_224145)")
    parser.add_argument("--root-cam",     default=None,  help="Root camera serial (default: first alphabetically)")
    parser.add_argument("--epochs",       type=int,   default=10000)
    parser.add_argument("--max-frames",   type=int,   default=50)
    parser.add_argument("--workers",      type=int,   default=40)
    parser.add_argument("--pnp-threshold",type=float, default=3.0)
    parser.add_argument("--lr-cams",      type=float, default=0.001)
    parser.add_argument("--lr-boards",    type=float, default=0.001)
    parser.add_argument("--device",       default=None)
    parser.add_argument("--log-dir",      default=None)
    args = parser.parse_args()

    run_extrinsic_calibration(
        recording_dir=args.recording_dir,
        intrinsics_id=args.intrinsics_id,
        root_cam=args.root_cam,
        num_epochs=args.epochs,
        max_frames=args.max_frames,
        max_workers=args.workers,
        pnp_threshold=args.pnp_threshold,
        lr_cams=args.lr_cams,
        lr_boards=args.lr_boards,
        device=args.device,
        log_dir=args.log_dir,
    )
