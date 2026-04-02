import dataclasses
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from cameras.camera import TorchCam
from calibration.marker import MarkerDetector


# ---------------------------------------------------------------------------
# Differentiable Rodrigues
# ---------------------------------------------------------------------------

def rodrigues_to_matrix(rvec: torch.Tensor) -> torch.Tensor:
    """Convert Rodrigues rotation vectors to rotation matrices.

    Args:
        rvec: (3,) or (B, 3) rotation vectors
    Returns:
        (3, 3) or (B, 3, 3) rotation matrices
    """
    unbatched = rvec.dim() == 1
    if unbatched:
        rvec = rvec.unsqueeze(0)

    B = rvec.shape[0]
    theta = rvec.norm(dim=1, keepdim=True)  # (B, 1)
    safe_theta = torch.where(theta < 1e-8, torch.ones_like(theta), theta)
    k = rvec / safe_theta  # (B, 3)

    # batched skew-symmetric
    zero = torch.zeros(B, dtype=rvec.dtype, device=rvec.device)
    K = torch.stack([
        zero, -k[:, 2], k[:, 1],
        k[:, 2], zero, -k[:, 0],
        -k[:, 1], k[:, 0], zero,
    ], dim=1).reshape(B, 3, 3)

    sin_t = torch.sin(safe_theta).unsqueeze(-1)   # (B, 1, 1)
    cos_t = torch.cos(safe_theta).unsqueeze(-1)
    I = torch.eye(3, dtype=rvec.dtype, device=rvec.device).unsqueeze(0)

    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)

    if unbatched:
        return R.squeeze(0)
    return R


# ---------------------------------------------------------------------------
# All board poses (vectorized)
# ---------------------------------------------------------------------------

class BoardPoses(nn.Module):
    """Holds all per-frame board poses as (N, 3) parameter tensors."""

    def __init__(self, rvecs: np.ndarray, tvecs: np.ndarray):
        """
        rvecs: (N, 3) array of Rodrigues vectors
        tvecs: (N, 3) array of translations
        """
        super().__init__()
        self.rvecs = nn.Parameter(torch.tensor(rvecs, dtype=torch.float64))
        self.tvecs = nn.Parameter(torch.tensor(tvecs, dtype=torch.float64))

    def transform(self, frame_indices: torch.Tensor, pts_3d: torch.Tensor) -> torch.Tensor:
        """Transform board points to camera frame for given frame indices.

        Args:
            frame_indices: (M,) long tensor — frame index per point
            pts_3d: (M, 3) board-frame 3D points
        Returns:
            (M, 3) camera-frame 3D points
        """
        rvecs = self.rvecs[frame_indices]  # (M, 3)
        tvecs = self.tvecs[frame_indices]  # (M, 3)
        R = rodrigues_to_matrix(rvecs)     # (M, 3, 3)
        # batched matmul: (M, 3, 3) @ (M, 3, 1) -> (M, 3, 1) -> (M, 3)
        return (R @ pts_3d.unsqueeze(-1)).squeeze(-1) + tvecs


# ---------------------------------------------------------------------------
# Per-frame observation container
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FrameObservations:
    frame_index: int
    charuco_pts_3d: torch.Tensor   # (Nc, 3)
    charuco_pts_2d: torch.Tensor   # (Nc, 2)
    aruco_pts_3d: torch.Tensor     # (Na, 3)
    aruco_pts_2d: torch.Tensor     # (Na, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_board(board_name: Optional[str] = None) -> MarkerDetector:
    boards_path = os.path.join(os.path.dirname(__file__), "boards.json")
    with open(boards_path) as f:
        data = json.load(f)

    board_cfg = None
    for b in data["boards"]:
        if board_name is None or b["name"] == board_name:
            board_cfg = b
            break
    if board_cfg is None:
        raise ValueError(f"Board '{board_name}' not found in {boards_path}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, board_cfg["aruco_dict"])
    )
    return MarkerDetector(
        board_cfg["layout"],
        board_cfg["square_length"],
        board_cfg["marker_length"],
        aruco_dict,
    )


def _extract_frame_observations(frame_df) -> FrameObservations:
    """Build observation tensors from a DataFrame group for one frame."""
    frame_index = int(frame_df.iloc[0]["frame_index"])

    # charuco corners — deduplicate by charuco_id
    seen_charuco = {}
    for _, row in frame_df.iterrows():
        cid = int(row["charuco_id"])
        if cid not in seen_charuco:
            seen_charuco[cid] = (row["corner_3d"], row["corner_2d"])

    if seen_charuco:
        c3d = np.stack([v[0] for v in seen_charuco.values()])
        c2d = np.stack([v[1] for v in seen_charuco.values()])
    else:
        c3d = np.zeros((0, 3))
        c2d = np.zeros((0, 2))

    # aruco marker corners — deduplicate by marker_id
    seen_markers = {}
    for _, row in frame_df.iterrows():
        for suffix in ("a", "b"):
            mid = int(row[f"marker_{suffix}_id"])
            if mid not in seen_markers:
                seen_markers[mid] = (
                    row[f"marker_{suffix}_corners_3d"],
                    row[f"marker_{suffix}_corners_2d"],
                )

    if seen_markers:
        a3d = np.concatenate([v[0] for v in seen_markers.values()], axis=0)
        a2d = np.concatenate([v[1] for v in seen_markers.values()], axis=0)
    else:
        a3d = np.zeros((0, 3))
        a2d = np.zeros((0, 2))

    return FrameObservations(
        frame_index=frame_index,
        charuco_pts_3d=torch.tensor(c3d, dtype=torch.float64),
        charuco_pts_2d=torch.tensor(c2d, dtype=torch.float64),
        aruco_pts_3d=torch.tensor(a3d, dtype=torch.float64),
        aruco_pts_2d=torch.tensor(a2d, dtype=torch.float64),
    )


def _init_pnp(
    obs: FrameObservations,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Run solvePnP, return (rvec, tvec, mean_error) or None."""
    pts_3d = np.concatenate([
        obs.charuco_pts_3d.numpy(),
        obs.aruco_pts_3d.numpy(),
    ], axis=0).astype(np.float64)
    pts_2d = np.concatenate([
        obs.charuco_pts_2d.numpy(),
        obs.aruco_pts_2d.numpy(),
    ], axis=0).astype(np.float64)

    if len(pts_3d) < 4:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None

    proj, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, dist_coeffs)
    proj = proj.reshape(-1, 2)
    errors = np.linalg.norm(proj - pts_2d, axis=1)
    mean_err = float(errors.mean())

    if mean_err > 50.0:
        return None
    return rvec, tvec, mean_err


def _compute_metrics(
    cam: TorchCam,
    board_poses: BoardPoses,
    frame_data: Dict[int, FrameObservations],
    fidx_to_int: Dict[int, int],
) -> dict:
    """Compute reprojection error metrics after optimization."""
    all_residuals = []
    all_types = []
    all_u = []
    all_v = []
    all_zdist = []

    dev = next(cam.parameters()).device
    with torch.no_grad():
        for fidx, obs in frame_data.items():
            fi = fidx_to_int[fidx]
            z_dist = float(board_poses.tvecs[fi, 2])
            fi_tensor = torch.full((1,), fi, dtype=torch.long, device=dev)

            for pts_3d, pts_2d, ptype in [
                (obs.charuco_pts_3d, obs.charuco_pts_2d, "charuco"),
                (obs.aruco_pts_3d, obs.aruco_pts_2d, "aruco"),
            ]:
                if pts_3d.shape[0] == 0:
                    continue
                pts_3d_dev = pts_3d.to(dev)
                pts_2d_dev = pts_2d.to(dev)
                fi_batch = fi_tensor.expand(pts_3d_dev.shape[0])
                cam_pts = board_poses.transform(fi_batch, pts_3d_dev)
                proj = cam.project_camera_points(cam_pts)
                res = (proj - pts_2d_dev).norm(dim=1)

                all_residuals.append(res.cpu().numpy())
                all_types.extend([ptype] * len(res))
                all_u.append(proj[:, 0].cpu().numpy())
                all_v.append(proj[:, 1].cpu().numpy())
                all_zdist.extend([z_dist] * len(res))

    residuals = np.concatenate(all_residuals)
    types = np.array(all_types)
    u_coords = np.concatenate(all_u)
    v_coords = np.concatenate(all_v)
    z_dists = np.array(all_zdist)

    def rmse(r):
        return float(np.sqrt((r ** 2).mean())) if len(r) > 0 else 0.0

    charuco_mask = types == "charuco"
    aruco_mask = types == "aruco"

    # image region (3x3 grid)
    u_frac = u_coords / cam.width
    v_frac = v_coords / cam.height
    center_mask = (u_frac > 1/3) & (u_frac < 2/3) & (v_frac > 1/3) & (v_frac < 2/3)
    corner_mask = ((u_frac <= 1/3) | (u_frac >= 2/3)) & ((v_frac <= 1/3) | (v_frac >= 2/3))
    edge_mask = ~center_mask & ~corner_mask

    # distance terciles
    if len(z_dists) > 0:
        t1, t2 = np.percentile(z_dists, [33.3, 66.6])
        near_mask = z_dists <= t1
        mid_mask = (z_dists > t1) & (z_dists <= t2)
        far_mask = z_dists > t2
    else:
        near_mask = mid_mask = far_mask = np.zeros(0, dtype=bool)

    return {
        "overall_rmse": rmse(residuals),
        "charuco_rmse": rmse(residuals[charuco_mask]),
        "aruco_rmse": rmse(residuals[aruco_mask]),
        "by_image_region": {
            "center": rmse(residuals[center_mask]),
            "edge": rmse(residuals[edge_mask]),
            "corner": rmse(residuals[corner_mask]),
        },
        "by_distance": {
            "near": rmse(residuals[near_mask]),
            "mid": rmse(residuals[mid_mask]),
            "far": rmse(residuals[far_mask]),
        },
        "num_frames": len(frame_data),
        "num_charuco_points": int(charuco_mask.sum()),
        "num_aruco_points": int(aruco_mask.sum()),
        "intrinsics": cam.to_dict()["intrinsics"],
    }


def _print_report(metrics: dict):
    print("\n" + "=" * 60)
    print("  INTRINSIC CALIBRATION REPORT")
    print("=" * 60)
    print(f"  Frames used:          {metrics['num_frames']}")
    print(f"  Charuco points:       {metrics['num_charuco_points']}")
    print(f"  ArUco points:         {metrics['num_aruco_points']}")
    print()
    print(f"  Overall RMSE:         {metrics['overall_rmse']:.4f} px")
    print(f"  Charuco RMSE:         {metrics['charuco_rmse']:.4f} px")
    print(f"  ArUco RMSE:           {metrics['aruco_rmse']:.4f} px")
    print()
    print("  By image region:")
    for region, val in metrics["by_image_region"].items():
        print(f"    {region:>8s}:           {val:.4f} px")
    print()
    print("  By distance to camera:")
    for dist, val in metrics["by_distance"].items():
        print(f"    {dist:>8s}:           {val:.4f} px")
    print()
    intr = metrics["intrinsics"]
    print("  Calibrated intrinsics:")
    print(f"    fx:                 {intr['fx']:.2f}")
    print(f"    fy:                 {intr['fy']:.2f}")
    print(f"    cx:                 {intr['cx']:.2f}")
    print(f"    cy:                 {intr['cy']:.2f}")
    dc = intr["dist_coeffs"]
    print(f"    k1:                 {dc[0]:.6f}")
    print(f"    k2:                 {dc[1]:.6f}")
    print(f"    t1:                 {dc[2]:.6f}")
    print(f"    t2:                 {dc[3]:.6f}")
    if len(dc) > 4:
        print(f"    k3:                 {dc[4]:.6f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_intrinsic_calibration(
    recording_dir: str,
    camera_serial: str,
    board_name: Optional[str] = None,
    num_epochs: int = 500,
    max_workers: int = 40,
    device: Optional[str] = None,
    lr_cam: float = 0.5,
    lr_dist: float = 0.01,
    lr_poses: float = 0.003,
    huber_charuco_delta: float = 0.5,
    huber_aruco_delta: float = 3.0,
    batch_size: int = 100000,
    aruco_weight: float = 1.0,
    detections_df: Optional[pd.DataFrame] = None,
    pnp_reproj_threshold: float = 3.0,
) -> dict:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Using device: {dev}")

    # -- Step A: Detection --------------------------------------------------
    if detections_df is not None:
        df = detections_df
    else:
        detector = _load_board(board_name)
        df = detector.detect_recording(recording_dir, camera_serial, camera_serial,
                                       max_workers=max_workers)
    if df.empty:
        raise RuntimeError("No detections found")

    # -- Step B: Initial camera ---------------------------------------------
    first_frame = sorted(
        d for d in os.listdir(os.path.join(recording_dir, camera_serial))
        if os.path.isdir(os.path.join(recording_dir, camera_serial, d))
    )[0]
    sample_img = cv2.imread(
        os.path.join(recording_dir, camera_serial, first_frame, "frame.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    h, w = sample_img.shape
    focal = float(max(w, h))

    intrinsics = {
        "fx": focal, "fy": focal,
        "cx": w / 2.0, "cy": h / 2.0,
        "width": w, "height": h,
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
    }
    cam = TorchCam(camera_serial, intrinsics, torch.eye(4, dtype=torch.float64))
    cam.to(dev)
    cam.pose.requires_grad_(False)

    camera_matrix = np.array([
        [focal, 0, w / 2.0],
        [0, focal, h / 2.0],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_np = np.zeros(4, dtype=np.float64)

    # -- Step C: PnP per frame ----------------------------------------------
    frame_groups = {fidx: grp for fidx, grp in df.groupby("frame_index")}
    frame_data: Dict[int, FrameObservations] = {}
    rvec_list = []
    tvec_list = []
    fidx_order = []
    pnp_errors = []

    print(f"\nRunning PnP on {len(frame_groups)} frames (reproj threshold={pnp_reproj_threshold:.1f} px)...")
    rejected_pnp = 0
    for fidx, grp in frame_groups.items():
        obs = _extract_frame_observations(grp)
        result = _init_pnp(obs, camera_matrix, dist_np)
        if result is None:
            continue
        rvec, tvec, err = result
        if err > pnp_reproj_threshold:
            rejected_pnp += 1
            continue
        pnp_errors.append(err)
        frame_data[fidx] = obs
        rvec_list.append(rvec.flatten())
        tvec_list.append(tvec.flatten())
        fidx_order.append(fidx)

    if not frame_data:
        raise RuntimeError("All frames rejected by PnP (error > 50px)")

    print(f"PnP: {len(frame_data)}/{len(frame_groups)} frames accepted, "
          f"{rejected_pnp} rejected by reproj threshold "
          f"(mean error {np.mean(pnp_errors):.2f} px)")

    # -- Step D: Build dataset and dataloader --------------------------------
    fidx_to_int = {fidx: i for i, fidx in enumerate(fidx_order)}
    board_poses = BoardPoses(np.stack(rvec_list), np.stack(tvec_list))
    board_poses.to(dev)

    # Flatten all observations: (frame_int, pt_3d[3], pt_2d[2], type)
    # type: 0=charuco, 1=aruco
    all_obs_rows = []
    for fidx, obs in frame_data.items():
        fi = fidx_to_int[fidx]
        for pts_3d, pts_2d, ptype in [
            (obs.charuco_pts_3d, obs.charuco_pts_2d, 0),
            (obs.aruco_pts_3d, obs.aruco_pts_2d, 1),
        ]:
            n = pts_3d.shape[0]
            if n == 0:
                continue
            rows = torch.zeros(n, 7, dtype=torch.float64)
            rows[:, 0] = fi
            rows[:, 1:4] = pts_3d
            rows[:, 4:6] = pts_2d
            rows[:, 6] = ptype
            all_obs_rows.append(rows)

    all_obs = torch.cat(all_obs_rows, dim=0).to(dev)
    dataset = torch.utils.data.TensorDataset(all_obs)
    n_charuco = int((all_obs[:, 6] == 0).sum())
    n_aruco = int((all_obs[:, 6] == 1).sum())
    print(f"\nTotal observations: {len(all_obs)} (charuco: {n_charuco}, aruco: {n_aruco})")

    huber_charuco = nn.HuberLoss(reduction="sum", delta=huber_charuco_delta)
    huber_aruco = nn.HuberLoss(reduction="sum", delta=huber_aruco_delta)

    optimizer = torch.optim.Adam([
        {"params": [cam.fx, cam.fy, cam.cx, cam.cy], "lr": lr_cam},
        {"params": [cam.dist_coeffs], "lr": lr_dist},
        {"params": board_poses.parameters(), "lr": lr_poses},
    ])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_rmse = float("inf")
    best_cam_state = None
    best_poses_state = None

    print(f"Optimizing ({num_epochs} epochs, {len(loader)} batches/epoch, bs={batch_size})...")
    for epoch in range(num_epochs):
        epoch_sq_err = 0.0
        epoch_n = 0
        for (batch,) in loader:
            optimizer.zero_grad()

            fi = batch[:, 0].long()
            pts_3d = batch[:, 1:4]
            pts_2d = batch[:, 4:6]
            ptype = batch[:, 6]

            # transform all points to camera frame in one batched op
            cam_pts = board_poses.transform(fi, pts_3d)
            proj = cam.project_camera_points(cam_pts)

            # split loss by type (aruco weighted down)
            charuco_mask = ptype == 0
            aruco_mask = ptype == 1
            loss = torch.tensor(0.0, dtype=torch.float64, device=dev)
            if charuco_mask.any():
                loss = loss + huber_charuco(proj[charuco_mask], pts_2d[charuco_mask])
            if aruco_mask.any():
                loss = loss + aruco_weight * huber_aruco(proj[aruco_mask], pts_2d[aruco_mask])

            loss.backward()
            optimizer.step()

            # track true RMSE (no grad needed, detach)
            with torch.no_grad():
                sq_err = ((proj - pts_2d) ** 2).sum().item()
                epoch_sq_err += sq_err
                epoch_n += len(batch)

        rmse = np.sqrt(epoch_sq_err / epoch_n)
        if rmse < best_rmse:
            best_rmse = rmse
            best_cam_state = {k: v.clone() for k, v in cam.state_dict().items()}
            best_poses_state = {k: v.clone() for k, v in board_poses.state_dict().items()}

        if epoch % 50 == 0 or epoch == num_epochs - 1:
            print(f"  epoch {epoch:>4d}  rmse={rmse:.4f} px  best={best_rmse:.4f} px  "
                  f"fx={cam.fx.item():.1f} fy={cam.fy.item():.1f} "
                  f"cx={cam.cx.item():.1f} cy={cam.cy.item():.1f}")

    # restore best checkpoint
    cam.load_state_dict(best_cam_state)
    board_poses.load_state_dict(best_poses_state)
    print(f"\nRestored best model (rmse={best_rmse:.4f} px)")

    # -- Step E: Metrics ----------------------------------------------------
    metrics = _compute_metrics(cam, board_poses, frame_data, fidx_to_int)
    _print_report(metrics)

    return {
        "metrics": metrics,
        "cam": cam,
        "board_poses": board_poses,
        "frame_data": frame_data,
        "fidx_to_int": fidx_to_int,
        "fidx_order": fidx_order,
        "recording_dir": recording_dir,
        "camera_serial": camera_serial,
    }


if __name__ == "__main__":
    import sys
    rec_dir = sys.argv[1] if len(sys.argv) > 1 else "scrap/20260321_180029"
    serial = sys.argv[2] if len(sys.argv) > 2 else "DA9128029"
    run_intrinsic_calibration(rec_dir, serial)
