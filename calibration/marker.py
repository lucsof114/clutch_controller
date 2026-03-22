import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd


def _segment_distance(pt, a, b):
    """Distance from point pt to the line segment a-b."""
    ab = b - a
    len_sq = float(ab @ ab)
    if len_sq < 1e-9:
        return float(np.linalg.norm(pt - a))
    t = max(0.0, min(1.0, float((pt - a) @ ab / len_sq)))
    proj = a + t * ab
    return float(np.linalg.norm(pt - proj))


class MarkerDetector:
    """Detects ArUco markers and charuco board corners, maps each board corner
    to a pair of ArUco IDs via perpendicular distance to centroid lines."""

    def __init__(
        self,
        board_layout: list,
        square_length: float,
        marker_length: float,
        aruco_dict=None,
    ):
        """
        board_layout: 2D list, e.g.
            [[-1, 24, -1, 25, -1],
             [26, -1, 27, -1, 28],
             [-1, 29, -1, 30, -1],
             [31, -1, 32, -1, 33],
             [-1, 34, -1, 35, -1]]
        square_length: checkerboard square side in metres
        marker_length: ArUco marker side in metres
        aruco_dict: cv2.aruco dictionary (default DICT_4X4_250)
        """
        self.layout = np.array(board_layout, dtype=np.int32)
        self.rows, self.cols = self.layout.shape
        self.square_length = square_length
        self.marker_length = marker_length

        if aruco_dict is None:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_dict = aruco_dict

        # extract marker IDs in row-major order (for CharucoBoard constructor)
        self.marker_ids = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.layout[r, c] != -1:
                    self.marker_ids.append(int(self.layout[r, c]))

        # build OpenCV board and detector
        self.board = cv2.aruco.CharucoBoard(
            (self.cols, self.rows),
            square_length,
            marker_length,
            aruco_dict,
            np.array(self.marker_ids, dtype=np.int32),
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

        # derive corner-to-marker-pair map and 3D positions
        # inner corners: (rows-1) x (cols-1) grid
        self._corner_pairs = {}   # charuco_id -> (marker_a, marker_b)
        self._corner_3d = {}      # charuco_id -> np.array(3,)

        center_x = self.cols * square_length / 2.0
        center_y = self.rows * square_length / 2.0

        corner_id = 0
        for r in range(self.rows - 1):
            for c in range(self.cols - 1):
                # 4 surrounding squares
                neighbours = [
                    self.layout[r, c],
                    self.layout[r, c + 1],
                    self.layout[r + 1, c],
                    self.layout[r + 1, c + 1],
                ]
                markers = [int(v) for v in neighbours if v != -1]

                if len(markers) == 2:
                    self._corner_pairs[corner_id] = tuple(sorted(markers))

                # 3D position: board center origin, x-left, y-up, z=0
                x_board = (c + 1) * square_length
                y_board = (r + 1) * square_length
                self._corner_3d[corner_id] = np.array([
                    center_x - x_board,   # x: positive left
                    center_y - y_board,    # y: positive up
                    0.0,
                ], dtype=np.float64)

                corner_id += 1

        # aruco marker 3D corners: aruco_id -> (4, 3) array
        # OpenCV corner order: top-left, top-right, bottom-right, bottom-left
        # In board coords (x-left, y-up): TL=(+h,+h), TR=(-h,+h), BR=(-h,-h), BL=(+h,-h)
        half = marker_length / 2.0
        self._marker_corners_3d = {}
        for r in range(self.rows):
            for c in range(self.cols):
                mid = int(self.layout[r, c])
                if mid == -1:
                    continue
                # cell center in board-centered coords
                cx = center_x - (c + 0.5) * square_length
                cy = center_y - (r + 0.5) * square_length
                self._marker_corners_3d[mid] = np.array([
                    [cx + half, cy + half, 0.0],  # top-left
                    [cx - half, cy + half, 0.0],  # top-right
                    [cx - half, cy - half, 0.0],  # bottom-right
                    [cx + half, cy - half, 0.0],  # bottom-left
                ], dtype=np.float64)

        # set of all valid marker pairs for line-segment matching
        self._valid_pairs = set(self._corner_pairs.values())

    def detect(self, image: np.ndarray, frame_index: int, camera_id: str) -> List[dict]:
        """Detect board corners in a single image and assign marker pairs.

        Returns list of dicts, one per detected board corner.
        """
        if image is None:
            return []

        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                self.detector.detectBoard(image)
        except Exception:
            return []

        if marker_ids is None or len(marker_ids) == 0:
            return []
        if charuco_ids is None or len(charuco_ids) == 0:
            return []

        # build marker lookup: aruco_id -> (centroid, quad)
        marker_info = {}
        for corners, mid in zip(marker_corners, marker_ids.flatten()):
            quad = corners.reshape(4, 2)
            marker_info[int(mid)] = {
                "centroid": quad.mean(axis=0),
                "corners": quad,
            }

        # build line segments for each valid pair where both markers detected
        pair_lines = []
        for pair in self._valid_pairs:
            a_id, b_id = pair
            if a_id in marker_info and b_id in marker_info:
                pair_lines.append((
                    pair,
                    marker_info[a_id]["centroid"],
                    marker_info[b_id]["centroid"],
                ))

        if not pair_lines:
            return []

        # assign each charuco corner to closest marker pair (greedy)
        results = []
        for pt, cid in zip(charuco_corners, charuco_ids.flatten()):
            cid = int(cid)
            pt_2d = pt.flatten()

            best_dist = np.inf
            best_pair = None
            for pair, ca, cb in pair_lines:
                d = _segment_distance(pt_2d, ca, cb)
                if d < best_dist:
                    best_dist = d
                    best_pair = pair

            if best_pair is None:
                continue

            a_id, b_id = best_pair
            results.append({
                "frame_index": frame_index,
                "camera_id": camera_id,
                "charuco_id": cid,
                "corner_2d": pt_2d,
                "corner_3d": self._corner_3d.get(cid, np.zeros(3)),
                "marker_pair": best_pair,
                "marker_a_id": a_id,
                "marker_b_id": b_id,
                "marker_a_corners_2d": marker_info[a_id]["corners"],
                "marker_b_corners_2d": marker_info[b_id]["corners"],
                "marker_a_corners_3d": self._marker_corners_3d[a_id],
                "marker_b_corners_3d": self._marker_corners_3d[b_id],
            })

        return results

    def detect_recording(
        self,
        recording_dir: str,
        camera_serial: str,
        camera_id: str,
        max_workers: int = 40,
    ) -> pd.DataFrame:
        """Run detection across all frames in a recording using a threadpool."""
        frames_dir = os.path.join(recording_dir, camera_serial)
        frame_dirs = sorted(glob(os.path.join(frames_dir, "*")))
        print(f"MarkerDetector: {len(frame_dirs)} frames | camera={camera_id} | workers={max_workers}")

        def _process(frame_dir):
            img_path = os.path.join(frame_dir, "frame.png")
            if not os.path.isfile(img_path):
                return []
            frame_idx = int(os.path.basename(frame_dir))
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            return self.detect(img, frame_idx, camera_id)

        all_rows = []
        detected = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process, d): d for d in frame_dirs}
            for i, fut in enumerate(as_completed(futures), 1):
                rows = fut.result()
                if rows:
                    detected += 1
                    all_rows.extend(rows)
                if i % 200 == 0 or i == len(futures):
                    print(f"  [{i}/{len(futures)}]  detected: {detected}  rows: {len(all_rows)}")

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        df = df.sort_values(["frame_index", "charuco_id"]).reset_index(drop=True)
        return df
