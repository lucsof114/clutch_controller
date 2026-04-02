import torch
import torch.nn as nn
import numpy as np
from typing import Union


class TorchCam(nn.Module):

    def __init__(self, camera_id: str, intrinsics: dict, extrinsics: torch.Tensor):
        """
        intrinsics: dict with keys fx, fy, cx, cy, width, height, dist_coeffs([k1,k2,t1,t2])
        extrinsics: (4, 4) tensor — camera-to-world transform
        """
        super().__init__()
        self.camera_id = camera_id
        self.width = int(intrinsics["width"])
        self.height = int(intrinsics["height"])

        self.fx = nn.Parameter(torch.tensor(float(intrinsics["fx"]), dtype=torch.float64))
        self.fy = nn.Parameter(torch.tensor(float(intrinsics["fy"]), dtype=torch.float64))
        self.cx = nn.Parameter(torch.tensor(float(intrinsics["cx"]), dtype=torch.float64))
        self.cy = nn.Parameter(torch.tensor(float(intrinsics["cy"]), dtype=torch.float64))

        dc = intrinsics.get("dist_coeffs", [0.0, 0.0, 0.0, 0.0])
        # Support 4 or 5 coefficients: [k1, k2, t1, t2] or [k1, k2, t1, t2, k3]
        if len(dc) < 5:
            dc = list(dc) + [0.0] * (5 - len(dc))
        self.dist_coeffs = nn.Parameter(torch.tensor(dc[:5], dtype=torch.float64))  # [k1, k2, t1, t2, k3]

        self.pose = nn.Parameter(extrinsics.to(torch.float64).clone())  # cam-to-world (4x4)

    # ---- pose ------------------------------------------------------------

    def get_pose(self) -> torch.Tensor:
        return self.pose

    def set_pose(self, pose: torch.Tensor):
        self.pose.data.copy_(pose.to(torch.float64))

    # ---- projection ------------------------------------------------------

    def project_world_points(self, pts: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """(N, 3) world points -> (N, 2) pixel coords.  NaN where no projection."""
        pts = _to_f64(pts, device=self.fx.device)
        # world -> camera: invert the cam-to-world pose
        T_world_to_cam = torch.inverse(self.pose)
        R = T_world_to_cam[:3, :3]
        t = T_world_to_cam[:3, 3]
        cam_pts = (R @ pts.T).T + t  # (N, 3)
        return self.project_camera_points(cam_pts)

    def project_camera_points(self, pts: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """(N, 3) camera-frame points -> (N, 2) pixel coords.  NaN where z <= 0."""
        pts = _to_f64(pts, device=self.fx.device)
        z = pts[:, 2]

        valid = z > 0
        safe_z = torch.where(valid, z, torch.ones_like(z))

        x = pts[:, 0] / safe_z
        y = pts[:, 1] / safe_z

        # radial + tangential distortion (OpenCV model, 5 coeffs)
        k1, k2, t1, t2, k3 = (self.dist_coeffs[i] for i in range(5))
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        xd = x * radial + 2.0 * t1 * x * y + t2 * (r2 + 2.0 * x * x)
        yd = y * radial + t1 * (r2 + 2.0 * y * y) + 2.0 * t2 * x * y

        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy

        out = torch.stack([u, v], dim=1)  # (N, 2)
        nan_mask = ~valid.unsqueeze(1).expand_as(out)
        out = torch.where(nan_mask, torch.tensor(float("nan"), dtype=out.dtype, device=out.device), out)
        return out

    # ---- serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "intrinsics": {
                "fx": self.fx.item(),
                "fy": self.fy.item(),
                "cx": self.cx.item(),
                "cy": self.cy.item(),
                "width": self.width,
                "height": self.height,
                "dist_coeffs": self.dist_coeffs.detach().tolist(),
            },
            "extrinsics": self.pose.detach().tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TorchCam":
        return cls(
            camera_id=d["camera_id"],
            intrinsics=d["intrinsics"],
            extrinsics=torch.tensor(d["extrinsics"], dtype=torch.float64),
        )


def _to_f64(pts: Union[torch.Tensor, np.ndarray], device: torch.device = None) -> torch.Tensor:
    if isinstance(pts, np.ndarray):
        t = torch.from_numpy(pts).to(torch.float64)
        return t.to(device) if device is not None else t
    return pts.to(torch.float64)
