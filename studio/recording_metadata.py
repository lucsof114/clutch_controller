"""
RecordingMetadata — standardised metadata written alongside every recording.
Saved as metadata.json in the recording root directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@dataclass
class CameraParameters:
    """1:1 map with the UI parameter controls."""
    exposure_us: float = 0.0
    gain_db: float = 0.0
    frame_rate: float = 0.0


@dataclass
class CameraRecordingInfo:
    ip: str
    model: str
    serial: str
    frame_count: int = 0
    total_size_bytes: int = 0
    parameters: CameraParameters = field(default_factory=CameraParameters)


@dataclass
class SyncMetadata:
    target_frequency: float           # requested trigger frequency (Hz)
    recording_frequency: float        # actual frequency: 1/mean(diff(timestamps))
    sync_count: int                   # len(timestamps)
    timestamps: list = field(default_factory=list)  # edge timestamps in microseconds


@dataclass
class RecordingMetadata:
    recording_id: str
    created_time: str                            # ISO 8601 UTC — when start_recording was called
    tags: list[str] = field(default_factory=list) # user-assigned labels e.g. ["intrinsic", "data"]
    start_time: Optional[str] = None             # ISO 8601 UTC — first frame received
    end_time: Optional[str] = None               # ISO 8601 UTC — last frame received
    duration_seconds: Optional[float] = None
    cameras: list[CameraRecordingInfo] = field(default_factory=list)
    sync: SyncMetadata | None = None
    warnings: list[str] = field(default_factory=list)

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def start(cls, recording_id: str, camera_infos: list[CameraRecordingInfo],
              tags: list[str] | None = None) -> "RecordingMetadata":
        return cls(
            recording_id=recording_id,
            created_time=_now_iso(),
            tags=tags or [],
            cameras=camera_infos,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def finalise(self, frame_counts: dict[str, int], size_bytes: dict[str, int],
                 first_frame_time: str | None = None, last_frame_time: str | None = None) -> None:
        """Call when recording stops. Fills timing and per-camera stats."""
        self.start_time = first_frame_time
        self.end_time = last_frame_time
        if first_frame_time and last_frame_time:
            t0 = datetime.fromisoformat(first_frame_time)
            t1 = datetime.fromisoformat(last_frame_time)
            self.duration_seconds = round((t1 - t0).total_seconds(), 3)
        for cam in self.cameras:
            cam.frame_count = frame_counts.get(cam.serial, 0)
            cam.total_size_bytes = size_bytes.get(cam.serial, 0)

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, directory: Path) -> Path:
        path = directory / "metadata.json"
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, directory: Path) -> "RecordingMetadata":
        d = json.loads((directory / "metadata.json").read_text())
        raw_cameras = d.pop("cameras", [])
        cameras = []
        for c in raw_cameras:
            params_data = c.pop("parameters", {})
            if isinstance(params_data, dict):
                params = CameraParameters(**params_data)
            else:
                params = params_data
            cameras.append(CameraRecordingInfo(**c, parameters=params))
        sync_data = d.pop("sync", None)
        sync = SyncMetadata(**sync_data) if sync_data else None
        warnings = d.pop("warnings", [])
        return cls(**d, cameras=cameras, sync=sync, warnings=warnings)
