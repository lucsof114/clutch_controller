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
class CameraRecordingInfo:
    camera_id: str
    model: str
    serial: str
    frame_count: int = 0
    total_size_bytes: int = 0
    parameters: dict = field(default_factory=dict)


@dataclass
class RecordingMetadata:
    recording_id: str
    start_time: str                          # ISO 8601 UTC
    end_time: Optional[str] = None           # ISO 8601 UTC, set on stop
    duration_seconds: Optional[float] = None
    cameras: list[CameraRecordingInfo] = field(default_factory=list)

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def start(cls, recording_id: str, camera_infos: list[CameraRecordingInfo]) -> "RecordingMetadata":
        return cls(
            recording_id=recording_id,
            start_time=_now_iso(),
            cameras=camera_infos,
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def finalise(self, frame_counts: dict[str, int], size_bytes: dict[str, int]) -> None:
        """Call when recording stops. Fills end_time, duration, per-camera stats."""
        self.end_time = _now_iso()
        t0 = datetime.fromisoformat(self.start_time)
        t1 = datetime.fromisoformat(self.end_time)
        self.duration_seconds = round((t1 - t0).total_seconds(), 3)
        for cam in self.cameras:
            cam.frame_count = frame_counts.get(cam.camera_id, 0)
            cam.total_size_bytes = size_bytes.get(cam.camera_id, 0)

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
        cameras = [CameraRecordingInfo(**c) for c in d.pop("cameras", [])]
        return cls(**d, cameras=cameras)
