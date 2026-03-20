"""RecordingManager — saver-thread-per-camera PNG recording system."""

from __future__ import annotations

import queue
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from recording_metadata import CameraRecordingInfo, RecordingMetadata, SyncMetadata

RECORDING_BASE = Path.home() / "Documents" / "clutch" / "clutch_db"
CATALOG_DIR    = RECORDING_BASE / "catalog"
QUEUE_MAXSIZE  = 120

log = logging.getLogger("recording_manager")


def _make_frame_id(n: int) -> str:
    """
    Sequential zero-padded frame directory name.
    Swap this function for timestamp-based logic when that system is ready.
    """
    return f"{n:06d}"


class RecordingManager:
    def __init__(self):
        self._recording_id: str | None = None
        self._metadata: Optional[RecordingMetadata] = None
        self._queues: dict[str, queue.Queue] = {}
        self._saver_threads: dict[str, threading.Thread] = {}
        self._frame_counts: dict[str, int] = {}
        self._size_bytes: dict[str, int] = {}

    @property
    def is_recording(self) -> bool:
        return self._recording_id is not None

    def start(
        self,
        camera_ids: list[str],
        camera_infos: Optional[list[CameraRecordingInfo]] = None,
    ) -> tuple[str, dict[str, queue.Queue]]:
        if self.is_recording:
            raise RuntimeError("Already recording")

        recording_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        RECORDING_BASE.mkdir(parents=True, exist_ok=True)

        if camera_infos is None:
            camera_infos = [CameraRecordingInfo(camera_id=cid, model="", serial="")
                            for cid in camera_ids]

        self._metadata = RecordingMetadata.start(recording_id, camera_infos)
        self._queues = {}
        self._saver_threads = {}
        self._frame_counts = {}
        self._size_bytes = {}

        # Build camera_id → serial lookup for directory naming
        self._serial_map = {info.camera_id: info.serial or info.camera_id
                            for info in camera_infos}
        serial_map = self._serial_map

        for cid in camera_ids:
            q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
            dir_name = serial_map.get(cid, cid)
            base_dir = RECORDING_BASE / recording_id / dir_name
            base_dir.mkdir(parents=True, exist_ok=True)

            t = threading.Thread(
                target=self._saver_loop,
                args=(cid, q, base_dir),
                daemon=False,
                name=f"saver-{cid}",
            )
            t.start()

            self._queues[cid] = q
            self._saver_threads[cid] = t

        self._recording_id = recording_id
        log.info("Recording started: %s  cameras=%s", recording_id, camera_ids)
        return recording_id, dict(self._queues)

    def stop(self, sync: SyncMetadata | None = None, warnings: list[str] | None = None) -> dict:
        if not self.is_recording:
            raise RuntimeError("Not recording")

        recording_id = self._recording_id
        self._recording_id = None  # block new pushes at manager level

        for q in self._queues.values():
            q.put(None)  # sentinel

        for cid, t in self._saver_threads.items():
            t.join()
            log.info("Saver thread joined for %s", cid)

        # Discard first frame from each camera (garbage from trigger arming)
        rec_dir = RECORDING_BASE / recording_id
        for cid in self._saver_threads:
            dir_name = self._serial_map.get(cid, cid)
            first_frame_dir = rec_dir / dir_name / _make_frame_id(0)
            if first_frame_dir.exists():
                import shutil
                try:
                    sz = sum(f.stat().st_size for f in first_frame_dir.iterdir())
                    shutil.rmtree(first_frame_dir)
                    self._frame_counts[cid] = max(self._frame_counts.get(cid, 0) - 1, 0)
                    self._size_bytes[cid] = max(self._size_bytes.get(cid, 0) - sz, 0)
                    log.info("Discarded first frame for %s", cid)
                except Exception as e:
                    log.warning("Failed to discard first frame for %s: %s", cid, e)

        # Finalise and persist metadata
        if self._metadata is not None:
            self._metadata.finalise(self._frame_counts, self._size_bytes)
            if sync is not None:
                self._metadata.sync = sync
            if warnings:
                self._metadata.warnings = warnings
            rec_dir = RECORDING_BASE / recording_id
            self._metadata.save(rec_dir)
            # Mirror to catalog as a flat <recording_id>.json
            CATALOG_DIR.mkdir(parents=True, exist_ok=True)
            import json as _json
            (CATALOG_DIR / f"{recording_id}.json").write_text(
                _json.dumps(self._metadata.to_dict(), indent=2)
            )
            log.info("Metadata written: %s/metadata.json + catalog", recording_id)

        result = {
            "recording_id": recording_id,
            "frame_counts": dict(self._frame_counts),
            "size_bytes": dict(self._size_bytes),
        }
        if self._metadata:
            result["metadata"] = self._metadata.to_dict()

        self._queues.clear()
        self._saver_threads.clear()
        self._frame_counts.clear()
        self._size_bytes.clear()
        self._metadata = None

        log.info("Recording stopped: %s  counts=%s", recording_id, result["frame_counts"])
        return result

    def _saver_loop(self, camera_id: str, q: queue.Queue, base_dir: Path):
        n = 0
        total_bytes = 0

        while True:
            try:
                frame = q.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:  # sentinel
                self._frame_counts[camera_id] = n
                self._size_bytes[camera_id] = total_bytes
                log.info("Saver done %s: %d frames, %.1f MB",
                         camera_id, n, total_bytes / 1_048_576)
                return

            frame_id = _make_frame_id(n)
            frame_dir = base_dir / frame_id
            frame_dir.mkdir(exist_ok=True)
            path = str(frame_dir / "frame.png")

            ok = cv2.imwrite(path, frame)
            if ok:
                try:
                    total_bytes += Path(path).stat().st_size
                except OSError:
                    pass
            else:
                log.error("cv2.imwrite failed: %s", path)

            q.task_done()
            n += 1
