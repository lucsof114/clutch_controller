"""RecordingManager — saver-thread-per-camera PNG recording system."""

import queue
import threading
import logging
from datetime import datetime
from pathlib import Path

import cv2

RECORDING_BASE = Path.home() / "Documents" / "clutch" / "clutch_db"
QUEUE_MAXSIZE = 120

log = logging.getLogger("recording_manager")


class RecordingManager:
    def __init__(self):
        self._recording_id: str | None = None
        self._queues: dict[str, queue.Queue] = {}
        self._saver_threads: dict[str, threading.Thread] = {}
        self._frame_counts: dict[str, int] = {}

    @property
    def is_recording(self) -> bool:
        return self._recording_id is not None

    def start(self, camera_ids: list[str]) -> tuple[str, dict[str, queue.Queue]]:
        if self.is_recording:
            raise RuntimeError("Already recording")

        recording_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        queues: dict[str, queue.Queue] = {}

        for camera_id in camera_ids:
            q: queue.Queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
            base_dir = RECORDING_BASE / recording_id / camera_id
            base_dir.mkdir(parents=True, exist_ok=True)

            t = threading.Thread(
                target=self._saver_loop,
                args=(camera_id, recording_id, q, base_dir),
                daemon=False,
            )
            t.start()

            queues[camera_id] = q
            self._saver_threads[camera_id] = t

        self._queues = queues
        self._recording_id = recording_id
        log.info("Recording started: %s  cameras=%s", recording_id, list(camera_ids))
        return recording_id, queues

    def stop(self) -> dict:
        if not self.is_recording:
            raise RuntimeError("Not recording")

        recording_id = self._recording_id
        self._recording_id = None  # block new pushes at manager level

        for q in self._queues.values():
            q.put(None)  # sentinel

        for camera_id, t in self._saver_threads.items():
            t.join()
            log.info("Saver thread joined for %s", camera_id)

        result = {
            "recording_id": recording_id,
            "frame_counts": dict(self._frame_counts),
        }

        self._queues.clear()
        self._saver_threads.clear()
        self._frame_counts.clear()

        log.info("Recording stopped: %s  frame_counts=%s", recording_id, result["frame_counts"])
        return result

    @staticmethod
    def _make_frame_id(n: int) -> str:
        return f"{n:06d}"

    def _saver_loop(self, camera_id: str, recording_id: str, q: queue.Queue, base_dir: Path):
        n = 0
        while True:
            try:
                frame = q.get(timeout=1.0)
            except queue.Empty:
                continue

            if frame is None:  # sentinel
                self._frame_counts[camera_id] = n
                log.info("Saver loop done for %s: %d frames written", camera_id, n)
                return

            frame_id = self._make_frame_id(n)
            frame_dir = base_dir / frame_id
            frame_dir.mkdir(exist_ok=True)
            path = str(frame_dir / "frame.png")

            ok = cv2.imwrite(path, frame)
            if not ok:
                log.error("cv2.imwrite failed for %s frame %s", camera_id, frame_id)

            n += 1
