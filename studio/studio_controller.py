"""
StudioController — orchestrator for synchronized multi-camera recording.

Coordinates Arduino (PWM trigger), PicoScope (edge timestamps), and cameras
(hardware Line0 trigger) into a single start/stop lifecycle.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time

from studio.arduino_controller.client import TriggerController
from studio.camera_manager import CameraManager
import numpy as np

from studio.pico_controller import PicoController, PicoResult
from studio.recording_manager import RecordingManager
from studio.recording_metadata import CameraParameters, CameraRecordingInfo, SyncMetadata

log = logging.getLogger("studio_controller")


class StudioController:
    def __init__(self):
        self._cam_mgr = CameraManager()
        self._trigger = TriggerController()
        self._pico = PicoController()
        self._recording_manager = RecordingManager()
        self._lock = threading.Lock()
        self._recording = False
        self._frequency_hz: float = 0.0
        self._camera_ids: list[str] = []

    # ── Connection lifecycle ─────────────────────────────────────────────────

    def connect(self):
        """Connect Arduino + open PicoScope."""
        if self._trigger.ser and self._trigger.ser.is_open and self._pico.is_open:
            log.info("Studio already connected")
            return
        if not self._trigger.connect():
            raise RuntimeError("Arduino connection failed")
        self._pico.open()
        log.info("Studio connected (Arduino + PicoScope)")

    def disconnect(self):
        """Tear down everything safely."""
        warnings = []

        if self._recording:
            try:
                self.stop_recording()
            except Exception as e:
                warnings.append(f"stop_recording during disconnect: {e}")

        try:
            self._pico.close()
        except Exception as e:
            warnings.append(f"PicoScope close: {e}")

        try:
            self._trigger.disconnect()
        except Exception as e:
            warnings.append(f"Arduino disconnect: {e}")

        if warnings:
            log.warning("Studio disconnect warnings: %s", warnings)
        log.info("Studio disconnected")

    def shutdown(self):
        """Full teardown: disconnect devices, shut down cameras."""
        self.disconnect()
        self._cam_mgr.shutdown()

    # ── Recording lifecycle ──────────────────────────────────────────────────

    def start_recording(self, frequency_hz: float = 30.0,
                        camera_ids: list[str] | None = None,
                        tags: list[str] | None = None) -> str:
        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording")

            # 0. Ensure devices are connected
            self.connect()

            # 1. Ensure Arduino is stopped
            self._ensure_sync_stopped()

            # 2. Determine target cameras (keyed by serial)
            if camera_ids is None:
                camera_ids = [serial for serial, cam in self._cam_mgr._cameras.items()
                              if cam.is_grabbing]
            if not camera_ids:
                raise RuntimeError("No cameras available for recording")

            # 3. Arm cameras with hardware Line0 trigger
            for cid in camera_ids:
                cam = self._cam_mgr.get_camera(cid)
                cam.configure(trigger_mode="line0")

            # Flush the one spurious free-run frame emitted on Line0 arm transition.
            # _record_queue is still None here so it is silently discarded.
            time.sleep(0.3)

            # 4. Build recording metadata from current camera state
            camera_infos = []
            for cid in camera_ids:
                cam = self._cam_mgr.get_camera(cid)
                raw = cam.get_parameters()
                if "ExposureTime" not in raw or "Gain" not in raw or "AcquisitionFrameRate" not in raw:
                    raise RuntimeError(f"Camera {cid} missing required parameters: {list(raw.keys())}")
                camera_infos.append(CameraRecordingInfo(
                    ip=cam.ip,
                    model=cam.model,
                    serial=cam.serial,
                    parameters=CameraParameters(
                        exposure_us=raw["ExposureTime"]["current"],
                        gain_db=raw["Gain"]["current"],
                        frame_rate=raw["AcquisitionFrameRate"]["current"],
                    ),
                ))

            # 5. Start recording (spawns saver threads)
            try:
                recording_id, queues = self._recording_manager.start(camera_ids, camera_infos, tags=tags)
                for cid, q in queues.items():
                    self._cam_mgr.get_camera(cid)._record_queue = q
            except Exception:
                self._rollback_start(camera_ids, pico=False, recording=False)
                raise

            # 6. Start PicoScope tracking
            try:
                log.info("Starting PicoScope tracking at %.1f Hz", frequency_hz)
                self._pico.start_tracking(frequency_hz)
            except Exception:
                self._rollback_start(camera_ids, pico=False, recording=True)
                raise

            # 7. Start Arduino trigger
            try:
                log.info("Starting Arduino at %.1f Hz", frequency_hz)
                ok = self._trigger.start(frequency_hz)
                log.info("Arduino start result: %s", ok)
                if not ok:
                    raise RuntimeError("Arduino start returned failure")
            except Exception:
                self._rollback_start(camera_ids, pico=True, recording=True)
                raise

            self._recording = True
            self._frequency_hz = frequency_hz
            self._camera_ids = list(camera_ids)
            log.info("Studio recording started: %s @ %.1f Hz, cameras=%s",
                     recording_id, frequency_hz, camera_ids)
            return recording_id

    def stop_recording(self) -> dict:
        with self._lock:
            if not self._recording:
                raise RuntimeError("Not recording")

            warnings: list[str] = []

            # 1. Stop Arduino (no more triggers from here)
            try:
                ok = self._trigger.stop()
                if not ok:
                    # Retry once
                    ok = self._trigger.stop()
                    if not ok:
                        warnings.append("Arduino stop failed after retry")
                # Verify via status
                try:
                    st = self._trigger.status()
                    if st.get("running"):
                        warnings.append("Arduino still reports running after stop")
                except Exception as e:
                    warnings.append(f"Arduino status check failed: {e}")
            except Exception as e:
                warnings.append(f"Arduino stop error: {e}")

            time.sleep(0.4)

            # 2. Stop PicoScope — all real edges are now accounted for
            pico_result: PicoResult | None = None
            try:
                pico_result = self._pico.stop_tracking()
                log.info("PicoScope result: %d edges, %d total samples",
                         len(pico_result.timestamps_us), pico_result.total_samples)
            except Exception as e:
                warnings.append(f"PicoScope stop_tracking error: {e}")

            # 3. Analyse timestamps and build sync metadata
            sync_meta: SyncMetadata | None = None
            if pico_result is not None:
                sync_meta = self._build_sync_metadata(pico_result)

            # 4. Disconnect recording queues (camera buffers already drained above),
            #    then drain saver threads
            self._stop_recording_queues(self._camera_ids)
            result = {}
            try:
                result = self._recording_manager.stop(
                    sync=sync_meta,
                    warnings=warnings if warnings else None,
                )
            except Exception as e:
                warnings.append(f"stop_recording error: {e}")

            # 5. Disarm cameras (back to free run) — only after savers have finished
            for cid in self._camera_ids:
                try:
                    self._cam_mgr.get_camera(cid).configure(trigger_mode=False)
                except Exception as e:
                    warnings.append(f"Disarm {cid}: {e}")

            self._recording = False
            self._camera_ids = []

            if sync_meta is not None:
                result["sync"] = {
                    "target_frequency": sync_meta.target_frequency,
                    "recording_frequency": sync_meta.recording_frequency,
                    "sync_count": sync_meta.sync_count,
                }

            if warnings:
                result["warnings"] = warnings
                log.warning("Studio stop warnings: %s", warnings)

            log.info("Studio recording stopped")
            return result

    # ── Status ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        arduino_status = {}
        try:
            if self._trigger.ser and self._trigger.ser.is_open:
                arduino_status = self._trigger.status()
        except Exception as e:
            arduino_status = {"error": str(e)}

        return {
            "arduino_connected": bool(self._trigger.ser and self._trigger.ser.is_open),
            "arduino": arduino_status,
            "pico_open": self._pico.is_open,
            "pico_tracking": self._pico.is_tracking,
            "recording": self._recording,
            "frequency_hz": self._frequency_hz if self._recording else None,
            "camera_ids": self._camera_ids if self._recording else [],
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def list_recordings(self) -> list[dict]:
        """List all recordings from the catalog."""
        catalog = self._recording_manager.catalog_dir
        if not catalog.exists():
            return []
        recordings = []
        for f in sorted(catalog.glob("*.json"), reverse=True):
            try:
                recordings.append(json.loads(f.read_text()))
            except Exception as e:
                log.warning("Failed to load catalog entry %s: %s", f.name, e)
        return recordings

    def set_camera_parameters(self, camera_id: str, **params):
        """Set parameters on a camera. Passes through to CameraManager."""
        cam = self._cam_mgr.get_camera(camera_id)

        config_map = {
            "exposure_us": "exposure_us",
            "gain": "gain",
            "frame_rate": "frame_rate",
            "use_hb": "use_hb",
            "gain_auto": "gain_auto",
            "exposure_auto": "exposure_auto",
            "trigger_mode": "trigger_mode",
        }

        config = {}
        generic = {}
        for k, v in params.items():
            if k in config_map:
                config[config_map[k]] = v
            else:
                generic[k] = v

        if config:
            cam.configure(**config)
        for k, v in generic.items():
            cam.set_parameter(k, v)

    def delete_recording(self, recording_id: str):
        """Delete a recording's data and catalog entry."""
        if "/" in recording_id or ".." in recording_id:
            raise ValueError("Invalid recording_id")

        errors = []
        rec_dir = self._recording_manager.recording_base / recording_id
        if rec_dir.exists():
            try:
                shutil.rmtree(rec_dir)
            except Exception as e:
                errors.append(str(e))

        catalog_file = self._recording_manager.catalog_dir / f"{recording_id}.json"
        if catalog_file.exists():
            try:
                catalog_file.unlink()
            except Exception as e:
                errors.append(str(e))

        if errors:
            raise RuntimeError("; ".join(errors))

    def _ensure_sync_stopped(self):
        """Send STOP to Arduino, verify it's not running."""
        try:
            self._trigger.stop()
            st = self._trigger.status()
            if st.get("running"):
                log.warning("Arduino still running after STOP — retrying")
                self._trigger.stop()
        except Exception as e:
            log.warning("_ensure_sync_stopped: %s", e)

    def _rollback_start(self, camera_ids: list[str], *, pico: bool, recording: bool):
        """Undo a partial start_recording in reverse order."""
        if pico:
            try:
                self._pico.stop_tracking()
            except Exception as e:
                log.warning("Rollback stop_tracking: %s", e)
        if recording:
            self._stop_recording_queues(camera_ids)
            try:
                self._recording_manager.stop()
            except Exception as e:
                log.warning("Rollback stop_recording: %s", e)
        for cid in camera_ids:
            try:
                self._cam_mgr.get_camera(cid).configure(trigger_mode=False)
            except Exception as e:
                log.warning("Rollback disarm %s: %s", cid, e)

    def _stop_recording_queues(self, camera_ids: list[str]):
        """Disconnect recording queues from cameras."""
        for cid in camera_ids:
            try:
                self._cam_mgr.get_camera(cid)._record_queue = None
            except Exception:
                pass

    def _build_sync_metadata(self, pico: PicoResult) -> SyncMetadata:
        """Analyse PicoScope timestamps and build SyncMetadata."""
        timestamps = pico.timestamps_us
        recording_frequency = 0.0
        if len(timestamps) > 1:
            intervals = np.diff(np.array(timestamps, dtype=float))
            recording_frequency = float(1_000_000.0 / intervals.mean())

        return SyncMetadata(
            target_frequency=self._frequency_hz,
            recording_frequency=recording_frequency,
            sync_count=len(timestamps),
            timestamps=timestamps,
        )
