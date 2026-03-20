"""
StudioController — orchestrator for synchronized multi-camera recording.

Coordinates Arduino (PWM trigger), PicoScope (edge timestamps), and cameras
(hardware Line0 trigger) into a single start/stop lifecycle.
"""

from __future__ import annotations

import logging
import threading

from arduino_controller.client import TriggerController
from camera_manager import CameraManager
from pico_controller import PicoController, SyncResult, SAMPLE_INTERVAL_US
from recording_metadata import SyncMetadata

log = logging.getLogger("studio_controller")


class StudioController:
    def __init__(self, camera_manager: CameraManager):
        self._cam_mgr = camera_manager
        self._trigger = TriggerController()
        self._pico = PicoController()
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

    # ── Recording lifecycle ──────────────────────────────────────────────────

    def start_recording(self, frequency_hz: float = 30.0,
                        camera_ids: list[str] | None = None) -> str:
        with self._lock:
            if self._recording:
                raise RuntimeError("Already recording")

            warnings = []

            # 0. Ensure devices are connected
            self.connect()

            # 1. Ensure Arduino is stopped
            self._ensure_sync_stopped()

            # 2. Determine target cameras
            if camera_ids is None:
                camera_ids = [ip for ip, cam in self._cam_mgr._cameras.items()
                              if cam.is_grabbing]
            if not camera_ids:
                raise RuntimeError("No cameras available for recording")

            # 3. Arm cameras with hardware Line0 trigger
            for cid in camera_ids:
                cam = self._cam_mgr.get_camera(cid)
                cam.configure(trigger_mode="line0")

            # 4. Start camera recording (spawns saver threads)
            try:
                recording_id = self._cam_mgr.start_recording(camera_ids)
            except Exception:
                # Rollback: disarm cameras
                for cid in camera_ids:
                    try:
                        self._cam_mgr.get_camera(cid).configure(trigger_mode=False)
                    except Exception as e:
                        log.warning("Rollback disarm %s: %s", cid, e)
                raise

            # 5. Start PicoScope tracking
            try:
                log.info("Starting PicoScope tracking at %.1f Hz", frequency_hz)
                self._pico.start_tracking(frequency_hz)
            except Exception:
                # Rollback: stop camera recording + disarm
                try:
                    self._cam_mgr.stop_recording()
                except Exception as e:
                    log.warning("Rollback stop_recording: %s", e)
                for cid in camera_ids:
                    try:
                        self._cam_mgr.get_camera(cid).configure(trigger_mode=False)
                    except Exception as e:
                        log.warning("Rollback disarm %s: %s", cid, e)
                raise

            # 6. Start Arduino trigger
            try:
                log.info("Starting Arduino at %.1f Hz", frequency_hz)
                ok = self._trigger.start(frequency_hz)
                log.info("Arduino start result: %s", ok)
                if not ok:
                    raise RuntimeError("Arduino start returned failure")
            except Exception:
                # Rollback: stop pico + stop camera recording + disarm
                try:
                    self._pico.stop_tracking()
                except Exception as e:
                    log.warning("Rollback stop_tracking: %s", e)
                try:
                    self._cam_mgr.stop_recording()
                except Exception as e:
                    log.warning("Rollback stop_recording: %s", e)
                for cid in camera_ids:
                    try:
                        self._cam_mgr.get_camera(cid).configure(trigger_mode=False)
                    except Exception as e:
                        log.warning("Rollback disarm %s: %s", cid, e)
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

            # 1. Stop Arduino
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

            # 2. Stop PicoScope (before cameras so we capture all edges)
            sync_result: SyncResult | None = None
            try:
                sync_result = self._pico.stop_tracking()
                log.info("PicoScope result: edges=%d, total_samples=%d, timing_ok=%s",
                         sync_result.edge_count, sync_result.total_samples, sync_result.timing_ok)
            except Exception as e:
                warnings.append(f"PicoScope stop_tracking error: {e}")

            # 3. Build sync metadata
            sync_meta: SyncMetadata | None = None
            if sync_result is not None:
                sync_meta = SyncMetadata(
                    frequency_hz=self._frequency_hz,
                    edge_count=sync_result.edge_count,
                    timing_ok=sync_result.timing_ok,
                    worst_error_us=sync_result.worst_error_us,
                    fault_intervals=sync_result.fault_intervals,
                    interval_stats=sync_result.interval_stats,
                    timestamps_us=sync_result.timestamps_us,
                    sample_interval_us=SAMPLE_INTERVAL_US,
                    total_samples=sync_result.total_samples,
                )

                if not sync_result.timing_ok:
                    warnings.append(
                        f"Sync timing fault: worst error {sync_result.worst_error_us:.1f} us"
                    )

            # 4. Stop camera recording (drains saver threads, writes metadata)
            result = {}
            try:
                result = self._cam_mgr.stop_recording(
                    sync=sync_meta,
                    warnings=warnings if warnings else None,
                )
            except Exception as e:
                warnings.append(f"Camera stop_recording error: {e}")

            # 5. Disarm cameras (back to free run)
            for cid in self._camera_ids:
                try:
                    self._cam_mgr.get_camera(cid).configure(trigger_mode=False)
                except Exception as e:
                    warnings.append(f"Disarm {cid}: {e}")

            self._recording = False
            self._camera_ids = []

            if sync_result is not None:
                result["sync"] = {
                    "edge_count": sync_result.edge_count,
                    "timing_ok": sync_result.timing_ok,
                    "worst_error_us": sync_result.worst_error_us,
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
