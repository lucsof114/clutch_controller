"""
CameraManager + CameraInstance — standalone multi-camera control for Hikrobot GigE cameras.
Requires MVS SDK installed at /opt/MVS with Python bindings.
"""

import sys
import queue
import threading
import time
import logging
import ctypes
from ctypes import byref, sizeof, memset, memmove, cast, POINTER, c_ubyte

import numpy as np
import cv2

MVS_SDK_PATH = "/opt/MVS/Samples/64/Python/MvImport"
sys.path.insert(0, MVS_SDK_PATH)

from MvCameraControl_class import *

from recording_manager import RecordingManager
from recording_metadata import CameraRecordingInfo, SyncMetadata

log = logging.getLogger("camera_manager")

HB_MONO8 = 0x81080001  # PixelType_Gvsp_HB_Mono8 from SDK PixelType_header.py
MONO8 = 0x01080001


def _ip_int_to_str(ip_int):
    return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}.{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"


def _decode_char(ctypes_char_array):
    raw = memoryview(ctypes_char_array).tobytes()
    null = raw.find(b'\x00')
    if null != -1:
        raw = raw[:null]
    return raw.decode('utf-8', errors='replace')


def _is_hb_pixel_format(pixel_type):
    return (pixel_type & 0x80000000) != 0


class CameraInstance:
    """Manages a single physical camera: open, configure, grab, encode."""

    def __init__(self, dev_info, ip, model, serial):
        self._dev_info = dev_info
        self.ip = ip
        self.model = model
        self.serial = serial

        self.cam = None
        self.is_open = False
        self.is_grabbing = False
        self.use_hb = True
        self.trigger_mode = False  # False = free run, True = triggered

        self._latest_frame = None   # JPEG bytes
        self._latest_raw = None     # numpy array
        self._frame_lock = threading.Lock()

        self._acq_thread = None
        self._stop_event = threading.Event()
        self._record_queue: queue.Queue | None = None

        self._acq_fps = 0.0
        self._frame_count = 0
        self._lost_packets = 0

    def open(self):
        if self.is_open:
            return
        self.cam = MvCamera()
        ret = self.cam.MV_CC_CreateHandle(self._dev_info)
        if ret != MV_OK:
            raise RuntimeError(f"CreateHandle failed for {self.ip}: 0x{ret:08X}")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != MV_OK:
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"OpenDevice failed for {self.ip}: 0x{ret:08X}")

        # Optimal packet size
        pkt = self.cam.MV_CC_GetOptimalPacketSize()
        if pkt > 0:
            self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt)

        # Zero inter-packet delay for max throughput
        self.cam.MV_CC_SetIntValue("GevSCPD", 0)

        # Free-run mode
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        # Apply pixel format — attempt HB_Mono8 for higher throughput, fall back to Mono8
        if self.use_hb:
            ret = self.cam.MV_CC_SetEnumValue("PixelFormat", HB_MONO8)
            if ret == MV_OK:
                log.info("Camera %s: PixelFormat = HB_Mono8 (compressed, higher fps)", self.ip)
            else:
                log.warning("Camera %s: HB_Mono8 not supported (0x%08X) — using Mono8; "
                            "max fps will be ~24 on 1GigE link", self.ip, ret)
                self.cam.MV_CC_SetEnumValue("PixelFormat", MONO8)
                self.use_hb = False
        else:
            self.cam.MV_CC_SetEnumValue("PixelFormat", MONO8)

        # Disable software frame rate cap so camera runs at its natural sensor maximum
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", False)

        self.is_open = True
        log.info("Opened camera %s (%s)", self.ip, self.model)

    def close(self):
        if not self.is_open:
            return
        if self.is_grabbing:
            self.stop_grabbing()
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()
        self.is_open = False
        self.cam = None
        log.info("Closed camera %s", self.ip)

    def configure(self, exposure_us=None, gain=None, frame_rate=None,
                  use_hb=None, gain_auto=None, exposure_auto=None,
                  trigger_mode=None):
        if not self.is_open:
            raise RuntimeError(f"Camera {self.ip} not open")

        needs_restart = False

        # Pixel format change requires stop/start
        if use_hb is not None and use_hb != self.use_hb:
            needs_restart = self.is_grabbing
            if needs_restart:
                self.stop_grabbing()
            fmt = HB_MONO8 if use_hb else MONO8
            ret = self.cam.MV_CC_SetEnumValue("PixelFormat", fmt)
            if ret == MV_OK:
                self.use_hb = use_hb
                log.info("Camera %s pixel format: %s", self.ip, "HB_Mono8" if use_hb else "Mono8")
            else:
                log.warning("Camera %s: set pixel format failed 0x%08X", self.ip, ret)

        # Exposure auto mode
        if exposure_auto is not None:
            # 0=Off, 1=Once, 2=Continuous
            self.cam.MV_CC_SetEnumValue("ExposureAuto", int(exposure_auto))

        # Exposure time (only effective when ExposureAuto=Off)
        if exposure_us is not None:
            self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
            self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_us))

        # Gain auto mode
        if gain_auto is not None:
            self.cam.MV_CC_SetEnumValue("GainAuto", int(gain_auto))

        # Gain (only effective when GainAuto=Off)
        if gain is not None:
            self.cam.MV_CC_SetEnumValue("GainAuto", 0)
            self.cam.MV_CC_SetFloatValue("Gain", float(gain))

        # Frame rate
        if frame_rate is not None:
            was_grabbing = self.is_grabbing
            if was_grabbing:
                self.stop_grabbing()
            self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(frame_rate))
            if was_grabbing:
                self.start_grabbing()
                needs_restart = False  # already restarted

        # Trigger mode — "line0" for hardware trigger, False for free run
        if trigger_mode is not None:
            if trigger_mode == "line0":
                self.cam.MV_CC_SetEnumValue("TriggerSource", MV_TRIGGER_SOURCE_LINE0)
            ret = self.cam.MV_CC_SetEnumValue(
                "TriggerMode", MV_TRIGGER_MODE_ON if trigger_mode else MV_TRIGGER_MODE_OFF
            )
            if ret == MV_OK:
                self.trigger_mode = trigger_mode
                log.info("Camera %s: TriggerMode = %s", self.ip,
                         "On (line0)" if trigger_mode else "Off (free run)")
            else:
                log.warning("Camera %s: set TriggerMode failed 0x%08X", self.ip, ret)

        if needs_restart:
            self.start_grabbing()

    def start_grabbing(self):
        if not self.is_open:
            raise RuntimeError(f"Camera {self.ip} not open")
        if self.is_grabbing:
            return

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            raise RuntimeError(f"StartGrabbing failed for {self.ip}: 0x{ret:08X}")

        self._stop_event.clear()
        self._acq_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self._acq_thread.start()
        self.is_grabbing = True
        log.info("Started grabbing on %s", self.ip)

    def stop_grabbing(self):
        if not self.is_grabbing:
            return
        self._stop_event.set()
        if self._acq_thread:
            self._acq_thread.join(timeout=3.0)
            self._acq_thread = None
        self.cam.MV_CC_StopGrabbing()
        self.is_grabbing = False
        log.info("Stopped grabbing on %s", self.ip)

    def _acquisition_loop(self):
        stOutFrame = MV_FRAME_OUT()
        memset(byref(stOutFrame), 0, sizeof(stOutFrame))

        # Pre-allocate HB decode buffer (worst case: full uncompressed frame)
        decode_buf_size = 2448 * 2048 * 3
        decode_buf = (c_ubyte * decode_buf_size)()

        fps_counter = 0
        fps_time = time.monotonic()

        while not self._stop_event.is_set():
            ret = self.cam.MV_CC_GetImageBuffer(stOutFrame, 1000)
            if ret != MV_OK:
                continue

            try:
                info = stOutFrame.stFrameInfo
                w, h = info.nWidth, info.nHeight

                if _is_hb_pixel_format(info.enPixelType):
                    # HB decode
                    stDecode = MV_CC_HB_DECODE_PARAM()
                    stDecode.pSrcBuf = stOutFrame.pBufAddr
                    stDecode.nSrcLen = info.nFrameLen
                    stDecode.pDstBuf = decode_buf
                    stDecode.nDstBufSize = decode_buf_size

                    dec_ret = self.cam.MV_CC_HBDecode(stDecode)
                    if dec_ret != MV_OK:
                        log.warning("HBDecode failed on %s: 0x%08X", self.ip, dec_ret)
                        continue

                    # Decoded data → numpy
                    n_bytes = stDecode.nDstBufLen
                    frame = np.ctypeslib.as_array(decode_buf, shape=(n_bytes,))
                    frame = frame[:w * h].reshape((h, w)).copy()
                else:
                    # Raw Mono8 — must copy before FreeImageBuffer
                    src = ctypes.cast(stOutFrame.pBufAddr, ctypes.POINTER(c_ubyte * (w * h))).contents
                    frame = np.frombuffer(src, dtype=np.uint8).reshape((h, w)).copy()

                # Push to recording queue if recording is active
                rq = self._record_queue
                if rq is not None:
                    try:
                        rq.put_nowait(frame)
                    except queue.Full:
                        log.debug("Record queue full for %s — frame dropped", self.ip)

                # JPEG encode for streaming
                _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg_bytes = jpeg_buf.tobytes()

                with self._frame_lock:
                    self._latest_frame = jpeg_bytes
                    self._latest_raw = frame

                self._frame_count += 1
                self._lost_packets += info.nLostPacket
                fps_counter += 1

                # FPS calculation every second
                now = time.monotonic()
                elapsed = now - fps_time
                if elapsed >= 1.0:
                    self._acq_fps = fps_counter / elapsed
                    fps_counter = 0
                    fps_time = now

            finally:
                self.cam.MV_CC_FreeImageBuffer(stOutFrame)

    def get_parameters(self):
        if not self.is_open:
            return {}

        params = {}

        # Float params
        for name in ("ExposureTime", "Gain", "AcquisitionFrameRate"):
            val = MVCC_FLOATVALUE()
            ret = self.cam.MV_CC_GetFloatValue(name, val)
            if ret == MV_OK:
                params[name] = {"current": val.fCurValue, "min": val.fMin, "max": val.fMax}

        # Pixel format
        val = MVCC_ENUMVALUE()
        ret = self.cam.MV_CC_GetEnumValue("PixelFormat", val)
        if ret == MV_OK:
            params["PixelFormat"] = val.nCurValue

        # Auto modes
        for name in ("ExposureAuto", "GainAuto"):
            val = MVCC_ENUMVALUE()
            ret = self.cam.MV_CC_GetEnumValue(name, val)
            if ret == MV_OK:
                params[name] = val.nCurValue

        # Runtime stats
        params["_stats"] = {
            "acq_fps": round(self._acq_fps, 2),
            "frame_count": self._frame_count,
            "lost_packets": self._lost_packets,
        }
        return params

    def set_parameter(self, name, value):
        if not self.is_open:
            raise RuntimeError(f"Camera {self.ip} not open")

        # Try float first, then int, then enum
        ret = self.cam.MV_CC_SetFloatValue(name, float(value))
        if ret == MV_OK:
            return
        ret = self.cam.MV_CC_SetIntValue(name, int(value))
        if ret == MV_OK:
            return
        ret = self.cam.MV_CC_SetEnumValue(name, int(value))
        if ret != MV_OK:
            raise RuntimeError(f"Failed to set {name}={value} on {self.ip}: 0x{ret:08X}")

    def get_latest_jpeg(self):
        with self._frame_lock:
            return self._latest_frame

    def get_snapshot_jpeg(self, quality=90):
        with self._frame_lock:
            raw = self._latest_raw
        if raw is None:
            return None
        _, buf = cv2.imencode('.jpg', raw, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()

    def status(self):
        return {
            "ip": self.ip,
            "model": self.model,
            "serial": self.serial,
            "is_open": self.is_open,
            "is_grabbing": self.is_grabbing,
            "use_hb": self.use_hb,
            "trigger_mode": self.trigger_mode,
            "acq_fps": round(self._acq_fps, 2),
            "frame_count": self._frame_count,
            "lost_packets": self._lost_packets,
        }


class CameraManager:
    """Singleton manager for all connected Hikrobot cameras."""

    def __init__(self):
        MvCamera.MV_CC_Initialize()
        self._cameras = {}  # ip_str -> CameraInstance
        self._recording_manager = RecordingManager()
        log.info("CameraManager initialized (SDK ready)")

    def discover(self):
        device_list = MV_CC_DEVICE_INFO_LIST()
        ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, device_list)
        if ret != MV_OK:
            raise RuntimeError(f"EnumDevices failed: 0x{ret:08X}")

        found = []
        for i in range(device_list.nDeviceNum):
            dev_info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if dev_info.nTLayerType != MV_GIGE_DEVICE:
                continue

            gige = dev_info.SpecialInfo.stGigEInfo
            ip = _ip_int_to_str(gige.nCurrentIp)
            model = _decode_char(gige.chModelName)
            serial = _decode_char(gige.chSerialNumber)

            if ip not in self._cameras:
                self._cameras[ip] = CameraInstance(dev_info, ip, model, serial)
            else:
                # Update dev_info in case it changed
                self._cameras[ip]._dev_info = dev_info
                self._cameras[ip].model = model
                self._cameras[ip].serial = serial

            found.append(ip)

        log.info("Discovered %d camera(s): %s", len(found), found)
        return found

    def get_camera(self, camera_id):
        cam = self._cameras.get(camera_id)
        if cam is None:
            raise KeyError(f"Camera {camera_id} not found. Run discover() first.")
        return cam

    def open_camera(self, camera_id, **config):
        cam = self.get_camera(camera_id)
        cam.open()
        if config:
            cam.configure(**config)
        return cam

    def close_camera(self, camera_id):
        cam = self.get_camera(camera_id)
        cam.close()

    def list_cameras(self):
        return [cam.status() for cam in self._cameras.values()]

    def start_recording(self, camera_ids=None) -> str:
        grabbing = [ip for ip, cam in self._cameras.items() if cam.is_grabbing]
        if not grabbing:
            raise RuntimeError("No cameras are currently grabbing")
        if camera_ids is None:
            camera_ids = grabbing
        else:
            for cid in camera_ids:
                if cid not in self._cameras:
                    raise KeyError(f"Camera {cid} not found")
                if not self._cameras[cid].is_grabbing:
                    raise RuntimeError(f"Camera {cid} is not grabbing")

        # Capture current camera params for metadata
        camera_infos = []
        for cid in camera_ids:
            cam = self._cameras[cid]
            params = cam.get_parameters()
            info = CameraRecordingInfo(
                camera_id=cid,
                model=cam.model,
                serial=cam.serial,
                parameters={
                    k: v["current"] if isinstance(v, dict) else v
                    for k, v in params.items()
                    if not k.startswith("_")
                },
            )
            camera_infos.append(info)

        recording_id, queues = self._recording_manager.start(camera_ids, camera_infos)
        for cid, q in queues.items():
            self._cameras[cid]._record_queue = q
        return recording_id

    def stop_recording(self, sync: SyncMetadata | None = None,
                        warnings: list[str] | None = None) -> dict:
        for cam in self._cameras.values():
            cam._record_queue = None
        return self._recording_manager.stop(sync=sync, warnings=warnings)

    def shutdown(self):
        if self._recording_manager.is_recording:
            try:
                self.stop_recording()
            except Exception as e:
                log.warning("Error stopping recording on shutdown: %s", e)
        for cam in self._cameras.values():
            try:
                cam.close()
            except Exception as e:
                log.warning("Error closing %s: %s", cam.ip, e)
        self._cameras.clear()
        MvCamera.MV_CC_Finalize()
        log.info("CameraManager shut down")
