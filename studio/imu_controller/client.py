#!/usr/bin/env python3
"""ImuController — drives the Teensy 4.1 ICM-42605 FSYNC-anchored IMU logger.

This is the controller-side counterpart to `teensy/icm6_usb_logger/icm6_usb_logger.ino`.
It is a purely *additive* consumer of the shared 30 Hz camera-sync edge: the same
Arduino edge that fans out to the camera Line0 trigger and the PicoScope is also wired
to the Teensy (pin 17, timestamped) and to every IMU FSYNC pin (tags the coincident
sample). The Teensy streams a 512-byte `IMUL` header + 32-byte binary records over USB.

Lifecycle (orchestrated by studio_controller, ordered around the Arduino):
    start(out_path)   -> 's' (clean stop) -> 'g' (stream) -> read IMUL header ->
                         spawn reader thread that writes the raw .bin and tallies stats.
                         MUST be called while the Arduino sync is OFF so the first
                         logged edge is seq 0, aligned with camera frame 0.
    stop()            -> 's' -> drain -> join reader -> return stats dict.

Alignment is done OFFLINE by the decoder from the saved .bin (capture-now/align-later);
this class only captures faithfully and surfaces validity.

Detection keys on the PJRC USB vendor id (0x16C0) so it never grabs the Arduino's port.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from pathlib import Path
from typing import Optional

import serial
import serial.tools.list_ports

log = logging.getLogger("imu_controller")

TEENSY_VID = 0x16C0          # PJRC / Teensyduino USB vendor id
BAUD = 2_000_000             # USB CDC ignores baud, but pyserial requires a value
MAGIC = b"IMUL"
HEADER_SIZE = 512
REC_SIZE = 32
HDR_FMT = "<4sHHIHBBBB"      # magic, version, record_size, f_cpu, rate, num_imus, gcfg, acfg, div
STATUS_IMU_ID = 0xFF
SYNC_IMU_ID = 0xF0
# firmware flag bits (icm6 v7)
FLAG_INVALID = 0x01          # STATUS sticky-invalid
FLAG_FSYNC_TAGGED = 0x02
FLAG_FIFO_RESET = 0x04
FLAG_BAD_HEADER = 0x08
DISC_MASK = FLAG_FIFO_RESET | FLAG_BAD_HEADER


def find_teensy_port() -> Optional[str]:
    """Return the serial device path of a connected Teensy, or None.

    Matches on the PJRC vendor id so the Arduino sync board (a different VID) is never
    mistaken for the IMU Teensy.
    """
    for p in serial.tools.list_ports.comports():
        if p.vid == TEENSY_VID:
            return p.device
    # Fallback for hosts that don't expose VID: a Teensyduino manufacturer string.
    for p in serial.tools.list_ports.comports():
        man = (p.manufacturer or "").lower()
        if "teensy" in man or "pjrc" in man:
            return p.device
    return None


class ImuController:
    """Optional IMU capture device. Absent hardware degrades to a no-op (camera-only)."""

    def __init__(self, port: Optional[str] = None):
        self._port = port
        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._out_path: Optional[Path] = None
        self._lock = threading.Lock()
        self._stats: dict = {}
        self._active = False

    # ------------------------------------------------------------------ detection
    def available(self) -> bool:
        """True if a Teensy is currently plugged in (auto-detect gate)."""
        return (self._port or find_teensy_port()) is not None

    @property
    def is_recording(self) -> bool:
        return self._active

    # ------------------------------------------------------------------ start
    def start(self, out_path: Path) -> bool:
        """Begin streaming the IMU log to ``out_path``.

        Returns True if the device handshook and the reader thread started, False if no
        Teensy was found or the header never arrived (caller falls back to camera-only).
        Call while the Arduino sync is OFF.
        """
        if self._active:
            raise RuntimeError("IMU already recording")

        port = self._port or find_teensy_port()
        if not port:
            log.info("No IMU Teensy detected; recording camera-only")
            return False

        try:
            ser = serial.Serial(port, BAUD, timeout=1.0)
        except serial.SerialException as e:
            log.warning("IMU Teensy open failed on %s: %s", port, e)
            return False

        header = self._handshake(ser)
        if header is None:
            log.warning("IMU Teensy on %s never sent the IMUL header; recording camera-only", port)
            ser.close()
            return False

        magic, version, rec_size, f_cpu, rate, num_imus, *_ = struct.unpack(
            HDR_FMT, header[:struct.calcsize(HDR_FMT)]
        )
        if rec_size != REC_SIZE:
            log.warning("IMU Teensy record size %d != %d; ignoring device", rec_size, REC_SIZE)
            ser.close()
            return False

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._out_path = out_path
        self._ser = ser
        self._stop.clear()
        self._stats = {
            "port": port,
            "version": version,
            "num_imus": num_imus,
            "rate_hz": rate,
            "bin_path": str(out_path),
            "imu_records": 0,
            "sync_edges": 0,
            "fsync_tags": 0,
            "discontinuities": 0,
            "fifo_overrun": 0,
            "valid": True,
        }

        f = open(out_path, "wb")
        f.write(header)

        self._thread = threading.Thread(
            target=self._reader_loop, args=(ser, f), name="imu-reader", daemon=True
        )
        self._active = True
        self._thread.start()
        log.info("IMU Teensy logging: %d IMUs @ %d Hz on %s -> %s",
                 num_imus, rate, port, out_path)
        return True

    def _handshake(self, ser: serial.Serial) -> Optional[bytes]:
        """Stop any prior stream, start a fresh one, return the 512-byte IMUL header."""
        ser.write(b"s")
        ser.flush()
        time.sleep(0.3)
        ser.reset_input_buffer()
        ser.write(b"g")
        ser.flush()

        window = b""
        t0 = time.time()
        while time.time() - t0 < 5.0:
            b = ser.read(1)
            if not b:
                continue
            window = (window + b)[-4:]
            if window == MAGIC:
                rest = b""
                while len(rest) < HEADER_SIZE - 4:
                    chunk = ser.read(HEADER_SIZE - 4 - len(rest))
                    if not chunk:
                        return None
                    rest += chunk
                return MAGIC + rest
        return None

    # ------------------------------------------------------------------ reader
    def _reader_loop(self, ser: serial.Serial, f):
        """Drain the USB stream to disk and tally validity until stop() drains it."""
        buf = bytearray()
        try:
            while True:
                data = ser.read(8192)
                if data:
                    f.write(data)
                    buf.extend(data)
                    while len(buf) >= REC_SIZE:
                        rec = bytes(buf[:REC_SIZE])
                        del buf[:REC_SIZE]
                        self._tally(rec)
                elif self._stop.is_set():
                    break
        except Exception as e:                                  # serial unplug, etc.
            log.warning("IMU reader stopped on error: %s", e)
            with self._lock:
                self._stats["valid"] = False
                self._stats["error"] = str(e)
        finally:
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    def _tally(self, rec: bytes):
        imu_id = rec[0]
        flags = rec[1]
        with self._lock:
            if imu_id == STATUS_IMU_ID:
                self._stats["fifo_overrun"] = struct.unpack_from("<I", rec, 4)[0]
                if flags & FLAG_INVALID:
                    self._stats["valid"] = False
            elif imu_id == SYNC_IMU_ID:
                self._stats["sync_edges"] += 1
            elif imu_id < self._stats["num_imus"]:
                self._stats["imu_records"] += 1
                if flags & FLAG_FSYNC_TAGGED:
                    self._stats["fsync_tags"] += 1
                if flags & DISC_MASK:
                    self._stats["discontinuities"] += 1
                    self._stats["valid"] = False

    # ------------------------------------------------------------------ stop
    def stop(self) -> dict:
        """Stop streaming, finalise the .bin, and return capture stats.

        Call AFTER the Arduino sync has stopped so the final edge and its FSYNC tag are
        captured. If the recording saw any discontinuity, the .bin is renamed *.INVALID.
        """
        if not self._active:
            return {}

        ser = self._ser
        try:
            ser.write(b"s")
            ser.flush()
        except Exception:
            pass
        time.sleep(0.3)                 # let the device's final flush land
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        try:
            ser.close()
        except Exception:
            pass

        with self._lock:
            stats = dict(self._stats)

        # Policy mirror of the host recorder: any discontinuity invalidates the file.
        if not stats.get("valid", True) and self._out_path and self._out_path.exists():
            invalid = self._out_path.with_suffix(self._out_path.suffix + ".INVALID")
            try:
                self._out_path.replace(invalid)
                stats["bin_path"] = str(invalid)
                log.warning("IMU recording INVALID (%d discontinuities) -> %s",
                            stats.get("discontinuities", 0), invalid)
            except OSError:
                pass

        self._active = False
        self._ser = None
        self._thread = None
        self._out_path = None
        log.info("IMU Teensy stopped: %d records, %d edges, %d fsync tags, valid=%s",
                 stats.get("imu_records", 0), stats.get("sync_edges", 0),
                 stats.get("fsync_tags", 0), stats.get("valid"))
        return stats
