#!/usr/bin/env python3
"""Host-side recorder for the Teensy imu_usb_logger sketch.

Opens the Teensy USB serial port, starts the stream ('g'), and writes the binary
records to a .bin file that is byte-identical to what the SD logger would write
(512-byte header followed by 32-byte records). Live per-IMU stats are derived
from the incoming records; the device's status sentinel records (imu_id 0xFF)
are printed and stripped, so the .bin stays pure data.

Decode later with the same decoder you'd use for SD .bin files.

Usage:
    pip install pyserial        # e.g. inside scrap/venv
    python imu_usb_recorder.py                       # autodetect port, Ctrl-C to stop
    python imu_usb_recorder.py --port /dev/cu.usbmodemXXXX --out run1.bin --duration 30
"""

import argparse
import glob
import signal
import struct
import sys
import time
from datetime import datetime

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial not installed. Run: pip install pyserial")

MAGIC = b"IMUL"
HEADER_SIZE = 512
REC_SIZE = 32
REC_FMT = "<BBHIIIhhhhhhhH"   # imu_id, flags, dropped_ts, isr_overflows, t_cyc, dt_cyc, ax,ay,az,gx,gy,gz,temp, i2c_read_us
HDR_FMT = "<4sHHIHBBBB"        # magic, version, record_size, f_cpu, sample_rate_hz, num_imus, gyro_cfg, accel_cfg, smplrt_div
ENTRY_FMT = "<BBBB"            # bus_index, addr, int_pin, reserved
STATUS_IMU_ID = 0xFF


def find_port():
    ports = sorted(glob.glob("/dev/cu.usbmodem*")) or sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def read_header(ser):
    """Stop any prior stream, send 'g', scan for MAGIC, return the 512-byte header."""
    # A previous aborted session may have left the device streaming; 'g' is then
    # ignored (already streaming) and no header is sent. Stop first for a clean start.
    ser.write(b"s")
    ser.flush()
    time.sleep(0.3)
    ser.reset_input_buffer()
    ser.write(b"g")
    ser.flush()

    # Scan byte-by-byte until we find MAGIC (skips any boot/banner text).
    window = b""
    t0 = time.time()
    while time.time() - t0 < 5.0:
        b = ser.read(1)
        if not b:
            continue
        window += b
        if len(window) > 4:
            window = window[-4:]
        if window == MAGIC:
            rest = b""
            while len(rest) < HEADER_SIZE - 4:
                chunk = ser.read(HEADER_SIZE - 4 - len(rest))
                if not chunk:
                    sys.exit("timed out reading header body")
                rest += chunk
            return MAGIC + rest
    sys.exit("never saw stream magic 'IMUL' (is the imu_usb_logger sketch flashed?)")


def parse_header(header):
    magic, version, rec_size, f_cpu, rate, num_imus, gcfg, acfg, div = struct.unpack(
        HDR_FMT, header[:struct.calcsize(HDR_FMT)]
    )
    entries = []
    off = struct.calcsize(HDR_FMT)
    for _ in range(num_imus):
        bus, addr, pin, _r = struct.unpack(ENTRY_FMT, header[off:off + 4])
        entries.append((bus, addr, pin))
        off += 4
    return {
        "version": version, "rec_size": rec_size, "f_cpu": f_cpu, "rate": rate,
        "num_imus": num_imus, "entries": entries,
    }


def main():
    ap = argparse.ArgumentParser(description="Record Teensy 6-IMU USB stream to .bin")
    ap.add_argument("--port", default=None, help="serial port (autodetect if omitted)")
    ap.add_argument("--out", default=None, help="output .bin (default imu_<timestamp>.bin)")
    ap.add_argument("--duration", type=float, default=0, help="seconds to record (0 = until Ctrl-C)")
    args = ap.parse_args()

    port = args.port or find_port()
    if not port:
        sys.exit("no Teensy serial port found (use --port)")
    out_path = args.out or datetime.now().strftime("imu_%Y%m%d_%H%M%S.bin")

    ser = serial.Serial(port, 2000000, timeout=1)
    print(f"port: {port}")
    header = read_header(ser)
    info = parse_header(header)
    if info["rec_size"] != REC_SIZE:
        sys.exit(f"unexpected record size {info['rec_size']}")
    print(f"header: {info['num_imus']} IMUs @ {info['rate']} Hz, F_CPU={info['f_cpu']/1e6:.0f} MHz")
    for idx, (bus, addr, pin) in enumerate(info["entries"]):
        print(f"  IMU {idx}: bus{bus} addr 0x{addr:02X} INT{pin}")

    n = info["num_imus"]
    counts = [0] * n
    errors = [0] * n
    drops = [0] * n
    last_counts = [0] * n
    last_overrun = 0
    bytes_written = 0
    recording_invalid = False     # set on any IMU FIFO_RESET/BAD_HEADER or sticky STATUS flag
    disc_count = 0
    # firmware flag bits (icm6 v7): bit0=STATUS invalid, bit2=FIFO_RESET, bit3=BAD_HEADER
    DISC_MASK = 0x0C

    f = open(out_path, "wb")
    f.write(header)
    print(f"recording -> {out_path}  (Ctrl-C to stop)\n")

    stop = {"flag": False}

    def handle_sigint(_sig, _frm):
        stop["flag"] = True
    signal.signal(signal.SIGINT, handle_sigint)

    buf = bytearray()
    t_start = time.time()
    t_report = t_start

    try:
        while not stop["flag"]:
            data = ser.read(8192)
            if data:
                buf.extend(data)
                # Parse whole records; write data records to file, handle status.
                while len(buf) >= REC_SIZE:
                    rec = bytes(buf[:REC_SIZE])
                    del buf[:REC_SIZE]
                    imu_id = rec[0]
                    flags = rec[1]
                    if imu_id == STATUS_IMU_ID:
                        _, _, _, overrun, uptime, total, *_ = struct.unpack(REC_FMT, rec)
                        last_overrun = overrun
                        if flags & 0x01:                 # sticky FLAG_INVALID
                            recording_invalid = True
                        continue  # do not write status records to the file
                    if imu_id < n:
                        counts[imu_id] += 1
                        if flags & DISC_MASK:            # FIFO_RESET / BAD_HEADER -> invalid
                            recording_invalid = True
                            disc_count += 1
                            if disc_count == 1:
                                print(f"\n*** DISCONTINUITY on IMU {imu_id} (flags=0x{flags:02x}) "
                                      f"-> RECORDING INVALID ***\n", flush=True)
                        drops[imu_id] += struct.unpack_from("<H", rec, 2)[0]
                    f.write(rec)
                    bytes_written += REC_SIZE

            now = time.time()
            if now - t_report >= 1.0:
                dt = now - t_report
                rates = [(counts[i] - last_counts[i]) / dt for i in range(n)]
                last_counts = counts[:]
                t_report = now
                rate_str = " ".join(f"{i}:{rates[i]:.0f}Hz" for i in range(n))
                print(f"[{now - t_start:6.1f}s] {rate_str}  "
                      f"err={sum(errors)} drop={sum(drops)} fifo_overrun={last_overrun} "
                      f"MB={bytes_written/1e6:.2f}", flush=True)

            if args.duration and now - t_start >= args.duration:
                break
    finally:
        ser.write(b"s")
        ser.flush()
        time.sleep(0.2)
        f.flush()
        f.close()
        ser.close()
        total = sum(counts)
        elapsed = time.time() - t_start
        # Policy: any FIFO_RESET/BAD_HEADER discontinuity invalidates the whole recording.
        if recording_invalid:
            bad = out_path + ".INVALID"
            try:
                import os
                os.replace(out_path, bad)
                out_path = bad
            except OSError:
                pass
            print(f"\n*** RECORDING INVALID *** ({disc_count} discontinuity record(s)) -> {out_path}")
        print(f"\nstopped. {total} records in {elapsed:.1f}s -> {out_path}")
        print(f"  per-IMU counts: {counts}")
        print(f"  dropped samples:{drops}  (total {sum(drops)})")
        print(f"  fifo overruns:  {last_overrun}")
        print(f"  recording valid: {not recording_invalid}")


if __name__ == "__main__":
    main()
