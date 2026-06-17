#!/usr/bin/env python3
"""Record a 10 s 6-IMU capture and analyze the 400 Hz signal.

Records from the Teensy imu_usb_logger (or analyzes an existing .bin) and reports,
per IMU and overall:

  Timing (the 400 Hz cadence)
    - effective record rate and interrupt rate (records+dropped)
    - measured sampling frequency and ppm offset vs 400 Hz
    - inter-sample jitter: mean/median/std/min/max/p2p + percentiles (us)
    - % of intervals within +/-5% of nominal, gap count, largest gap
    - ASCII histogram of dt
  Integrity
    - dropped samples, I2C read errors, ISR queue overflows
  Signal
    - per-axis accel/gyro bias (mean) and noise (std), |a|, temperature
    - FFT spectrum (top peaks) up to the 200 Hz Nyquist

Usage:
    ./venv/bin/python teensy/imu_400hz_test.py                  # record 10 s, analyze
    ./venv/bin/python teensy/imu_400hz_test.py --duration 20
    ./venv/bin/python teensy/imu_400hz_test.py --bin run1.bin   # analyze existing file
"""

import argparse
import glob
import struct
import sys
import time
from datetime import datetime

import numpy as np

try:
    import serial  # pyserial (only needed when recording)
except ImportError:
    serial = None

MAGIC = b"IMUL"
HEADER_SIZE = 512
REC_SIZE = 32
STATUS_IMU_ID = 0xFF
HDR_FMT = "<4sHHIHBBBB"
ENTRY_FMT = "<BBBB"

ACCEL_LSB_PER_G = 8192.0
GYRO_LSB_PER_DPS = 16.4
NOMINAL_HZ = 400.0

REC_DTYPE = np.dtype([
    ("imu_id", "u1"), ("flags", "u1"), ("dropped_ts", "<u2"), ("isr_overflows", "<u4"),
    ("t_cyc", "<u4"), ("dt_cyc", "<u4"),
    ("ax", "<i2"), ("ay", "<i2"), ("az", "<i2"),
    ("gx", "<i2"), ("gy", "<i2"), ("gz", "<i2"),
    ("temp", "<i2"), ("i2c_read_us", "<u2"),
])
assert REC_DTYPE.itemsize == REC_SIZE


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def find_port():
    ports = sorted(glob.glob("/dev/cu.usbmodem*")) or sorted(glob.glob("/dev/ttyACM*"))
    return ports[0] if ports else None


def scan_header(ser):
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
                    sys.exit("timed out reading header body")
                rest += chunk
            return MAGIC + rest
    sys.exit("never saw stream magic 'IMUL' (is imu_usb_logger flashed? Serial Monitor closed?)")


def record(port, seconds, out_path):
    if serial is None:
        sys.exit("pyserial not installed; run: pip install pyserial (or use --bin)")
    ser = serial.Serial(port, 2000000, timeout=1)
    ser.write(b"s"); ser.flush(); time.sleep(0.3); ser.reset_input_buffer()
    ser.write(b"g"); ser.flush()
    header = scan_header(ser)
    with open(out_path, "wb") as f:
        f.write(header)
        print(f"recording {seconds:.0f}s -> {out_path}")
        t0 = time.time()
        while time.time() - t0 < seconds:
            data = ser.read(16384)
            if data:
                f.write(data)
    ser.write(b"s"); ser.flush(); time.sleep(0.2)
    ser.close()
    return out_path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load(path):
    raw = open(path, "rb").read()
    if raw[:4] != MAGIC:
        sys.exit(f"{path}: bad magic {raw[:4]!r}")
    magic, ver, recsz, f_cpu, rate, n_imus, gcfg, acfg, div = struct.unpack(
        HDR_FMT, raw[:struct.calcsize(HDR_FMT)])
    entries = []
    off = struct.calcsize(HDR_FMT)
    for _ in range(n_imus):
        bus, addr, pin, _ = struct.unpack(ENTRY_FMT, raw[off:off + 4])
        entries.append((bus, addr, pin)); off += 4
    body = raw[HEADER_SIZE:]
    n = len(body) // REC_SIZE
    arr = np.frombuffer(body[:n * REC_SIZE], dtype=REC_DTYPE)
    arr = arr[arr["imu_id"] != STATUS_IMU_ID]
    return {"f_cpu": f_cpu, "rate": rate, "n_imus": n_imus, "entries": entries, "rec": arr}


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def ascii_hist(values, bins=20, width=46, unit="us"):
    if len(values) == 0:
        return ["  (no data)"]
    counts, edges = np.histogram(values, bins=bins)
    peak = counts.max() or 1
    out = []
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(round(width * c / peak))
        out.append(f"  {lo:8.1f}-{hi:8.1f} {unit} | {bar} {c}")
    return out


def top_peaks(sig, fs, k=3, fmin=1.0):
    """Return list of (freq_hz, rel_power) for the k strongest spectral peaks."""
    sig = sig - sig.mean()
    if len(sig) < 16 or not np.any(sig):
        return []
    win = np.hanning(len(sig))
    F = np.fft.rfft(sig * win)
    psd = (np.abs(F) ** 2)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    keep = freqs >= fmin
    psd, freqs = psd[keep], freqs[keep]
    if psd.size == 0:
        return []
    total = psd.sum() or 1.0
    idx = np.argsort(psd)[::-1][:k]
    return [(float(freqs[i]), float(psd[i] / total)) for i in idx]


def analyze(data):
    f_cpu = data["f_cpu"]
    rec = data["rec"]
    nominal_us = 1e6 / NOMINAL_HZ
    print("\n" + "=" * 72)
    print(f"6-IMU 400 Hz analysis   F_CPU={f_cpu/1e6:.0f} MHz   nominal dt={nominal_us:.1f} us")
    print(f"total records: {len(rec)}")
    print("=" * 72)

    verdicts = []
    for idx, (bus, addr, pin) in enumerate(data["entries"]):
        sub = rec[rec["imu_id"] == idx]
        print(f"\n----- IMU {idx}  (bus{bus} 0x{addr:02X} INT{pin}) -----")
        if len(sub) < 10:
            print("  too few samples (check wiring / interrupts)")
            verdicts.append(("IMU %d" % idx, False, "no data"))
            continue

        n = len(sub)
        dropped = int(sub["dropped_ts"].sum())
        errors = int((sub["flags"] & 1).sum())
        isr_ovf = int(sub["isr_overflows"].max())

        # Relative wrap-free timeline from the device's wrap-safe dt_cyc.
        dt_cyc = sub["dt_cyc"].astype(np.int64)        # dt_cyc[0] == 0 (first sample)
        t_rel = np.cumsum(dt_cyc)
        elapsed = t_rel[-1] / f_cpu
        rec_rate = (n - 1) / elapsed if elapsed > 0 else 0
        irq_rate = (n - 1 + dropped) / elapsed if elapsed > 0 else 0

        # Jitter: only consecutive intervals (no dropped samples between them).
        consec = sub["dropped_ts"][1:] == 0
        dt_us = (dt_cyc[1:][consec] / f_cpu) * 1e6
        mean = float(dt_us.mean()); med = float(np.median(dt_us))
        std = float(dt_us.std()); lo = float(dt_us.min()); hi = float(dt_us.max())
        p1, p99 = np.percentile(dt_us, [1, 99])
        meas_hz = 1e6 / mean
        ppm = (meas_hz - NOMINAL_HZ) / NOMINAL_HZ * 1e6
        within = float(np.mean(np.abs(dt_us - nominal_us) < 0.05 * nominal_us) * 100)

        # Gaps (dropped events): intervals spanning more than 1.5 nominal periods.
        gap_cyc = dt_cyc[1:][sub["dropped_ts"][1:] > 0]
        n_gaps = int(gap_cyc.size)
        max_gap_us = float(gap_cyc.max() / f_cpu * 1e6) if n_gaps else 0.0

        print(f"  samples           {n}   over {elapsed:6.3f}s")
        print(f"  record rate       {rec_rate:7.2f} Hz   (interrupt rate {irq_rate:7.2f} Hz)")
        print(f"  measured freq     {meas_hz:7.3f} Hz   ({ppm:+.0f} ppm vs 400)")
        print(f"  dt mean/median    {mean:7.2f} / {med:7.2f} us   (nominal {nominal_us:.1f})")
        print(f"  dt jitter (std)   {std:7.2f} us   min {lo:.1f}  max {hi:.1f}  p2p {hi-lo:.1f}")
        print(f"  dt p1..p99        {p1:7.2f} .. {p99:.2f} us")
        print(f"  within +/-5%      {within:6.1f}%")
        print(f"  dropped samples   {dropped}   ({dropped/(n+dropped)*100:.2f}%)   gaps {n_gaps}  max gap {max_gap_us:.0f} us")
        print(f"  I2C read errors   {errors}     ISR overflows {isr_ovf}")
        print(f"  i2c read time     mean {sub['i2c_read_us'].mean():.0f} us  max {sub['i2c_read_us'].max()} us")

        # Signal stats from good reads only.
        ok = (sub["flags"] & 1) == 0
        acc = np.stack([sub["ax"][ok], sub["ay"][ok], sub["az"][ok]], 1) / ACCEL_LSB_PER_G
        gyr = np.stack([sub["gx"][ok], sub["gy"][ok], sub["gz"][ok]], 1) / GYRO_LSB_PER_DPS
        amag = np.linalg.norm(acc, axis=1)
        tc = sub["temp"][ok] / 340.0 + 36.53
        print(f"  accel bias (g)    x{acc[:,0].mean():+.3f} y{acc[:,1].mean():+.3f} z{acc[:,2].mean():+.3f}   |a|={amag.mean():.3f}")
        print(f"  accel noise (mg)  x{acc[:,0].std()*1e3:5.1f} y{acc[:,1].std()*1e3:5.1f} z{acc[:,2].std()*1e3:5.1f}")
        print(f"  gyro bias (dps)   x{gyr[:,0].mean():+.2f} y{gyr[:,1].mean():+.2f} z{gyr[:,2].mean():+.2f}")
        print(f"  gyro noise (dps)  x{gyr[:,0].std():5.2f} y{gyr[:,1].std():5.2f} z{gyr[:,2].std():5.2f}   temp {tc.mean():.1f}C")

        # Spectrum (approximate: treat as uniform at the measured rate).
        peaks_a = top_peaks(acc[:, 2], meas_hz)
        peaks_g = top_peaks(np.linalg.norm(gyr, axis=1), meas_hz)
        fa = "  ".join(f"{f:.1f}Hz({p*100:.0f}%)" for f, p in peaks_a) or "flat"
        fg = "  ".join(f"{f:.1f}Hz({p*100:.0f}%)" for f, p in peaks_g) or "flat"
        print(f"  accel-z peaks     {fa}")
        print(f"  gyro-|w| peaks    {fg}")

        print("  dt histogram:")
        for line in ascii_hist(dt_us):
            print(line)

        ok_rate = abs(meas_hz - NOMINAL_HZ) / NOMINAL_HZ < 0.02
        ok_drop = dropped / (n + dropped) < 0.01
        passed = ok_rate and ok_drop and isr_ovf == 0
        why = []
        if not ok_rate: why.append("rate")
        if not ok_drop: why.append(f"{dropped/(n+dropped)*100:.1f}% drops")
        if isr_ovf: why.append("isr overflow")
        if errors: why.append(f"{errors} rd errors")
        verdicts.append((f"IMU {idx}", passed, ", ".join(why) if why else "clean"))

    print("\n" + "=" * 72)
    print("VERDICT (400 Hz signal)")
    for name, passed, why in verdicts:
        print(f"  {name}: {'PASS' if passed else 'CHECK'}  - {why}")
    allpass = all(p for _, p, _ in verdicts)
    print(f"\n  overall: {'ALL PASS' if allpass else 'see CHECK items above'}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description="Record 10 s of 6 IMUs and analyze the 400 Hz signal")
    ap.add_argument("--port", default=None, help="serial port (autodetect if omitted)")
    ap.add_argument("--duration", type=float, default=10.0, help="record seconds (default 10)")
    ap.add_argument("--bin", default=None, help="analyze an existing .bin instead of recording")
    ap.add_argument("--out", default=None, help="output .bin path when recording")
    args = ap.parse_args()

    if args.bin:
        path = args.bin
    else:
        port = args.port or find_port()
        if not port:
            sys.exit("no Teensy port found (use --port, or --bin to analyze a file)")
        path = args.out or datetime.now().strftime("imu_%Y%m%d_%H%M%S.bin")
        print(f"port: {port}")
        record(port, args.duration, path)

    analyze(load(path))


if __name__ == "__main__":
    main()
