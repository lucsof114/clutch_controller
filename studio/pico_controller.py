"""
PicoController — PicoScope edge-detection controller for sync signal validation.

Producer-consumer architecture:
- Poll thread (daemon) runs continuously between open() and close()
- start_tracking/stop_tracking can be called multiple times within one session
- Callback + _active_callback module-level ref to prevent GC

Requires: LD_PRELOAD=/opt/picoscope/lib/libpicoipp.so
"""

from __future__ import annotations

import ctypes
from ctypes import c_int16, c_int32, c_uint32, CFUNCTYPE, POINTER
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from picosdk.ps2000 import ps2000

log = logging.getLogger("pico_controller")

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_INTERVAL_US = 10
OVERVIEW_SIZE = 15000
SAMPLES_PER_SEC = 1_000_000 / SAMPLE_INTERVAL_US
THRESHOLD_ADC = int(0.8 * 32767.0 / 10.0)  # 0.8V threshold in ±10V range (signal peaks ~1.6V)

CALLBACK_TYPE = CFUNCTYPE(
    None, POINTER(POINTER(c_int16)), c_int16, c_uint32, c_int16, c_int16, c_uint32
)

_SHUTDOWN = None
_active_callback = None


@dataclass
class PicoResult:
    """Raw output from PicoScope edge detection."""
    timestamps_us: list[float]
    total_samples: int = 0
    raw_samples: "np.ndarray | None" = None   # only populated in debug mode


# ── Internal data structures ─────────────────────────────────────────────────
@dataclass
class _Chunk:
    data: np.ndarray


@dataclass
class _ProcessingState:
    freq_hz: float
    confirm_k: int = 5              # consecutive samples required to confirm a signal state change
    edge_timestamps: list = field(default_factory=list)  # sample indices of confirmed falling edges
    total_samples: int = 0
    signal_state: str = 'LOW'       # confirmed state: 'LOW' or 'HIGH'
    pending_count: int = 0          # consecutive samples in the pending new state
    pending_start: int = 0          # sample index where the pending transition began
    peak_adc: int = 0
    save_raw: bool = False
    raw_chunks: list = field(default_factory=list)


# ── Core algorithm: k-confirmation state machine ─────────────────────────────
def _process_chunk(chunk: _Chunk, state: _ProcessingState):
    """
    Detect falling edges using a k-consecutive-sample confirmation scheme.

    A transition is only accepted after confirm_k consecutive samples on the
    new side of the threshold.  The recorded timestamp is the sample index of
    the FIRST sample that crossed the threshold (the actual edge), not the
    k-th confirming sample.

    This rejects narrow ringing spikes that don't sustain long enough to pass
    the confirmation window without any look-ahead or period-dependent tuning.
    """
    arr = chunk.data
    n = len(arr)
    if n == 0:
        return

    chunk_max = int(arr.max())
    if chunk_max > state.peak_adc:
        state.peak_adc = chunk_max

    if state.save_raw:
        state.raw_chunks.append(arr.copy())

    base = state.total_samples
    K = state.confirm_k

    for i in range(n):
        above = int(arr[i]) >= THRESHOLD_ADC
        sample_idx = base + i

        if state.signal_state == 'LOW':
            # Waiting for K consecutive HIGH samples to confirm signal went HIGH
            if above:
                if state.pending_count == 0:
                    state.pending_start = sample_idx
                state.pending_count += 1
                if state.pending_count >= K:
                    state.signal_state = 'HIGH'
                    state.pending_count = 0
            else:
                state.pending_count = 0

        else:  # 'HIGH'
            # Waiting for K consecutive LOW samples to confirm falling edge
            if not above:
                if state.pending_count == 0:
                    # Record timestamp of the FIRST low sample — the actual edge
                    state.pending_start = sample_idx
                state.pending_count += 1
                if state.pending_count >= K:
                    state.edge_timestamps.append(state.pending_start)
                    state.signal_state = 'LOW'
                    state.pending_count = 0
            else:
                state.pending_count = 0

    state.total_samples = base + n


def _worker_loop(q: queue.Queue, state: _ProcessingState, stop_event: threading.Event):
    while True:
        try:
            chunk = q.get(timeout=0.05)
        except queue.Empty:
            if stop_event.is_set():
                while not q.empty():
                    chunk = q.get_nowait()
                    if chunk is _SHUTDOWN:
                        return
                    _process_chunk(chunk, state)
                return
            continue

        if chunk is _SHUTDOWN:
            return

        _process_chunk(chunk, state)


def _make_callback(q: queue.Queue):
    def cb(overviewBuffers, overflow, triggeredAt, triggered, auto_stop, nValues):
        n = int(nValues)
        if n <= 0 or not overviewBuffers:
            return
        ptr = overviewBuffers[0]
        if not ptr:
            return
        arr = np.ctypeslib.as_array(ptr, shape=(n,)).copy()
        q.put(_Chunk(data=arr))
    return cb


# ── PicoController ───────────────────────────────────────────────────────────
class PicoController:
    def __init__(self):
        self._handle: int = 0
        self._is_open = False
        self._is_tracking = False

        self._poll_thread: threading.Thread | None = None
        self._poll_stop = threading.Event()

        self._worker_thread: threading.Thread | None = None
        self._worker_stop = threading.Event()
        self._queue: queue.Queue | None = None
        self._state: _ProcessingState | None = None
        self._freq_hz: float = 0.0
        self._c_cb = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def is_tracking(self) -> bool:
        return self._is_tracking

    def open(self):
        """Open device, configure channels, start streaming, start poll thread."""
        if self._is_open:
            return

        handle = ps2000._open_unit()
        if handle <= 0:
            raise RuntimeError(f"PicoScope open failed: {handle}")
        self._handle = handle

        ps2000._set_channel(c_int16(handle), c_int16(0), c_int16(1), c_int16(1), c_int16(9))  # Ch A, DC, ±10V
        ps2000._set_channel(c_int16(handle), c_int16(1), c_int16(0), c_int16(1), c_int16(9))  # Ch B off
        ps2000._set_trigger(c_int16(handle), c_int16(5), c_int16(0), c_int16(0), c_int16(0), c_int16(0))

        status = ps2000._run_streaming_ns(
            c_int16(handle), c_uint32(SAMPLE_INTERVAL_US), c_int32(3),
            c_uint32(100000), c_int16(0), c_uint32(1), c_uint32(OVERVIEW_SIZE),
        )
        if status != 1:
            ps2000._close_unit(c_int16(handle))
            raise RuntimeError(f"PicoScope streaming start failed: {status}")

        self._queue = queue.Queue(maxsize=0)
        raw_cb = _make_callback(self._queue)
        self._c_cb = CALLBACK_TYPE(raw_cb)
        global _active_callback
        _active_callback = self._c_cb

        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        self._is_open = True
        log.info("PicoScope opened and streaming (handle=%d)", handle)

    def close(self):
        """Stop streaming, close device."""
        if not self._is_open:
            return

        if self._is_tracking:
            self.stop_tracking()

        self._poll_stop.set()
        if self._poll_thread:
            self._poll_thread.join(timeout=3.0)
            self._poll_thread = None

        ps2000._stop(c_int16(self._handle))
        ps2000._close_unit(c_int16(self._handle))

        global _active_callback
        _active_callback = None
        self._c_cb = None
        self._queue = None
        self._handle = 0
        self._is_open = False
        log.info("PicoScope closed")

    def start_tracking(self, freq_hz: float, debug: bool = False, confirm_k: int = 5):
        """Start worker thread to detect falling edges.

        Args:
            freq_hz:   Expected trigger frequency (used only for diagnostics).
            debug:     If True, collect every raw ADC sample into PicoResult.raw_samples.
            confirm_k: Consecutive samples required on each side to confirm a transition.
        """
        if not self._is_open:
            raise RuntimeError("PicoScope not open")
        if self._is_tracking:
            raise RuntimeError("Already tracking")

        self._freq_hz = freq_hz
        self._state = _ProcessingState(freq_hz=freq_hz, confirm_k=confirm_k,
                                       save_raw=debug)
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=_worker_loop,
            args=(self._queue, self._state, self._worker_stop),
            daemon=True,
        )
        self._worker_thread.start()
        self._is_tracking = True
        log.info("PicoScope tracking started at %.1f Hz", freq_hz)

    def stop_tracking(self) -> PicoResult:
        """Stop worker, return raw edge timestamps."""
        if not self._is_tracking:
            raise RuntimeError("Not tracking")

        self._worker_stop.set()
        self._queue.put(_SHUTDOWN)
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)
            self._worker_thread = None

        self._is_tracking = False

        state = self._state
        timestamps_us = [t * SAMPLE_INTERVAL_US for t in state.edge_timestamps]

        raw_samples = None
        if state.save_raw and state.raw_chunks:
            raw_samples = np.concatenate(state.raw_chunks)

        result = PicoResult(
            timestamps_us=timestamps_us,
            total_samples=state.total_samples,
            raw_samples=raw_samples,
        )

        peak_v = state.peak_adc * 10.0 / 32767.0
        if state.peak_adc == 0:
            log.warning("PicoScope: peak ADC=0 — Channel A appears to have no signal. "
                        "Check that the probe is connected to the Arduino trigger output pin.")
        elif len(timestamps_us) == 0:
            log.warning("PicoScope: peak ADC=%d (%.2fV) — signal present but no edges confirmed "
                        "(threshold=%.1fV, confirm_k=%d). Check signal level.",
                        state.peak_adc, peak_v, THRESHOLD_ADC * 10.0 / 32767.0, state.confirm_k)
        else:
            log.info("PicoScope: peak ADC=%d (%.2fV)", state.peak_adc, peak_v)

        log.info("PicoScope tracking stopped: %d edges, %d total samples",
                 len(timestamps_us), state.total_samples)
        self._state = None
        return result

    def _poll_loop(self):
        h = c_int16(self._handle)
        cb = self._c_cb
        while not self._poll_stop.is_set():
            try:
                ps2000._get_streaming_last_values(h, cb)
            except Exception as e:
                log.warning("PicoScope poll error: %s", e)
            time.sleep(0.005)
