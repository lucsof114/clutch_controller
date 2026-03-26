# CLAUDE.md — clutch_controller

## Project overview

Multi-camera control system for Hikrobot MV-CS050-10GM-PRO GigE cameras.
- Linux node `10.0.50.2` (Ubuntu) — Flask server, MVS SDK 4.7.0.3 at `/opt/MVS`
- Mac `10.0.50.1` — web UI at `http://10.0.50.2:5000`

## Required environment

```bash
export MVCAM_COMMON_RUNENV=/opt/MVS/lib  # already in ~/.zshrc
source venv/bin/activate
pip install -e .
python -m studio.server
```

If port 5000 is in use: `fuser -k 5000/tcp`

## Architecture

```
studio/                          — installable Python package (pip install -e .)
  server.py                      — Flask API (thin wrapper, all logic in the classes below)
  camera_manager.py              — CameraInstance + CameraManager (SDK layer, acquisition threads)
  recording_manager.py           — RecordingManager (queue-based disk writer, per-camera saver threads)
  recording_metadata.py          — RecordingMetadata dataclass (standardised JSON metadata)
  studio_controller.py           — orchestrator: Arduino + PicoScope + cameras
  pico_controller.py             — PicoScope sync signal validation
  arduino_controller/            — Arduino PWM trigger (firmware + serial client)
  templates/index.html           — Web UI (cameras tab + recordings tab, self-contained)
```

## SDK notes

- Python bindings: `/opt/MVS/Samples/64/Python/MvImport`
- HB lossless compression setup (required order):
  1. `PixelFormat = Mono8` (must be 8-bit first)
  2. `ImageCompressionMode = HB` (enum value 2)
  3. `HighBandwidthMode = Burst` (enum value 1) — critical: tells camera to use compressed bandwidth for rate calc
- With HB Burst: ~35 fps at full 2448×2048 over 1GigE (~3.4x compression ratio)
- Without HB: max ~24 fps at 2448×2048 Mono8 (1GigE bandwidth ceiling)
- Frames arrive with HB pixel type (bit 31 set) and must be decoded via `MV_CC_HBDecode`
- `AcquisitionFrameRateEnable` must be disabled to allow the camera to run at its natural sensor maximum

## Recording design

- Acquisition thread: `put_nowait` → drop on full (never blocks)
- Saver thread: drains queue → `cv2.imwrite` PNG
- `Queue(maxsize=120)` per camera
- `_make_frame_id(n)` in `studio/recording_manager.py` is intentionally isolated — swap for timestamp-based logic when ready
- Directory naming uses camera serial number (not IP)

## Storage layout

```
~/Documents/clutch/clutch_db/
  <recording_id>/          # e.g. 20260314_173433
    metadata.json
    <serial>/              # e.g. DA9128029
      000000/frame.png
      000001/frame.png
  catalog/
    <recording_id>.json    # flat copy for fast listing
```

## UI behaviour

- On load: auto-discover → auto-open → auto-start all cameras
- Play/pause is browser-side only — camera keeps grabbing; pause just disconnects the MJPEG `<img src>`
- Params panel: slider + text input linked bidirectionally; dirty state tracked; single Apply button submits
- Trigger mode: Free Run / Triggered segmented control; Fire button appears in triggered mode

## Common pitfalls

- Don't set PixelFormat while another process holds the camera (`MV_E_CALLORDER`)
- `autoStart()` must refresh camera status *after* the `/start` call (not just after `/open`) to avoid stuck "Opening stream…" spinner
- `RecordingMetadata.save()` expects a directory path, not a file path
- Catalog entries are written by `recording_manager.stop()`, not by `server.py`
