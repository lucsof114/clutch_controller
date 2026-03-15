# clutch_controller

Multi-camera control system for Hikrobot MV-CS050-10GM-PRO GigE cameras. A Flask server runs on a Linux node (`10.0.50.2`) with the MVS SDK installed. Cameras are controlled from a Mac (`10.0.50.1`) via a web UI.

## Architecture

```
camera_manager.py     — CameraInstance + CameraManager (SDK layer, acquisition threads)
recording_manager.py  — RecordingManager (queue-based disk writer, per-camera saver threads)
recording_metadata.py — RecordingMetadata dataclass (standardised JSON metadata)
server.py             — Flask API (thin wrapper, all logic in the classes above)
templates/index.html  — Web UI (cameras tab + recordings tab, self-contained)
```

## Hardware

| Component | Detail |
|---|---|
| Camera | Hikrobot MV-CS050-10GM-PRO (IMX264, 2448×2048, Mono8) |
| Interface | 1GigE (camera PHY) via 2.5G switch |
| Max fps | ~24 fps at full resolution over 1GigE (bandwidth ceiling) |
| HB mode | Not supported by current firmware (`PixelType_Gvsp_HB_Mono8` = `0x81080001`) |
| Linux node | `10.0.50.2`, Ubuntu, MVS SDK 4.7.0.3 at `/opt/MVS` |

## Setup

```bash
# Install Python deps
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Required env var (add to ~/.zshrc for permanence)
export MVCAM_COMMON_RUNENV=/opt/MVS/lib

# Run
python server.py
```

Open `http://10.0.50.2:5000` from the Mac.

## API

### Cameras

| Method | Route | Description |
|---|---|---|
| GET | `/api/cameras?discover=true` | Discover / list cameras |
| POST | `/api/cameras/<id>/open` | Open camera |
| POST | `/api/cameras/<id>/close` | Close camera |
| POST | `/api/cameras/<id>/start` | Start grabbing |
| POST | `/api/cameras/<id>/stop` | Stop grabbing |
| GET | `/api/cameras/<id>/parameters` | Get parameters + stats |
| POST | `/api/cameras/<id>/parameters` | Set parameters (JSON body) |
| GET | `/api/cameras/<id>/stream` | MJPEG stream |
| GET | `/api/cameras/<id>/snapshot` | Single JPEG |
| POST | `/api/cameras/<id>/trigger` | Fire software trigger |

**Parameters body keys:** `exposure_us`, `gain`, `frame_rate`, `trigger_mode`, `use_hb`, `gain_auto`, `exposure_auto`

### Recordings

| Method | Route | Description |
|---|---|---|
| GET | `/api/recordings` | List all recordings (from catalog) |
| POST | `/api/recordings/start` | Start recording all grabbing cameras |
| POST | `/api/recordings/stop` | Stop recording, flush to disk |
| DELETE | `/api/recordings/<id>` | Delete recording + catalog entry |

## Storage

```
~/Documents/clutch/clutch_db/
  <recording_id>/               # e.g. 20260314_173433
    metadata.json               # RecordingMetadata
    <camera_serial>/            # e.g. DA9128029
      000000/frame.png
      000001/frame.png
      ...
  catalog/
    <recording_id>.json         # flat copy of each recording's metadata
```

Frame IDs are zero-padded sequential integers (`_make_frame_id` in `recording_manager.py`), designed to be swapped for timestamp-based IDs.

## Recording metadata

`metadata.json` schema:

```json
{
  "recording_id": "20260314_173433",
  "start_time": "2026-03-14T17:34:33.030452+00:00",
  "end_time":   "2026-03-14T17:34:36.581950+00:00",
  "duration_seconds": 3.551,
  "cameras": [
    {
      "camera_id": "10.0.50.50",
      "model": "MV-CS050-10GM-PRO",
      "serial": "DA9128029",
      "frame_count": 50,
      "total_size_bytes": 146200168,
      "parameters": {
        "ExposureTime": 5000.0,
        "Gain": 0.0,
        "AcquisitionFrameRate": 24.0,
        "PixelFormat": 17301505
      }
    }
  ]
}
```

## Camera trigger modes

Toggle between **Free Run** and **Triggered** (software trigger) in the UI params panel. In triggered mode, the **Fire** button sends a `TriggerSoftware` command to capture one frame.

## Known limitations

- HB compression not available on current camera firmware — max fps capped at ~24 over 1GigE
- Firmware update pending from Hikrobot support (requested for serial DA9128029)
- Frame IDs are sequential integers; timestamp-based IDs planned
