"""Flask API server — thin wrapper around CameraManager."""

import json
import shutil
import time
import logging
import atexit
from pathlib import Path

from flask import Flask, jsonify, request, Response, render_template

from camera_manager import CameraManager
from recording_manager import CATALOG_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("server")

app = Flask(__name__)
mgr = CameraManager()
atexit.register(mgr.shutdown)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cameras")
def list_cameras():
    if request.args.get("discover") == "true":
        mgr.discover()
    return jsonify(mgr.list_cameras())


@app.route("/api/cameras/<camera_id>/open", methods=["POST"])
def open_camera(camera_id):
    config = request.get_json(silent=True) or {}
    try:
        mgr.open_camera(camera_id, **config)
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/close", methods=["POST"])
def close_camera(camera_id):
    try:
        mgr.close_camera(camera_id)
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/start", methods=["POST"])
def start_grabbing(camera_id):
    try:
        cam = mgr.get_camera(camera_id)
        cam.start_grabbing()
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/stop", methods=["POST"])
def stop_grabbing(camera_id):
    try:
        cam = mgr.get_camera(camera_id)
        cam.stop_grabbing()
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/parameters")
def get_parameters(camera_id):
    try:
        cam = mgr.get_camera(camera_id)
        return jsonify(cam.get_parameters())
    except (KeyError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/parameters", methods=["POST"])
def set_parameters(camera_id):
    try:
        cam = mgr.get_camera(camera_id)
        data = request.get_json(force=True)

        # Map friendly names to configure() kwargs
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
        for k, v in data.items():
            if k in config_map:
                config[config_map[k]] = v
            else:
                generic[k] = v

        if config:
            cam.configure(**config)
        for k, v in generic.items():
            cam.set_parameter(k, v)

        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/stream")
def stream(camera_id):
    try:
        cam = mgr.get_camera(camera_id)
    except KeyError as e:
        return str(e), 404

    def generate():
        while True:
            jpeg = cam.get_latest_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
            time.sleep(1 / 15)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/recordings")
def list_recordings():
    if not CATALOG_DIR.exists():
        return jsonify([])
    recordings = []
    for f in sorted(CATALOG_DIR.glob("*.json"), reverse=True):
        try:
            recordings.append(json.loads(f.read_text()))
        except Exception:
            pass
    return jsonify(recordings)


@app.route("/api/recordings/<recording_id>", methods=["DELETE"])
def delete_recording(recording_id):
    from recording_manager import RECORDING_BASE, CATALOG_DIR
    # Sanitise: recording_id must be a plain timestamp string, no path traversal
    if "/" in recording_id or ".." in recording_id:
        return jsonify({"ok": False, "error": "Invalid recording_id"}), 400
    errors = []
    rec_dir = RECORDING_BASE / recording_id
    if rec_dir.exists():
        try:
            shutil.rmtree(rec_dir)
        except Exception as e:
            errors.append(str(e))
    catalog_file = CATALOG_DIR / f"{recording_id}.json"
    if catalog_file.exists():
        try:
            catalog_file.unlink()
        except Exception as e:
            errors.append(str(e))
    if errors:
        return jsonify({"ok": False, "error": "; ".join(errors)}), 500
    return jsonify({"ok": True})


@app.route("/api/recordings/start", methods=["POST"])
def start_recording():
    data = request.get_json(silent=True) or {}
    camera_ids = data.get("camera_ids")
    try:
        recording_id = mgr.start_recording(camera_ids)
        return jsonify({"ok": True, "recording_id": recording_id})
    except (RuntimeError, KeyError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/recordings/stop", methods=["POST"])
def stop_recording():
    try:
        result = mgr.stop_recording()
        return jsonify({"ok": True, **result})
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/trigger", methods=["POST"])
def fire_trigger(camera_id):
    try:
        cam = mgr.get_camera(camera_id)
        cam.fire_software_trigger()
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/snapshot")
def snapshot(camera_id):
    try:
        cam = mgr.get_camera(camera_id)
        quality = int(request.args.get("quality", 90))
        jpeg = cam.get_snapshot_jpeg(quality)
        if jpeg is None:
            return "No frame available", 503
        return Response(jpeg, mimetype="image/jpeg")
    except (KeyError, RuntimeError) as e:
        return str(e), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
