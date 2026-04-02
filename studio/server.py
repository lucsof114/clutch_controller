"""Flask API server — thin wrapper around StudioController."""

import time
import logging
import atexit

import cv2
from flask import Flask, jsonify, request, Response, render_template

from studio.studio_controller import StudioController

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("server")

app = Flask(__name__)
studio = StudioController()


def _shutdown():
    try:
        studio.shutdown()
    except Exception as e:
        log.warning("Shutdown error: %s", e)


atexit.register(_shutdown)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/cameras")
def list_cameras():
    if request.args.get("discover") == "true":
        studio._cam_mgr.discover()
    return jsonify(studio._cam_mgr.list_cameras())


@app.route("/api/cameras/<camera_id>/open", methods=["POST"])
def open_camera(camera_id):
    config = request.get_json(silent=True) or {}
    try:
        studio._cam_mgr.open_camera(camera_id, **config)
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/close", methods=["POST"])
def close_camera(camera_id):
    try:
        studio._cam_mgr.close_camera(camera_id)
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/reboot", methods=["POST"])
def reboot_camera(camera_id):
    try:
        studio._cam_mgr.reboot_camera(camera_id)
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/start", methods=["POST"])
def start_grabbing(camera_id):
    try:
        cam = studio._cam_mgr.get_camera(camera_id)
        cam.start_grabbing()
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/stop", methods=["POST"])
def stop_grabbing(camera_id):
    try:
        cam = studio._cam_mgr.get_camera(camera_id)
        cam.stop_grabbing()
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/parameters")
def get_parameters(camera_id):
    try:
        cam = studio._cam_mgr.get_camera(camera_id)
        return jsonify(cam.get_parameters())
    except (KeyError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/cameras/<camera_id>/parameters", methods=["POST"])
def set_parameters(camera_id):
    try:
        data = request.get_json(force=True)
        studio.set_camera_parameters(camera_id, **data)
        return jsonify({"ok": True})
    except (KeyError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/stream")
def stream(camera_id):
    try:
        cam = studio._cam_mgr.get_camera(camera_id)
    except KeyError as e:
        return str(e), 404

    def generate():
        q = cam.subscribe_stream()
        try:
            while True:
                try:
                    frame = q.get(timeout=5.0)
                except Exception:
                    continue
                _, jpeg_buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpeg = jpeg_buf.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
                    b"\r\n" + jpeg + b"\r\n"
                )
        finally:
            cam.unsubscribe_stream(q)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/recordings")
def list_recordings():
    return jsonify(studio.list_recordings())


@app.route("/api/recordings/<recording_id>", methods=["DELETE"])
def delete_recording(recording_id):
    try:
        studio.delete_recording(recording_id)
        return jsonify({"ok": True})
    except (ValueError, RuntimeError) as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/recordings/start", methods=["POST"])
def start_recording():
    """All recordings go through the studio pipeline."""
    data = request.get_json(silent=True) or {}
    frequency_hz = data.get("frequency_hz", 30.0)
    camera_ids = data.get("camera_ids")
    tags = data.get("tags") or []
    try:
        recording_id = studio.start_recording(frequency_hz=frequency_hz,
                                               camera_ids=camera_ids,
                                               tags=tags)
        return jsonify({"ok": True, "recording_id": recording_id})
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/recordings/stop", methods=["POST"])
def stop_recording():
    try:
        result = studio.stop_recording()
        return jsonify({"ok": True, **result})
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/cameras/<camera_id>/snapshot")
def snapshot(camera_id):
    try:
        cam = studio._cam_mgr.get_camera(camera_id)
        quality = int(request.args.get("quality", 90))
        jpeg = cam.get_snapshot_jpeg(quality)
        if jpeg is None:
            return "No frame available", 503
        return Response(jpeg, mimetype="image/jpeg")
    except (KeyError, RuntimeError) as e:
        return str(e), 400


# ── Studio routes ────────────────────────────────────────────────────────────

@app.route("/api/studio/connect", methods=["POST"])
def studio_connect():
    try:
        studio.connect()
        return jsonify({"ok": True})
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/studio/disconnect", methods=["POST"])
def studio_disconnect():
    try:
        studio.disconnect()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/studio/start", methods=["POST"])
def studio_start():
    data = request.get_json(silent=True) or {}
    frequency_hz = data.get("frequency_hz", 30.0)
    camera_ids = data.get("camera_ids")
    tags = data.get("tags") or []
    try:
        recording_id = studio.start_recording(frequency_hz=frequency_hz,
                                               camera_ids=camera_ids,
                                               tags=tags)
        return jsonify({"ok": True, "recording_id": recording_id})
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/studio/stop", methods=["POST"])
def studio_stop():
    try:
        result = studio.stop_recording()
        return jsonify({"ok": True, **result})
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/studio/status")
def studio_status():
    return jsonify(studio.status())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
