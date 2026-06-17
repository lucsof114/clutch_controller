#!/usr/bin/env python3
"""Live web graph for the icm6_usb_logger binary stream (v7).

Does the 'g'/'s' handshake, reads the IMUL header, parses the 32-byte records, and
plots one IMU's accel (g) and gyro (dps) live at http://localhost:8771.

    ./venv/bin/python teensy/icm_live_stream.py            # autodetect, IMU 0
    ./venv/bin/python teensy/icm_live_stream.py --imu 2 --port /dev/cu.usbmodemXXXX
"""
import argparse, glob, json, struct, sys, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed (use ./venv/bin/python)")

MAGIC = b"IMUL"
HEADER_SIZE = 512
REC_SIZE = 32
ACCEL_LSB_PER_G = 2048.0     # ICM-42605 ±16 g
GYRO_LSB_PER_DPS = 16.4      # ICM-42605 ±2000 dps
DECIM = 5                    # 500 Hz / 5 = 100 Hz to the plot
WIN = 600                    # ~6 s window

buf = deque(maxlen=WIN)
lock = threading.Lock()
info = {"tagged": 0, "rate": 0.0}


def find_port():
    p = sorted(glob.glob("/dev/cu.usbmodem*")) or sorted(glob.glob("/dev/ttyACM*"))
    return p[0] if p else None


def reader(port, imu_sel):
    ser = serial.Serial(port, 2000000, timeout=1)
    ser.write(b"s"); ser.flush(); time.sleep(0.3); ser.reset_input_buffer()
    ser.write(b"g"); ser.flush()
    # lock onto the 512-byte IMUL header
    win = b""
    t0 = time.time()
    while win != MAGIC and time.time() - t0 < 5:
        b = ser.read(1)
        if b:
            win = (win + b)[-4:]
    rest = b""
    while len(rest) < HEADER_SIZE - 4:
        rest += ser.read(HEADER_SIZE - 4 - len(rest))

    raw = bytearray()
    deci = 0
    last_t = time.time(); last_n = 0; n = 0
    while True:
        data = ser.read(8192)
        if data:
            raw.extend(data)
            while len(raw) >= REC_SIZE:
                rec = bytes(raw[:REC_SIZE]); del raw[:REC_SIZE]
                iid = rec[0]
                if iid == imu_sel:
                    n += 1
                    if rec[1] & 0x02:          # FSYNC-tagged
                        info["tagged"] += 1
                    deci += 1
                    if deci >= DECIM:
                        deci = 0
                        ax, ay, az, gx, gy, gz = struct.unpack_from("<6h", rec, 16)
                        with lock:
                            buf.append([ax / ACCEL_LSB_PER_G, ay / ACCEL_LSB_PER_G,
                                        az / ACCEL_LSB_PER_G, gx / GYRO_LSB_PER_DPS,
                                        gy / GYRO_LSB_PER_DPS, gz / GYRO_LSB_PER_DPS])
        now = time.time()
        if now - last_t >= 1.0:
            info["rate"] = (n - last_n) / (now - last_t)
            last_n = n; last_t = now


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>ICM live</title>
<style>body{background:#111;color:#ddd;font:14px monospace;margin:0;padding:12px}
h3{margin:6px 0}canvas{background:#000;border:1px solid #333;width:100%;height:260px;display:block}</style></head><body>
<h3>Accel (g) <span style=color:#e55>X</span> <span style=color:#5d5>Y</span> <span style=color:#59f>Z</span> <span id=ainfo></span></h3>
<canvas id=acc></canvas>
<h3>Gyro (dps) <span style=color:#e55>X</span> <span style=color:#5d5>Y</span> <span style=color:#59f>Z</span> <span id=ginfo></span></h3>
<canvas id=gyr></canvas>
<script>
const COL=['#e55','#5d5','#59f'];
function fit(c){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;}
function draw(cv,info,rows,off){
 const ctx=cv.getContext('2d'),W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
 if(rows.length<2)return;
 let mn=1e9,mx=-1e9;
 for(const r of rows)for(let k=0;k<3;k++){const y=r[off+k];if(y<mn)mn=y;if(y>mx)mx=y;}
 if(mx-mn<1e-6){mx+=1;mn-=1;}const pad=(mx-mn)*0.1;mn-=pad;mx+=pad;
 ctx.strokeStyle='#222';ctx.beginPath();const zy=H-(0-mn)/(mx-mn)*H;ctx.moveTo(0,zy);ctx.lineTo(W,zy);ctx.stroke();
 for(let k=0;k<3;k++){ctx.strokeStyle=COL[k];ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();
  for(let i=0;i<rows.length;i++){const x=i/(rows.length-1)*W;const y=H-(rows[i][off+k]-mn)/(mx-mn)*H;i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();}
 info.textContent='['+mn.toFixed(2)+'..'+mx.toFixed(2)+'] n='+rows.length;
}
const acc=document.getElementById('acc'),gyr=document.getElementById('gyr');
function rs(){fit(acc);fit(gyr);}addEventListener('resize',rs);rs();
async function tick(){
 try{const j=await(await fetch('/data')).json();
  document.title='ICM '+j.rate.toFixed(0)+'Hz';
  draw(acc,document.getElementById('ainfo'),j.s,0);
  draw(gyr,document.getElementById('ginfo'),j.s,3);}catch(e){}
 setTimeout(tick,80);
}
tick();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/data"):
            with lock:
                s = list(buf)
            body = json.dumps({"s": s, "rate": info["rate"], "tagged": info["tagged"]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--imu", type=int, default=0, help="IMU id to plot (default 0)")
    ap.add_argument("--http", type=int, default=8771)
    a = ap.parse_args()
    port = a.port or find_port()
    if not port:
        sys.exit("no Teensy serial port found")
    threading.Thread(target=reader, args=(port, a.imu), daemon=True).start()
    print(f"serial {port}  IMU {a.imu}  ->  http://localhost:{a.http}")
    ThreadingHTTPServer(("127.0.0.1", a.http), H).serve_forever()


if __name__ == "__main__":
    main()
