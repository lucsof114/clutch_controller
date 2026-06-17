#!/usr/bin/env python3
"""Live web graph for the ICM-42605 stream (D,ax,ay,az,gx,gy,gz at 100 Hz).

Reads the Teensy serial port, parses 'D,' lines, and serves a self-contained
rolling line chart at http://localhost:8770 (accel in g, gyro in dps).

    ./venv/bin/python teensy/imu_live_plot.py            # autodetect port
    ./venv/bin/python teensy/imu_live_plot.py --port /dev/cu.usbmodemXXXX
"""
import argparse, glob, json, sys, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed (use ./venv/bin/python)")

WIN = 600  # samples shown (~6 s at 100 Hz)
buf = deque(maxlen=WIN)
lock = threading.Lock()


def find_port():
    p = sorted(glob.glob("/dev/cu.usbmodem*")) or sorted(glob.glob("/dev/ttyACM*"))
    return p[0] if p else None


def reader(port):
    ser = serial.Serial(port, 2000000, timeout=1)
    while True:
        line = ser.readline().decode(errors="replace").strip()
        if not line.startswith("D,"):
            continue
        try:
            v = [float(x) for x in line[2:].split(",")]
        except ValueError:
            continue
        if len(v) == 6:
            with lock:
                buf.append(v)


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>ICM-42605 live</title>
<style>body{background:#111;color:#ddd;font:14px monospace;margin:0;padding:12px}
h3{margin:6px 0}canvas{background:#000;border:1px solid #333;width:100%;height:260px;display:block}
.leg span{display:inline-block;margin-right:14px}</style></head><body>
<h3>Accel (g) &nbsp;<span class=leg><span style=color:#e55>X</span><span style=color:#5d5>Y</span><span style=color:#59f>Z</span></span> <span id=ainfo></span></h3>
<canvas id=acc></canvas>
<h3>Gyro (dps) &nbsp;<span class=leg><span style=color:#e55>X</span><span style=color:#5d5>Y</span><span style=color:#59f>Z</span></span> <span id=ginfo></span></h3>
<canvas id=gyr></canvas>
<script>
const COL=['#e55','#5d5','#59f'];
function fit(c){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio;}
function draw(cv,info,rows,off){
 const ctx=cv.getContext('2d'),W=cv.width,H=cv.height;ctx.clearRect(0,0,W,H);
 if(rows.length<2){return;}
 let mn=1e9,mx=-1e9;
 for(const r of rows)for(let k=0;k<3;k++){const y=r[off+k];if(y<mn)mn=y;if(y>mx)mx=y;}
 if(mx-mn<1e-6){mx+=1;mn-=1;}const pad=(mx-mn)*0.1;mn-=pad;mx+=pad;
 // zero line + scale
 ctx.strokeStyle='#222';ctx.lineWidth=1;ctx.beginPath();
 const zy=H-(0-mn)/(mx-mn)*H;ctx.moveTo(0,zy);ctx.lineTo(W,zy);ctx.stroke();
 for(let k=0;k<3;k++){ctx.strokeStyle=COL[k];ctx.lineWidth=1.5*devicePixelRatio;ctx.beginPath();
  for(let i=0;i<rows.length;i++){const x=i/(rows.length-1)*W;const y=H-(rows[i][off+k]-mn)/(mx-mn)*H;
   i?ctx.lineTo(x,y):ctx.moveTo(x,y);}ctx.stroke();}
 info.textContent='['+mn.toFixed(2)+' .. '+mx.toFixed(2)+']  n='+rows.length;
}
const acc=document.getElementById('acc'),gyr=document.getElementById('gyr');
function rs(){fit(acc);fit(gyr);}window.addEventListener('resize',rs);rs();
async function tick(){
 try{const r=await fetch('/data');const j=await r.json();
  draw(acc,document.getElementById('ainfo'),j.s,0);
  draw(gyr,document.getElementById('ginfo'),j.s,3);}catch(e){}
 setTimeout(tick,80);
}
tick();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/data"):
            with lock:
                s = list(buf)
            body = json.dumps({"s": s}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None)
    ap.add_argument("--http", type=int, default=8770)
    a = ap.parse_args()
    port = a.port or find_port()
    if not port:
        sys.exit("no Teensy serial port found")
    threading.Thread(target=reader, args=(port,), daemon=True).start()
    print(f"serial {port}  ->  http://localhost:{a.http}")
    ThreadingHTTPServer(("127.0.0.1", a.http), H).serve_forever()


if __name__ == "__main__":
    main()
