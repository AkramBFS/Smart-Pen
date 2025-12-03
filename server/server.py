# server/server.py

import asyncio
import threading
import json
import csv
import os
import time
import traceback
from datetime import datetime
from typing import Set, Any

from flask import Flask, request, jsonify, render_template
import websockets

# ---- Flask app paths (keep as you had them) ----
app = Flask(__name__, template_folder="../web/templates", static_folder="../web/static")

# -------------------- Shared State --------------------
recording = False
recording_lock = threading.Lock()
current_meta = {}
csv_file = None
csv_writer = None

# connected dashboard clients (type Any to avoid deprecated typed class warning)
ui_clients: Set[Any] = set()

RECORDINGS_DIR = "../recordings"
os.makedirs(RECORDINGS_DIR, exist_ok=True)


# -------------------- Flask Routes --------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_recording():
    global recording, current_meta, csv_file, csv_writer

    payload = request.get_json() or request.form.to_dict()

    student = payload.get("student") or payload.get("name")
    sid     = payload.get("id") or payload.get("ID")
    hand    = payload.get("hand")
    sample  = payload.get("sample")
    quality = payload.get("quality")

    if not (student and sid and hand and sample and quality):
        return jsonify({"error": "missing fields: student,id,hand,sample,quality"}), 400

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_st = "".join(c if c.isalnum() or c in "-_" else "_" for c in student)
    safe_samp = "".join(c if c.isalnum() or c in "-_" else "_" for c in sample)

    fname = f"{safe_st}_{sid}_{safe_samp}_{quality}_{hand}_{ts}.csv"
    path = os.path.join(RECORDINGS_DIR, fname)

    with recording_lock:
        csv_file = open(path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)

        csv_writer.writerow([
            "student", "student_id", "hand", "sample", "quality",
            "timestamp_ms", "ax", "ay", "az", "gx", "gy", "gz"
        ])
        csv_file.flush()

        current_meta = {
            "student": student,
            "student_id": sid,
            "hand": hand,
            "sample": sample,
            "quality": quality,
            "filename": path
        }
        recording = True

    print(f"[INFO] Recording STARTED -> {path}")
    return jsonify({"status": "recording_started", "file": path})


@app.route("/stop", methods=["POST"])
def stop_recording():
    global recording, csv_file, csv_writer, current_meta

    with recording_lock:
        recording = False
        if csv_file:
            csv_file.close()

        meta = current_meta
        current_meta = {}
        csv_file = None
        csv_writer = None

    print("[INFO] Recording STOPPED")
    return jsonify({
        "status": "recording_stopped",
        "file": meta.get("filename") if meta else None
    })


@app.route("/status", methods=["GET"])
def status():
    with recording_lock:
        return jsonify({
            "record": recording,
            "meta": current_meta
        })


# -------------------- WebSocket Server (robust for many websockets versions) ----

async def handle_ws(websocket, path=None):
    """
    Robust handler: accept either (websocket, path) or (websocket,) calling conventions.
    Be defensive about where 'path' is stored and protect the handler from exceptions so
    one error does not kill the connection (causing ESP to disconnect).
    """
    global csv_writer, csv_file, recording

    # Try to obtain path in multiple ways (compat for websockets v11..v15)
    if path is None:
        # websockets >= 11 expose .path or .request.path; probe safely
        try:
            path = getattr(websocket, "path", None)
        except Exception:
            path = None
    if path is None:
        try:
            req = getattr(websocket, "request", None)
            if req is not None:
                path = getattr(req, "path", None)
        except Exception:
            path = None
    if path is None:
        # fallback default
        path = "/"

    print(f"[WS] Client connected on: {path}")

    # Top-level try so any unexpected exception will be logged and won't kill the handler
    try:
        # UI dashboard clients (subscribe to broadcasts)
        if path == "/ui":
            ui_clients.add(websocket)
            try:
                async for _ in websocket:
                    # UI is not expected to send messages; keep loop alive to detect disconnects
                    pass
            except websockets.ConnectionClosed:
                pass
            except Exception:
                print("[WS] UI handler error:")
                traceback.print_exc()
            finally:
                ui_clients.discard(websocket)
                print("[WS] UI client disconnected")
            return

        # ESP32 data stream
        if path == "/esp":
            try:
                async for msg in websocket:
                    # debug: show raw message (trim to reasonable size)
                    try:
                        raw = msg if len(msg) < 800 else msg[:800] + "..."
                    except Exception:
                        raw = "<unreadable>"
                    print("[WS][esp] raw:", raw)

                    data = None

                    # Try parse JSON first
                    try:
                        data = json.loads(msg)
                    except Exception:
                        # fallback: CSV-style parsing (timestamp,ax,ay,az,gx,gy,gz)
                        parts = msg.strip().split(",")
                        if len(parts) >= 7:
                            try:
                                data = {
                                    "timestamp_ms": int(parts[0]),
                                    "ax": float(parts[1]),
                                    "ay": float(parts[2]),
                                    "az": float(parts[3]),
                                    "gx": float(parts[4]),
                                    "gy": float(parts[5]),
                                    "gz": float(parts[6]),
                                }
                            except Exception:
                                # malformed numeric conversion -> skip
                                print("[WS][esp] malformed numeric conversion; skipping")
                                continue
                        else:
                            # message too short -> skip
                            print("[WS][esp] message too short; skipping")
                            continue

                    # Normalize timestamp: if device sent a small millis-since-boot value,
                    # replace with server epoch ms so browser displays a sane date.
                    try:
                        ts_val = int(data.get("timestamp_ms", 0))
                        # threshold: anything less than 1e12 (approx 2001-09-09 in ms) is probably millis-since-boot
                        if ts_val < 1_000_000_000_000:
                            data["timestamp_ms"] = int(time.time() * 1000)
                    except Exception:
                        data["timestamp_ms"] = int(time.time() * 1000)

                    # broadcast to UI clients (non-blocking best-effort)
                    packet = json.dumps(data)
                    dead = []
                    for c in set(ui_clients):
                        try:
                            await c.send(packet)
                        except Exception:
                            dead.append(c)
                    for d in dead:
                        ui_clients.discard(d)

                    # write to CSV if recording
                    with recording_lock:
                        if recording and csv_writer:
                            try:
                                csv_writer.writerow([
                                    current_meta.get("student"),
                                    current_meta.get("student_id"),
                                    current_meta.get("hand"),
                                    current_meta.get("sample"),
                                    current_meta.get("quality"),
                                    data.get("timestamp_ms"),
                                    data.get("ax"),
                                    data.get("ay"),
                                    data.get("az"),
                                    data.get("gx"),
                                    data.get("gy"),
                                    data.get("gz"),
                                ])
                                csv_file.flush()
                            except Exception:
                                print("[WS] CSV write failed")
                                traceback.print_exc()
            except websockets.ConnectionClosed:
                print("[WS] ESP disconnected")
            except Exception:
                print("[WS] ESP handler error:")
                traceback.print_exc()
            return

        # Unknown path -> close politely
        try:
            await websocket.close()
        except Exception:
            pass

    except Exception:
        print("[WS] handler top-level error:")
        traceback.print_exc()
        try:
            await websocket.close()
        except Exception:
            pass


# start WebSocket server
def start_ws_server(host="0.0.0.0", port=8765):
    async def main():
        # websockets.serve will call handle_ws with either (websocket) or (websocket, path)
        async with websockets.serve(handle_ws, host, port):
            print(f"[WS] Listening on ws://{host}:{port}")
            await asyncio.Future()  # run forever

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())


# -------------------- Start Both Servers --------------------

if __name__ == "__main__":
    # run websocket server in background thread (daemon)
    threading.Thread(target=start_ws_server, daemon=True).start()
    print("[INFO] Flask running on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
