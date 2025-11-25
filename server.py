from flask import Flask, request, jsonify, render_template
import csv
import time
import os
from datetime import datetime

app = Flask(__name__)

# GLOBAL VARIABLES
recording = False
csv_file = None
csv_writer = None
current_label = None  # This will store the "Name" from the form
current_id = None     # This will store the "ID" date from the form


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start", methods=["POST"])
def start_recording():
    global recording, csv_file, csv_writer, current_label, current_id

    # Get form data from the webpage
    current_label = request.form.get("name")
    current_id = request.form.get("ID")

    # Sanity check
    if not current_label or not current_id:
        return "Missing label or ID!", 400

    # Make directory if not exists
    if not os.path.exists("recordings"):
        os.makedirs("recordings")

    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recordings/{current_label}_{current_id}_{timestamp}.csv"

    # Open CSV file
    csv_file = open(filename, "w", newline="")
    csv_writer = csv.writer(csv_file)

    # Write header
    csv_writer.writerow(["timestamp", "ax", "ay", "az", "gx", "gy", "gz", "label"])

    recording = True
    print(f"[INFO] Recording started: {filename}")

    return jsonify({"status": "recording_started", "file": filename})


@app.route("/stop", methods=["POST"])
def stop_recording():
    global recording, csv_file

    recording = False

    if csv_file:
        csv_file.close()
        csv_file = None

    print("[INFO] Recording stopped.")
    return jsonify({"status": "recording_stopped"})


@app.route("/data", methods=["POST"])
def receive_data():
    global recording, csv_writer, current_label

    if not recording:
        return jsonify({"status": "not_recording"}), 200

    data = request.get_json()

    if not data:
        return "Bad JSON", 400

    # Extract readings
    ax = data.get("ax")
    ay = data.get("ay")
    az = data.get("az")
    gx = data.get("gx")
    gy = data.get("gy")
    gz = data.get("gz")

    # Write a row with timestamp
    csv_writer.writerow([time.time(), ax, ay, az, gx, gy, gz, current_label])

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



