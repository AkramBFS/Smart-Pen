🖊️ Smart Pen – Arabic Handwriting Proficiency Assessment ( Assignment / Mini-Project)
Overview

This project is a Smart Pen system designed to assess Arabic handwriting proficiency using motion data collected from an ESP32 + MPU6050 (clone-safe) sensor mounted on a pen.

Instead of analyzing images of handwriting, this project analyzes time-series motion data (accelerometer + gyroscope) recorded while writing Arabic words and sentences.
A machine learning model (Random Forest & MLP) classifies handwriting quality as:

Good

Bad
(“Medium” was merged into “Good” for model stability and dataset balance)

This project was built under tight academic constraints and deadlines, prioritizing correct data collection, clean preprocessing, and explainable ML results.

System Architecture (High-Level)
ESP32 + MPU6050
        │
        │  WebSocket (JSON @ ~20Hz)
        ▼
Python Server (Flask + WebSockets)
        │
        ├── Live Web Dashboard
        ├── CSV Recording (per sample)
        ▼
Offline ML Pipeline
        │
        ├── Feature Extraction
        ├── Label Encoding + Normalization
        ├── Model Training (RF + MLP)
        ▼
Trained Models (.joblib / .h5)

Hardware Setup
Components

ESP32-WROOM

MPU6050 (clone / knockoff supported)

Mounted at the tip of the pen (important for signal quality)

Important MPU Constraint

Many cheap MPU6050 clones fail on single-byte I²C reads.

✅ This project uses clone-safe multi-byte reads exclusively.

ESP32 Firmware
Features

Clone-safe MPU6050 access

WiFi auto-reconnect

WebSocket streaming to server

JSON sensor packets

Robust against WiFi drops

Data Sent
{
  "timestamp_ms": 124904,
  "ax": -2164,
  "ay": -14092,
  "az": 8744,
  "gx": 60,
  "gy": 222,
  "gz": -18
}

Key Design Decisions

Uses millis() for timestamps

Server normalizes timestamps to epoch time

Sends data at ~20 Hz

Does not buffer data locally (simplicity + reliability)

📁 Location:

esp32/esp32_ws_knockoff_mpu/esp32_ws_knockoff_mpu.ino

Python Server
Technologies Used

Flask → HTTP control & UI

websockets → Real-time data streaming

CSV → Raw data storage (no preprocessing here)

Responsibilities

Serve web interface

Start/stop recording

Receive ESP32 sensor data

Broadcast live data to UI

Save recordings to CSV

WebSocket Paths
Path	Purpose
/esp	ESP32 sensor stream
/ui	Live dashboard clients
Recording Format (CSV)

Each writing sample is stored as one CSV file.



📁 Location:

server/server.py
recordings/

Web Interface
Capabilities

Enter metadata (student, hand, word, quality)

Start / stop recording

Live sensor readings via WebSocket

No data processing in the browser (intentional)

Why Simple?

UI is not the focus

Eliminates JS-side bugs affecting data quality

All ML happens offline

📁 Location:

web/templates/index.html
web/static/main.js
web/static/style.css

Dataset Design
Current Dataset

6 writers

~150 samples per writer

Labels: good, bad

Arabic words and sentences

Balanced as much as possible

Label Decision

medium samples were converted to good

Reason:

Small dataset

3-class classification unstable

Professor required MLP + Random Forest for the assignment

Feature Extraction Strategy (Option A)
Why Option A?

Instead of segmenting strokes or letters:

✅ Treat each CSV file as one sample

This avoids:

Stroke boundary detection

Sliding window complexity

Overfitting on small datasets

Extracted Features (Per Axis)

For each of the 6 axes:
ax, ay, az, gx, gy, gz

Feature	Why it Matters
Mean	Overall posture / tilt bias
Standard Deviation	Shakiness indicator (most important)
Min / Max	Extreme motion
RMS	Energy of movement
Zero Crossing Rate	Hand corrections / instability
Path Length	Movement inefficiency

➡ Total Features:
6 axes × 7 features = 42 features per sample

Machine Learning Models
Random Forest (Primary)

Handles small datasets well

Robust to noisy features

Provides feature importance

Saved as:

models/random_forest.joblib

MLP (Secondary)

Simple architecture

Demonstrates neural network usage

Avoids overfitting

Architecture:

Input (42)
→ Dense(32, ReLU)
→ Dense(16, ReLU)
→ Dense(1, Sigmoid)


Saved as:

models/mlp_model.h5

Training Pipeline

Load all CSV files

Extract features (Option A)

Encode labels (good=1, bad=0)

Normalize features

Train/Test split with stratification

Train Random Forest

Train MLP

Save models

stratify=y is used to prevent label imbalance in test sets.

Has an extra CNN_LSTM model just for testing, has similar results to ML models. Couldn't expect any better due to the small dataset.
