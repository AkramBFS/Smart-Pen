import os
import pandas as pd
import numpy as np

RAW_DATA_DIR = "data"
OUTPUT_CSV = "training_features_extended.csv"

ACC_AXES = ["ax", "ay", "az"]
GYRO_AXES = ["gx", "gy", "gz"]
AXES = ACC_AXES + GYRO_AXES

MIN_SAMPLES = 20


def rms(x):
    return np.sqrt(np.mean(x ** 2))


def zero_crossing_rate(x):
    mean_centered = x - np.mean(x)
    return np.sum(np.diff(np.sign(mean_centered)) != 0) / len(x)


def path_length(x):
    return np.sum(np.abs(np.diff(x)))


def jerk_proxy(x):
    return np.mean(np.abs(np.diff(x)))


rows = []

for file in os.listdir(RAW_DATA_DIR):
    if not file.endswith(".csv"):
        continue

    try:
        df = pd.read_csv(os.path.join(RAW_DATA_DIR, file))
        df = df.dropna(subset=AXES)

        if len(df) < MIN_SAMPLES:
            raise ValueError("Too few samples")

        row = {
            "student_id": df["student_id"].iloc[0],
            "quality": df["quality"].iloc[0].strip().lower(),
            "duration_ms": df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0],
        }

        rms_values = {}

        # Base + jerk
        for axis in AXES:
            signal = df[axis].values

            row[f"{axis}_mean"] = np.mean(signal)
            row[f"{axis}_std"] = np.std(signal, ddof=1)
            row[f"{axis}_min"] = np.min(signal)
            row[f"{axis}_max"] = np.max(signal)
            row[f"{axis}_rms"] = rms(signal)
            row[f"{axis}_zcr"] = zero_crossing_rate(signal)
            row[f"{axis}_path"] = path_length(signal)
            row[f"{axis}_jerk"] = jerk_proxy(signal)

            rms_values[axis] = row[f"{axis}_rms"]

        # Velocity proxy (same as jerk magnitude but conceptually separated)
        for axis in AXES:
            row[f"{axis}_velocity_proxy"] = np.mean(np.abs(np.diff(df[axis].values)))

        # Accel vs Gyro energy ratio
        accel_energy = sum(rms_values[a] for a in ACC_AXES)
        gyro_energy = sum(rms_values[g] for g in GYRO_AXES)
        row["accel_gyro_energy_ratio"] = accel_energy / (gyro_energy + 1e-6)

        # Axis correlations
        row["ax_ay_corr"] = np.corrcoef(df["ax"], df["ay"])[0, 1]
        row["ay_az_corr"] = np.corrcoef(df["ay"], df["az"])[0, 1]
        row["gx_gy_corr"] = np.corrcoef(df["gx"], df["gy"])[0, 1]
        row["gy_gz_corr"] = np.corrcoef(df["gy"], df["gz"])[0, 1]

        # Gyro-to-Accel dominance per axis
        for a, g in zip(ACC_AXES, GYRO_AXES):
            row[f"{g}_to_{a}_ratio"] = rms_values[g] / (rms_values[a] + 1e-6)

        rows.append(row)

    except Exception as e:
        print(f"[SKIP] {file}: {e}")


features_df = pd.DataFrame(rows)
features_df.to_csv(OUTPUT_CSV, index=False)

print(f"[OK] Extended feature extraction complete → {OUTPUT_CSV}")
print(f"Samples processed: {len(features_df)}")
