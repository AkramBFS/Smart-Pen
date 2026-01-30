import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

# =============================
# Configuration & Loading
# =============================
MODEL_DIR = "models"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model_extended.joblib")
MLP_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model_extended.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "mlp_scaler_extended.joblib")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.joblib")

print("Loading models...")
RF_MODEL = joblib.load(RF_MODEL_PATH)
MLP_MODEL = load_model(MLP_MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
TRAINED_FEATURES = joblib.load(FEATURE_NAMES_PATH)

AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
ACC_AXES = ["ax", "ay", "az"]
GYRO_AXES = ["gx", "gy", "gz"]

# =============================
# Feature Extraction
# =============================
def extract_features_demo(filepath):
    df = pd.read_csv(filepath).dropna(subset=AXES)
    if len(df) < 20:
        raise ValueError("Sample too short")

    row = {}
    rms_values = {}

    for axis in AXES:
        sig = df[axis].values
        row[f"{axis}_mean"] = np.mean(sig)
        row[f"{axis}_std"] = np.std(sig, ddof=1)
        row[f"{axis}_min"] = np.min(sig)
        row[f"{axis}_max"] = np.max(sig)
        row[f"{axis}_rms"] = np.sqrt(np.mean(sig ** 2))
        row[f"{axis}_zcr"] = np.sum(np.diff(np.sign(sig - np.mean(sig))) != 0) / len(sig)

        diffs = np.diff(sig)
        row[f"{axis}_path"] = np.sum(np.abs(diffs))
        row[f"{axis}_jerk"] = np.mean(np.abs(diffs))
        row[f"{axis}_velocity_proxy"] = row[f"{axis}_jerk"]

        rms_values[axis] = row[f"{axis}_rms"]

    row["duration_ms"] = df["timestamp_ms"].iloc[-1] - df["timestamp_ms"].iloc[0]

    row["accel_gyro_energy_ratio"] = (
        sum(rms_values[a] for a in ACC_AXES) /
        (sum(rms_values[g] for g in GYRO_AXES) + 1e-6)
    )

    row["ax_ay_corr"] = np.corrcoef(df["ax"], df["ay"])[0, 1]
    row["ay_az_corr"] = np.corrcoef(df["ay"], df["az"])[0, 1]
    row["gx_gy_corr"] = np.corrcoef(df["gx"], df["gy"])[0, 1]
    row["gy_gz_corr"] = np.corrcoef(df["gy"], df["gz"])[0, 1]

    for a, g in zip(ACC_AXES, GYRO_AXES):
        row[f"{g}_to_{a}_ratio"] = rms_values[g] / (rms_values[a] + 1e-6)

    return pd.DataFrame([row])[TRAINED_FEATURES]

# =============================
# Batch Processing
# =============================
dir_path = input("\nEnter the directory path for the student's samples: ")

if not os.path.isdir(dir_path):
    print("Invalid directory!")
    exit()

files = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
print(f"Found {len(files)} samples. Processing...")

results = []

for f in files:
    full_path = os.path.join(dir_path, f)

    if "good" in f.lower():
        true_label = 1
    elif "bad" in f.lower():
        true_label = 0
    else:
        true_label = None

    try:
        feat = extract_features_demo(full_path)
        X = feat.values

        rf_pred = RF_MODEL.predict(X)[0]

        X_scaled = SCALER.transform(X)
        mlp_pred = int(MLP_MODEL.predict(X_scaled, verbose=0)[0][0] > 0.5)

        results.append({
            "file": f,
            "truth": true_label,
            "rf_pred": rf_pred,
            "mlp_pred": mlp_pred,
            "rf_correct": rf_pred == true_label if true_label is not None else None,
            "mlp_correct": mlp_pred == true_label if true_label is not None else None
        })

    except Exception as e:
        print(f"Error processing {f}: {e}")

res_df = pd.DataFrame(results)
res_df = res_df[res_df.truth.notnull()]  # drop unlabeled

# =============================
# Derived Error Flags
# =============================
res_df["rf_fp"] = (res_df.truth == 0) & (res_df.rf_pred == 1)
res_df["rf_fn"] = (res_df.truth == 1) & (res_df.rf_pred == 0)

res_df["mlp_fp"] = (res_df.truth == 0) & (res_df.mlp_pred == 1)
res_df["mlp_fn"] = (res_df.truth == 1) & (res_df.mlp_pred == 0)

res_df["both_wrong"] = (res_df.rf_correct == False) & (res_df.mlp_correct == False)
res_df["disagree"] = res_df.rf_pred != res_df.mlp_pred

# =============================
# Reporting Utilities
# =============================
def print_confusion(name, df, pred_col):
    tp = ((df.truth == 1) & (df[pred_col] == 1)).sum()
    tn = ((df.truth == 0) & (df[pred_col] == 0)).sum()
    fp = ((df.truth == 0) & (df[pred_col] == 1)).sum()
    fn = ((df.truth == 1) & (df[pred_col] == 0)).sum()

    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0
    accuracy = (tp + tn) / len(df) * 100

    print(f"\n[{name}]")
    print(f"Accuracy : {accuracy:.1f}%")
    print(f"Precision: {precision:.1f}% (GOOD correctness)")
    print(f"Recall   : {recall:.1f}%")
    print("Confusion Matrix:")
    print(f"  TP (Good → Good): {tp}")
    print(f"  TN (Bad → Bad) : {tn}")
    print(f"  FP (Bad → Good): {fp}  ⚠️")
    print(f"  FN (Good → Bad): {fn}")

# =============================
# Final Report
# =============================
print("\n" + "=" * 90)
print(f"ENHANCED BATCH REPORT: {os.path.basename(dir_path)}")
print("=" * 90)

print_confusion("Random Forest", res_df, "rf_pred")
print_confusion("Neural Network", res_df, "mlp_pred")

print("\n" + "-" * 90)
print("CROSS-MODEL ANALYSIS")
print("-" * 90)

print(f"Total samples           : {len(res_df)}")
print(f"Both models wrong       : {res_df.both_wrong.sum()}")
print(f"RF wrong / MLP correct  : {((~res_df.rf_correct) & res_df.mlp_correct).sum()}")
print(f"MLP wrong / RF correct  : {((~res_df.mlp_correct) & res_df.rf_correct).sum()}")
print(f"Model disagreements     : {res_df.disagree.sum()}")

print("\n⚠️  CRITICAL FALSE POSITIVES (Bad → Good)")
print(f"RF false positives      : {res_df.rf_fp.sum()}")
print(f"MLP false positives     : {res_df.mlp_fp.sum()}")
print(f"Both FP on same file    : {((res_df.rf_fp) & (res_df.mlp_fp)).sum()}")

critical = res_df[(res_df.rf_fp) | (res_df.mlp_fp)]

if len(critical):
    print("\nFILES WHERE BAD SAMPLES WERE APPROVED AS GOOD")
    print("-" * 90)
    print(f"{'FILE':<40} | RF | MLP")
    print("-" * 90)
    for _, r in critical.iterrows():
        rf = "GOOD*" if r.rf_fp else "OK"
        mlp = "GOOD*" if r.mlp_fp else "OK"
        print(f"{r.file[:40]:<40} | {rf:<5} | {mlp:<5}")
else:
    print("\nNo BAD samples were incorrectly approved as GOOD.")

print("=" * 90)
