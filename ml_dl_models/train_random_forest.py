import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# =============================
# Configuration
# =============================
CSV_PATH = "training_features_extended.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

ACC_AXES = ["ax", "ay", "az"]
GYRO_AXES = ["gx", "gy", "gz"]
AXES = ACC_AXES + GYRO_AXES

EXTENDED_FEATURES = []
for axis in AXES:
    EXTENDED_FEATURES.extend([
        f"{axis}_mean", f"{axis}_std", f"{axis}_min", f"{axis}_max", 
        f"{axis}_rms", f"{axis}_zcr", f"{axis}_path", f"{axis}_jerk",
        f"{axis}_velocity_proxy"
    ])
EXTENDED_FEATURES.extend(["duration_ms", "accel_gyro_energy_ratio", "ax_ay_corr", "ay_az_corr", "gx_gy_corr", "gy_gz_corr"])
for a, g in zip(ACC_AXES, GYRO_AXES):
    EXTENDED_FEATURES.append(f"{g}_to_{a}_ratio")

# =============================
# Load Data
# =============================
df = pd.read_csv(CSV_PATH)
if "label" not in df.columns:
    df["label"] = df["quality"].map({"good": 1, "bad": 0})
    df = df.dropna(subset=["label"])

X = df[EXTENDED_FEATURES].values
y = df["label"].values.astype(int)
groups = df["student_id"].values

# --- LOGO CV (Omitted for brevity, keep your loop here) ---

# =============================
# Final Training & Saving
# =============================
print("\nTraining final Random Forest model on all data...")
final_rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
final_rf.fit(X, y)

# Save the model
joblib.dump(final_rf, os.path.join(MODEL_DIR, "rf_model_extended.joblib"))
# Save the feature list (very useful for the demo script later!)
joblib.dump(EXTENDED_FEATURES, os.path.join(MODEL_DIR, "feature_names.joblib"))

print(f"[OK] Random Forest model saved in {MODEL_DIR}")