import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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
print("\nTraining final MLP model on all data...")
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)

final_model = Sequential([
    Dense(64, activation="relu", input_shape=(X_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])
final_model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
final_model.fit(X_scaled, y, epochs=80, batch_size=16, verbose=0)

# Save the model and the scaler
final_model.save(os.path.join(MODEL_DIR, "mlp_model_extended.h5"))
joblib.dump(final_scaler, os.path.join(MODEL_DIR, "mlp_scaler_extended.joblib"))

print(f"[OK] MLP model and scaler saved in {MODEL_DIR}")