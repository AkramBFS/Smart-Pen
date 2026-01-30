import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from logo_evaluation_utils import evaluate_logo

# Update this to your actual generated features file
CSV_PATH = "training_features_extended.csv" 

# 1. Dynamically build the Extended Feature List
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]
EXTENDED_FEATURES = []

for axis in AXES:
    EXTENDED_FEATURES.extend([
        f"{axis}_mean", f"{axis}_std", f"{axis}_min", f"{axis}_max", f"{axis}_rms",
        f"{axis}_zcr",   # Per-axis Zero Crossing Rate
        f"{axis}_path",  # Per-axis Path Length
        f"{axis}_jerk",  # Per-axis Jerk (Shakiness)
    ])

# Add the specific student-level and dispersion features
# Note: Ensure your extractor used these exact names
EXTENDED_FEATURES.append("duration_ms") 

# Load Data
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"Error: {CSV_PATH} not found. Please run extract_features.py first.")
    exit()

# 2. Label Mapping (Good/Bad -> 1/0)
if "label" not in df.columns:
    df["label"] = df["quality"].map({"good": 1, "bad": 0})
    df = df.dropna(subset=["label"])

X = df[EXTENDED_FEATURES].values
y = df["label"].values.astype(int)
groups = df["student_id"].values

# -------- RF Evaluation --------
print("Evaluating Random Forest (Extended Features)...")
rf_metrics, rf_cm = evaluate_logo(
    model_builder=lambda: RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced"
    ),
    X=X,
    y=y,
    groups=groups
)

# -------- MLP Evaluation --------
print("Evaluating MLP Neural Network (Extended Features)...")
mlp_metrics, mlp_cm = evaluate_logo(
    model_builder=lambda: MLPClassifier(
        hidden_layer_sizes=(32,16),
        max_iter=1000,
        random_state=42
    ),
    X=X,
    y=y,
    groups=groups,
    scaler=StandardScaler()
)

# -------- Final Results --------
print("\n" + "="*30)
print("EXTENDED FEATURES RESULTS (LOGO)")
print("="*30)

for name, metrics in [("RF", rf_metrics), ("MLP", mlp_metrics)]:
    print(f"--- {name} ---")
    for k, v in metrics.items():
        # Using nanmean to ignore folds where ROC-AUC couldn't be calculated
        val = np.nanmean(v)
        print(f"{k:<10}: {val:.3f}")
    print()