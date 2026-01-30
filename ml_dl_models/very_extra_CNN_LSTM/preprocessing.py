import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- CONFIG ---
RAW_DATA_DIR = "data" 
WINDOW_SIZE = 40  # 2 seconds at 20Hz
STEP_SIZE = 10    # 50% overlap for more data
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]

def load_and_window_data(directory):
    all_windows = []
    all_labels = []
    scaler = StandardScaler()

    for file in os.listdir(directory):
        if not file.endswith(".csv"): continue
        
        # Load raw sensor data
        df = pd.read_csv(os.path.join(directory, file)).dropna(subset=AXES)
        if len(df) < WINDOW_SIZE: continue
        
        # Determine label (0 for bad, 1 for good)
        label = 1 if "good" in file.lower() else 0
        
        # Pre-process: Scale the raw values (Important for CNN)
        # Note: In a real scenario, fit the scaler on train only, but this is for simplicity
        raw_values = scaler.fit_transform(df[AXES].values)
        
        # Create Sliding Windows
        for i in range(0, len(raw_values) - WINDOW_SIZE, STEP_SIZE):
            window = raw_values[i : i + WINDOW_SIZE]
            all_windows.append(window)
            all_labels.append(label)
            
    return np.array(all_windows), np.array(all_labels)

# --- EXECUTION ---
print("Loading and windowing raw CSV files...")
X, y = load_and_window_data(RAW_DATA_DIR)

# Split into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}") # Expecting (N, 40, 6)

# Build the CNN-LSTM (Using the function from the previous message)
model = build_cnn_lstm_model(input_shape=(WINDOW_SIZE, 6))

# Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)