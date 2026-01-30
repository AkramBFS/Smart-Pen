import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# =============================
# 1. Configuration
# =============================
RAW_DATA_DIR = "data" 
WINDOW_SIZE = 40  
STEP_SIZE = 10    
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]

# =============================
# 2. Data Loading & LOSO Prep
# =============================
def load_data_loso(directory):
    all_windows = []
    all_labels = []
    all_groups = [] # To track Student IDs
    scaler = StandardScaler()

    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    
    for file in files:
        # Extract student_id (Assumes filename starts with ID, e.g., '1-11-2026_...')
        student_id = file.split('_')[0]
        
        df = pd.read_csv(os.path.join(directory, file)).dropna(subset=AXES)
        if len(df) < WINDOW_SIZE:
            continue
        
        label = 1 if "good" in file.lower() else 0
        scaled_data = scaler.fit_transform(df[AXES].values)
        
        for i in range(0, len(scaled_data) - WINDOW_SIZE, STEP_SIZE):
            window = scaled_data[i : i + WINDOW_SIZE]
            all_windows.append(window)
            all_labels.append(label)
            all_groups.append(student_id)
            
    return np.array(all_windows), np.array(all_labels), np.array(all_groups)

# =============================
# 3. Model Architecture
# =============================
def build_cnn_lstm_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =============================
# 4. LOSO Execution Loop
# =============================
if __name__ == "__main__":
    print("🔄 Loading data for Leave-One-Student-Out Validation...")
    X, y, groups = load_data_loso(RAW_DATA_DIR)
    
    logo = LeaveOneGroupOut()
    overall_accuracies = []

    # Get unique students
    unique_students = np.unique(groups)
    print(f"✅ Found {len(unique_students)} unique students: {unique_students}")

    

    for train_idx, test_idx in logo.split(X, y, groups):
        current_student = groups[test_idx][0]
        print(f"\n--- Training: Excluding Student {current_student} ---")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Build a fresh model for each fold
        model = build_cnn_lstm_model(input_shape=(WINDOW_SIZE, 6))
        
        # Train (Using fewer epochs for the CV loop to save time)
        model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
        
        # Evaluate
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        overall_accuracies.append(acc)
        print(f"🎯 Accuracy for Student {current_student}: {acc:.4f}")

    print("\n" + "="*30)
    print(f"FINAL LOSO AVG ACCURACY: {np.mean(overall_accuracies):.4f}")
    print("="*30)

    # Final step: Train on ALL data and save for the demo
    print("\n💾 Training final production model on all students...")
    final_model = build_cnn_lstm_model(input_shape=(WINDOW_SIZE, 6))
    final_model.fit(X, y, epochs=20, batch_size=32, verbose=1)
    
    os.makedirs("models", exist_ok=True)
    final_model.save("models/cnn_lstm_production.h5")
    print("✅ Final model saved to models/cnn_lstm_production.h5")