import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten

# =============================
# 1. Data Augmentation Functions
# =============================
def add_noise(data, noise_level=0.01):
    """Adds tiny random jitters to the sensor data."""
    noise = np.random.randn(*data.shape) * noise_level
    return data + noise

def scale_data(data, scaling_factor=(0.9, 1.1)):
    """Slightly strengthens or weakens the signal intensity."""
    factor = np.random.uniform(scaling_factor[0], scaling_factor[1])
    return data * factor

# =============================
# 2. Loading & Preprocessing
# =============================
RAW_DATA_DIR = "data" 
WINDOW_SIZE = 40  # 2 seconds at 20Hz
STEP_SIZE = 10    # 50% overlap
AXES = ["ax", "ay", "az", "gx", "gy", "gz"]

def load_data(directory):
    X, y = [], []
    scaler = StandardScaler()

    # Get list of files
    files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    
    for file in files:
        df = pd.read_csv(os.path.join(directory, file)).dropna(subset=AXES)
        if len(df) < WINDOW_SIZE: continue
        
        label = 1 if "good" in file.lower() else 0
        
        # Scale the data first
        scaled_values = scaler.fit_transform(df[AXES].values)
        
        # Windowing loop
        for i in range(0, len(scaled_values) - WINDOW_SIZE, STEP_SIZE):
            window = scaled_values[i : i + WINDOW_SIZE]
            X.append(window)
            y.append(label)
            
            # --- DATA AUGMENTATION ---
            # For every real window, we create 2 augmented versions
            # This triples your dataset size!
            X.append(add_noise(window))
            y.append(label)
            
            X.append(scale_data(window))
            y.append(label)
            
    return np.array(X), np.array(y)

# =============================
# 3. Model Architecture
# =============================
def build_cnn_lstm(input_shape):
    model = Sequential([
        # Stage 1: CNN Feature Extraction
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Stage 2: LSTM Sequence Learning
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        
        # Stage 3: Classification
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =============================
# 4. Main Execution
# =============================
if __name__ == "__main__":
    print("🔄 Loading data and applying Augmentation...")
    X, y = load_data(RAW_DATA_DIR)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"✅ Total Windows for Training: {len(X_train)}")
    print(f"✅ Input Shape: {X_train.shape[1:]}")

    # Build and Train
    model = build_cnn_lstm(input_shape=(WINDOW_SIZE, 6))
    
    

    print("\n🚀 Starting Deep Learning Training...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=50, 
        batch_size=32,
        verbose=1
    )

    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_lstm_smartpen.h5")
    print("\n💾 Model saved as models/cnn_lstm_smartpen.h5")