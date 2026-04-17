import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.audio_utils import load_and_preprocess
from src.model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_project_model(data_dir, epochs=50):
    """
    MAX ACCURACY TRAINER: Processes all 8,732 UrbanSound8K samples.
    """
    print(f"--- UrbanSound8K HIGH ACCURACY Training Process Started ---")
    
    # 1. Load Metadata
    csv_path = os.path.join(data_dir, "UrbanSound8K.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_dir, "metadata", "UrbanSound8K.csv")
        if not os.path.exists(csv_path):
            print(f"Error: Could not find UrbanSound8K.csv")
            return
        
    df = pd.read_csv(csv_path)
    print(f"Metadata loaded: {len(df)} samples available.")

    # 2. Extract Features & Labels
    features = []
    labels = []
    
    print(f"Preprocessing FULL DATASET (8732 files). This will take 10-15 minutes...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        fold = row['fold']
        file_name = row['slice_file_name']
        label = row['classID']
        
        file_path = os.path.join(data_dir, f"fold{fold}", file_name)
        if not os.path.exists(file_path):
            file_path = os.path.join(data_dir, "audio", f"fold{fold}", file_name)
        
        if os.path.exists(file_path):
            try:
                feat = load_and_preprocess(file_path)
                features.append(feat)
                labels.append(label)
            except Exception:
                continue

    X = np.array(features)
    y = np.array(labels)
    
    print(f"\nData Loading Complete. Final Shape: {X.shape}")
    
    # 3. Train/Val Split (80/20)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Build Model
    model = create_model(num_classes=10)
    
    # 5. Callbacks for Best Accuracy
    # Save the best version only
    checkpoint = ModelCheckpoint(
        "models/urban_noise_xai_model.h5", 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    # Stop if it stops improving to save time
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("\nStarting Training for Maximum Accuracy...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs, 
        batch_size=32,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    
    # 6. Final Save and Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy']); plt.title('Final Accuracy Curves'); plt.legend(['Train', 'Val'])
    plt.subplot(1, 2, 2); plt.plot(history.history['loss']); plt.plot(history.history['val_loss']); plt.title('Final Loss Curves'); plt.legend(['Train', 'Val'])
    plt.savefig("results/training_metrics.png")
    
    print("\nSuccess! High-Accuracy model saved to models/urban_noise_xai_model.h5")

if __name__ == "__main__":
    train_project_model("data/")
