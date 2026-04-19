import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.audio_utils import load_and_preprocess
from src.model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from src.train import plot_confusion_matrix

def train_synthetic():
    data_dir = "./data"
    classes = ["air_conditioner", "car_horn", "children_playing", "dog_bark", 
               "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]
    
    features = []
    labels = []
    
    print("--- Training on Synthetic Data ---")
    
    for i, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.exists(cls_path):
            continue
            
        for file_name in os.listdir(cls_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(cls_path, file_name)
                try:
                    feat = load_and_preprocess(file_path)
                    features.append(feat)
                    labels.append(i)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

    X = np.array(features)
    y = np.array(labels)
    
    print(f"Loaded {len(X)} samples.")
    
    if len(X) == 0:
        print("No data found. Run generate_test_data.py first.")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = create_model(num_classes=10)
    
    checkpoint = ModelCheckpoint(
        "models/urban_noise_xai_model.h5", 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max', 
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=10, 
        batch_size=8,
        callbacks=[checkpoint],
        verbose=1
    )
    
    # Save training metrics plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history.history['accuracy']); plt.plot(history.history['val_accuracy']); plt.title('Accuracy'); plt.legend(['Train', 'Val'])
    plt.subplot(1, 2, 2); plt.plot(history.history['loss']); plt.plot(history.history['val_loss']); plt.title('Loss'); plt.legend(['Train', 'Val'])
    plt.savefig("results/training_metrics.png")
    
    # Generate Confusion Matrix
    print("\nGenerating Confusion Matrix...")
    y_val_pred = model.predict(X_val)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    plot_confusion_matrix(y_val, y_val_pred_classes, classes)
    
    print("Synthetic model trained and saved.")

if __name__ == "__main__":
    train_synthetic()
