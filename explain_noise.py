import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os

# 1. PREPROCESSING: Audio to Mel-Spectrogram
def preprocess_audio(file_path, duration=4, sr=22050):
    """
    Loads audio and converts to a Mel-Spectrogram (the 'image' for our CNN).
    """
    # Load audio (trim/pad to 4 seconds)
    audio, _ = librosa.load(file_path, duration=duration, sr=sr)
    if len(audio) < sr * duration:
        audio = np.pad(audio, (0, sr * duration - len(audio)))
    
    # Generate Mel-Spectrogram
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Normalize for the model (-1 to 1)
    norm_spectrogram = (log_spectrogram - np.min(log_spectrogram)) / (np.max(log_spectrogram) - np.min(log_spectrogram))
    
    # Add channel dimension (H, W, 1)
    return norm_spectrogram[..., np.newaxis]

# 2. MODEL: A CNN Architecture compatible with Grad-CAM
def build_cnn_model(input_shape=(128, 173, 1), num_classes=10):
    """
    CNN model with a specific 'last_conv_layer' for Grad-CAM to hook into.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', name="last_conv_layer"), # Target this layer
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# 3. EXPLAINABLE AI (XAI): Grad-CAM Algorithm
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    The Grad-CAM logic: Calculates gradients of the predicted class 
    w.r.t the last convolutional layer's feature maps.
    """
    # Create a sub-model that outputs both the last conv layer and the final prediction
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the class output w.r.t the conv layer output
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by its importance (gradient)
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap (0 to 1) for visualization
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# 4. VISUALIZATION: Overlay Heatmap on Spectrogram
def save_and_display_gradcam(img, heatmap, cam_path="explanation.png", alpha=0.4):
    """
    Overlays the XAI heatmap on the original spectrogram image.
    """
    # Rescale heatmap to original image size
    img = np.uint8(255 * img)
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap for the heatmap
    jet = cv2.applyColorMap(cv2.resize(heatmap, (img.shape[1], img.shape[0])), cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original (convert grayscale img to RGB first)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    superimposed_img = jet * alpha + img_rgb
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the result
    superimposed_img.save(cam_path)
    print(f"Explanation saved to: {cam_path}")

# Example Usage (Placeholder)
if __name__ == "__main__":
    print("XAI Noise Classification Framework Loaded.")
    print("Project Directory: C:\\Users\\srika\\Desktop\\UrbanNoiseXAI")
    
    # Instructions for the user:
    # 1. Place your UrbanSound8K .wav files in a folder.
    # 2. Train the model using the architecture above.
    # 3. Call make_gradcam_heatmap to see 'why' a sound was classified.
