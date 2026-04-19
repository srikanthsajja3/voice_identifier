import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.audio_utils import load_and_preprocess
from src.model import load_trained_model
from src.explain import generate_gradcam_heatmap, overlay_heatmap

CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", 
    "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

def generate_xai_grid(model_path="models/urban_noise_xai_model.h5", data_dir="data/"):
    if not os.path.exists(model_path):
        print("Model not found.")
        return
    
    model = load_trained_model(model_path)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    print("Generating XAI Heatmap Grid...")
    
    for i, cls in enumerate(CLASSES):
        cls_path = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith(".wav")]
        if not files:
            continue
            
        audio_path = os.path.join(cls_path, files[0])
        img = load_and_preprocess(audio_path)
        img_batch = img[np.newaxis]
        
        # Predict
        preds = model.predict(img_batch)
        pred_idx = np.argmax(preds[0])
        confidence = preds[0][pred_idx]
        
        # Heatmap
        heatmap = generate_gradcam_heatmap(img_batch, model, "last_conv")
        result_img = overlay_heatmap(img, heatmap)
        
        axes[i].imshow(result_img)
        axes[i].set_title(f"True: {cls}\nPred: {CLASSES[pred_idx]} ({confidence:.2f})")
        axes[i].axis('off')
        
    plt.tight_layout()
    output_path = "results/xai_heatmap_matrix.png"
    plt.savefig(output_path)
    print(f"XAI Heatmap Matrix saved to {output_path}")

if __name__ == "__main__":
    generate_xai_grid()
