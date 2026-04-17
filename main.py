import argparse
import os
import numpy as np
from src.audio_utils import load_and_preprocess
from src.model import load_trained_model
from src.explain import generate_gradcam_heatmap, overlay_heatmap
from src.train import train_project_model

# UrbanSound8K Classes
CLASSES = [
    "air_conditioner", "car_horn", "children_playing", "dog_bark", 
    "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
]

def main():
    parser = argparse.ArgumentParser(description="Urban Noise XAI: Explainable Audio Classification")
    parser.add_argument("--mode", choices=["train", "explain"], required=True, help="Run in 'train' or 'explain' mode.")
    parser.add_argument("--audio", type=str, help="Path to the .wav file (required for explain mode).")
    parser.add_argument("--model", type=str, default="models/urban_noise_xai_model.h5", help="Path to saved model.")
    
    args = parser.parse_args()

    if args.mode == "train":
        train_project_model("data/")

    elif args.mode == "explain":
        if not args.audio:
            print("Error: Please provide an audio file with --audio <path>")
            return
        
        # 1. Preprocess the audio file
        img = load_and_preprocess(args.audio)
        img_batch = img[np.newaxis] # Add batch dimension
        
        # 2. Load the model
        if not os.path.exists(args.model):
            print(f"Error: Model not found at {args.model}. Train it first.")
            return
        model = load_trained_model(args.model)
        
        # 3. Predict the class
        preds = model.predict(img_batch)
        class_idx = np.argmax(preds[0])
        print(f"Predicted Class: {CLASSES[class_idx]} (Confidence: {preds[0][class_idx]:.2f})")
        
        # 4. Generate the Grad-CAM Heatmap
        heatmap = generate_gradcam_heatmap(img_batch, model, "last_conv")
        
        # 5. Overlay and Save
        result_img = overlay_heatmap(img, heatmap)
        output_name = f"results/{os.path.basename(args.audio)}_explained.png"
        result_img.save(output_name)
        print(f"XAI Visualization saved to: {output_name}")

if __name__ == "__main__":
    main()
