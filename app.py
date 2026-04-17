from flask import Flask, render_template, request, jsonify
import os
import numpy as np
import datetime
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.audio_utils import load_and_preprocess
from src.model import load_trained_model
from src.explain import generate_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)
MODEL_PATH = "models/urban_noise_xai_model.h5"
CLASSES = ["Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", "Drilling", "Engine Idling", "Gunshot", "Jackhammer", "Siren", "Street Music"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
        
    file = request.files['audio']
    filepath = os.path.join("static/uploads", "current_audio.wav")
    file.save(filepath)

    # 1. Load Model & Predict
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet"}), 500
        
    model = load_trained_model(MODEL_PATH)
    img = load_and_preprocess(filepath)
    img_batch = img[np.newaxis]
    preds = model.predict(img_batch)
    class_idx = np.argmax(preds[0])
    result = CLASSES[class_idx]
    
    # 2. Generate XAI Heatmap
    heatmap = generate_gradcam_heatmap(img_batch, model, "last_conv")
    explained_img = overlay_heatmap(img, heatmap)
    explained_path = "static/uploads/explanation.png"
    explained_img.save(explained_path)

    # 3. Future Trend Estimation (Heuristic-based)
    # We simulate a "Smart City" trend where construction and traffic noise are increasing
    trends = []
    for i, cls in enumerate(CLASSES):
        base_increase = np.random.uniform(1.5, 8.0)
        # Increase construction noise more (Jackhammer, Drilling)
        if cls in ["Jackhammer", "Drilling", "Siren"]:
            base_increase += 5.0
        trends.append({"label": cls, "value": round(base_increase, 1)})

    return jsonify({
        "prediction": result,
        "confidence": round(float(np.max(preds[0])) * 100, 2),
        "explanation_url": "/" + explained_path + "?v=" + str(datetime.datetime.now().timestamp()),
        "trends": trends
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
