from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import numpy as np
import datetime
import sys

# Ensure src is in path
sys.path.append(os.getcwd())

from src.audio_utils import load_and_preprocess, save_waveform, save_mfcc
from src.model import load_trained_model
from src.explain import generate_gradcam_heatmap, overlay_heatmap

app = Flask(__name__)
MODEL_PATH = "models/urban_noise_xai_model.h5"
CLASSES = ["Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", "Drilling", "Engine Idling", "Gunshot", "Jackhammer", "Siren", "Street Music"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/metrics')
def metrics():
    timestamp = str(datetime.datetime.now().timestamp())
    cm_exists = os.path.exists("results/confusion_matrix.png")
    xai_exists = os.path.exists("results/xai_heatmap_matrix.png")
    return render_template('metrics.html', 
                           cm_url="/results/confusion_matrix.png?v=" + timestamp,
                           xai_url="/results/xai_heatmap_matrix.png?v=" + timestamp,
                           cm_exists=cm_exists,
                           xai_exists=xai_exists)

@app.route('/results/<path:filename>')
def serve_results(filename):
    return send_from_directory('results', filename)

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
    
    # Generate all probabilities for chart
    all_probs = [{"label": CLASSES[i], "score": float(preds[0][i])} for i in range(len(CLASSES))]

    # 2. Generate Visualizations
    save_waveform(filepath, "static/uploads/waveform.png")
    save_mfcc(filepath, "static/uploads/mfcc.png")
    
    heatmap = generate_gradcam_heatmap(img_batch, model, "last_conv")
    explained_img = overlay_heatmap(img, heatmap)
    explained_path = "static/uploads/explanation.png"
    explained_img.save(explained_path)

    timestamp = str(datetime.datetime.now().timestamp())

    # 3. Future Trend Estimation (Heuristic-based)
    trends = []
    for i, cls in enumerate(CLASSES):
        base_increase = np.random.uniform(1.5, 8.0)
        if cls in ["Jackhammer", "Drilling", "Siren"]:
            base_increase += 5.0
        trends.append({"label": cls, "value": round(base_increase, 1)})

    return jsonify({
        "prediction": result,
        "confidence": round(float(np.max(preds[0])) * 100, 2),
        "probs": all_probs,
        "waveform_url": "/static/uploads/waveform.png?v=" + timestamp,
        "mfcc_url": "/static/uploads/mfcc.png?v=" + timestamp,
        "explanation_url": "/" + explained_path + "?v=" + timestamp,
        "trends": trends
    })

if __name__ == '__main__':
    app.run(debug=True, port=8080)
