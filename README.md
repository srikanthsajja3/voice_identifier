# Urban Noise Source Classification with Explainable AI (XAI)

This project extends standard Deep Learning noise classification by adding **Explainable AI** via **Grad-CAM**. It identifies *where* in the frequency spectrum the model is looking to make its prediction.

## Project Structure
- `data/`: Place your UrbanSound8K audio files here in subfolders (e.g., `data/siren/`, `data/dog_bark/`).
- `models/`: Trained models will be saved here as `.h5` files.
- `results/`: Grad-CAM heatmap images will be saved here.
- `src/`: Core logic for audio preprocessing, CNN architecture, training, and Grad-CAM.
- `main.py`: Entry point for training and explaining.

## Setup Instructions
1. Install [Python 3.8+](https://www.python.org/downloads/)
2. Open terminal in this folder and run:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
Place your dataset in the `data/` folder and run:
```bash
python main.py --mode train
```

### 2. Getting an Explanation (XAI)
To see *why* the model classified a specific file:
```bash
python main.py --mode explain --audio "data/siren/123.wav"
```
Check the `results/` folder for the visualization.

## Technical Details
- **Architecture**: 2D Convolutional Neural Network (CNN).
- **Features**: Mel-frequency spectrograms (128x173).
- **Explainability**: Grad-CAM (Gradient-weighted Class Activation Mapping) targeting the `last_conv` layer.
