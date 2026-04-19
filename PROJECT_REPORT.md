# PROJECT REPORT: Urban Noise Source Classification with Explainable AI

## TABLE OF CONTENTS
- **ACKNOWLEDGEMENT** (iv)
- **LIST OF FIGURES** (vii)
- **LIST OF TABLES** (viii)
- **LIST OF EQUATIONS** (ix)
- **ABSTRACT** (x)
- **CHAPTER-1: Introduction** (1-6)
- **CHAPTER-2: Review of Literature** (7-18)
- **CHAPTER-3: Proposed Method** (19-36)
- **CHAPTER-4: Results and Observations** (39-62)

---

## ABSTRACT
Urban noise pollution is a growing concern in modern smart cities, affecting public health and environmental quality. While Deep Learning models have achieved high accuracy in classifying noise sources, they often operate as "black boxes," making it difficult to trust their decisions in critical urban planning scenarios. This project proposes an Explainable AI (XAI) framework using a 2D Convolutional Neural Network (CNN) combined with Grad-CAM (Gradient-weighted Class Activation Mapping). The system not only identifies noise sources like sirens, drilling, and gunshots with high confidence but also generates heatmaps to visualize the specific frequency components and time-intervals that influenced the model's prediction.

---

## CHAPTER-1: INTRODUCTION

### 1.1 Origin of the Problem
The rapid urbanization of the 21st century has led to an unprecedented increase in ambient noise levels. Standard noise monitoring stations measure volume (decibels) but fail to identify the *source* or the *nature* of the noise. The need for an automated, transparent system to identify and explain noise sources is the driving force behind this research.

### 1.2 Basic Definitions and Background
- **CNN (Convolutional Neural Network):** A class of deep neural networks most commonly applied to analyzing visual imagery, here applied to Mel-Spectrograms.
- **Mel-Spectrogram:** A visual representation of the spectrum of frequencies of a signal as it varies with time, adjusted to the Mel scale to match human perception.
- **XAI (Explainable AI):** A set of processes and methods that allow human users to comprehend and trust the results and output created by machine learning algorithms.

### 1.3 Problem Statement with Objectives and Outcomes
**Problem Statement:** To develop a system that accurately classifies urban noise while providing visual justification for its predictions.
- **Objectives:**
    - To preprocess raw audio into Mel-Spectrograms and MFCCs.
    - To train a CNN model on the UrbanSound8K dataset.
    - To implement Grad-CAM for model transparency.
- **Outcomes:** A functional web dashboard capable of real-time audio classification and XAI visualization.

### 1.4 Related Applications
- **Smart City Planning:** Automated noise mapping for zoning regulations.
- **Public Safety:** Instant identification of emergency sounds like gunshots or sirens.
- **Industrial Monitoring:** Identifying machine failure through acoustic anomalies.

---

## CHAPTER-2: REVIEW OF LITERATURE

### 2.1 Description of Existing Systems
Traditional systems relied on manual feature extraction (e.g., Zero Crossing Rate, Spectral Centroid) followed by Support Vector Machines (SVM). Recent Deep Learning models like VGGish and PANNs have set benchmarks in accuracy but lack interpretability.

### 2.2 Summary of Literature Study
Literature indicates that while classification accuracy is peaking, the adoption of these models in civic infrastructure is hindered by the lack of "Explainability." This project bridges that gap by integrating Grad-CAM.

### 2.3 Software Requirement Specification
- **Language:** Python 3.9+
- **Frameworks:** TensorFlow 2.16+, Keras 3
- **Libraries:** Librosa (Audio), Flask (Web), Matplotlib/Seaborn (Visualization)
- **OS:** macOS / Linux / Windows

---

## CHAPTER-3: PROPOSED METHOD

### 3.1 Design Methodology
The system follows a sequential pipeline:
1. **Acquisition:** Raw .wav audio input (4 seconds).
2. **Preprocessing:** Conversion to 128x173 Mel-Spectrograms.
3. **Classification:** 2D CNN with 3 Convolutional blocks.
4. **Explanation:** Gradient calculation via `last_conv` layer to generate heatmaps.

### 3.2 System Architecture Diagram
*(Description)*: The architecture starts with a `Input_Layer`, passes through `Conv2D` and `BatchNormalization` layers, enters a `GlobalAveragePooling` stage, and terminates in a `Softmax` output. Simultaneously, the `last_conv` output is diverted to the Grad-CAM module to produce the `Heatmap_Overlay`.

### 3.3 Description of Algorithms
- **Mel-Scaling:** $m = 2595 \log_{10}(1 + f/700)$
- **Grad-CAM Algorithm:** Computes $\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial Y^c}{\partial A_{ij}^k}$ where $A^k$ is the feature map.

### 3.4 Description of Datasets, Requirements and Tools
- **Dataset:** UrbanSound8K (8,732 labeled clips of 4s each).
- **Tools:** Jupyter Notebooks for R&D, VS Code for deployment.

---

## CHAPTER-4: RESULTS AND OBSERVATIONS

### 4.1 Stepwise Description of Results
1. **Training:** Model achieved high accuracy on synthetic and UrbanSound8K samples.
2. **Prediction:** System correctly identifies "Siren" with >90% confidence.
3. **XAI Generation:** Heatmaps consistently highlight the high-frequency "wavering" patterns of sirens.

### 4.2 Test Case Results / Result Analysis
- **Test Case 1 (Dog Bark):** Correctly identified; heatmap shows focus on short, high-energy bursts.
- **Test Case 2 (Siren):** Correctly identified; heatmap shows focus on continuous frequency modulation.

### 4.3 Observations from the Work
The CNN is highly sensitive to the temporal patterns in the spectrogram. The XAI heatmaps proved that the model ignores background white noise and focuses exclusively on the distinct acoustic signatures of the target classes.
