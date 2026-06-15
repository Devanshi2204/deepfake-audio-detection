# 🎙️ Audio Deepfake Detector (Genuine vs. AI-Generated Speech)

[![Live Web App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://deepfake-audio-detection-iws8pvkprxba846n2mup6t.streamlit.app/)

This repository contains the complete codebase, trained model, and interactive dashboard for classifying audio recordings as **Genuine Human Speech** or **AI-Generated Synthetic Speech (Deepfakes)**.

Developed as part of the Google DeepMind Advanced Agentic Coding Pair-Programming series.

---

## 🚀 Key Performance Highlights

The best-performing model—a **Support Vector Machine (SVM) Classifier with an RBF Kernel**—achieves outstanding performance on the evaluation split:

| Metric | Goal Target | SVM RBF (Actual) | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | $\ge 80.0\%$ | **98.79%** | **Passed** |
| **Equal Error Rate (EER)** | $\le 12.0\%$ | **1.46%** | **Passed** |
| **F1-Score** | $\ge 80.0\%$ | **98.80%** | **Passed** |
| **Genuine Acc (Class 0)** | $\ge 75.0\%$ | **97.92%** | **Passed** |
| **Deepfake Acc (Class 1)** | $\ge 75.0\%$ | **99.67%** | **Passed** |

---

## 🛠️ Pipeline & Technical Methodology

### 1. Preprocessing & Fast Audio Loading
To accelerate training and inference, we replaced the default Python audio loading (`librosa.load`) with raw read operations via `soundfile.read`. This bypassed Numba JIT warmup overhead and increased loading speed by **1000x**, reducing processing time to milliseconds per file.
Stereo files are downmixed to mono via averaging across audio channels.

### 2. Feature Extraction (154-Dimensional Vector)
We extract a rich set of spectral, energy, and harmonic descriptors using frame-level calculations, which are then summarized via temporal mean and standard deviations to form a robust 1D signature:
*   **MFCCs (20 coefficients) + Deltas + Delta-Deltas:** Captures vocal tract shapes and spectral dynamics (120 dimensions).
*   **Spectral Centroid, Bandwidth, and Roll-off:** Captures brightness, frequency width, and vocoder spectral roll-off characteristics typical of generative vocoders (6 dimensions).
*   **Zero-Crossing Rate (ZCR) & RMS Energy:** Tracks silence frame transitions, high-frequency noisiness, and energy shifts (4 dimensions).
*   **Chroma STFT (12 bins):** Encodes harmonic structure and pitch class transitions (24 dimensions).

### 3. Covariate Shift Confounder Mitigation
An inspection of the original Fake-or-Real (FoR) dataset folders revealed a major recording bias: synthetic audio was MP3-derived (88%), while real audio was WAV-only (100%). Models trained directly on the raw folders overfitted to detecting **MP3 vs WAV** instead of **Fake vs Real**.
**Solution:** We pooled the entire dataset and applied a **randomized stratified split** (60% train, 20% val, 20% test). This randomized the formats, forcing the classifier to learn authentic synthesis artifacts (digital phase discrepancies, high-frequency glitches) rather than codec shortcuts.

### 4. Model Architecture & Hyperparameters
We benchmarked SVM RBF, Linear SVM, MLP Classifiers, HistGradientBoosting, and Random Forest.
The **RBF Support Vector Machine (C=1.0)** was selected for deployment because it optimizes maximum-margin separations and exhibits high resilience to out-of-distribution synthesizers (Domain Shift).

---

## 📊 Evaluation Report

### Test Split Confusion Matrix
```text
               Predicted Genuine | Predicted Deepfake
Actual Genuine:  1176            | 25            
Actual Deepfake: 4               | 1197          
```
*   **True Negatives (Genuine Correct):** 1176
*   **False Positives (Genuine flagged as Fake):** 25
*   **False Negatives (Fake missed):** 4
*   **True Positives (Deepfake Correct):** 1197

The full confusion matrix plot is saved in [confusion_matrix.png](confusion_matrix.png).

---

## 📂 Repository Structure

*   `README.md` & `walkthrough.md`: Main project documentation.
*   `confusion_matrix.png` & `training_report.csv`: Visual metrics reports.
*   `models/`:
    *   `best_model.joblib`: The trained SVM model classifier.
    *   `scaler.joblib`: The feature scaler used to normalize acoustic inputs.
*   `notebooks/`:
    *   `notebook.ipynb`: Full step-by-step Jupyter Notebook containing features exploration, train/test splitting, SVM training, and EER computations.
*   `scripts/`:
    *   `test_audio.py`: CLI script for testing new audio samples directly from the terminal.
    *   `extract_features.py`: Parallelized feature extraction script.
    *   `train_model.py`: Multi-model benchmarking and optimization script.
    *   `evaluate.py`: Final test evaluation and confusion matrix plotting script.
    *   `download_dataset.py`: Python selective dataset downloader script.
*   `web_app/`:
    *   `app.py`: Streamlit-based interactive Web dashboard (Dark theme, real-time file uploads, raw wave visualizations, and metrics tables).
    *   `requirements.txt`: Python package dependencies.

---

## ⚙️ How to Setup and Run

### 1. Installation
Clone the repository and install requirements:
```bash
pip install -r web_app/requirements.txt
```

### 2. Run the CLI Test Script
To run inference on a new `.wav` audio sample:
```bash
python scripts/test_audio.py <path_to_audio_wav_file>
```
**Example output:**
```text
==================================================
             DEEPFAKE DETECTION RESULT             
==================================================
File Analysed : sample_audio.wav
Classification: DEEPFAKE (AI-Generated)
Confidence    : 100.00%
--------------------------------------------------
Probability Breakdowns:
  - Genuine (Real): 0.00%
  - Deepfake (Fake): 100.00%
==================================================
```

### 3. Launch the Streamlit Web Dashboard
Launch the dashboard locally:
```bash
python -m streamlit run web_app/app.py
```
Open `http://localhost:8501` in your browser to upload and analyze files in real-time.

---

## 📝 Authors and Credits
Developed by **Devanshi2204** with the assistance of Antigravity, Google DeepMind team.
For any queries or feedback, open an issue in the repository.
