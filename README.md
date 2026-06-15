# Deepfake Audio Detection

[![Live Web App](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://deepfake-audio-detection-iws8pvkprxba846n2mup6t.streamlit.app/)

## Overview
With the rapid advancement of generative AI, synthetic speech cloning has become highly realistic, posing significant security and integrity challenges. We developed this Deepfake Audio Detection system to automatically identify AI-generated synthetic speech and distinguish it from genuine human recordings. By analyzing high-frequency vocoder distortions, voicing discontinuities, and spectral anomalies, our pipeline provides a reliable, secure tool to verify voice authenticity.

---

## Key Features
*   **Acoustic Feature Extraction:** Extracts 154-dimensional audio feature vectors consisting of Mel-Frequency Cepstral Coefficients (MFCCs), Delta & Delta-Delta coefficients, Spectral Centroid, Spectral Bandwidth, Spectral Roll-off, RMS Energy, Zero-Crossing Rate, and Chroma STFT.
*   **Bias Mitigation Pipeline:** Employs a stratified random splitting workflow to balance MP3 vs. WAV codec distribution, forcing the models to learn actual synthesis artifacts rather than file-format shortcuts.
*   **Real-Time Dashboard:** Features an interactive web interface for uploading files, playing audio, displaying prediction confidence scores, and plotting signal waveforms.
*   **CLI Inference Utility:** Supports command-line prediction on individual audio files.

---

## Model Performance
We benchmarked several classifiers, with the Support Vector Machine (RBF SVM) showing the strongest generalization performance on the test set:

| Metric | Target | RBF SVM (Ours) | Status |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | $\ge 80.0\%$ | **98.79%** | **Passed** |
| **Equal Error Rate (EER)** | $\le 12.0\%$ | **1.46%** | **Passed** |
| **F1-Score** | $\ge 80.0\%$ | **98.80%** | **Passed** |
| **Genuine Accuracy** | $\ge 75.0\%$ | **97.92%** | **Passed** |
| **Deepfake Accuracy** | $\ge 75.0\%$ | **99.67%** | **Passed** |

---

## Repository Structure
The project is structured logically into subdirectories:

```text
deepfake-audio-detection/
├── .gitignore
├── README.md
├── walkthrough.md
├── confusion_matrix.png
├── evaluation_report.txt
├── training_report.csv
├── extracted_features.csv
├── file100.wav_16k.wav_norm.wav_mono.wav_silence.wav
├── models/
│   ├── best_model.joblib
│   └── scaler.joblib
├── notebooks/
│   └── notebook.ipynb
├── scripts/
│   ├── test_audio.py
│   ├── extract_features.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── download_dataset.py
└── web_app/
    ├── app.py
    └── requirements.txt
```

---

## How to Run

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then install the package requirements:
```bash
pip install -r web_app/requirements.txt
```

### 2. Run the Streamlit Web Application
To start the interactive web application locally:
```bash
python -m streamlit run web_app/app.py
```
Open `http://localhost:8501` in your browser.

### 3. Run CLI Test Script
To test individual audio files via the command line:
```bash
python scripts/test_audio.py <path_to_audio_wav_file>
```
