#!/usr/bin/env python
"""
Deepfake Audio Detector - CLI Testing Script
Usage:
    python test_prediction.py <path_to_audio_wav_file>
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
import joblib

# Load Model and Scaler
MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler.joblib"

def extract_file_features(filepath):
    try:
        # Load audio using soundfile for high speed
        y, sr = sf.read(filepath, dtype='float32')
        if len(y) == 0:
            print("Error: Audio file is empty.")
            return None
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        # Feature extraction
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Aggregate features
        features = {}
        for i in range(20):
            features[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
            features[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
            features[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
            features[f'mfcc_delta2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
            features[f'mfcc_delta2_{i}_std'] = float(np.std(mfcc_delta2[i]))
            
        features['spec_cent_mean'] = float(np.mean(spec_cent))
        features['spec_cent_std'] = float(np.std(spec_cent))
        features['spec_bw_mean'] = float(np.mean(spec_bw))
        features['spec_bw_std'] = float(np.std(spec_bw))
        features['spec_roll_mean'] = float(np.mean(spec_roll))
        features['spec_roll_std'] = float(np.std(spec_roll))
        features['rms_mean'] = float(np.mean(rms))
        features['rms_std'] = float(np.std(rms))
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        for i in range(12):
            features[f'chroma_{i}_mean'] = float(np.mean(chroma[i]))
            features[f'chroma_{i}_std'] = float(np.std(chroma[i]))
            
        # Reconstruct exactly in training column order
        feature_keys = []
        for i in range(20):
            feature_keys.extend([
                f'mfcc_{i}_mean', f'mfcc_{i}_std',
                f'mfcc_delta_{i}_mean', f'mfcc_delta_{i}_std',
                f'mfcc_delta2_{i}_mean', f'mfcc_delta2_{i}_std'
            ])
        feature_keys.extend([
            'spec_cent_mean', 'spec_cent_std',
            'spec_bw_mean', 'spec_bw_std',
            'spec_roll_mean', 'spec_roll_std',
            'rms_mean', 'rms_std',
            'zcr_mean', 'zcr_std'
        ])
        for i in range(12):
            feature_keys.extend([
                f'chroma_{i}_mean', f'chroma_{i}_std'
            ])
            
        vector = [features[k] for k in feature_keys]
        return np.array(vector).reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("\n[!] Error: Please specify the audio file path.")
        print("Usage: python test_prediction.py <path_to_audio_wav_file>\n")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"\n[!] Error: File '{audio_path}' does not exist.\n")
        sys.exit(1)
        
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"\n[!] Error: Model artifacts ({MODEL_PATH} or {SCALER_PATH}) not found in the current directory.")
        print("Please train the model first or verify file placement.\n")
        sys.exit(1)
        
    print(f"\nLoading model: {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    print(f"Analyzing audio file: {audio_path}...")
    features_vec = extract_file_features(audio_path)
    
    if features_vec is None:
        sys.exit(1)
        
    # Scale features
    features_scaled = scaler.transform(features_vec)
    
    # Run prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Class mapping: 0 -> Genuine, 1 -> Deepfake
    outcome = "DEEPFAKE (AI-Generated)" if prediction == 1 else "GENUINE (Human Speech)"
    confidence = probabilities[prediction] * 100
    
    print("\n" + "="*50)
    print("             DEEPFAKE DETECTION RESULT             ")
    print("="*50)
    print(f"File Analysed : {os.path.basename(audio_path)}")
    print(f"Classification: {outcome}")
    print(f"Confidence    : {confidence:.2f}%")
    print("-"*50)
    print(f"Probability Breakdowns:")
    print(f"  - Genuine (Real): {probabilities[0]*100:.2f}%")
    print(f"  - Deepfake (Fake): {probabilities[1]*100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
