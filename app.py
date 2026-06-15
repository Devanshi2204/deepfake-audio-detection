import streamlit as st
import os
import tempfile
import soundfile as sf
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

# Set page config for a premium dashboard layout
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for styling (Dark Theme, Glassmorphism, Rounded Cards)
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Premium Header */
    .main-header {
        background: linear-gradient(135deg, #1E1E2F 0%, #111122 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #333344;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    
    /* Result Cards */
    .result-card-genuine {
        background: linear-gradient(135deg, #1A3E2F 0%, #0E251C 100%);
        border: 2px solid #2ECC71;
        padding: 2rem;
        border-radius: 15px;
        color: #2ECC71;
        text-align: center;
        box-shadow: 0 4px 20px rgba(46, 204, 113, 0.2);
    }
    .result-card-deepfake {
        background: linear-gradient(135deg, #4A1E1E 0%, #2A0F0F 100%);
        border: 2px solid #E74C3C;
        padding: 2rem;
        border-radius: 15px;
        color: #E74C3C;
        text-align: center;
        box-shadow: 0 4px 20px rgba(231, 76, 60, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to extract features in the exact training order
def extract_file_features(filepath):
    try:
        y, sr = sf.read(filepath, dtype='float32')
        if len(y) == 0:
            return None, None, None
        
        # Ensure mono
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
        return np.array(vector).reshape(1, -1), y, sr
    except Exception as e:
        st.error(f"Error during audio processing: {e}")
        return None, None, None

def main():
    # Sidebar
    st.sidebar.title("🎙️ Audio Deepfake Detector")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Information")
    st.sidebar.markdown("""
    * **Classifier:** Support Vector Machine (SVC)
    * **Kernel:** Radial Basis Function (RBF)
    * **Acoustic Features:** 154 dimensions (MFCCs, Spectral Centroids, Roll-off, Bandwidth, Chroma, RMS, ZCR)
    """)
    st.sidebar.subheader("Benchmark Metrics")
    st.sidebar.markdown("""
    * **Test Accuracy:** **98.79%**
    * **Equal Error Rate (EER):** **1.46%**
    * **F1-Score:** **98.80%**
    """)
    st.sidebar.markdown("---")
    st.sidebar.caption("Google DeepMind Advanced Agentic Coding Project")

    # Main layout
    st.markdown('<div class="main-header"><h1>🎙️ Deepfake Voice Audio Detector</h1><p style="font-size:1.15rem; color:#A0A0B0;">Analyze audio recordings to verify if they are genuine human speech or AI-generated synthetic deepfakes using state-of-the-art acoustic feature extraction and maximum-margin classifications.</p></div>', unsafe_allow_html=True)

    # Load Model and Scaler
    model_path = "best_model.joblib"
    scaler_path = "scaler.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.error("Error: Trained model files (best_model.joblib/scaler.joblib) not found. Run training script first.")
        return

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    # File uploader
    uploaded_file = st.file_uploader("Upload a speech audio file (.wav)", type=["wav"])

    if uploaded_file is not None:
        # Create tabs
        tab1, tab2 = st.tabs(["🔍 Detection Result", "📊 Acoustic Analysis"])
        
        with st.spinner("Extracting features and running model inference..."):
            # Write to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
                
            features_vec, y, sr = extract_file_features(tmp_path)
            
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            if features_vec is not None:
                # Scale features
                features_scaled = scaler.transform(features_vec)
                
                # Predict
                pred_label = model.predict(features_scaled)[0]
                prob = model.predict_proba(features_scaled)[0]
                
                # Probability of being deepfake
                fake_prob = prob[1]
                
                with tab1:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("Classification Outcome")
                        if pred_label == 1:
                            st.markdown(f"""
                            <div class="result-card-deepfake">
                                <h2>🚨 DEEPFAKE DETECTED</h2>
                                <p style="font-size:1.2rem; margin-top:10px;">The model has identified this voice recording as <b>AI-Generated Synthetic Speech</b>.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card-genuine">
                                <h2>✅ GENUINE SPEECH</h2>
                                <p style="font-size:1.2rem; margin-top:10px;">The model has verified this voice recording as <b>Genuine Human Speech</b>.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                        # Audio Player
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.subheader("Play Audio")
                        st.audio(uploaded_file, format="audio/wav")

                    with col2:
                        st.subheader("Confidence Scores")
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Confidence indicator bar
                        st.write(f"**Deepfake Probability:** {fake_prob*100:.2f}%")
                        st.progress(float(fake_prob))
                        
                        st.write(f"**Genuine Probability:** {(1-fake_prob)*100:.2f}%")
                        st.progress(float(1-fake_prob))
                        
                        # Explain features
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.info("""
                        **How the model decides:**
                        * **MFCCs:** Scans the vocal tract resonances to identify synthesized articulation anomalies.
                        * **Spectral Centroid & Bandwidth:** Analyzes high-frequency vocoder distortions (digital noise) typical of neural speech generators.
                        * **Zero-Crossing Rate:** Detects background synthesis silent-frame discontinuities.
                        """)
                
                with tab2:
                    st.subheader("Acoustic Feature Visualizations")
                    
                    # Waveform plot
                    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0E1117")
                    ax.set_facecolor("#1E1E2F")
                    
                    # Compute time array
                    time_axis = np.linspace(0, len(y) / sr, num=len(y))
                    ax.plot(time_axis, y, color="#00B0FF", alpha=0.8, linewidth=0.8)
                    ax.set_title("Speech Signal Waveform", color="white", fontsize=12, fontweight="bold", pad=10)
                    ax.set_xlabel("Time (seconds)", color="white")
                    ax.set_ylabel("Amplitude", color="white")
                    ax.tick_params(colors="white")
                    ax.grid(True, color="#444455", linestyle="--", alpha=0.5)
                    
                    st.pyplot(fig)
                    
                    # Key acoustic values metrics table
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("Extracted Acoustic Metrics Summary")
                    
                    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y=y))
                    spec_cent_mean = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    spec_roll_mean = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
                    
                    metrics_data = {
                        "Acoustic Parameter": [
                            "Zero Crossing Rate (ZCR)",
                            "Spectral Centroid",
                            "Spectral Roll-off"
                        ],
                        "Value": [
                            f"{zcr_mean:.4f}",
                            f"{spec_cent_mean:.2f} Hz",
                            f"{spec_roll_mean:.2f} Hz"
                        ],
                        "Acoustic Explanation": [
                            "Measures signal noisiness and voiceless speech transitions.",
                            "Represents the average frequency or brightness of the sound.",
                            "The frequency threshold below which 95% of spectral energy lies."
                        ]
                    }
                    st.table(pd.DataFrame(metrics_data))

if __name__ == '__main__':
    main()
