import os
import sys
import glob
import random
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from joblib import Parallel, delayed
import time

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def extract_single_file_features(filepath, label):
    """
    Extracts acoustic features from a single WAV file using soundfile.read.
    """
    try:
        # Load audio using soundfile
        y, sr = sf.read(filepath, dtype='float32')
        if len(y) == 0:
            return None
        
        # Ensure mono
        if len(y.shape) > 1:
            y = np.mean(y, axis=1)
            
        # 1. MFCC (20 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        # 2. Delta MFCC
        mfcc_delta = librosa.feature.delta(mfcc)
        # 3. Delta-Delta MFCC
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 4. Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        # 5. Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        # 6. Spectral Roll-off
        spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # 7. RMS Energy
        rms = librosa.feature.rms(y=y)
        # 8. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        
        # 9. Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Aggregate features via mean and standard deviation over time axis
        features = {
            'filepath': filepath,
            'label': label,        # 1 for fake, 0 for real
        }
        
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
            
        return features
    except Exception as e:
        return None

def process_file_wrapper(item):
    filepath, label = item
    res = extract_single_file_features(filepath, label)
    return res

def main():
    base_dir = r"for-norm/for-norm"
    if not os.path.exists(base_dir):
        print(f"Error: Dataset directory {base_dir} not found.", flush=True)
        sys.exit(1)
            
    print(f"Scanning and pooling dataset from: {base_dir}", flush=True)
    
    # Pool all fake and real files from all directories
    all_fake_files = []
    all_real_files = []
    
    for split_name in ['training', 'validation', 'testing']:
        fake_path = os.path.join(base_dir, split_name, 'fake', '*.wav')
        real_path = os.path.join(base_dir, split_name, 'real', '*.wav')
        all_fake_files.extend(glob.glob(fake_path))
        all_real_files.extend(glob.glob(real_path))
        
    print(f"Total pooled fake files: {len(all_fake_files)}", flush=True)
    print(f"Total pooled real files: {len(all_real_files)}", flush=True)
    
    # We want a balanced dataset of 12,000 files (6,000 fake, 6,000 real)
    target_fake_count = 6000
    target_real_count = 6000
    
    sampled_fake = random.sample(all_fake_files, target_fake_count)
    sampled_real = random.sample(all_real_files, target_real_count)
    
    # Create the item list (filepath, label)
    items = []
    for f in sampled_fake:
        items.append((f, 1))
    for f in sampled_real:
        items.append((f, 0))
        
    # Shuffle the items
    random.shuffle(items)
    
    # Run feature extraction in parallel using joblib
    print(f"Starting feature extraction for {len(items)} balanced files...", flush=True)
    start_time = time.time()
    
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_file_wrapper)(item) for item in items
    )
    
    # Filter out None values
    valid_results = [r for r in results if r is not None]
    
    elapsed_time = time.time() - start_time
    print(f"Feature extraction completed in {elapsed_time:.2f} seconds.", flush=True)
    print(f"Successfully extracted features for {len(valid_results)} files.", flush=True)
    
    # Create DataFrame
    df = pd.DataFrame(valid_results)
    
    # Assign stratified random splits: 60% training, 20% validation, 20% testing
    # To do this correctly, we group by label and assign split
    df['split'] = 'training' # default
    
    for label_val in [0, 1]:
        label_indices = df[df['label'] == label_val].index.tolist()
        random.shuffle(label_indices)
        
        n_total = len(label_indices)
        n_train = int(0.6 * n_total)
        n_val = int(0.2 * n_total)
        
        train_idx = label_indices[:n_train]
        val_idx = label_indices[n_train:n_train + n_val]
        test_idx = label_indices[n_train + n_val:]
        
        df.loc[train_idx, 'split'] = 'training'
        df.loc[val_idx, 'split'] = 'validation'
        df.loc[test_idx, 'split'] = 'testing'
        
    output_csv = "extracted_features.csv"
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}. Shape: {df.shape}", flush=True)
    print(df.groupby(['split', 'label']).size(), flush=True)

if __name__ == '__main__':
    main()
