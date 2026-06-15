import os
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'

def load_paths(filename):
    with open(filename, 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
    return paths

def main():
    if not os.path.exists('norm_paths.txt'):
        print("Error: norm_paths.txt not found!", file=sys.stderr)
        sys.exit(1)
        
    paths = load_paths('norm_paths.txt')
    print(f"Total paths loaded: {len(paths)}")
    
    # Group paths by split and label
    groups = {
        'training/fake': [],
        'training/real': [],
        'validation/fake': [],
        'validation/real': [],
        'testing/fake': [],
        'testing/real': []
    }
    
    for path in paths:
        # Expected path format: for-norm/for-norm/testing/fake/file602.wav_16k.wav_norm.wav_mono.wav_silence.wav
        parts = path.split('/')
        if len(parts) >= 5:
            split = parts[2] # testing, training, validation
            label = parts[3] # fake, real
            key = f"{split}/{label}"
            if key in groups:
                groups[key].append(path)
                
    for key, val in groups.items():
        print(f"Group {key}: {len(val)} files available.")
        
    # Define how many files we want to download per group
    # Let's do a fast but extremely robust subset:
    # 1500 per class for training, 500 per class for validation, 500 per class for testing.
    # Total = 5000 files.
    subset_sizes = {
        'training/fake': 1500,
        'training/real': 1500,
        'validation/fake': 500,
        'validation/real': 500,
        'testing/fake': 500,
        'testing/real': 500
    }
    
    selected_files = []
    # Seed for reproducibility
    random.seed(42)
    
    for key, size in subset_sizes.items():
        available = groups[key]
        if len(available) < size:
            print(f"Warning: requested {size} files for {key}, but only {len(available)} available. Downloading all.")
            chosen = available
        else:
            chosen = random.sample(available, size)
        
        # Target folder on local disk
        target_dir = os.path.join('data', key)
        os.makedirs(target_dir, exist_ok=True)
        
        for p in chosen:
            selected_files.append((p, target_dir))
            
    print(f"Total files selected for download: {len(selected_files)}")
    
    # Download helper
    def download_file(item):
        path_in_dataset, local_dir = item
        filename = path_in_dataset.split('/')[-1]
        local_filepath = os.path.join(local_dir, filename)
        
        # Check if already exists to avoid re-downloading
        if os.path.exists(local_filepath):
            return True, path_in_dataset, False
            
        try:
            api.dataset_download_file(dataset, path_in_dataset, path=local_dir)
            # Kaggle API downloads it as file_name.zip if it's zipped or saves it directly.
            # Usually it downloads the raw wav file if it's single.
            return True, path_in_dataset, True
        except Exception as e:
            return False, path_in_dataset, str(e)

    # Download in parallel using ThreadPoolExecutor
    num_threads = 32
    print(f"Starting parallel download with {num_threads} threads...")
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    already_exists = 0
    new_downloads = 0
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(download_file, item): item for item in selected_files}
        
        for future in as_completed(futures):
            success, path, info = future.result()
            if success:
                success_count += 1
                if info: # True if newly downloaded
                    new_downloads += 1
                else:
                    already_exists += 1
            else:
                fail_count += 1
                print(f"Failed to download {path}: {info}", file=sys.stderr)
                
            total_processed = success_count + fail_count
            if total_processed % 500 == 0:
                elapsed = time.time() - start_time
                speed = total_processed / elapsed if elapsed > 0 else 0
                print(f"Progress: {total_processed}/{len(selected_files)} | Success: {success_count} | Fail: {fail_count} | Speed: {speed:.1f} files/sec")
                
    total_time = time.time() - start_time
    print(f"Finished download in {total_time:.2f} seconds.")
    print(f"Successful: {success_count} (New: {new_downloads}, Cached: {already_exists})")
    print(f"Failed: {fail_count}")

if __name__ == "__main__":
    main()
