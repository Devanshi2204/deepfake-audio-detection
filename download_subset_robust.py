import os
import sys
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import urllib.error

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Authentication failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'
    
    if not os.path.exists('norm_paths.txt'):
        print("Error: norm_paths.txt not found!", file=sys.stderr)
        sys.exit(1)
        
    with open('norm_paths.txt', 'r') as f:
        paths = [line.strip() for line in f if line.strip()]
        
    groups = {
        'training/fake': [],
        'training/real': [],
        'validation/fake': [],
        'validation/real': [],
        'testing/fake': [],
        'testing/real': []
    }
    
    for path in paths:
        parts = path.split('/')
        if len(parts) >= 5:
            split = parts[2]
            label = parts[3]
            key = f"{split}/{label}"
            if key in groups:
                groups[key].append(path)
                
    # Select 300 per class for training, 100 for validation, 100 for testing.
    # Total = 1000 files.
    subset_sizes = {
        'training/fake': 300,
        'training/real': 300,
        'validation/fake': 100,
        'validation/real': 100,
        'testing/fake': 100,
        'testing/real': 100
    }
    
    random.seed(42)
    selected_files = []
    
    for key, size in subset_sizes.items():
        available = groups[key]
        chosen = random.sample(available, min(len(available), size))
        target_dir = os.path.join('data', key)
        os.makedirs(target_dir, exist_ok=True)
        for p in chosen:
            selected_files.append((p, target_dir))
            
    print(f"Total files selected: {len(selected_files)}")
    
    import threading
    rate_lock = threading.Lock()
    last_request_time = [0.0]
    
    # Thread-safe download function with rate limiting and backoff
    def download_file(item):
        path_in_ds, local_dir = item
        filename = path_in_ds.split('/')[-1]
        local_filepath = os.path.join(local_dir, filename)
        
        if os.path.exists(local_filepath):
            return True, path_in_ds, False
            
        max_retries = 5
        backoff = 1.0
        
        for attempt in range(max_retries):
            # Enforce request spacing
            with rate_lock:
                now = time.time()
                elapsed = now - last_request_time[0]
                # spacing of 0.25 seconds between requests (4 requests per second max)
                if elapsed < 0.25:
                    time.sleep(0.25 - elapsed)
                last_request_time[0] = time.time()
                
            try:
                # Call kaggle api to download
                api.dataset_download_file(dataset, path_in_ds, path=local_dir)
                return True, path_in_ds, True
            except Exception as e:
                err_msg = str(e)
                if '429' in err_msg or '404' in err_msg or 'Too Many Requests' in err_msg:
                    # Sleep and backoff
                    print(f"Rate limited or error on {filename} (attempt {attempt+1}/{max_retries}): {err_msg}. Backing off {backoff:.1f}s...")
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    # Print unexpected error and retry
                    print(f"Error on {filename} (attempt {attempt+1}/{max_retries}): {err_msg}. Retrying...")
                    time.sleep(1.0)
                    
        return False, path_in_ds, f"Failed after {max_retries} attempts"

    # Use 4 threads to prevent hitting rate-limiting thresholds
    num_workers = 4
    print(f"Starting rate-limited parallel download using {num_workers} threads...")
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    cached_count = 0
    new_downloads = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_file, item): item for item in selected_files}
        
        for future in as_completed(futures):
            success, path, info = future.result()
            if success:
                success_count += 1
                if info:
                    new_downloads += 1
                else:
                    cached_count += 1
            else:
                fail_count += 1
                print(f"Permanently failed to download {path}: {info}", file=sys.stderr)
                
            total_processed = success_count + fail_count
            if total_processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {total_processed}/{len(selected_files)} | Success: {success_count} (New: {new_downloads}, Cached: {cached_count}) | Fail: {fail_count} | Elapsed: {elapsed/60:.2f} min")
                
    print(f"Download complete in {(time.time() - start_time)/60:.2f} minutes.")
    print(f"Total Success: {success_count} | Total Fail: {fail_count}")

if __name__ == "__main__":
    main()
