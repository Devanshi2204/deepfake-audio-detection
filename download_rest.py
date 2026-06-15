import os
import sys
import requests
import zipfile
import time

username = 'devanshi78'
key = 'cab3af2761e387b397589ec38fd6ad8b'
dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'
zip_name = "the-fake-or-real-dataset.zip"

def download_with_resume():
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    
    # Check if there is already a partially downloaded file
    downloaded = 0
    if os.path.exists(zip_name):
        downloaded = os.path.getsize(zip_name)
        print(f"Resuming download from byte {downloaded} ({downloaded / (1024*1024):.1f} MB)...", flush=True)
    else:
        print("Starting download from scratch...", flush=True)
        
    headers = {}
    if downloaded > 0:
        headers['Range'] = f"bytes={downloaded}-"
        
    try:
        print("Requesting redirect URL from Kaggle...", flush=True)
        r_init = requests.get(url, auth=(username, key), allow_redirects=False)
        if r_init.status_code == 302:
            gcs_url = r_init.headers.get('Location')
        elif r_init.status_code == 200:
            gcs_url = url
        else:
            print(f"API Error (Status {r_init.status_code}): {r_init.text}", file=sys.stderr, flush=True)
            return False, downloaded
            
        print("Connecting to GCS storage...", flush=True)
        response = requests.get(gcs_url, headers=headers, stream=True, timeout=30)
        
        # 200 = Full Content, 206 = Partial Content
        if response.status_code not in [200, 206]:
            print(f"GCS Storage Error (Status {response.status_code})", file=sys.stderr, flush=True)
            return False, downloaded
            
    except Exception as e:
        print(f"Failed to initiate connection: {e}", file=sys.stderr, flush=True)
        return False, downloaded
        
    mode = 'ab' if response.status_code == 206 else 'wb'
    if response.status_code == 200 and downloaded > 0:
        print("Server did not support Range (returned 200 instead of 206), restarting from scratch.", flush=True)
        downloaded = 0
        
    # Content-Length for 206 is only the remaining bytes
    remaining_size = int(response.headers.get('content-length', 0))
    total_size = remaining_size + downloaded
    print(f"Stream active. Mode: {mode}. Remaining: {remaining_size / (1024*1024):.1f} MB | Total Target: {total_size / (1024*1024):.1f} MB", flush=True)
    
    start_time = time.time()
    chunk_size = 1024 * 1024 # 1 MB
    last_print = time.time()
    
    try:
        with open(zip_name, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print progress every 50 MB or 10 seconds
                    now = time.time()
                    if downloaded % (50 * 1024 * 1024) == 0 or now - last_print > 10 or downloaded == total_size:
                        elapsed = now - start_time
                        speed = (downloaded - (downloaded - len(chunk))) / (now - last_print) if (now - last_print) > 0 else 0
                        # Overall speed
                        overall_speed = downloaded / (1024 * 1024 * elapsed) if elapsed > 0 else 0
                        percent = (downloaded / total_size) * 100 if total_size > 0 else 0
                        print(f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%) | Speed: {overall_speed:.2f} MB/s", flush=True)
                        last_print = now
        return downloaded == total_size, downloaded
    except Exception as e:
        print(f"Download interrupted: {e}", file=sys.stderr, flush=True)
        return False, downloaded

def main():
    while True:
        success, downloaded = download_with_resume()
        if success:
            print("Download completed successfully!", flush=True)
            break
        else:
            print("Download interrupted. Retrying in 10 seconds...", flush=True)
            time.sleep(10)
            
    print("Extracting 'for-norm' folder from zip file...", flush=True)
    extract_start = time.time()
    try:
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            namelist = zip_ref.namelist()
            target_files = [f for f in namelist if 'for-norm' in f.lower()]
            total_target = len(target_files)
            print(f"Total files to extract: {total_target}", flush=True)
            
            extracted = 0
            for file in target_files:
                zip_ref.extract(file, path='.')
                extracted += 1
                if extracted % 2000 == 0 or extracted == total_target:
                    print(f"Extracted {extracted}/{total_target} files...", flush=True)
                    
        print(f"Extraction complete in {(time.time() - extract_start)/60:.2f} minutes.", flush=True)
        print("Cleaning up ZIP file...", flush=True)
        os.remove(zip_name)
        print("Cleanup complete. Dataset is ready!", flush=True)
    except Exception as e:
        print(f"Error during extraction: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
