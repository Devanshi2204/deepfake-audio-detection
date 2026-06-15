import os
import sys
import zipfile
import time

# Set Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

def main():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print(f"Kaggle API Authentication failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'
    dest_path = '.'
    
    print("Starting download of dataset zip file (approx. 17 GB)...")
    print("This may take some time depending on connection speed.")
    start_time = time.time()
    
    try:
        # Download files without unzipping automatically to save disk space and control extraction
        api.dataset_download_files(dataset, path=dest_path, unzip=False)
        download_time = time.time() - start_time
        print(f"Download finished in {download_time/60:.2f} minutes.")
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Search for the downloaded zip file name
    zip_name = None
    for file in os.listdir(dest_path):
        if file.endswith('.zip') and 'the-fake-or-real' in file.lower():
            zip_name = file
            break
            
    if not zip_name:
        # Fallback to general zip search
        for file in os.listdir(dest_path):
            if file.endswith('.zip'):
                zip_name = file
                break
                
    if not zip_name:
        print("Error: Downloaded zip file not found in current directory.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Found downloaded zip file: {zip_name}")
    print("Extracting only 'for-norm' folder files to conserve disk space...")
    
    extract_start = time.time()
    zip_path = os.path.join(dest_path, zip_name)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            namelist = zip_ref.namelist()
            print(f"Total files in archive: {len(namelist)}")
            
            # Filter for files inside 'for-norm'
            # Note: We do case-insensitive match for 'for-norm'
            target_files = [f for f in namelist if 'for-norm' in f.lower()]
            total_target = len(target_files)
            print(f"Target files to extract (belonging to for-norm): {total_target}")
            
            if total_target == 0:
                print("Warning: No files containing 'for-norm' were found in the zip namelist.", file=sys.stderr)
                print("Available root directories in zip:")
                roots = set(f.split('/')[0] for f in namelist if '/' in f)
                print(roots)
                print("Extracting all files instead...", file=sys.stderr)
                target_files = namelist
                total_target = len(target_files)
                
            extracted_count = 0
            for file in target_files:
                zip_ref.extract(file, path=dest_path)
                extracted_count += 1
                if extracted_count % 1000 == 0 or extracted_count == total_target:
                    print(f"Extracted {extracted_count}/{total_target} files...")
                    
        print(f"Extraction completed in {(time.time() - extract_start)/60:.2f} minutes.")
        
        # Clean up zip file
        print("Cleaning up ZIP file to free disk space...")
        os.remove(zip_path)
        print("ZIP file deleted. Directory structure is ready.")
        
    except Exception as e:
        print(f"Error during extraction/cleanup: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
