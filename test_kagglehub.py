import os
import sys
import time

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

try:
    import kagglehub
    print("Kagglehub imported successfully.")
except ImportError:
    print("Installing kagglehub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "kagglehub"])
    import kagglehub
    print("Kagglehub installed and imported.")

print("Starting download of 'mohammedabdeldayem/the-fake-or-real-dataset' via kagglehub...")
start_time = time.time()
try:
    # This downloads and extracts the dataset, caching it locally
    path = kagglehub.dataset_download('mohammedabdeldayem/the-fake-or-real-dataset')
    print("Download and extraction complete!")
    print(f"Dataset path: {path}")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
except Exception as e:
    import traceback
    traceback.print_exc()
