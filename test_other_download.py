import os
import sys

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# Try downloading a small 12MB news dataset
dataset = 'jillanisofttech/fake-or-real-news'

print(f"Attempting to download a test dataset: {dataset}...", flush=True)
try:
    api.dataset_download_files(dataset, path='.', unzip=False)
    print("Success! Download completed for test dataset.", flush=True)
    # Cleanup downloaded test file
    for file in os.listdir('.'):
        if 'fake-or-real-news' in file.lower() and file.endswith('.zip'):
            os.remove(file)
            print("Cleaned up test file.", flush=True)
except Exception as e:
    print("Failed with error:", type(e), e, flush=True)
