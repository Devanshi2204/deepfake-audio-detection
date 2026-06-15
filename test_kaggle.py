import os
import traceback

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    print("Downloading dataset...")
    api.dataset_download_files('mohammedabdeldayem/the-fake-or-real-dataset', path='.', unzip=False)
    print("Done!")
except Exception as e:
    traceback.print_exc()
