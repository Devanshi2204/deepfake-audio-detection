import os
import sys

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

import kagglehub

print("Downloading version 1 of the dataset explicitly...")
try:
    path = kagglehub.dataset_download('mohammedabdeldayem/the-fake-or-real-dataset/versions/1')
    print("Success! Dataset path:", path)
except Exception as e:
    import traceback
    traceback.print_exc()
