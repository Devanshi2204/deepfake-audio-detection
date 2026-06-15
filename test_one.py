import os
import sys

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'
path_in_ds = "for-norm/for-norm/testing/fake/file100.wav_16k.wav_norm.wav_mono.wav_silence.wav"

print("Attempting to download:", path_in_ds, flush=True)
try:
    api.dataset_download_file(dataset, path_in_ds, path='.')
    print("Success!", flush=True)
except Exception as e:
    print("Failed with error:", type(e), e, flush=True)
