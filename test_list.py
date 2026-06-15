import os

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'
path = "for-norm/for-norm/training/fake/file100.wav_16k.wav_norm.wav_mono.wav_silence.wav"

try:
    api.dataset_download_file(dataset, path, path='.')
    print("Success training file downloaded!")
except Exception as e:
    print(f"Failed: {e}")
