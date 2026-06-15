import urllib.request
import time
import os

url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3"
headers = {'User-Agent': 'Mozilla/5.0'}

print("Sending request to Edinburgh DataShare...")
req = urllib.request.Request(url, headers=headers)

start_time = time.time()
try:
    with urllib.request.urlopen(req) as response:
        print("Connected! Reading first 10 MB of the dataset zip...")
        # Read 10 MB in chunks
        chunk_size = 1024 * 1024
        total_read = 0
        while total_read < 10 * 1024 * 1024:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            total_read += len(chunk)
            
    elapsed = time.time() - start_time
    speed_mb = (total_read / (1024 * 1024)) / elapsed
    print(f"Read {total_read/(1024*1024):.1f} MB in {elapsed:.2f} seconds. Speed: {speed_mb:.2f} MB/s")
except Exception as e:
    print(f"Failed to connect/download: {e}")
