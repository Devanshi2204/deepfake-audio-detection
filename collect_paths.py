import os
import sys
import time

os.environ['KAGGLE_USERNAME'] = 'devanshi78'
os.environ['KAGGLE_KEY'] = 'cab3af2761e387b397589ec38fd6ad8b'

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'

print("Scanning pages to find files in 'for-norm'...")
start_time = time.time()
page_token = None
collected_paths = []

page_count = 0
while True:
    # Use max page size of 200
    result = api.dataset_list_files(dataset, page_token=page_token, page_size=200)
    if not result.files:
        break
        
    page_count += 1
    
    first_file = result.files[0].name
    last_file = result.files[-1].name
    
    # Check if this page contains 'for-norm'
    page_has_norm = False
    for file in result.files:
        if 'for-norm' in file.name:
            collected_paths.append(file.name)
            page_has_norm = True
            
    if page_count % 20 == 0:
        print(f"Scanned {page_count} pages. First on page: {first_file}. Collected {len(collected_paths)} norm files.")
        
    # If we have passed 'for-norm' (e.g. we reached 'for-original'), we can stop
    if not page_has_norm and len(collected_paths) > 0:
        print(f"Reached end of for-norm files. First file here: {first_file}")
        break
        
    page_token = getattr(result, 'nextPageToken', None)
    if not page_token:
        break

print(f"Completed scan in {time.time() - start_time:.2f} seconds.")
print(f"Total pages scanned: {page_count}")
print(f"Total 'for-norm' files collected: {len(collected_paths)}")

# Save to a text file
with open('norm_paths.txt', 'w') as f:
    for path in collected_paths:
        f.write(path + '\n')
print("Saved paths to 'norm_paths.txt'.")
