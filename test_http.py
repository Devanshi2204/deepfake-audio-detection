import os
import requests

username = 'devanshi78'
key = 'cab3af2761e387b397589ec38fd6ad8b'

dataset = 'mohammedabdeldayem/the-fake-or-real-dataset'

# Direct dataset download URL
url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"

print("Sending GET request to:", url)
try:
    response = requests.get(url, auth=(username, key), allow_redirects=False)
    print("Status Code:", response.status_code)
    print("Response Headers:")
    for k, v in response.headers.items():
        print(f"  {k}: {v}")
    
    if response.status_code >= 300 and response.status_code < 400:
        print("Redirect Location:", response.headers.get('Location'))
        
    print("\nResponse Content (first 500 chars):")
    print(response.text[:500])
except Exception as e:
    print("Request failed:", e)
