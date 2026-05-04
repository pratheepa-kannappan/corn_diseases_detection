import torch
import zipfile
import os

# Try to repair by reading as zipfile
try:
    with zipfile.ZipFile('best_model.pth', 'r') as z:
        print("Files inside:", z.namelist())
        z.extractall('model_extracted')
    print("Extraction successful!")
except Exception as e:
    print(f"Error: {e}")

# Check file size
size = os.path.getsize('best_model.pth')
print(f"File size: {size / 1024 / 1024:.2f} MB")
