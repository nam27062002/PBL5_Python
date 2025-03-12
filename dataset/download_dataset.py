import kagglehub
import shutil
import os

dest = './raw'
path = kagglehub.dataset_download("lexset/synthetic-asl-alphabet")
print(path)
os.makedirs(dest, exist_ok=True)
shutil.move(path, dest)
path = dest
print("Path to dataset files:", path)