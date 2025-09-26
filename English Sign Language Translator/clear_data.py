import os
import shutil

DATA_DIR = './data'

if os.path.exists(DATA_DIR):
    for subdir in os.listdir(DATA_DIR):
        subdir_path = os.path.join(DATA_DIR, subdir)
        if os.path.isdir(subdir_path):
            shutil.rmtree(subdir_path)
    print("✅ All images deleted, empty data folder remains.")
else:
    print("⚠️ No data folder found.")
