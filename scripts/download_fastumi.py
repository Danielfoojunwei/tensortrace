
from huggingface_hub import hf_hub_download
import os

# Create data directory
os.makedirs("data/fastumi/pouring", exist_ok=True)
os.makedirs("data/fastumi/screwing", exist_ok=True)

# List of files to download (Small sample for empirical test)
files = [
    {"repo": "IPEC-COMMUNITY/FastUMI-Data", "filename": "data/fastumi/pouring/episode_0.hdf5", "local": "data/fastumi/pouring/episode_0.hdf5"},
    {"repo": "IPEC-COMMUNITY/FastUMI-Data", "filename": "data/fastumi/screwing/episode_0.hdf5", "local": "data/fastumi/screwing/episode_0.hdf5"}
]

# Note: In a real environment, we'd need to handle auth if it's private.
# The user's metadata suggests they want to see the real thing.

print("Starting download of FastUMI samples...")
for f in files:
    try:
        # We use a try-except because some files might be named differently in the actual repo structure
        # I will first list the files in the repo to be sure
        pass
    except Exception as e:
        print(f"Error downloading {f['filename']}: {e}")

# Revised strategy: Use the hub API to list and download first N HDF5 files
from huggingface_hub import HfApi
api = HfApi()
repo_id = "IPEC-COMMUNITY/FastUMI-Data"

try:
    all_files = api.list_repo_files(repo_id, repo_type="dataset")
    hdf5_files = [f for f in all_files if f.endswith(".hdf5")]
    print(f"Found {len(hdf5_files)} HDF5 files.")
    
    # Download top 2 files for different tasks if possible, or just the first two
    for i in range(min(2, len(hdf5_files))):
        target = hdf5_files[i]
        local_path = os.path.join("data/fastumi", target)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {target}...")
        hf_hub_download(repo_id=repo_id, filename=target, local_dir="data/fastumi", repo_type="dataset")
        print(f"Saved to {local_path}")
        
except Exception as e:
    print(f"Hub error: {e}")
