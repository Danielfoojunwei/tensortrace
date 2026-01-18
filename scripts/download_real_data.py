
from huggingface_hub import hf_hub_download
import os

repo_id = "FastUMIPro/example_data_fastumi_pro_raw"
dest_dir = "data/fastumi_pro"
os.makedirs(dest_dir, exist_ok=True)

# Selection of real data files (Verified paths from listing)
files_to_get = [
    "task1/session_1/left_hand_250801DR48FP25002314/RGB_Images/video.mp4",
    "task1/session_1/left_hand_250801DR48FP25002314/Merged_Trajectory/merged_trajectory.txt"
]

print(f"Downloading real FastUMI Pro samples from {repo_id}...")
for f in files_to_get:
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=f, local_dir=dest_dir, repo_type="dataset")
        print(f"Verified: {local_path}")
    except Exception as e:
        print(f"Failed to download {f}: {e}")
