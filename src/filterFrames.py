import shelve
import numpy as np
import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Create output directories if they don't exist
camera_dir = 'Camera'
lidar_dir = 'Lidar'
data_dir = 'Data'
os.makedirs(camera_dir, exist_ok=True)
os.makedirs(lidar_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

REQUIRED_MODALITIES = ["camera_data", "lidar_data", "imu", "collision"]

def save_single_frame(frame_id, data):
    missing = [mod for mod in REQUIRED_MODALITIES if mod not in data]

    if not missing:
        # Save RGB camera image
        path_img = f'{camera_dir}/camera_{frame_id:06d}.png'
        bgr_array = data['camera_data']
        rgb_array = bgr_array[:, :, ::-1]  # BGR to RGB
        img = Image.fromarray(rgb_array)
        img.save(path_img)
        data['camera_path'] = path_img
        del data['camera_data']

        # Save LiDAR data
        path_lidar = f'{lidar_dir}/lidar_{frame_id:06d}.npy'
        np.save(path_lidar, data['lidar_data'])
        data['lidar_path'] = path_lidar
        del data['lidar_data']

        return frame_id, data, None
    else:
        return frame_id, None, missing

def filter_raw_frames(input_file='Data/raw_data'):
    frame_data = shelve.open(input_file)

    valid_frame_data = {}
    dropped_frames = {}
    total_frames_seen = set()

    print("Filtering frame data...")
    items = [(int(k), v) for k, v in frame_data.items()] # Convert keys to integers for sorting/ordering
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda item: save_single_frame(*item), items))

    for frame_id, data, missing in results:
        total_frames_seen.add(frame_id)
        if data:
            valid_frame_data[frame_id] = data
        else:
            dropped_frames[frame_id] = missing

    # Save valid frames
    with open(f'{data_dir}/filtered_data.json', 'w') as f:
        json.dump(valid_frame_data, f, indent=2)
    print("Filtered frame data saved.")

    # Save summary
    with open(f'{data_dir}/frame_summary.json', 'w') as f:
        json.dump({
            "total_frames_played": len(total_frames_seen),
            "stored_frames": len(valid_frame_data),
            "dropped_frames": len(dropped_frames),
            "missing_data": {str(fid): mods for fid, mods in sorted(dropped_frames.items())}
        }, f, indent=2)
    print("Frame drop summary saved.")

    frame_data.close()

if __name__ == "__main__":
    filter_raw_frames()
