import pickle
import numpy as np
import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# Create output directories if they don't exist
os.makedirs('Camera', exist_ok=True)
os.makedirs('Lidar', exist_ok=True)
os.makedirs('Logs', exist_ok=True)

def save_single_frame(frame_id, data):
    # Save only if all required modalities are present
    if all(k in data for k in ["camera_data", "lidar_data", "imu", "gnss"]):
        # Save RGB camera image
        path_img = f'Camera/camera_{frame_id:06d}.png'
        bgr_array = data['camera_data']
        rgb_array = bgr_array[:, :, ::-1]  # BGR to RGB
        img = Image.fromarray(rgb_array)
        img.save(path_img)
        data['camera_path'] = path_img
        del data['camera_data']

        # Save LiDAR data
        path_lidar = f'Lidar/lidar_{frame_id:06d}.npy'
        np.save(path_lidar, data['lidar_data'])
        data['lidar_path'] = path_lidar
        del data['lidar_data']

        return frame_id, data
    else:
        return frame_id, None

def process_raw_frames(input_file='Logs/raw_frame_data.pkl'):
    with open(input_file, 'rb') as f:
        frame_data = pickle.load(f)

    valid_frame_data = {}
    dropped_frames = []
    total_frames_seen = set()

    print("Processing frame data...")
    items = list(frame_data.items())
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda item: save_single_frame(*item), items))

    for frame_id, data in results:
        total_frames_seen.add(frame_id)
        if data:
            valid_frame_data[frame_id] = data
        else:
            dropped_frames.append(frame_id)

    with open('Logs/simulation_log.json', 'w') as f:
        json.dump(valid_frame_data, f, indent=2)
    print("Frame data log saved.")
    with open('Logs/frame_summary.json', 'w') as f:
        json.dump({
            "total_frames_played": len(total_frames_seen),
            "stored_frames": len(valid_frame_data),
            "dropped_frames": len(dropped_frames),
            "dropped_frame_ids": sorted(dropped_frames)
        }, f, indent=2)
    print("Frame drop summary saved.")

    print("Frame data processing complete.")

if __name__ == "__main__":
    process_raw_frames()
