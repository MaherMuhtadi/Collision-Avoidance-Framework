import shelve
import numpy as np
import os
import json
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

WINDOW, FPS = 3, 30
REQUIRED_MODALITIES = ["camera_data", "lidar_data", "imu"]
raw_dir = 'SensorData'
extraction_dir = 'ExtractedData'

def label(frame_data, collision_frame, window=WINDOW, fps=FPS):
    if collision_frame is None:
        for fid, data in frame_data.items():
            data["collision"] = 0
        return
    window_frames = min(int(window*fps), len(frame_data)-1)
    for fid, data in frame_data.items():
        if collision_frame - window_frames <= fid <= collision_frame:
            data['collision'] = 1
        else:
            data['collision'] = 0

def discover_shelve_bases(base_dir="SensorData"):
    if not os.path.isdir(base_dir):
        return []
    bases = set()
    for fname in os.listdir(base_dir):
        fpath = os.path.join(base_dir, fname)
        root, ext = os.path.splitext(fpath)
        if ext.lower() == '.db':
            bases.add(root)
    return bases

def save_single_frame(frame_id, data, camera_dir, lidar_dir):
    missing = [mod for mod in REQUIRED_MODALITIES if mod not in data]

    if not missing:
        # Save RGB camera image
        path_img = os.path.join(camera_dir, f'camera_{frame_id:06d}.png')
        bgr_array = data['camera_data']
        rgb_array = bgr_array[:, :, ::-1]  # BGR to RGB
        img = Image.fromarray(rgb_array)
        img.save(path_img)
        data['camera_path'] = path_img
        del data['camera_data']

        # Save LiDAR data
        path_lidar = os.path.join(lidar_dir, f'lidar_{frame_id:06d}.npy')
        np.save(path_lidar, data['lidar_data'])
        data['lidar_path'] = path_lidar
        del data['lidar_data']

        return frame_id, data, None
    else:
        return frame_id, None, missing

def process_single_run(shelve_dir, out_dir):
    run_name = os.path.basename(shelve_dir.rstrip(os.sep))
    run_dir = os.path.join(out_dir, run_name)
    camera_dir = os.path.join(run_dir, 'Camera')
    lidar_dir  = os.path.join(run_dir, 'Lidar')
    os.makedirs(camera_dir, exist_ok=True)
    os.makedirs(lidar_dir,  exist_ok=True)

    valid_frame_data = {}
    dropped_frames = {}
    total_frames_seen = set()
    final_frame = None

    print(f"\nProcessing {run_name}...")
    try:
        with shelve.open(shelve_dir, flag='r') as frame_data:
            collision_ids = [int(fid) for fid, d in frame_data.items() if d.get('collision') == 1]
            if collision_ids:
                final_frame = min(collision_ids)
            # Convert keys to integers for ordering and truncate list until the first collision
            items = [(int(k), v) for k, v in frame_data.items() if final_frame is None or int(k) <= final_frame]
    except Exception as e:
        print(f"Error: {e}")

    print("Filtering and saving frame data...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        def _worker(item):
            return save_single_frame(item[0], item[1], camera_dir, lidar_dir)
        results = sorted(list(executor.map(_worker, items)))

    for frame_id, data, missing in results:
        total_frames_seen.add(frame_id)
        if data:
            valid_frame_data[frame_id] = data
        else:
            dropped_frames[frame_id] = missing

    print("Adding labels...")
    label(valid_frame_data, final_frame)

    # Save valid frames metadata
    with open(os.path.join(run_dir, 'extracted_data.json'), 'w') as f:
        json.dump(valid_frame_data, f, indent=2)

    # Save summary
    summary = {
        "total_frames_played": len(total_frames_seen),
        "stored_frames": len(valid_frame_data),
        "dropped_frames": len(dropped_frames),
        "collision_frame": final_frame,
        "missing_data": {str(fid): mods for fid, mods in sorted(dropped_frames.items())}
    }
    with open(os.path.join(run_dir, 'frame_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved data to '{out_dir}'.")

def extract_data(in_dir=raw_dir, out_dir=extraction_dir):
    os.makedirs(out_dir, exist_ok=True)
    runs = discover_shelve_bases(in_dir)
    if not runs:
        print(f"No shelve runs found in '{in_dir}'.")
    for run in runs:
        process_single_run(run, out_dir)

if __name__ == "__main__":
    extract_data()
