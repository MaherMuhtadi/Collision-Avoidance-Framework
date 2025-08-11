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
    present = {
        "camera_data": "camera_data" in data,
        "lidar_data": "lidar_data" in data,
        "imu": "imu" in data,
    }
    missing = [k for k,v in present.items() if not v]

    if not present["imu"]:
        data['imu'] = None

    # Camera: save if present, else path None
    if present["camera_data"]:
        path_img = os.path.join(camera_dir, f"camera_{frame_id:06d}.png")
        bgr = data["camera_data"]
        rgb = bgr[:, :, ::-1]  # BGR -> RGB
        Image.fromarray(rgb).save(path_img)
        data["camera_path"] = path_img
        del data["camera_data"]
    else:
        data["camera_path"] = None

    # LiDAR: save if present, else path None
    if present["lidar_data"]:
        path_lidar = os.path.join(lidar_dir, f"lidar_{frame_id:06d}.npy")
        np.save(path_lidar, data["lidar_data"])
        data["lidar_path"] = path_lidar
        del data["lidar_data"]
    else:
        data["lidar_path"] = None

    data["modalities"] = {
        "imu": present["imu"],
        "camera": present["camera_data"],
        "lidar": present["lidar_data"]
    }

    return frame_id, data, missing

def process_single_run(shelve_dir, out_dir):
    run_name = os.path.basename(shelve_dir.rstrip(os.sep))
    run_dir = os.path.join(out_dir, run_name)
    camera_dir = os.path.join(run_dir, 'Camera')
    lidar_dir  = os.path.join(run_dir, 'Lidar')
    os.makedirs(camera_dir, exist_ok=True)
    os.makedirs(lidar_dir,  exist_ok=True)

    final_frame_data = {}
    padded_frames = {}
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

    print("Padding and saving frame data...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        def _worker(item):
            return save_single_frame(item[0], item[1], camera_dir, lidar_dir)
        results = sorted(list(executor.map(_worker, items)))

    for frame_id, data, missing in results:
        total_frames_seen.add(frame_id)
        final_frame_data[frame_id] = data
        if missing:
            padded_frames[frame_id] = missing

    print("Adding labels...")
    label(final_frame_data, final_frame)

    # Save valid frames metadata
    with open(os.path.join(run_dir, 'extracted_data.json'), 'w') as f:
        json.dump(final_frame_data, f, indent=2)

    # Save summary
    summary = {
        "frames_played": len(total_frames_seen),
        "padded_frames": len(padded_frames),
        "collision_frame_id": final_frame,
        "missing_modalities": {str(fid): mods for fid, mods in sorted(padded_frames.items())}
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
