import os
import json
import math
import numpy as np
from PIL import Image

IN_DIR = "ExtractedData"
OUT_DIR = "PreprocessedData"
JSON_FILE = "extracted_data.json"
IMG_SIZE = 224
LIDAR_H = 64
LIDAR_W = 1024
FOV_UP = 15.0
FOV_DOWN = -25.0
LIDAR_MAX_RANGE = 50.0
REQUIRE_ALL_MODALITIES = True

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def runs(in_dir):
    out = []
    if not os.path.isdir(in_dir):
        return out
    for n in os.listdir(in_dir):
        d = os.path.join(in_dir, n)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, JSON_FILE)):
            out.append(d)
    return out

def load_meta(run_dir):
    with open(os.path.join(run_dir, JSON_FILE), "r") as f:
        data = json.load(f)
    items = [(int(k), v) for k, v in data.items()]
    return items

def preprocess_rgb(p):
    with Image.open(p) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(a, (2, 0, 1)).astype(np.float32)

def preprocess_lidar(pts):
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("LiDAR array must be [N,3+]")
    x, y, z = pts[:,0].astype(np.float32), pts[:,1].astype(np.float32), pts[:,2].astype(np.float32)
    r = np.sqrt(x*x + y*y + z*z)
    az = np.arctan2(y, x)
    elev = np.arctan2(z, np.sqrt(x*x + y*y))
    fup, fdown = math.radians(FOV_UP), math.radians(FOV_DOWN)
    mask = (elev >= fdown) & (elev <= fup) & np.isfinite(r) & (r > 0)
    if not np.any(mask):
        return np.zeros((LIDAR_H, LIDAR_W), dtype=np.float32)
    r, az, elev = r[mask], az[mask], elev[mask]
    u = (az + math.pi) / (2.0 * math.pi)
    v = 1.0 - (elev - fdown) / (fup - fdown)
    ui = np.clip((u * LIDAR_W).astype(np.int32), 0, LIDAR_W - 1)
    vi = np.clip((v * LIDAR_H).astype(np.int32), 0, LIDAR_H - 1)
    out = np.full((LIDAR_H * LIDAR_W,), np.inf, dtype=np.float32)
    rn = np.clip(r / float(LIDAR_MAX_RANGE), 0.0, 1.0)
    lin = vi * LIDAR_W + ui
    np.minimum.at(out, lin, rn)
    out = out.reshape(LIDAR_H, LIDAR_W)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)

def imu_stack(items):
    out = []
    for _, d in items:
        imu = d.get("imu")
        if not imu: continue
        acc, gyro = imu.get("acc"), imu.get("gyro")
        if acc and gyro and len(acc)==3 and len(gyro)==3:
            out.append(np.array(list(acc)+list(gyro), dtype=np.float32))
    return np.stack(out, axis=0) if out else np.zeros((0,6), dtype=np.float32)

def imu_stats(m):
    if m.size == 0:
        return np.zeros((6,), dtype=np.float32), np.ones((6,), dtype=np.float32)
    mean = m.mean(axis=0).astype(np.float32)
    std = m.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std

def imu_norm(imu, mean, std):
    acc = imu.get("acc", [0.0,0.0,0.0])
    gyro = imu.get("gyro", [0.0,0.0,0.0])
    v = np.array(list(acc)+list(gyro), dtype=np.float32)
    return ((v - mean) / std).astype(np.float32)

def process_run(run_dir):
    items = load_meta(run_dir)
    mean, std = imu_stats(imu_stack(items))
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    out_run = os.path.join(OUT_DIR, run_name)
    rgb_dir = os.path.join(out_run, "Camera")
    lidar_dir = os.path.join(out_run, "Lidar")
    imu_dir = os.path.join(out_run, "IMU")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)
    os.makedirs(imu_dir, exist_ok=True)
    with open(os.path.join(out_run, "imu_stats.json"), "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    index, labels = {}, {}
    kept, skipped = 0, 0
    for fid, data in items:
        cam_p = data.get("camera_path")
        lidar_p = data.get("lidar_path")
        imu_d = data.get("imu")

        if REQUIRE_ALL_MODALITIES and not (cam_p and lidar_p and imu_d):
            skipped += 1
            continue

        rec = {}
        try:
            if cam_p:
                rgb = preprocess_rgb(cam_p)
                rp = os.path.join(rgb_dir, f"rgb_{fid:06d}.npy"); np.save(rp, rgb); rec["rgb"] = rp
            if lidar_p:
                rim = preprocess_lidar(np.load(lidar_p))
                lp = os.path.join(lidar_dir, f"lidar_{fid:06d}.npy"); np.save(lp, rim); rec["lidar_range"] = lp
            if imu_d:
                iv = imu_norm(imu_d, mean, std)
                ip = os.path.join(imu_dir, f"imu_{fid:06d}.npy"); np.save(ip, iv); rec["imu_norm"] = ip
        except Exception:
            skipped += 1
            continue

        rec["label"] = int(data.get("collision", 0))
        index[str(fid)] = rec
        labels[str(fid)] = rec["label"]
        kept += 1

    with open(os.path.join(out_run, "preprocessed_data.json"), "w") as f:
        json.dump(index, f, indent=2)
    with open(os.path.join(out_run, "data_summary.json"), "w") as f:
        json.dump({
            "frames_seen": len(items),
            "frames_processed": kept,
            "frames_skipped": skipped,
        }, f, indent=2)

    print(f"'{run_name}' processed and saved to '{out_run}'")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    found = runs(IN_DIR)
    if not found:
        print(f"No runs found in '{IN_DIR}'."); return
    for rd in found:
        process_run(rd)

if __name__ == "__main__":
    main()
