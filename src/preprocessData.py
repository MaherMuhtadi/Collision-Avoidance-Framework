import os
import json
import math
import numpy as np
from PIL import Image

IN_DIR = "ExtractedData"
OUT_DIR = "PreprocessedData"
JSON_FILE = "extracted_data.json"
IMG_SIZE = 224

with open('lidar_settings.json', 'r') as f:
    lidar_settings = json.load(f)
LIDAR_H = int(lidar_settings["CHANNELS"])
LIDAR_W = max(1, int(int(lidar_settings["PPS"]) / max(float(lidar_settings["ROT_FREQ"]), 1e-6) / max(LIDAR_H, 1)))
FOV_UP = float(lidar_settings["FOV_UP"])
FOV_DOWN = float(lidar_settings["FOV_DOWN"])
LIDAR_MAX_RANGE = float(lidar_settings["MAX_RANGE"])

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
    m = (elev >= fdown) & (elev <= fup) & np.isfinite(r) & (r > 0)

    # No points -> all zeros (range and mask)
    if not np.any(m):
        return (np.zeros((LIDAR_H, LIDAR_W), dtype=np.float32),
                np.zeros((LIDAR_H, LIDAR_W), dtype=np.uint8))

    r, az, elev = r[m], az[m], elev[m]
    u = (az + math.pi) / (2.0 * math.pi)
    v = 1.0 - (elev - fdown) / (fup - fdown)
    ui = np.clip((u * LIDAR_W).astype(np.int32), 0, LIDAR_W - 1)
    vi = np.clip((v * LIDAR_H).astype(np.int32), 0, LIDAR_H - 1)

    flat = np.full((LIDAR_H * LIDAR_W,), np.inf, dtype=np.float32)
    rn = np.clip(r / float(LIDAR_MAX_RANGE), 0.0, 1.0)
    lin = vi * LIDAR_W + ui
    np.minimum.at(flat, lin, rn)

    rim = flat.reshape(LIDAR_H, LIDAR_W)
    valid = np.isfinite(rim)
    rim[~valid] = 0.0
    return rim.astype(np.float32), valid.astype(np.uint8)

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

def process_run(run_dir, global_mean, global_std):
    items = load_meta(run_dir)
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    out_run = os.path.join(OUT_DIR, run_name)
    rgb_dir = os.path.join(out_run, "Camera")
    lidar_dir = os.path.join(out_run, "Lidar")
    imu_dir = os.path.join(out_run, "IMU")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(lidar_dir, exist_ok=True)
    os.makedirs(imu_dir, exist_ok=True)

    index, labels = {}, {}
    kept, skipped = 0, 0
    miss_rgb = miss_lidar = miss_imu = 0

    for fid, data in items:
        cam_p   = data.get("camera_path")
        lidar_p = data.get("lidar_path")
        imu_d   = data.get("imu")

        rec = {}
        # 1 = present, 0 = padded
        m_rgb   = 1 if cam_p   else 0
        m_lidar = 1 if lidar_p else 0
        m_imu   = 1 if imu_d   else 0

        try:
            if m_rgb:
                rgb = preprocess_rgb(cam_p)
            else:
                rgb = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
                miss_rgb += 1
            rp = os.path.join(rgb_dir, f"rgb_{fid:06d}.npy")
            np.save(rp, rgb)
            rec["rgb"] = rp

            if m_lidar:
                rim, vmask = preprocess_lidar(np.load(lidar_p))
            else:
                rim, vmask = (np.zeros((LIDAR_H, LIDAR_W), dtype=np.float32),
                              np.zeros((LIDAR_H, LIDAR_W), dtype=np.uint8))
                miss_lidar += 1
            lp  = os.path.join(lidar_dir, f"lidar_{fid:06d}.npy")
            np.save(lp,  rim)
            rec["lidar_range"] = lp
            lvp = os.path.join(lidar_dir, f"lidar_valid_{fid:06d}.npy")
            np.save(lvp, vmask)
            rec["lidar_valid"] = lvp

            if m_imu:
                iv = imu_norm(imu_d, global_mean, global_std)
            else:
                iv = np.zeros((6,), dtype=np.float32)
                miss_imu += 1
            ip = os.path.join(imu_dir, f"imu_{fid:06d}.npy")
            np.save(ip, iv)
            rec["imu"] = ip

        except Exception:
            skipped += 1
            continue

        # Frame-level presence mask the model can read
        rec["modality_mask"] = {"rgb": m_rgb, "lidar": m_lidar, "imu": m_imu}

        rec["label"] = int(data.get("collision", 0))
        index[str(fid)] = rec
        labels[str(fid)] = rec["label"]
        kept += 1

    # When writing the summary, include missing counts
    with open(os.path.join(out_run, "preprocessed_data.json"), "w") as f:
        json.dump(index, f, indent=2)
    with open(os.path.join(out_run, "summary.json"), "w") as f:
        json.dump({
            "frames_seen": len(items),
            "frames_processed": kept,
            "frames_skipped": skipped,
            "missing_modalities_counts": {"rgb": miss_rgb, "lidar": miss_lidar, "imu": miss_imu}
        }, f, indent=2)

    print(f"'{run_name}' processed and saved to '{out_run}'")

def compute_global_imu_stats(run_dirs):
    # Numerically stable running mean/std (Welford) across all frames
    count = 0
    mean = np.zeros(6, dtype=np.float64)
    M2   = np.zeros(6, dtype=np.float64)
    for rd in run_dirs:
        items = load_meta(rd)
        m = imu_stack(items)  # [N,6]
        if m.size == 0: 
            continue
        for v in m:
            count += 1
            delta = v - mean
            mean += delta / count
            delta2 = v - mean
            M2 += delta * delta2
    if count == 0:
        mean_f = np.zeros((6,), dtype=np.float32)
        std_f  = np.ones((6,), dtype=np.float32)
    else:
        var = (M2 / max(count - 1, 1))
        std_f  = np.sqrt(var).astype(np.float32)
        std_f  = np.where(std_f < 1e-6, 1.0, std_f)
        mean_f = mean.astype(np.float32)
    return mean_f, std_f

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    found = runs(IN_DIR)
    if not found:
        print(f"No runs found in '{IN_DIR}'.")
        return
    g_mean, g_std = compute_global_imu_stats(found)
    with open(os.path.join(OUT_DIR, "imu_stats_global.json"), "w") as f:
        json.dump({"mean": g_mean.tolist(), "std": g_std.tolist()}, f, indent=2)
    for rd in found:
        process_run(rd, g_mean, g_std)

if __name__ == "__main__":
    main()
