import os
import json
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torchvision import models
except Exception:
    models = None

DATASET_DIR = "Dataset"
PREPROCESSED_ROOT = "PreprocessedData"
OUT_DIR = "Tokens"
EMBED_DIM = 384
BETA = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_PRETRAINED_CAMERA = True
FREEZE_BACKBONES = True
BATCH_SIZE = 8

SEED = 777
torch.manual_seed(SEED)
np.random.seed(SEED)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def slugify(s: str) -> str:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"{s.replace(os.sep, '_')}_{h}"

def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


_RUN_INDEX: Dict[str, Dict[int, dict]] = {}

def _get_run_index(run: str) -> Dict[int, dict]:
    """Load and cache frame mapping from preprocessed_data.json."""
    if run in _RUN_INDEX:
        return _RUN_INDEX[run]
    pre_json = os.path.join(PREPROCESSED_ROOT, run, "preprocessed_data.json")
    if not os.path.isfile(pre_json):
        raise FileNotFoundError(f"Expected per-run preprocessed file not found: {pre_json}")
    with open(pre_json, "r") as f:
        data = json.load(f)
    mapping: Dict[int, dict] = {}
    if isinstance(data, dict):
        for k, rec in data.items():
            try:
                fid = int(k)
            except Exception:
                continue
            mapping[fid] = rec
    elif isinstance(data, list):
        for rec in data:
            try:
                fid = int(rec.get("frame_id"))
            except Exception:
                continue
            mapping[fid] = rec
    else:
        raise ValueError("Unsupported format for preprocessed_data.json (expected dict or list).")
    _RUN_INDEX[run] = mapping
    return mapping

def _parse_modality_mask(mm) -> Tuple[bool, bool, bool]:
    """Return (image, lidar, imu) availability booleans."""  
    if isinstance(mm, dict):
        return bool(mm.get("image", 1)), bool(mm.get("lidar", 1)), bool(mm.get("imu", 1))
    if isinstance(mm, (list, tuple)) and len(mm) >= 3:
        return bool(mm[0]), bool(mm[1]), bool(mm[2])
    return True, True, True

def get_frame_ids(sample: Dict) -> List[int]:
    """Extract frame IDs from window sample."""
    fids = sample.get("frame_ids")
    if isinstance(fids, list) and len(fids) > 0:
        return [int(x) for x in fids]
    if "start_frame" in sample and "end_frame" in sample:
        s = int(sample["start_frame"]); e = int(sample["end_frame"])
        if e < s:
            raise ValueError(f"end_frame {e} < start_frame {s}")
        return list(range(s, e + 1))
    if "end_fid" in sample and "window_frames" in sample:
        e = int(sample["end_fid"]); T = int(sample["window_frames"])
        return list(range(e - T + 1, e + 1))
    return []

def sinusoidal_position_encoding(length: int, dim: int, device: str) -> torch.Tensor:
    """Standard transformer sinusoidal positional encoding."""
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    position = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class ResNet18Trunk(nn.Module):
    """ResNet-18 trunk returning 512-d features."""
    def __init__(self, in_ch: int = 3, pretrained: bool = False):
        super().__init__()
        if models is None:
            raise RuntimeError("torchvision is required for ResNet backbones but isn't available.")

        if pretrained and in_ch == 3:
            try:
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet18(weights=None)

        if in_ch != 3:
            old = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                                     stride=old.stride, padding=old.padding, bias=False)
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class CameraEmbedding(nn.Module):
    def __init__(self, out_dim=EMBED_DIM, pretrained=True, freeze=True):
        super().__init__()
        self.trunk = ResNet18Trunk(in_ch=3, pretrained=pretrained)
        self.proj = nn.Linear(512, out_dim)
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        e = self.proj(feat)
        return e

class LiDAREmbedding(nn.Module):
    def __init__(self, in_ch=2, out_dim=EMBED_DIM, freeze=True):
        super().__init__()
        self.trunk = ResNet18Trunk(in_ch=in_ch, pretrained=False)
        self.proj = nn.Linear(512, out_dim)
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.trunk(x)
        e = self.proj(feat)
        return e

class IMUEmbedding(nn.Module):
    def __init__(self, out_dim=EMBED_DIM, hidden=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class AvailabilityEmbedding(nn.Module):
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(4, out_dim)
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.proj(a)

class MultiModalTokenizer(nn.Module):
    """Multi-modal tokenizer with carry-forward imputation and temporal encoding."""
    def __init__(self, embed_dim=EMBED_DIM, beta=BETA,
                 use_pretrained_camera=USE_PRETRAINED_CAMERA, freeze_backbones=FREEZE_BACKBONES,
                 device=DEVICE):
        super().__init__()
        self.device = torch.device(device)
        self.embed_dim = embed_dim
        self.beta = beta

        self.image_enc = CameraEmbedding(out_dim=embed_dim,
                                         pretrained=use_pretrained_camera,
                                         freeze=freeze_backbones)
        self.lidar_enc = LiDAREmbedding(in_ch=2, out_dim=embed_dim,
                                        freeze=freeze_backbones)
        self.imu_enc   = IMUEmbedding(out_dim=embed_dim)
        self.avail_enc = AvailabilityEmbedding(out_dim=embed_dim)

        self.e_miss_image = nn.Parameter(torch.zeros(embed_dim))
        self.e_miss_lidar = nn.Parameter(torch.zeros(embed_dim))
        self.e_miss_imu   = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.e_miss_image, mean=0.0, std=0.02)
        nn.init.normal_(self.e_miss_lidar, mean=0.0, std=0.02)
        nn.init.normal_(self.e_miss_imu,   mean=0.0, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def _carry_forward(self, embeds: torch.Tensor, avail: torch.Tensor, e_miss: torch.Tensor) -> torch.Tensor:
        """Apply carry-forward imputation with decay when modality is missing."""
        out = embeds.clone()
        prev = None
        for t in range(out.shape[0]):
            if avail[t] >= 0.5:
                prev = out[t]
            else:
                if prev is None:
                    out[t] = e_miss
                else:
                    out[t] = self.beta * prev + (1 - self.beta) * e_miss
        return out

    @torch.no_grad()
    def _embed_modalities(self,
                          imgs: torch.Tensor, a_img: torch.Tensor,
                          rims: torch.Tensor, a_lidar: torch.Tensor,
                          imus: torch.Tensor, a_imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute embeddings for present modalities."""
        B = imgs.shape[0]
        e_img = torch.zeros((B, self.embed_dim), device=self.device)
        if a_img.any():
            idx = torch.nonzero(a_img, as_tuple=False).squeeze(1)
            e_img[idx] = self.image_enc(imgs[idx])

        e_lidar = torch.zeros((B, self.embed_dim), device=self.device)
        if a_lidar.any():
            idx = torch.nonzero(a_lidar, as_tuple=False).squeeze(1)
            e_lidar[idx] = self.lidar_enc(rims[idx])

        e_imu = torch.zeros((B, self.embed_dim), device=self.device)
        if a_imu.any():
            idx = torch.nonzero(a_imu, as_tuple=False).squeeze(1)
            e_imu[idx] = self.imu_enc(imus[idx])

        return e_img, e_lidar, e_imu

    @torch.no_grad()
    def tokenize_window(self, window: Dict) -> Dict:
        """Tokenize one window into multi-modal tokens."""
        frame_ids: List[int] = get_frame_ids(window)
        if not frame_ids:
            frame_ids = [int(x) for x in window.get("frame_ids", [])]
        T = len(frame_ids)

        has_explicit_files = all(k in window for k in ["image", "lidar_range", "lidar_mask", "imu", "modality_masks"])
        imgs, rims, imus = [], [], []
        a_img_list, a_lidar_list, a_imu_list = [], [], []

        if has_explicit_files:
            for t in range(T):
                img_arr = np.load(window["image"][t]).astype(np.float32)
                imgs.append(torch.from_numpy(img_arr))
                rim = np.load(window["lidar_range"][t]).astype(np.float32)
                vmask = np.load(window["lidar_mask"][t]).astype(np.uint8)
                rim2 = np.stack([rim, vmask.astype(np.float32)], axis=0)
                rims.append(torch.from_numpy(rim2))
                imu = np.load(window["imu"][t]).astype(np.float32)
                imus.append(torch.from_numpy(imu))
                mm = window["modality_masks"][t]
                a_img_list.append(bool(mm["image"]) if isinstance(mm, dict) else bool(mm[0]))
                a_lidar_list.append(bool(mm["lidar"]) if isinstance(mm, dict) else bool(mm[1]))
                a_imu_list.append(bool(mm["imu"]) if isinstance(mm, dict) else bool(mm[2]))
        else:
            run = window.get("run")
            if not run:
                raise KeyError("Missing 'run' in window sample; required for new format.")
            index = _get_run_index(run)
            for fid in frame_ids:
                rec = index.get(int(fid))
                if rec is None:
                    raise KeyError(f"Frame id {fid} not found in preprocessed index for run '{run}'.")
                img_arr = np.load(rec["image"]).astype(np.float32)
                imgs.append(torch.from_numpy(img_arr))
                rim = np.load(rec["lidar_range"]).astype(np.float32)
                vmask = np.load(rec["lidar_mask"]).astype(np.uint8)
                rim2 = np.stack([rim, vmask.astype(np.float32)], axis=0)
                rims.append(torch.from_numpy(rim2))
                imu = np.load(rec["imu"]).astype(np.float32)
                imus.append(torch.from_numpy(imu))
                mm = rec.get("modality_mask", rec.get("modality_masks", None))
                a_img, a_lidar, a_imu = _parse_modality_mask(mm)
                a_img_list.append(a_img); a_lidar_list.append(a_lidar); a_imu_list.append(a_imu)
        imgs_t = torch.stack(imgs, dim=0).to(torch.float32).to(self.device)
        rims_t = torch.stack(rims, dim=0).to(torch.float32).to(self.device)
        imus_t = torch.stack(imus, dim=0).to(torch.float32).to(self.device)

        a_img = torch.tensor(a_img_list, dtype=torch.float32, device=self.device)
        a_lidar = torch.tensor(a_lidar_list, dtype=torch.float32, device=self.device)
        a_imu = torch.tensor(a_imu_list, dtype=torch.float32, device=self.device)

        e_img, e_lidar, e_imu = self._embed_modalities(imgs_t, a_img, rims_t, a_lidar, imus_t, a_imu)

        e_img   = self._carry_forward(e_img, a_img,   self.e_miss_image)
        e_lidar = self._carry_forward(e_lidar, a_lidar, self.e_miss_lidar)
        e_imu   = self._carry_forward(e_imu, a_imu,   self.e_miss_imu)

        avail_vec = torch.stack([a_img, a_lidar, a_imu, (a_img + a_lidar + a_imu)], dim=1)
        e_avail = self.avail_enc(avail_vec)
        cls = self.cls_token.expand(T, -1)

        pe = sinusoidal_position_encoding(T, self.embed_dim, self.device)

        tokens_per_frame = torch.stack([e_img, e_lidar, e_imu, e_avail, cls], dim=1)
        tokens_per_frame = tokens_per_frame + pe.unsqueeze(1)

        masks = {
            "modality": torch.stack([a_img, a_lidar, a_imu], dim=1).to(torch.uint8).cpu().numpy(),
        }
        meta = {
            "T": T,
            "label": int(window.get("label", 0)),
            "frame_ids": frame_ids,
        }
        return {"tokens": tokens_per_frame.contiguous().cpu(), "mask": masks, "meta": meta}

@dataclass
class SplitStats:
    windows: int = 0
    positives: int = 0
    negatives: int = 0
    bad_len: int = 0

def read_expected_window_len(dataset_dir: str) -> int:
    stats_json = os.path.join(dataset_dir, "stats.json")
    if not os.path.isfile(stats_json):
        raise FileNotFoundError(f"stats.json not found under {dataset_dir}. Did you run createFrameWindows.py?")
    with open(stats_json, "r") as f:
        j = json.load(f)
    cfg = j.get("config") if isinstance(j, dict) else None
    if isinstance(cfg, dict) and "window_frames" in cfg:
        return int(cfg["window_frames"])
    return int(j.get("window_frames", 0))

def process_split(split: str, model: MultiModalTokenizer, expected_T: int) -> SplitStats:
    in_path = os.path.join(DATASET_DIR, f"{split}.jsonl")
    if not os.path.isfile(in_path):
        print(f"[WARN] {in_path} not found. Skipping split '{split}'.")
        return SplitStats()

    out_split_dir = os.path.join(OUT_DIR, split)
    ensure_dir(out_split_dir)

    st = SplitStats()
    manifest = []
    for sample in load_jsonl(in_path):
        fids = get_frame_ids(sample)
        T = len(fids)
        if T != expected_T:
            st.bad_len += 1
        st.windows += 1
        if int(sample.get("label", 0)) == 1:
            st.positives += 1
        else:
            st.negatives += 1

        pack = model.tokenize_window(sample)

        run = sample.get("run", "run")
        end_fid = int(sample.get("end_frame", sample.get("end_fid", -1)))
        if end_fid == -1:
            end_fid = fids[-1] if fids else -1
        base = f"{slugify(run)}__endframe_{end_fid:06d}.pt"
        out_fp = os.path.join(out_split_dir, base)

        torch.save(pack, out_fp)

        manifest.append({
            "path": out_fp,
            "run": run,
            "end_fid": end_fid,
            "label": int(sample["label"]),
            "T": T
        })

    with open(os.path.join(out_split_dir, "manifest.jsonl"), "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    return st

def main():
    ensure_dir(OUT_DIR)

    expected_T = read_expected_window_len(DATASET_DIR)
    print(f"[Info] Expected frames per window: {expected_T}")

    model = MultiModalTokenizer(
        embed_dim=EMBED_DIM,
        beta=BETA,
        use_pretrained_camera=USE_PRETRAINED_CAMERA,
        freeze_backbones=FREEZE_BACKBONES,
        device=DEVICE,
    )

    totals = {}
    for split in ["train", "val", "test"]:
        print(f"\n=== Processing split: {split} ===")
        stats = process_split(split, model, expected_T)
        totals[split] = {
            "windows": stats.windows,
            "positives": stats.positives,
            "negatives": stats.negatives,
            "frames_per_window_mismatches": stats.bad_len
        }
        print(f"[{split}] windows={stats.windows} (+)={stats.positives} (-)={stats.negatives} "
              f"bad_len={stats.bad_len}")

    summary = {
        "expected_frames_per_window": expected_T,
        "splits": totals
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nDone. Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
