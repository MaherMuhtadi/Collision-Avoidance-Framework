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
    models = None  # If torchvision isn't available, raise a helpful error at runtime.

DATASET_DIR = "Dataset"              # where {train,val,test}.jsonl and stats.json live
PREPROCESSED_ROOT = "PreprocessedData"
OUT_DIR = "Tokens"                # output root for token files and summary
EMBED_DIM = 384
BETA = 0.5                           # decay for carry-forward when modality missing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_PRETRAINED_CAMERA = True         # uses torchvision's resnet18 weights for camera
FREEZE_BACKBONES = True              # set False if you want to fine-tune backbones during tokenization (not typical)
BATCH_SIZE = 8                       # per-window microbatch for speed/memory (we embed per-frame; keep small)

# If you want fully deterministic outputs across runs:
SEED = 777
torch.manual_seed(SEED)
np.random.seed(SEED)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def slugify(s: str) -> str:
    # stable short filename component
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"{s.replace(os.sep, '_')}_{h}"

def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def sinusoidal_position_encoding(length: int, dim: int, device: str) -> torch.Tensor:
    """Standard transformer sinusoidal positional encoding (per frame)."""
    # pe[t, 2i] = sin(t / 10000^(2i/d)), pe[t, 2i+1] = cos(t / 10000^(2i/d))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    position = torch.arange(0, length, device=device, dtype=torch.float32).unsqueeze(1)  # [T,1]
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [T, D]

class ResNet18Trunk(nn.Module):
    """ResNet-18 trunk up to (and incl.) global avg pool; returns 512-d feature."""
    def __init__(self, in_ch: int = 3, pretrained: bool = False):
        super().__init__()
        if models is None:
            raise RuntimeError("torchvision is required for ResNet backbones but isn't available.")

        # Build a standard resnet18 and adjust conv1 if needed.
        if pretrained and in_ch == 3:
            try:
                # torchvision >= 0.13 style
                resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                # older fallback
                resnet = models.resnet18(pretrained=True)
        else:
            resnet = models.resnet18(weights=None)

        if in_ch != 3:
            # Replace first conv to accept in_ch (e.g., 2 for LiDAR range+mask)
            old = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                                     stride=old.stride, padding=old.padding, bias=False)
            # Kaiming init; if we wanted to "seed" from RGB, we could average or copy, but here we init fresh.
            nn.init.kaiming_normal_(resnet.conv1.weight, mode='fan_out', nonlinearity='relu')

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool  # outputs [B, 512, 1, 1]

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
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
        # x: [B,3,H,W] normalized to ImageNet stats already (as in preprocessing)
        feat = self.trunk(x)            # [B,512]
        e = self.proj(feat)             # [B,384]
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
        # x: [B,2,H,W]  (range, validity)
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
        # x: [B,6] (already normalized globally in preprocessing)
        return self.mlp(x)

class AvailabilityEmbedding(nn.Module):
    def __init__(self, out_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Linear(4, out_dim)
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: [B,4] where a = [a_img, a_lidar, a_imu, sum(a)]
        return self.proj(a)

class MultiModalTokenizer(nn.Module):
    """
    Builds tokens per frame: [image, lidar, imu, avail, cls] (5 tokens, each 384-d).
    Handles carry-forward imputation with learned e_miss per modality and decay BETA.
    Adds temporal sinusoidal positional encodings (same PE added to all tokens in a frame).
    """
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

        # Learnable "missing" embeddings for each modality
        self.e_miss_image = nn.Parameter(torch.zeros(embed_dim))
        self.e_miss_lidar = nn.Parameter(torch.zeros(embed_dim))
        self.e_miss_imu   = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.e_miss_image, mean=0.0, std=0.02)
        nn.init.normal_(self.e_miss_lidar, mean=0.0, std=0.02)
        nn.init.normal_(self.e_miss_imu,   mean=0.0, std=0.02)

        # Learnable CLS token (per frame we replicate this vector)
        self.cls_token = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

        self.to(self.device)
        self.eval()

    @torch.no_grad()
    def _embed_modalities(self,
                          imgs: torch.Tensor, a_img: torch.Tensor,
                          rims: torch.Tensor, a_lidar: torch.Tensor,
                          imus: torch.Tensor, a_imu: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute em for present modalities; returns image, lidar, imu embeddings each [B, D].
        We only compute encoders where availability=1 to avoid unnecessary work.
        """
        B = imgs.shape[0]
        # Camera
        e_img = torch.zeros((B, self.embed_dim), device=self.device)
        if a_img.any():
            idx = torch.nonzero(a_img, as_tuple=False).squeeze(1)
            e_img[idx] = self.image_enc(imgs[idx])

        # LiDAR
        e_lidar = torch.zeros((B, self.embed_dim), device=self.device)
        if a_lidar.any():
            idx = torch.nonzero(a_lidar, as_tuple=False).squeeze(1)
            e_lidar[idx] = self.lidar_enc(rims[idx])

        # IMU
        e_imu = torch.zeros((B, self.embed_dim), device=self.device)
        if a_imu.any():
            idx = torch.nonzero(a_imu, as_tuple=False).squeeze(1)
            e_imu[idx] = self.imu_enc(imus[idx])

        return e_img, e_lidar, e_imu

    @torch.no_grad()
    def tokenize_window(self, window: Dict) -> Dict:
        """
        window: a dict from the JSONL (one line), fields:
            frame_ids, image[], lidar_range[], lidar_mask[], imu[], modality_masks[], label
        Returns:
            {
              "tokens": Tensor [T*5, D],
              "mask":   dict with modality mask [T,3] and availability [T,4],
              "meta":   dict
            }
        """
        frame_ids: List[int] = window["frame_ids"]
        T = len(frame_ids)

        # Load per-frame arrays into tensors on device
        imgs, rims, imus = [], [], []
        a_img_list, a_lidar_list, a_imu_list = [], [], []
        for t in range(T):
            # image
            img_arr = np.load(window["image"][t]).astype(np.float32)  # [3,224,224] already normalized
            imgs.append(torch.from_numpy(img_arr))
            # lidar (range + validity)
            rim = np.load(window["lidar_range"][t]).astype(np.float32)  # [H,W]
            vmask = np.load(window["lidar_mask"][t]).astype(np.uint8)   # [H,W]
            rim2 = np.stack([rim, vmask.astype(np.float32)], axis=0)    # [2,H,W]
            rims.append(torch.from_numpy(rim2))
            # imu (6,)
            imu = np.load(window["imu"][t]).astype(np.float32)
            imus.append(torch.from_numpy(imu))
            # masks
            mm = window["modality_masks"][t]
            a_img_list.append(1 if int(mm.get("image", 1)) == 1 else 0)
            a_lidar_list.append(1 if int(mm.get("lidar", 1)) == 1 else 0)
            a_imu_list.append(1 if int(mm.get("imu",   1)) == 1 else 0)

        imgs  = torch.stack(imgs,  dim=0).to(self.device)  # [T,3,H,W]
        rims  = torch.stack(rims,  dim=0).to(self.device)  # [T,2,H,W]
        imus  = torch.stack(imus,  dim=0).to(self.device)  # [T,6]
        a_img   = torch.tensor(a_img_list,   device=self.device, dtype=torch.bool)  # [T]
        a_lidar = torch.tensor(a_lidar_list, device=self.device, dtype=torch.bool)  # [T]
        a_imu   = torch.tensor(a_imu_list,   device=self.device, dtype=torch.bool)  # [T]

        # Compute availability embedding input per frame
        a_float = torch.stack([
            a_img.float(), a_lidar.float(), a_imu.float(),
            (a_img.float() + a_lidar.float() + a_imu.float())
        ], dim=1)  # [T,4]
        e_avail = self.avail_enc(a_float)  # [T, D]

        # Embed present modalities
        e_img_raw, e_lidar_raw, e_imu_raw = self._embed_modalities(imgs, a_img, rims, a_lidar, imus, a_imu)  # [T,D] each

        # Carry-forward imputation per modality
        def impute(e_raw: torch.Tensor, avail: torch.BoolTensor, e_miss: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(e_raw)
            prev_valid = None  # previous valid embedding vector
            for t in range(T):
                if avail[t]:
                    out[t] = e_raw[t]
                    prev_valid = e_raw[t]
                else:
                    if prev_valid is None:
                        out[t] = e_miss
                    else:
                        out[t] = self.beta * prev_valid + (1.0 - self.beta) * e_miss
            return out

        e_img   = impute(e_img_raw,   a_img,   self.e_miss_image)  # [T,D]
        e_lidar = impute(e_lidar_raw, a_lidar, self.e_miss_lidar)  # [T,D]
        e_imu   = impute(e_imu_raw,   a_imu,   self.e_miss_imu)    # [T,D]

        # Build per-frame token set: [image, lidar, imu, avail, cls]
        cls = self.cls_token.unsqueeze(0).expand(T, -1)  # [T,D]
        tokens_per_frame = torch.stack([e_img, e_lidar, e_imu, e_avail, cls], dim=1)  # [T,5,D]

        # Add temporal positional encoding (same PE for all tokens of a frame)
        pe = sinusoidal_position_encoding(T, self.embed_dim, device=self.device)  # [T,D]
        tokens_per_frame = tokens_per_frame + pe.unsqueeze(1)  # broadcast to [T,5,D]

        # Store per-frame tokens: image,lidar,imu,avail,cls
        tokens = tokens_per_frame.contiguous()  # [T, 5, D]

        # Build masks for downstream use (e.g., attention key biasing when am=0)
        masks = {
            "modality": torch.stack([a_img, a_lidar, a_imu], dim=1).to(torch.uint8).cpu().numpy(),  # [T,3]
        }

        meta = {
            "T": T,
            "label": int(window["label"]),
            "frame_ids": window["frame_ids"],
        }

        return {"tokens": tokens.cpu(), "mask": masks, "meta": meta}

@dataclass
class SplitStats:
    windows: int = 0
    positives: int = 0
    negatives: int = 0
    bad_len: int = 0  # windows whose length != expected

def read_expected_window_len(dataset_dir: str) -> int:
    stats_json = os.path.join(dataset_dir, "stats.json")
    if not os.path.isfile(stats_json):
        raise FileNotFoundError(f"stats.json not found under {dataset_dir}. Did you run createFrameWindows.py?")
    with open(stats_json, "r") as f:
        j = json.load(f)
    return int(j.get("window_frames", 0))

def process_split(split: str, model: MultiModalTokenizer, expected_T: int) -> SplitStats:
    in_path = os.path.join(DATASET_DIR, f"{split}.jsonl")
    if not os.path.isfile(in_path):
        print(f"[WARN] {in_path} not found. Skipping split '{split}'.")
        return SplitStats()

    out_split_dir = os.path.join(OUT_DIR, split)
    ensure_dir(out_split_dir)

    st = SplitStats()
    manifest = []  # JSONL rows with minimal metadata & file path
    for sample in load_jsonl(in_path):
        T = len(sample.get("frame_ids", []))
        if T != expected_T:
            st.bad_len += 1
        st.windows += 1
        if int(sample.get("label", 0)) == 1:
            st.positives += 1
        else:
            st.negatives += 1

        # Tokenize this window
        pack = model.tokenize_window(sample)  # dict with tokens, mask, meta

        run = sample.get("run", "run")
        end_fid = int(sample.get("end_fid", -1))
        base = f"{slugify(run)}__endfid_{end_fid:06d}.pt"
        out_fp = os.path.join(out_split_dir, base)

        torch.save(pack, out_fp)

        manifest.append({
            "path": out_fp,
            "run": run,
            "end_fid": end_fid,
            "label": int(sample["label"]),
            "T": T
        })

    # Write a tiny manifest for convenience
    with open(os.path.join(out_split_dir, "manifest.jsonl"), "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    return st

def main():
    ensure_dir(OUT_DIR)

    expected_T = read_expected_window_len(DATASET_DIR)  # verify consistency with dataset creator
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

    # Save summary including expected frames per window
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
