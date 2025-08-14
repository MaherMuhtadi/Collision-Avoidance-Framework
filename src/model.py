
"""
End-to-end training for Dynamic Transformer Encoder + Binary Head on Carla tokens.

Pipeline expected (already provided by user):
- extractData.py -> preprocessData.py -> createFrameWindows.py -> embeddingLayer.py

This script consumes the tokenized windows produced by embeddingLayer.py under Tokens/.
It trains the dynamic transformer encoder (with optional static mode) and a binary
prediction head, computes metrics, and writes results to disk.

Configuration is done via GLOBAL VARIABLES below (no CLI args).
"""

# =========================
# ====== CONFIG (GLOBALS)
# =========================

TOKENS_ROOT = "Tokens"           # root with {train,val,test}/manifest.jsonl and *.pt token packs
RESULTS_DIR = "Results"          # where to write metrics.json, logs, and model checkpoint
MODEL_CKPT = "model.pt"          # filename for model weights inside RESULTS_DIR
LOG_CSV = "train_log.csv"        # per-epoch log
METRICS_JSON = "metrics.json"    # final metrics dump
DEVICE = "cuda"                  # "cuda" or "cpu"; auto-fallback if unavailable
SEED = 42

# Data / Training
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 2e-4

# Transformer
D_MODEL = 384
N_HEADS = 6
FF_DIM = 1024
N_LAYERS = 6
DROPOUT = 0.1

# === Aggregation / Calibration Toggles ===
# Pooling strategy for CLS across frames: "last"|"mean"|"max"
POOL_STRATEGY = "attn"
# Enable temperature scaling (calibration) based on validation set
ENABLE_TEMPERATURE_SCALING = True

# Dynamic skipping
USE_DYNAMIC = False          # Toggle dynamic (True) vs static (False) encoder
SOFT_SKIP_TRAIN = True      # If True, use soft gating during training: out = s*x + (1-s)*layer(x)
HARD_SKIP_AT_EVAL = True    # At evaluation, use hard 0.5 thresholding for skipping

# Misc
NEG_INF = -1e9              # attention bias for keys from missing-modality tokens
NUM_TOKENS_PER_FRAME = 5    # [image, lidar, imu, avail, cls]
IDX_IMAGE, IDX_LIDAR, IDX_IMU, IDX_AVAIL, IDX_CLS = 0, 1, 2, 3, 4

import os
import json
import math
import csv
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# --------------------
# Utils
# --------------------

def safe_load(path, map_location="cpu"):
    # Works on PyTorch 2.6+; falls back cleanly on older versions.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def load_jsonl(path: str):
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def manifest_path(split: str) -> str:
    return os.path.join(TOKENS_ROOT, split, "manifest.jsonl")

def has_split(split: str) -> bool:
    return os.path.isfile(manifest_path(split))

# --------------------
# Dataset
# --------------------

@dataclass
class Batch:
    tokens: torch.Tensor         # [B, S, D]
    key_padding_mask: torch.Tensor  # [B, S] boolean; True => mask that key (discourage attention to it)
    T: int
    labels: torch.Tensor         # [B] float32 in {0,1}
    frame_indices: torch.Tensor  # [S] (common indexing mapping per-frame per batch item)
    # Indices cache helpers
    cls_indices: torch.Tensor    # [T] positions of CLS tokens for each frame (within an example)

class TokenWindowDataset(torch.utils.data.Dataset):
    def __init__(self, split: str):
        self.split = split
        self.rows = []
        mp = manifest_path(split)
        if not os.path.isfile(mp):
            raise FileNotFoundError(f"Manifest not found for split '{split}': {mp}")
        for row in load_jsonl(mp):
            self.rows.append(row)

        if len(self.rows) == 0:
            raise RuntimeError(f"No rows in manifest for split '{split}'")

        # Infer T from first pack
        first_pack = safe_load(self.rows[0]["path"], map_location="cpu")
        T = int(first_pack["meta"]["T"])
        D = int(first_pack["tokens"].shape[-1])
        self.T = T
        self.D = D
        self.S = T * NUM_TOKENS_PER_FRAME

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        pack = safe_load(row["path"], map_location="cpu")  # dict with: tokens [T,5,D], mask {'modality':[T,3]}, meta
        tokens = pack["tokens"].float()                     # [T,5,D]
        T = tokens.shape[0]
        assert T == self.T, f"Window length mismatch: expected {self.T} got {T}"
        D = tokens.shape[-1]
        tokens = tokens.view(T * NUM_TOKENS_PER_FRAME, D)  # [S, D]
        label = float(pack["meta"]["label"])

        # Build key-padding mask (mask keys that come from missing modalities)
        # mask["modality"] is a numpy array [T,3] with 1 present / 0 missing for (image, lidar, imu)
        m = pack["mask"]["modality"].astype(np.uint8)  # [T,3]
        key_mask = np.zeros((T * NUM_TOKENS_PER_FRAME,), dtype=np.bool_)  # False => keep; True => mask
        for t in range(T):
            base = t * NUM_TOKENS_PER_FRAME
            # If modality missing, mask its token as a KEY
            if m[t, 0] == 0:
                key_mask[base + IDX_IMAGE] = True
            if m[t, 1] == 0:
                key_mask[base + IDX_LIDAR] = True
            if m[t, 2] == 0:
                key_mask[base + IDX_IMU] = True
            # Do NOT mask avail or cls tokens
        key_mask = torch.from_numpy(key_mask)  # [S]

        # Precompute indices for CLS and AVAIL tokens per frame
        cls_indices = torch.tensor([t * NUM_TOKENS_PER_FRAME + IDX_CLS for t in range(T)], dtype=torch.long)
        avail_indices = torch.tensor([t * NUM_TOKENS_PER_FRAME + IDX_AVAIL for t in range(T)], dtype=torch.long)

        # Pack
        out = {
            "tokens": tokens,                                  # [S, D]
            "key_padding_mask": key_mask,                      # [S]
            "label": torch.tensor(label, dtype=torch.float32), # []
            "T": T,
            "cls_indices": cls_indices,                        # [T]
            "avail_indices": avail_indices,                    # [T]
        }
        return out

def collate_fn(batch_rows: List[Dict]) -> Batch:
    B = len(batch_rows)
    T = batch_rows[0]["T"]
    D = batch_rows[0]["tokens"].shape[-1]
    S = T * NUM_TOKENS_PER_FRAME

    tokens = torch.stack([r["tokens"] for r in batch_rows], dim=0)                 # [B,S,D]
    key_padding_mask = torch.stack([r["key_padding_mask"] for r in batch_rows], 0) # [B,S]
    labels = torch.stack([r["label"] for r in batch_rows], dim=0)                  # [B]

    frame_indices = torch.arange(S, dtype=torch.long)  # same mapping for all in-batch examples
    cls_indices = batch_rows[0]["cls_indices"]         # [T]

    return Batch(tokens=tokens,
                 key_padding_mask=key_padding_mask.bool(),
                 T=T, labels=labels,
                 frame_indices=frame_indices,
                 cls_indices=cls_indices)

# --------------------
# Model: Dynamic Transformer Encoder + Binary Head
# --------------------

class DynamicTransformerLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

    # Match PyTorch’s signature
    def forward(self, x: torch.Tensor, src_mask=None, src_key_padding_mask=None):
        return self.layer(
            x,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )


class OracleGate(nn.Module):
    """
    Oracle gating network q_j* taking current CLS, skip history, and availability embedding (per-frame).
    Produces skip probability s_j in [0,1] via sigmoid.
    """
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, hidden=256):
        super().__init__()
        self.n_layers = n_layers
        self.fc1 = nn.Linear(d_model * 2 + n_layers, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, cls_vec: torch.Tensor, avail_vec: torch.Tensor, skip_hist: torch.Tensor) -> torch.Tensor:
        """
        cls_vec:   [B, T, D]
        avail_vec: [B, T, D]
        skip_hist: [B, T, n_layers]  (we'll zero future layers; only first j-1 slots filled)
        Returns:   [B, T] skip probability in [0,1]
        """
        x = torch.cat([cls_vec, avail_vec, skip_hist], dim=-1)  # [B, T, 2D + n_layers]
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)).squeeze(-1)  # [B, T]
        return x


class DynamicTransformerEncoder(nn.Module):
    """
    Implements dynamic layer skipping as described in Documentation.pdf Section B.2.
    - Uses oracle gates per layer (with sigmoid skip score).
    - During training: soft gating (optional) for gradient flow.
    - During eval: hard threshold at 0.5 (optional).
    - Attention discourages keys from missing-modality tokens using key_padding_mask (per Section B.2).
    """
    def __init__(self, d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS, ff_dim=FF_DIM, dropout=DROPOUT,
                 use_dynamic=USE_DYNAMIC, soft_skip_train=SOFT_SKIP_TRAIN, hard_skip_at_eval=HARD_SKIP_AT_EVAL):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_dynamic = use_dynamic
        self.soft_skip_train = soft_skip_train
        self.hard_skip_at_eval = hard_skip_at_eval

        self.layers = nn.ModuleList([DynamicTransformerLayer(d_model, n_heads, ff_dim, dropout) for _ in range(n_layers)])
        self.gates = nn.ModuleList([OracleGate(d_model=d_model, n_layers=n_layers) for _ in range(n_layers)])

        # Running stats
        self.register_buffer("last_skip_probs", None)  # updated on forward: [B, T, n_layers]

    def _build_attn_mask(self, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert key_padding_mask [B, S] (True => mask key) into additive attn mask usable by Transformer.
        We'll build a per-batch mask [B, S, S] with columns for masked keys set to NEG_INF.
        """
        B, S = key_padding_mask.shape
        base = torch.zeros((B, S, S), device=key_padding_mask.device, dtype=torch.float32)
        # For each batch item, set entire column j to NEG_INF if key_padding_mask[b, j] == True
        # base[:, :, j] += NEG_INF where masked
        mask_cols = key_padding_mask.unsqueeze(1).expand(B, S, S)  # [B, S, S] mask over columns
        base = base.masked_fill(mask_cols, NEG_INF)
        return base  # [B, S, S]

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor, T: int) -> Tuple[torch.Tensor, Dict]:
        """
        tokens: [B, S, D] where S = T*5 (order: per frame [image,lidar,imu,avail,cls])
        key_padding_mask: [B, S] (True to discourage attending to that key)
        Returns:
            out: [B, S, D]
            info: dict with skip_probs [B, T, L], exec_rates_per_layer [L]
        """
        B, S, D = tokens.shape
        assert D == self.d_model

        # Helper indices for avail and cls per frame (same mapping across batch)
        cls_idx = torch.arange(T, device=tokens.device) * NUM_TOKENS_PER_FRAME + IDX_CLS     # [T]
        avail_idx = torch.arange(T, device=tokens.device) * NUM_TOKENS_PER_FRAME + IDX_AVAIL # [T]

        x = tokens  # [B, S, D]

        # Keep skip history per frame
        skip_hist = torch.zeros((B, T, self.n_layers), device=tokens.device, dtype=torch.float32)

        for j in range(self.n_layers):
            # Current per-frame vectors (before layer j): CLS and AVAIL
            cls_vec   = x[:, cls_idx, :]     # [B, T, D]
            avail_vec = x[:, avail_idx, :]   # [B, T, D]

            # Compute skip probabilities
            s = self.gates[j](cls_vec, avail_vec, skip_hist)  # [B, T] in [0,1]
            skip_hist[:, :, j] = s  # record

            # Decide execution weights per frame
            if not self.use_dynamic:
                # Static: everyone executes (weight_exec = 1.0)
                w_exec = torch.ones_like(s)
            else:
                if self.training and self.soft_skip_train:
                    # Soft: weight_exec = (1 - s)
                    w_exec = 1.0 - s
                else:
                    # Hard: threshold at 0.5
                    w_exec = (s < 0.5).float()

            # Expand per-token to [B, S] (copy each frame's weight to its 5 tokens)
            w_exec_tokens = w_exec.repeat_interleave(NUM_TOKENS_PER_FRAME, dim=1)  # [B, S]
            assert w_exec_tokens.shape[1] == S

            # Run the transformer layer
            # The TransformerEncoderLayer accepts attn_mask broadcastable to (B, S, S).
            # We'll pass our per-batch additive mask.
            y = self.layers[j](x, src_mask=None, src_key_padding_mask=key_padding_mask)

            # Merge based on execution weight:
            # soft: out = w*x_layer + (1-w)*x  ; hard: w in {0,1} does select
            w = w_exec_tokens.unsqueeze(-1)  # [B, S, 1]
            x = w * y + (1.0 - w) * x

        # Save last skip probs for inspection
        self.last_skip_probs = skip_hist.detach()

        # Compute exec rates
        exec_rates = []
        for j in range(self.n_layers):
            if not self.use_dynamic:
                exec_rates.append(1.0)
            else:
                if self.training and self.soft_skip_train:
                    # Expected execution rate = mean(1 - s)
                    exec_rates.append(float((1.0 - skip_hist[:, :, j]).mean().item()))
                else:
                    exec_rates.append(float(((skip_hist[:, :, j] < 0.5).float().mean().item())))

        info = {
            "skip_probs": skip_hist,           # [B, T, L]
            "exec_rates_per_layer": exec_rates # list length L
        }
        return x, info


class BinaryHead(nn.Module):
    def __init__(self, d_model=D_MODEL):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1)  # logits
        )

    def forward(self, cls_vec: torch.Tensor) -> torch.Tensor:
        # cls_vec: [B, D] (typically from last frame)
        return self.ff(cls_vec).squeeze(-1)  # [B] logits


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DynamicTransformerEncoder()
        self.head = BinaryHead()

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            logits: [B]
            aux: dict with skip info, etc.
        """
        x = batch.tokens  # [B, S, D]
        kpm = batch.key_padding_mask  # [B, S]
        T = batch.T
        out, info = self.backbone(x, kpm, T)  # [B, S, D]

        # Take CLS of LAST frame for binary prediction (consistent with end-of-window labeling)
        cls_idx = batch.cls_indices  # [T]
        cls_seq = out[:, cls_idx, :]  # [B, T, D]
        if POOL_STRATEGY == "mean":
            cls_vec = cls_seq.mean(dim=1)
        elif POOL_STRATEGY == "max":
            cls_vec = cls_seq.max(dim=1).values
        else:  # "last" (default)
            cls_vec = cls_seq[:, -1, :]
        logits = self.head(cls_vec)        # [B]
        return logits, info

# --------------------
# Metrics
# --------------------

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def binary_classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict:
    """
    y_true:  (N,) in {0,1}
    y_score: (N,) in [0,1] (probabilities)
    """
    y_pred = (y_score >= threshold).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    # ROC-AUC (rank-based trapezoid)
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    # Cum sums
    cum_tp = np.cumsum(y_true_sorted == 1)
    cum_fp = np.cumsum(y_true_sorted == 0)
    P = max(int((y_true == 1).sum()), 1)
    N = max(int((y_true == 0).sum()), 1)
    tpr = cum_tp / P
    fpr = cum_fp / N
    # Prepend start (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    auc = np.trapz(tpr, fpr)

    # PR-AUC (Average Precision)
    # Compute precision/recall at each threshold
    precision_points = cum_tp / np.maximum((np.arange(1, len(y_true_sorted) + 1)), 1)
    recall_points = cum_tp / P
    # Prepend starting point
    precision_points = np.concatenate([[precision], precision_points])
    recall_points = np.concatenate([[0.0], recall_points])
    # Integrate precision over recall
    ap = 0.0
    for i in range(1, len(recall_points)):
        ap += precision_points[i] * (recall_points[i] - recall_points[i - 1])

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def collect_scores(model, loader, device, return_logits=False):
    """Return (labels, probabilities[, logits]) for a loader using current model parameters."""
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            b = move_batch_to_device(batch, device)
            l, _ = model(b)
            logits_list.append(l.detach().cpu().numpy())
            labels_list.append(b.labels.detach().cpu().numpy())
    if not logits_list:
        empty = (np.zeros((0,)), np.zeros((0,)))
        return (*empty, np.zeros((0,))) if return_logits else empty
    logits = np.concatenate(logits_list)
    labels = np.concatenate(labels_list)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return (labels, probs, logits) if return_logits else (labels, probs)

def find_best_threshold(y_true, y_prob, metric="f1"):
    """Grid-search a threshold on y_prob to maximize F1 (default) or Youden's J on a validation set."""
    thr_grid = np.unique(y_prob)
    if thr_grid.size > 400:
        thr_grid = np.linspace(0.0, 1.0, 401)
    best_score, best_thr = -1.0, 0.5
    for thr in thr_grid:
        y_pred = (y_prob >= thr)
        tp = np.sum((y_pred==1) & (y_true==1))
        fp = np.sum((y_pred==1) & (y_true==0))
        fn = np.sum((y_pred==0) & (y_true==1))
        tn = np.sum((y_pred==0) & (y_true==0))
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1 = 0.0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
        youdenJ = rec - (fp / max(fp+tn, 1))
        score = f1 if metric.lower()=="f1" else youdenJ
        if score > best_score:
            best_score, best_thr = score, float(thr)
    return best_thr

def metrics_with_threshold(y, p, thr):
    m = binary_classification_metrics(y, p, threshold=thr)
    m["threshold_used"] = float(thr)
    return m


def fit_temperature(val_logits_np, val_labels_np, max_iter=200, lr=0.01):
    """Fit a scalar temperature T on validation logits to minimize BCE with labels.
    Returns float(T)."""
    import torch
    L = torch.from_numpy(val_logits_np).float().unsqueeze(1)  # [N,1]
    y = torch.from_numpy(val_labels_np).float().unsqueeze(1)  # [N,1]
    T = torch.nn.Parameter(torch.ones(1))  # start at 1.0
    bce = torch.nn.BCEWithLogitsLoss()

    # Prefer LBFGS for stable optimization of one parameter; fall back to SGD
    try:
        opt = torch.optim.LBFGS([T], lr=1.0, max_iter=50)
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = bce(L / T.clamp_min(1e-3), y)
            loss.backward()
            return loss
        opt.step(closure)
    except Exception:
        opt = torch.optim.SGD([T], lr=lr)
        for _ in range(max_iter):
            opt.zero_grad(set_to_none=True)
            loss = bce(L / T.clamp_min(1e-3), y)
            loss.backward()
            opt.step()

    return float(T.detach().clamp_min(1e-3))

# --------------------
# Training / Eval
# --------------------

@dataclass
class EpochResult:
    loss: float
    metrics: Dict
    exec_rates: List[float]

def move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
    return Batch(tokens=batch.tokens.to(device),
                 key_padding_mask=batch.key_padding_mask.to(device),
                 T=batch.T,
                 labels=batch.labels.to(device),
                 frame_indices=batch.frame_indices.to(device),
                 cls_indices=batch.cls_indices.to(device))

def run_one_epoch(model: Model, loader, device, optimizer=None) -> EpochResult:
    train_mode = optimizer is not None
    model.train(mode=train_mode)

    all_logits = []
    all_labels = []
    losses = []

    exec_rates_accum = np.zeros((N_LAYERS,), dtype=np.float64)
    exec_rates_count = 0

    bce = nn.BCEWithLogitsLoss()

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits, info = model(batch)
        labels = batch.labels

        loss = bce(logits, labels)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Collect
        losses.append(float(loss.item()))
        all_logits.append(logits.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

        exec_rates_accum += np.array(info["exec_rates_per_layer"], dtype=np.float64)
        exec_rates_count += 1

    all_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,), dtype=np.float32)
    all_labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.float32)
    probs = sigmoid_np(all_logits)
    metrics = binary_classification_metrics(all_labels, probs)

    exec_rates = (exec_rates_accum / max(exec_rates_count, 1)).tolist()
    return EpochResult(loss=float(np.mean(losses) if losses else 0.0),
                       metrics=metrics,
                       exec_rates=exec_rates)

def evaluate(model: Model, loader, device) -> EpochResult:
    # Switch to eval; optionally hard gating at eval time
    model.eval()
    # If using dynamic and requested hard skip at eval, temporarily disable soft training behavior
    prev_soft_flag = model.backbone.soft_skip_train
    prev_train_flag = model.training
    model.backbone.soft_skip_train = False  # no soft mixing
    with torch.no_grad():
        res = run_one_epoch(model, loader, device, optimizer=None)
    # Restore flag
    model.backbone.soft_skip_train = prev_soft_flag
    model.train(prev_train_flag)
    return res

# --------------------
# Main
# --------------------

def main():
    set_seed(SEED)
    ensure_dir(RESULTS_DIR)

    # Device
    dev = torch.device(DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[Info] Using device: {dev}")

    # Datasets / Loaders (existence checked by Dataset itself)
    train_loader = None
    if has_split("train"):
        train_ds = TokenWindowDataset("train")
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                                   shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)
        T = train_ds.T
    else:
        raise SystemExit("No train split found under Tokens/train/manifest.jsonl")

    val_loader = None
    if has_split("val"):
        val_ds = TokenWindowDataset("val")
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE,
                                                 shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
    test_loader = None
    if has_split("test"):
        test_ds = TokenWindowDataset("test")
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE,
                                                  shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)

    # Model
    model = Model().to(dev)
    # Ensure encoder toggles reflect globals
    model.backbone.use_dynamic = USE_DYNAMIC
    model.backbone.soft_skip_train = SOFT_SKIP_TRAIN
    model.backbone.hard_skip_at_eval = HARD_SKIP_AT_EVAL

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # CSV log header
    log_path = os.path.join(RESULTS_DIR, LOG_CSV)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "split", "loss", "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"] + [f"exec_rate_L{j+1}" for j in range(N_LAYERS)])

    best_val_auc = -1.0
    best_state = None

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        print(f"\n==== Epoch {epoch} ====")
        # Train
        train_res = run_one_epoch(model, train_loader, dev, optimizer=optimizer)
        print(f"[Train] loss={train_res.loss:.4f} AUC={train_res.metrics['roc_auc']:.4f} F1={train_res.metrics['f1']:.4f}")
        # Log
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, "train", train_res.loss,
                        train_res.metrics["accuracy"], train_res.metrics["precision"], train_res.metrics["recall"],
                        train_res.metrics["f1"], train_res.metrics["roc_auc"], train_res.metrics["average_precision"]] + train_res.exec_rates)

        # Validate (if available)
        if val_loader is not None:
            val_res = evaluate(model, val_loader, dev)
            print(f"[Val]   loss={val_res.loss:.4f} AUC={val_res.metrics['roc_auc']:.4f} F1={val_res.metrics['f1']:.4f}")
            with open(log_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([epoch, "val", val_res.loss,
                            val_res.metrics["accuracy"], val_res.metrics["precision"], val_res.metrics["recall"],
                            val_res.metrics["f1"], val_res.metrics["roc_auc"], val_res.metrics["average_precision"]] + val_res.exec_rates)

            # Track best model by val AUC
            if val_res.metrics["roc_auc"] > best_val_auc:
                best_val_auc = val_res.metrics["roc_auc"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Save best (or last) model
    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    ckpt_path = os.path.join(RESULTS_DIR, MODEL_CKPT)
    torch.save({"model_state": best_state,
                "config": {
                    "D_MODEL": D_MODEL, "N_HEADS": N_HEADS, "FF_DIM": FF_DIM, "N_LAYERS": N_LAYERS,
                    "USE_DYNAMIC": USE_DYNAMIC, "SOFT_SKIP_TRAIN": SOFT_SKIP_TRAIN, "HARD_SKIP_AT_EVAL": HARD_SKIP_AT_EVAL
                }}, ckpt_path)
    print(f"[Info] Saved checkpoint to {ckpt_path}")

    # Evaluate on test (if available)
    final_metrics = {}
    if test_loader is not None:
        # Load best state for fair test
        model.load_state_dict(best_state)
        test_res = evaluate(model, test_loader, dev)
        print(f"[Test]  loss={test_res.loss:.4f} AUC={test_res.metrics['roc_auc']:.4f} F1={test_res.metrics['f1']:.4f}")
        final_metrics["test"] = {**test_res.metrics,
                                 "exec_rates_per_layer": test_res.exec_rates}
    # Also include last train/val
    final_metrics["train_last"] = {**train_res.metrics, "exec_rates_per_layer": train_res.exec_rates}
    if val_loader is not None:
        final_metrics["val_last"] = {**val_res.metrics, "exec_rates_per_layer": val_res.exec_rates}

    # === Threshold tuning on validation set (choose threshold on val, then apply to test) ===
    val_thr = 0.5
    if val_loader is not None:
        yv, pv = collect_scores(model, val_loader, dev)
        if len(pv) > 0:
            val_thr = find_best_threshold(yv, pv, metric="f1")
            print(f"[Val] chosen threshold = {val_thr:.3f}")
            final_metrics["val_last_tuned"] = metrics_with_threshold(yv, pv, val_thr)

    if test_loader is not None:
        yt, pt = collect_scores(model, test_loader, dev)
        final_metrics["test_tuned"] = metrics_with_threshold(yt, pt, val_thr)
        print(f"[Sanity] Test fraction >=0.5: {float((pt>=0.5).mean()):.3f}")
        if val_loader is not None:
            print(f"[Sanity] Val  fraction >=0.5: {float((pv>=0.5).mean()):.3f}")

    # Save the chosen threshold(s)
    final_metrics.setdefault("thresholds", {})["val_best_f1"] = float(val_thr)

    

    # === Temperature scaling calibration (fit T on val, then evaluate calibrated metrics) ===
    if ENABLE_TEMPERATURE_SCALING and val_loader is not None:
        yv2, pv2, v_logits = collect_scores(model, val_loader, dev, return_logits=True)
        if len(v_logits) > 0:
            Tcal = fit_temperature(v_logits, yv2)
            print(f"[Calib] temperature T = {Tcal:.3f}")
            val_probs_cal = sigmoid_np(v_logits / max(Tcal, 1e-3))
            val_thr_cal = find_best_threshold(yv2, val_probs_cal, metric="f1")
            final_metrics["val_last_calibrated"] = binary_classification_metrics(yv2, val_probs_cal)
            final_metrics["val_last_calibrated"]["temperature_used"] = float(Tcal)
            final_metrics["val_last_calibrated_tuned"] = metrics_with_threshold(yv2, val_probs_cal, val_thr_cal)
            final_metrics.setdefault("thresholds", {})["val_best_f1_calibrated"] = float(val_thr_cal)
            final_metrics.setdefault("calibration", {})["temperature"] = float(Tcal)
            if test_loader is not None:
                yt2, pt2, t_logits = collect_scores(model, test_loader, dev, return_logits=True)
                test_probs_cal = sigmoid_np(t_logits / max(Tcal, 1e-3))
                final_metrics["test_calibrated_tuned"] = metrics_with_threshold(yt2, test_probs_cal, val_thr_cal)
# Compute approximate compute savings with dynamic (relative to static)
    # Expected fraction executed = mean over layers of exec rate
    if USE_DYNAMIC:
        def avg_exec(d):
            return float(np.mean(d["exec_rates_per_layer"])) if "exec_rates_per_layer" in d else 1.0
        dyn_train = avg_exec(final_metrics["train_last"])
        dyn_val = avg_exec(final_metrics.get("val_last", {}))
        dyn_test = avg_exec(final_metrics.get("test", {}))
        final_metrics["approx_compute"] = {
            "expected_fraction_layers_executed": {
                "train_last": dyn_train,
                "val_last": dyn_val if dyn_val != 0.0 else dyn_train,
                "test": dyn_test if dyn_test != 0.0 else dyn_train
            },
            "expected_speedup_vs_static": {
                "train_last": (1.0 / max(dyn_train, 1e-6)),
                "val_last": (1.0 / max(dyn_val if dyn_val != 0.0 else dyn_train, 1e-6)),
                "test": (1.0 / max(dyn_test if dyn_test != 0.0 else dyn_train, 1e-6)),
            }
        }

    # Write metrics.json
    metrics_path = os.path.join(RESULTS_DIR, METRICS_JSON)
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[Info] Wrote metrics to {metrics_path}")

if __name__ == "__main__":
    main()
