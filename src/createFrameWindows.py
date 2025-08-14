import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

PRE_DIR: str = "PreprocessedData"      # Root with per-run preprocessed_data.json
OUT_DIR: str = "Dataset"               # Where to write {train,val,test}.jsonl and stats.json

# Window spec: set exactly one of WINDOW_FRAMES or WINDOW_SECONDS (or leave both None for default 1s at FPS)
WINDOW_FRAMES: int | None = None       # e.g., 30 for a 1-second window at 30 FPS
WINDOW_SECONDS: float | None = 2.0     # If set, requires FPS
FPS: float = 30.0

POS_STRIDE: int = 1                    # Stride for positive windows
NEG_STRIDE: int = 1                    # Stride for negative windows

TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

SEED: int = 7


def _find_runs(pre_dir: str) -> List[str]:
    runs = []
    if not os.path.isdir(pre_dir):
        return runs
    for name in sorted(os.listdir(pre_dir)):
        d = os.path.join(pre_dir, name)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "preprocessed_data.json")):
            runs.append(d)
    return runs


def _load_run(run_dir: str) -> Tuple[str, List[int], Dict[int, dict]]:
    """Return run_name, sorted frame ids, and frame->record dict"""
    run_name = os.path.basename(run_dir.rstrip(os.sep))
    jpath = os.path.join(run_dir, "preprocessed_data.json")
    with open(jpath, "r") as f:
        data = json.load(f)
    # keys are strings of frame ids
    items = [(int(k), v) for k, v in data.items()]
    items.sort(key=lambda x: x[0])
    frame_ids = [k for k, _ in items]
    recs = {k: v for k, v in items}
    return run_name, frame_ids, recs


def _split_runs_by_collision(run_dirs: List[str], seed: int, ratios: Tuple[float, float, float]):
    """Stratified split by run (collision vs non-collision)."""
    train_r, val_r, test_r = ratios
    assert abs(train_r + val_r + test_r - 1.0) < 1e-6, "Ratios must sum to 1.0"

    collision_runs, non_collision_runs = [], []
    for rd in run_dirs:
        _, fids, recs = _load_run(rd)
        # A run is a "collision run" if ANY frame has label == 1
        has_pos = any(int(recs[f]["label"]) == 1 for f in fids)
        (collision_runs if has_pos else non_collision_runs).append(rd)

    def strat_split(group: List[str]) -> Tuple[List[str], List[str], List[str]]:
        random.shuffle(group)
        n = len(group)
        n_train = int(round(n * train_r))
        n_val = int(round(n * val_r))
        # ensure total stays n
        n_test = max(0, n - n_train - n_val)
        return group[:n_train], group[n_train:n_train+n_val], group[n_train+n_val:]

    random.seed(seed)
    c_tr, c_va, c_te = strat_split(collision_runs)
    n_tr, n_va, n_te = strat_split(non_collision_runs)

    train = c_tr + n_tr
    val   = c_va + n_va
    test  = c_te + n_te
    # deterministic order
    train.sort()
    val.sort()
    test.sort()

    split_summary = {
        "counts_by_group": {
            "collision": {"train": len(c_tr), "val": len(c_va), "test": len(c_te), "total": len(collision_runs)},
            "non_collision": {"train": len(n_tr), "val": len(n_va), "test": len(n_te), "total": len(non_collision_runs)},
        },
        "runs": {
            "train": [os.path.basename(r) for r in train],
            "val":   [os.path.basename(r) for r in val],
            "test":  [os.path.basename(r) for r in test],
        }
    }
    return train, val, test, split_summary


def _windowize(frame_ids: List[int], window: int, stride: int = 1) -> List[List[int]]:
    """Return list of windows (each is a list of frame ids) ending at each end index, step=stride"""
    if window <= 0:
        raise ValueError("window must be >= 1")
    out = []
    for end_idx in range(window - 1, len(frame_ids), stride):
        start_idx = end_idx - window + 1
        out.append(frame_ids[start_idx: end_idx + 1])
    return out


def _build_windows_for_run(run_dir: str, window: int, pos_stride: int, neg_stride: int):
    """
    Returns:
        pos_windows: list of (frame_ids_window, end_fid)
        neg_windows: list of (frame_ids_window, end_fid)
    """
    run_name, fids, recs = _load_run(run_dir)

    # Build two sets of windows with different strides for pos vs neg if desired
    pos_wins = _windowize(fids, window=window, stride=pos_stride)
    neg_wins = _windowize(fids, window=window, stride=neg_stride)

    def label_of_end(win):
        end_fid = win[-1]
        return int(recs[end_fid]["label"])

    pos_windows = [(win, win[-1]) for win in pos_wins if label_of_end(win) == 1]
    neg_windows = [(win, win[-1]) for win in neg_wins if label_of_end(win) == 0]
    return run_name, pos_windows, neg_windows, recs


def _serialize_window_sample(run_name: str, win: List[int], end_fid: int, recs: Dict[int, dict]) -> dict:
    sample = {
        "run": run_name,
        "end_fid": int(end_fid),
        "frame_ids": [int(f) for f in win],
        "image": [],
        "lidar_range": [],
        "lidar_mask": [],
        "imu": [],
        "modality_masks": [],
        "label": int(recs[end_fid]["label"]),
    }
    for fid in win:
        r = recs[fid]
        sample["image"].append(r["image"])
        sample["lidar_range"].append(r["lidar_range"])
        sample["lidar_mask"].append(r["lidar_mask"])
        sample["imu"].append(r["imu"])
        # store mask as dict to preserve modality names
        mm = r.get("modality_mask", {"image": 1, "lidar": 1, "imu": 1})
        sample["modality_masks"].append({k: int(v) for k, v in mm.items()})
    return sample


def _round_robin_balance(neg_sources: Dict[str, List[Tuple[List[int], int]]], target: int, seed: int):
    """
    Sample negatives across runs in a round-robin fashion to improve diversity.
    neg_sources: run_name -> list of (win, end_fid)
    """
    random.seed(seed)
    # Shuffle within each run
    for k in neg_sources:
        random.shuffle(neg_sources[k])
    # Round-robin selection
    picks = []
    keys = list(neg_sources.keys())
    idx_map = {k: 0 for k in keys}
    while len(picks) < target and keys:
        next_keys = []
        for k in keys:
            i = idx_map[k]
            if i < len(neg_sources[k]):
                picks.append((k, neg_sources[k][i]))
                idx_map[k] = i + 1
                if len(picks) >= target:
                    break
                next_keys.append(k)  # keep key in rotation
        keys = next_keys
        if not keys:  # ran out early
            break
    return picks


def build_dataset(
    pre_dir: str,
    out_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    window_frames: int,
    pos_stride: int,
    neg_stride: int,
    seed: int,
):
    os.makedirs(out_dir, exist_ok=True)
    run_dirs = _find_runs(pre_dir)
    if not run_dirs:
        raise SystemExit(f"No runs found under '{pre_dir}'. Expected subdirs with preprocessed_data.json")

    # Split runs by collision presence (stratified) into train/val/test
    train_runs, val_runs, test_runs, split_summary = _split_runs_by_collision(
        run_dirs, seed=seed, ratios=(train_ratio, val_ratio, test_ratio)
    )

    # Windowize & collect per split
    def collect(runs):
        pos, neg, by_run_recs = [], [], {}
        by_run_pos, by_run_neg = defaultdict(list), defaultdict(list)
        for rd in runs:
            run_name, p_wins, n_wins, recs = _build_windows_for_run(
                rd, window=window_frames, pos_stride=pos_stride, neg_stride=neg_stride
            )
            by_run_recs[run_name] = recs
            for win, end_fid in p_wins:
                by_run_pos[run_name].append((win, end_fid))
                pos.append((run_name, win, end_fid))
            for win, end_fid in n_wins:
                by_run_neg[run_name].append((win, end_fid))
                neg.append((run_name, win, end_fid))
        return pos, neg, by_run_pos, by_run_neg, by_run_recs

    pos_tr, neg_tr, by_pos_tr, by_neg_tr, recs_tr = collect(train_runs)
    pos_va, neg_va, by_pos_va, by_neg_va, recs_va = collect(val_runs)
    pos_te, neg_te, by_pos_te, by_neg_te, recs_te = collect(test_runs)

    # Balance by undersampling negatives to match positives count (per split)
    def balance_and_write(split_name, pos_list, neg_list, by_neg, recs_map):
        # determine target
        num_pos = len(pos_list)
        num_neg = len(neg_list)
        if num_pos == 0 and num_neg == 0:
            return [], {"pos": 0, "neg": 0, "chosen_neg": 0}

        # choose negatives
        chosen_neg = []
        if num_pos == 0:
            # no positives; keep up to some cap of negatives? we'll keep them all
            chosen_neg = neg_list
        else:
            # build mapping run_name -> windows for round-robin sampling
            neg_sources = defaultdict(list)
            for rn, win, fid in neg_list:
                neg_sources[rn].append((win, fid))
            picks = _round_robin_balance(neg_sources, target=num_pos, seed=seed)
            for rn, (win, fid) in picks:
                chosen_neg.append((rn, win, fid))

        # Combine pos + chosen_neg and serialize samples
        all_samples = []
        for rn, win, fid in pos_list:
            recs = recs_map[rn]
            all_samples.append(_serialize_window_sample(rn, win, fid, recs))
        for rn, win, fid in chosen_neg:
            recs = recs_map[rn]
            all_samples.append(_serialize_window_sample(rn, win, fid, recs))

        # shuffle for training
        random.Random(seed).shuffle(all_samples)

        # write jsonl
        out_p = os.path.join(OUT_DIR if out_dir is None else out_dir, f"{split_name}.jsonl")
        with open(out_p, "w") as f:
            for s in all_samples:
                f.write(json.dumps(s) + "\n")

        stats = {
            "pos": len(pos_list),
            "neg": len(neg_list),
            "chosen_neg": len(chosen_neg),
            "total_written": len(all_samples),
            "path": out_p,
        }
        return all_samples, stats

    # Write splits
    all_tr, st_tr = balance_and_write("train", pos_tr, neg_tr, by_neg_tr, recs_tr)
    all_va, st_va = balance_and_write("val",   pos_va, neg_va, by_neg_va, recs_va)
    all_te, st_te = balance_and_write("test",  pos_te, neg_te, by_neg_te, recs_te)

    # Write stats.json
    global_stats = {
        "split_summary": split_summary,
        "window_frames": window_frames,
        "pos_stride": pos_stride,
        "neg_stride": neg_stride,
        "stats_by_split": {
            "train": st_tr, "val": st_va, "test": st_te
        }
    }
    with open(os.path.join(OUT_DIR if out_dir is None else out_dir, "stats.json"), "w") as f:
        json.dump(global_stats, f, indent=2)

    print("Done.")
    print(json.dumps(global_stats, indent=2))


def _compute_window_frames(window_frames: int | None, window_seconds: float | None, fps: float) -> int:
    """Resolve to a concrete window length (in frames)."""
    if window_frames is None and window_seconds is None:
        return int(round(fps * 1.0))  # default: 1 second
    if window_frames is not None:
        wf = int(window_frames)
    else:
        if fps is None:
            raise SystemExit("WINDOW_SECONDS requires FPS")
        wf = int(round(float(window_seconds) * float(fps)))
    if wf <= 0:
        raise SystemExit("Window must be >= 1 frame")
    return wf


def main():
    # Normalize ratios to sum to 1.0 if they don't already
    s = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if s <= 0:
        raise SystemExit("Split ratios must sum to a positive value")
    train_r = TRAIN_RATIO / s
    val_r   = VAL_RATIO / s
    test_r  = TEST_RATIO / s

    window_frames = _compute_window_frames(WINDOW_FRAMES, WINDOW_SECONDS, FPS)

    build_dataset(
        pre_dir=PRE_DIR,
        out_dir=OUT_DIR,
        train_ratio=train_r,
        val_ratio=val_r,
        test_ratio=test_r,
        window_frames=window_frames,
        pos_stride=int(POS_STRIDE),
        neg_stride=int(NEG_STRIDE),
        seed=int(SEED),
    )


if __name__ == "__main__":
    main()
