"""
Dataset creation for CARLA collision prediction using my runs. Feel free to modify according to your runs.

Collision scenarios (positive samples):
- RearEnd: Vehicle approaches from behind and collides
- SideCollision: Vehicle collides from the side (lane change)
- BlindSpot: Vehicle in blind spot causes collision during lane change

Non-collision scenarios (negative samples):
- EmptyRoad: Clear road with no other vehicles
- LightTraffic: Sparse traffic, no dangerous situations
- HeavyTraffic: Dense traffic requiring careful navigation
- Tailgating: Following vehicle closely but maintaining safe distance
- Overtake: Safe overtaking maneuvers without collision
"""

from __future__ import annotations
import os
import json
import random
import re
from collections import defaultdict, Counter
from typing import Dict, List

# Configuration
PRE_DIR: str = "PreprocessedData"
OUT_DIR: str = "Dataset"
PRE_JSON_NAME: str = "preprocessed_data.json"

WINDOW_FRAMES: int | None = None
WINDOW_SECONDS: float | None = 2.0
FPS: float = 30.0

POS_STRIDE: int = 3
NEG_STRIDE: int = 8

TRAIN_RATIO: float = 0.65
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.20

SEED: int = 13

# Optional: hard-pin specific runs to splits (these reflect the suggested swaps)
# Any listed runs that are not present will be ignored.
SPLIT_OVERRIDES = {
    "train": {"BlindSpot6", "Tailgating4"},
    "val": {"SideCollision4"},
    "test": {"RearEnd1", "Overtake4"},
}

COLLISION_SCENARIOS = {"RearEnd", "SideCollision", "BlindSpot"}
NONCOLLISION_SCENARIOS = {"EmptyRoad", "LightTraffic", "HeavyTraffic", "Tailgating", "Overtake"}


def _prefix(run_name: str) -> str:
    """Extract alphabetic prefix from run name."""
    m = re.match(r"[A-Za-z]+", run_name)
    return m.group(0) if m else run_name


def _is_collision(prefix: str) -> bool:
    """Check if scenario prefix indicates a collision scenario."""
    return prefix in COLLISION_SCENARIOS


def _list_runs(pre_dir: str) -> List[str]:
    """List run directory names."""
    if not os.path.isdir(pre_dir):
        return []
    return sorted([d for d in os.listdir(pre_dir) if os.path.isdir(os.path.join(pre_dir, d))])


def _load_run_frames(pre_dir: str, run: str) -> List[tuple[int, int]]:
    """Load (frame_id, label) pairs for a run."""
    p = os.path.join(pre_dir, run, PRE_JSON_NAME)
    if not os.path.isfile(p):
        return []
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except Exception:
        return []
    items = []
    if isinstance(data, dict):
        for k, rec in data.items():
            try:
                fid = int(k)
            except Exception:
                continue
            lab = int(rec.get("label", 0))
            items.append((fid, lab))
    elif isinstance(data, list):
        for rec in data:
            try:
                fid = int(rec.get("frame_id"))
                lab = int(rec.get("label", 0))
                items.append((fid, lab))
            except Exception:
                continue
    items.sort(key=lambda x: x[0])
    return items


def _scenario_buckets(runs: List[str]) -> Dict[tuple[bool, str], List[str]]:
    """Group runs by (is_collision, scenario_prefix)."""
    buckets: Dict[tuple[bool, str], List[str]] = defaultdict(list)
    for r in runs:
        pref = _prefix(r)
        coll = _is_collision(pref)
        buckets[(coll, pref)].append(r)
    for k in buckets:
        buckets[k].sort()
    return buckets


def _apply_overrides(runs: List[str], split: Dict[str, List[str]]) -> None:
    """Apply SPLIT_OVERRIDES to assign specific runs to splits."""
    all_runs = set(runs)
    # Remove any already assigned to avoid duplicates
    assigned = set(sum(split.values(), []))
    for target, names in SPLIT_OVERRIDES.items():
        for name in names:
            if name in all_runs and name not in assigned:
                split[target].append(name)
                assigned.add(name)


def _scenario_aware_split(runs: List[str],
                          train_r: float, val_r: float, test_r: float,
                          seed: int) -> Dict[str, List[str]]:
    """Create splits ensuring scenario coverage while honoring target ratios."""
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}

    _apply_overrides(runs, split)

    already = set(sum(split.values(), []))
    pool = [r for r in runs if r not in already]
    rng.shuffle(pool)

    buckets = _scenario_buckets(pool)
    order = ["train", "val", "test"]
    for key, group in buckets.items():
        i = 0
        for r in group:
            split[order[i % 3]].append(r)
            i += 1
    total = len(runs)
    target_counts = {
        "train": int(round(total * train_r)),
        "val": int(round(total * val_r)),
        "test": int(round(total * test_r)),
    }
    diff = total - sum(target_counts.values())
    if diff != 0:
        target_counts["train"] += diff

    def coverage_ok(s: List[str]) -> bool:
        prefs = {_prefix(x) for x in s}
        coll_covered = len(prefs & COLLISION_SCENARIOS) >= min(len(COLLISION_SCENARIOS), len(prefs & COLLISION_SCENARIOS))
        noncoll_covered = len(prefs & NONCOLLISION_SCENARIOS) >= min(len(NONCOLLISION_SCENARIOS), len(prefs & NONCOLLISION_SCENARIOS))
        return True if not s else (len(prefs) >= 2 and (coll_covered or any(_is_collision(_prefix(x)) for x in s) == False))

    def take_from(src: str, dst: str):
        for i, r in enumerate(split[src]):
            pref = _prefix(r)
            if sum(1 for x in split[src] if _prefix(x) == pref) > 1:
                split[dst].append(r)
                del split[src][i]
                return True
        return False

    changed = True
    while changed:
        changed = False
        for k in ["train", "val", "test"]:
            while len(split[k]) > target_counts[k] + 1:
                dst = min(["train", "val", "test"], key=lambda x: len(split[x]) - target_counts[x])
                if dst == k:
                    break
                if take_from(k, dst):
                    changed = True
                else:
                    break

    return split


def _ensure_min_test_coverage(split: Dict[str, List[str]]) -> None:
    """Ensure test split includes at least one run per collision scenario."""
    test_prefs = {_prefix(r) for r in split["test"]}
    for needed in sorted(COLLISION_SCENARIOS):
        if needed not in test_prefs:
            for src in ("train", "val"):
                for i, r in enumerate(split[src]):
                    if _prefix(r) == needed:
                        split["test"].append(r)
                        del split[src][i]
                        test_prefs.add(needed)
                        break
                if needed in test_prefs:
                    break


def _make_windows_for_run(frames: List[tuple[int, int]], window_frames: int,
                          pos_stride: int, neg_stride: int) -> Dict[str, List[dict]]:
    """Create sliding windows from frames, returning positive and negative samples."""
    n = len(frames)
    if n == 0 or window_frames <= 0:
        return {"pos": [], "neg": []}

    pos, neg = [], []
    i = window_frames - 1
    while i < n:
        start_idx = i - window_frames + 1
        end_fid, lab = frames[i]
        start_fid = frames[start_idx][0]
        rec = {"start_frame": start_fid, "end_frame": end_fid, "label": int(lab)}
        if lab == 1:
            pos.append(rec)
            i += max(1, pos_stride)
        else:
            neg.append(rec)
            i += max(1, neg_stride)
    return {"pos": pos, "neg": neg}


def _write_jsonl(path: str, rows: List[dict]) -> None:
    """Write data rows to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build_dataset(pre_dir: str = PRE_DIR,
                  out_dir: str = OUT_DIR,
                  train_ratio: float = TRAIN_RATIO,
                  val_ratio: float = VAL_RATIO,
                  test_ratio: float = TEST_RATIO,
                  window_frames: int | None = WINDOW_FRAMES,
                  pos_stride: int = POS_STRIDE,
                  neg_stride: int = NEG_STRIDE,
                  seed: int = SEED) -> None:
    """Build train/val/test dataset splits from preprocessed data."""

    if window_frames is None:
        if WINDOW_SECONDS is None:
            window_frames = int(FPS)
        else:
            window_frames = int(round(FPS * float(WINDOW_SECONDS)))
    assert window_frames > 0, "window_frames must be > 0"

    runs = _list_runs(pre_dir)
    if not runs:
        print(f"[WARN] No runs found in {pre_dir}")
        return

    split = _scenario_aware_split(runs, train_ratio, val_ratio, test_ratio, seed)
    _ensure_min_test_coverage(split)

    per_run_windows: Dict[str, Dict[str, List[dict]]] = {}
    run_meta: Dict[str, dict] = {}
    for r in runs:
        frames = _load_run_frames(pre_dir, r)
        windows = _make_windows_for_run(frames, window_frames, pos_stride, neg_stride)
        per_run_windows[r] = windows
        run_meta[r] = {"scenario": _prefix(r), "is_collision": int(_is_collision(_prefix(r)))}
    final = {"train": [], "val": [], "test": []}
    stats = {"split": {}, "per_scenario": {}}

    for split_name in ("train", "val", "test"):
        pos_rows, neg_rows = [], []
        for r in split[split_name]:
            meta = run_meta[r]
            for rec in per_run_windows[r]["pos"]:
                pos_rows.append({"run": r, **rec, **meta})
            for rec in per_run_windows[r]["neg"]:
                neg_rows.append({"run": r, **rec, **meta})

        rng = random.Random(seed + hash(split_name) % (2**16))
        needed = len(pos_rows)
        if len(neg_rows) > needed and needed > 0:
            neg_rows = rng.sample(neg_rows, needed)

        rows = pos_rows + neg_rows
        rng.shuffle(rows)
        final[split_name] = rows

        stats["split"][split_name] = {
            "runs": split[split_name],
            "pos": len(pos_rows),
            "neg": len(neg_rows),
            "total": len(rows),
        }

        sc_counts = Counter([r["scenario"] for r in rows])
        stats["per_scenario"][split_name] = dict(sc_counts)
    os.makedirs(out_dir, exist_ok=True)
    for k in ("train", "val", "test"):
        _write_jsonl(os.path.join(out_dir, f"{k}.jsonl"), final[k])

    stats["config"] = {
        "window_frames": window_frames,
        "pos_stride": pos_stride,
        "neg_stride": neg_stride,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "seed": seed,
        "pre_dir": pre_dir,
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("=== Split summary ===")
    for k in ("train", "val", "test"):
        s = stats["split"][k]
        print(f"{k}: runs={len(s['runs'])}  pos={s['pos']}  neg={s['neg']}  total={s['total']}")
        print(f"  scenarios: {', '.join(sorted(stats['per_scenario'][k].keys()))}")
    print(f"Wrote {out_dir}/{{train,val,test}}.jsonl and {out_dir}/stats.json")


def main():
    """Entry point for dataset creation."""
    random.seed(SEED)
    build_dataset()


if __name__ == "__main__":
    main()
