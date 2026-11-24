import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

OUT = Path(__file__).resolve().parent
WORK = OUT.parent
ROOT = WORK.parent
CACHE_ROOT = Path("E:/main_W_window_20261110")
INPUT = WORK / "crop_slice_20261107"
BGA = WORK / "bgA_20261109"
N2 = WORK / "side_N2_L50tail_index_20261110"
TASK_L = WORK / "side_L_wifimix_20261110"
TASK_M = WORK / "side_M_notchfill_20261110"
TASK_H = WORK / "train_stage1_L20_20261108"
M_CACHE = Path("E:/side_M_notchfill_20261110")
H_CACHE = Path("E:/drone_lab_spec_L20_20261108")
PROTECTED = (INPUT, TASK_H, BGA, TASK_L, TASK_M, N2)
WINDOWS = ("10", "20", "30", "40", "50t")
BANDS = ("2.4GHz", "5.8GHz")
SPLITS = ("train", "val", "test")
SEEDS = (2026, 2027, 2028)
FS = {"2.4GHz": 61_440_000, "5.8GHz": 100_000_000}
FC = {"2.4GHz": 2430.0, "5.8GHz": 5800.0}
FLAGS = ("lack_feature", "primary_weak", "power_shift", "floor_raised", "interference")
LIMIT = 20_000_000_000

sys.path.insert(0, str(ROOT / "dronerfa_baseline"))
from model import ResNet18

torch.set_num_threads(4)


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def writecsv(path, records):
    records = list(records)
    if not records:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def kept(row):
    return row["keep"] == "True"


def order(row):
    return row["block_file"], int(row["start"])


def index(window):
    if window == "50t":
        return rows(N2 / "index_L50t.csv")
    return rows(INPUT / f"index_L{window}.csv")


def bg_index(window):
    if window != "50t":
        return rows(BGA / f"index_bgA_L{window}.csv")
    original = rows(BGA / "index_bgA_L50.csv")
    rejected_blocks = {Path(row["block_file"]).stem for row in original if not kept(row)}
    mirror = []
    for row in index(window):
        if row["band"] != "2.4GHz" or row["dataset"] != "data":
            continue
        stem = Path(row["block_file"]).stem + "__bgA_s0"
        item = {k: row[k] for k in original[0] if k in row}
        item.update(
            block_file=str(Path("E:/drone_lab_bgA_20261109/2.4GHz") / (stem + ".npy")),
            dataset="dataA_mirror", label="background", model="none",
            state="unknown", bw_label="unknown", sample_id="bgA_" + row["sample_id"],
            keep=str(stem not in rejected_blocks), paired_sample_id=row["sample_id"],
            mirror_shift_mhz="0", on_original_grid=row["on_original_grid"],
        )
        item.update(dict.fromkeys(FLAGS, "False"))
        mirror.append(item)
    return mirror


def select(records, band, split):
    return [r for r in records if r["band"] == band and r["split"] == split and kept(r)]


class IQReader:
    """Map one source block at a time and return an owned slice."""
    def __init__(self):
        self.path = None
        self.block = None

    def __call__(self, row):
        path, start = order(row)
        if path != self.path:
            self.block = np.load(path, mmap_mode="r")
            self.path = path
        size = int(row["n_samples"])
        iq = np.array(self.block[start:start + size])
        if iq.shape != (size,) or iq.dtype != np.complex64:
            raise OSError(f"Invalid IQ slice {path} {start} {size}: {iq.shape} {iq.dtype}")
        return iq


def background_iq(records):
    result = [None] * len(records)
    read = IQReader()
    for i in sorted(range(len(records)), key=lambda i: order(records[i])):
        result[i] = read(records[i])
    return result


def train_groups(signal, mirror):
    # L uses the H metadata order for the v3 background draw pool.
    background = sorted((r for r in signal if r["dataset"] == "v3" and r["label"] == "background"), key=order)
    rng = np.random.default_rng(20261110)
    groups = {}
    for name, records in (("drone", [r for r in signal if r["dataset"] == "data"]), ("bg", mirror)):
        assignments = [(float(rng.uniform(-5, 15)), int(rng.integers(0, len(background)))) for _ in records]
        groups[name] = (records, assignments)
    return background, groups


def paired(records, assignments, background, mixing, consume=None):
    iq = background_iq(background)
    read = IQReader()
    pairs = [None] * len(records)
    for i in sorted(range(len(records)), key=lambda i: order(records[i])):
        row = records[i]
        x, j = assignments[i]
        z, gain, pd, pb = mixing(read(row), iq[j], x)
        pairs[i] = dict(
            sample_id=row["sample_id"], background_sample_id=background[j]["sample_id"],
            background_group_id=background[j]["group_id"], X=x, g=gain, P_d=pd, P_b=pb,
            actual_power_ratio_db=float(10 * np.log10(gain * gain * pd / pb)),
        )
        if consume is not None:
            consume(i, z)
    return pairs


def coverage(records):
    blocks = defaultdict(list)
    for row in records:
        start = int(row["start"])
        blocks[row["block_file"]].append((start, start + int(row["n_samples"])))
    total = 0
    for intervals in blocks.values():
        end = 0
        for start, stop in sorted(intervals):
            total += max(0, stop - max(start, end))
            end = max(end, stop)
    return total


def inventory(window, raw, mirror):
    groups = defaultdict(list)
    for row in raw + mirror:
        groups[tuple(row[k] for k in ("band", "dataset", "label", "split"))].append(row)
    return [
        dict(L=window, band=band, dataset=dataset, label=label, split=split,
             n=len(group), keep=sum(map(kept, group)),
             keep_unique_s=coverage([r for r in group if kept(r)]) / FS[band])
        for (band, dataset, label, split), group in sorted(groups.items())
    ]


def protected_hashes():
    result = {}
    for folder in PROTECTED:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    result[path.relative_to(WORK).as_posix()] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def disk_guard(extra=0):
    used = sum(p.stat().st_size for root in (OUT, CACHE_ROOT) for p in root.rglob("*") if p.is_file())
    if used + extra > LIMIT:
        raise OSError(f"Output disk limit: {used} + {extra} > {LIMIT}")
    return used


def window_dirs(window):
    out, cache = OUT / f"L{window}", CACHE_ROOT / f"L{window}"
    out.mkdir(exist_ok=True)
    cache.mkdir(exist_ok=True)
    return out, cache


def train_window():
    return os.environ["WINDOW_W_L"]
