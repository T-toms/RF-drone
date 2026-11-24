"""Micro-benchmark: how fast is the frozen window reader (input side only)?"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
BASE_DIR = PROJECT_ROOT / "dronerfa_baseline"
os.environ.setdefault("WINDIR", r"C:\Windows")
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

import argparse

import numpy as np
import torch

import verify_cascade3_evaluation as V

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="heldout")
parser.add_argument("--condition", default="clean")
parser.add_argument("--windows", type=int, default=100)
args = parser.parse_args()

config = V.read_json(V.CONFIG_PATH)
split = "heldout" if args.dataset == "heldout" else "test"
key = "heldout_index" if args.dataset == "heldout" else "old_test_index"
index_path = PROJECT_ROOT / str(config["datasets"][key])
noise_seed = int(config["noise"]["seed"])

from dronerfa_data import DroneRFaWindowDataset

snr = None if args.condition == "clean" else float(args.condition)
dataset = DroneRFaWindowDataset(
    index_path, split=split, fixed_snr=snr, apply_fixed_snr=snr is not None, seed=noise_seed
)
import hashlib

digest = hashlib.sha256()
started = time.perf_counter()
for position in range(args.windows):
    tensor, label = dataset[position]
    digest.update(np.ascontiguousarray(tensor[:, :4096].numpy()).tobytes())
elapsed = time.perf_counter() - started
dataset.close()
print(
    f"reader-only {args.windows} windows of {args.dataset}/{args.condition} in {elapsed:.1f}s "
    f"({args.windows / elapsed:.1f} win/s); projected full split 1820 -> {1820 / (args.windows / elapsed):.0f}s",
    flush=True,
)
