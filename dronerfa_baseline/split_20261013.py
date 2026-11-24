"""Dataset split of 2026-10-13: ordinary train / val / test, about 6:2:2.

Replaces every earlier split (no supplementary test set, no frozen files).
For each class separately the files are shuffled with a fixed seed and split
6:2:2 (val and test = round(0.2 * n), at least 1). The split is per file, not
per window: windows of one file are cut from one continuous recording and are
nearly identical, so splitting windows would put near-copies in train and test.

  8 files  -> 4 / 2 / 2      12 files -> 8 / 2 / 2
  13 files -> 7 / 3 / 3      16 files -> 10 / 3 / 3

The resulting assignment is written to split_20261013.json by write_split().
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SPLIT_SEED = 2026
RATIO = (0.6, 0.2, 0.2)
SPLIT_FILE = Path(__file__).resolve().parent / "split_20261013.json"

NAMES = {
    "T0000": "背景", "T0001": "Phantom 3", "T0010": "Phantom 4 Pro", "T0011": "Matrice 200",
    "T0100": "Matrice 100", "T0101": "Air 2S", "T0110": "Mini 3 Pro", "T0111": "Inspire 2",
    "T1000": "Mavic Pro", "T1001": "Mini 2", "T1010": "Mavic 3", "T1011": "Matrice 300",
    "T1100": "Phantom 4 Pro RTK",
}

# Matrice 300 files without a video link (control link only), for reporting.
NO_VIDEO = {"T1011_S0111", "T1011_S1000", "T1011_S1001", "T1011_S1010", "T1011_S1011", "T1011_S1100"}


def build_split(stems: list[str]) -> dict[str, str]:
    by_target: dict[str, list[str]] = {}
    for s in sorted(stems):
        if s.split("_")[0] in NAMES:
            by_target.setdefault(s.split("_")[0], []).append(s)
    split: dict[str, str] = {}
    for target in sorted(by_target):
        files = by_target[target][:]
        random.Random(f"{SPLIT_SEED}-{target}").shuffle(files)
        n = len(files)
        n_val = n_test = max(1, round(RATIO[1] * n))
        for i, s in enumerate(files):
            split[s] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return split


def write_split(stems: list[str]) -> dict[str, str]:
    split = build_split(stems)
    SPLIT_FILE.write_text(json.dumps(dict(sorted(split.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    return split
