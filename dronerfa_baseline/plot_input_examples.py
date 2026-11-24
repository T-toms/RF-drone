"""Model-input spectrograms with real axes: one example window per class.

Reads the cached clean window images (precompute_dualband_cache.py, row 0 =
lowest frequency) and draws them with time in ms and absolute frequency in MHz
(RF0 2390-2490 MHz, RF1 5750-5850 MHz; 865-965 / 2390-2490 MHz for the two
915 MHz remote controls). Colour = per-window standardised log power, exactly
what the model sees.

Usage: python plot_input_examples.py --cache DIR --out FILE.png [--window-ms 15]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from split_20261013 import NAMES

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
                     "font.size": 8})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-ms", type=int, default=15)
    p.add_argument("--index", type=int, default=20, help="window index taken from each file")
    args = p.parse_args()
    stems = sorted(d.name for d in args.cache.iterdir() if (d / "meta.json").exists())
    first = {}
    for s in stems:
        first.setdefault(s.split("_")[0], s)
    targets = [t for t in NAMES if t in first]
    fig, axes = plt.subplots(len(targets), 2, figsize=(9, 1.55 * len(targets)), squeeze=False)
    L = args.window_ms
    for row, t in enumerate(targets):
        stem = first[t]
        meta = json.loads((args.cache / stem / "meta.json").read_text(encoding="utf-8"))
        img = np.load(args.cache / stem / f"win{L}.npy", mmap_mode="r")[args.index].astype(np.float32)
        for ch in (0, 1):
            ax = axes[row, ch]
            lo, hi = meta["freq_range_mhz"][ch]
            ax.imshow(img[ch], origin="lower", aspect="auto", cmap="viridis", extent=(0, L, lo, hi),
                      vmin=-1.5, vmax=4)
            ax.set_yticks([lo + 10, (lo + hi) / 2, hi - 10])
            if row == len(targets) - 1:
                ax.set_xlabel("时间 (ms)")
            if ch == 0:
                ax.set_ylabel(f"{NAMES[t]}\n频率 (MHz)", fontsize=7.5)
            if row == 0:
                ax.set_title(f"RF{ch}（{'2.4' if ch == 0 else '5.8'} GHz 频段）", fontsize=9)
            ax.text(0.01, 0.95, stem, transform=ax.transAxes, color="white", fontsize=6, va="top")
    fig.suptitle(f"模型输入时频图示例（{L} ms 窗口，纵轴为真实频率）", fontsize=10, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)


if __name__ == "__main__":
    main()
