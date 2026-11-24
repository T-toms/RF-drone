"""Whole-file time-frequency overview from the dual-band cache (ms_max.npy).

One PNG per file: top RF0 (2.4 GHz band), bottom RF1 (5.8 GHz band); x = time
over the full recording, y = absolute frequency; each column is the max-hold
spectrum of 1 ms, so short hopping bursts stay visible. Dashed lines mark the
0.1 s recording-block boundaries (the data is not continuous across them).

Usage:
  python plot_file_overview.py --cache D:/dronerfa_cache_20261013/files --out DIR [stems...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
                     "font.size": 9})

NAMES = {
    "T0000": "背景", "T0001": "Phantom 3", "T0010": "Phantom 4 Pro", "T0011": "Matrice 200",
    "T0100": "Matrice 100", "T0101": "Air 2S", "T0110": "Mini 3 Pro", "T0111": "Inspire 2",
    "T1000": "Mavic Pro", "T1001": "Mini 2", "T1010": "Mavic 3", "T1011": "Matrice 300",
    "T1100": "Phantom 4 Pro RTK",
}


def load(cache: Path, stem: str):
    d = cache / stem
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return meta, np.load(d / "ms_max.npy").astype(np.float32)


def plot_overview(cache: Path, stem: str, out: Path, flagged: dict[str, list[int]] | None = None) -> Path:
    meta, ms_max = load(cache, stem)
    n_ms = ms_max.shape[0]
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.4), sharex=True)
    for ch, ax in enumerate(axes):
        img = ms_max[:, ch, :].T  # [freq, time]
        lo, hi = meta["freq_range_mhz"][ch]
        vmin = float(np.percentile(img, 5))
        vmax = float(np.percentile(img, 99.8))
        ax.imshow(img, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax,
                  extent=(0, n_ms, lo, hi), interpolation="nearest")
        for b in range(1, meta["n_blocks"]):
            ax.axvline(b * 100, color="white", lw=0.6, ls="--", alpha=0.6)
        ax.set_ylabel(f"RF{ch} 频率 (MHz)")
        if flagged:
            for L, idx in flagged.items():
                starts = meta["window_starts"][L]
                for i in idx:
                    s = starts[i] / 100_000
                    ax.axvspan(s, s + int(L), color="red", alpha=0.25, lw=0)
    target = stem.split("_")[0]
    axes[0].set_title(f"{stem}  {NAMES.get(target, target)}  全文件 {n_ms} ms（每列为 1 ms 内的最大值谱；虚线为 0.1 s 存盘分界）",
                      loc="left", fontsize=10)
    axes[1].set_xlabel("时间 (ms)")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stem}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("stems", nargs="*")
    args = parser.parse_args()
    stems = args.stems or sorted(p.name for p in args.cache.iterdir() if (p / "meta.json").exists())
    for stem in stems:
        print(plot_overview(args.cache, stem, args.out))


if __name__ == "__main__":
    main()
