#!/usr/bin/env python3
"""Read-only: the model's own input image of one window, clean vs -10/-12/-15 dB.

Uses the same RMS normalisation, AWGN and SpectrogramFrontEnd (n_fft 512, hop 256,
256x256, per-image standardisation) as Stage2. Validation files only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from dronerfa_data import add_complex_awgn, read_vector_slice, rms_normalize
from model import SpectrogramFrontEnd

PICKS = [("T0001_D00_S0101.mat", 50, "Phantom 3（室外）"),
         ("T1010_S0101.mat", 50, "Mavic 3（室内）"),
         ("T1011_S0101.mat", 50, "Matrice 300（室内）")]
CONDS = [(None, "不加噪声"), (-10.0, "−10 dB"), (-12.0, "−12 dB"), (-15.0, "−15 dB")]
WINDOW = 1_000_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    frontend = SpectrogramFrontEnd(n_fft=512, hop_length=256, image_size=256).eval()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
                         "font.size": 9})

    fig, axes = plt.subplots(len(PICKS), len(CONDS), figsize=(11, 2.6 * len(PICKS)), squeeze=False)
    for i, (fn, w, name) in enumerate(PICKS):
        with h5py.File(args.data_root / fn, "r") as h:
            s = w * WINDOW
            iq = rms_normalize(read_vector_slice(h["RF0_I"], s, WINDOW) + 1j * read_vector_slice(h["RF0_Q"], s, WINDOW))
        for j, (snr, cname) in enumerate(CONDS):
            x = iq if snr is None else add_complex_awgn(iq, snr, np.random.default_rng(12026))
            t = torch.from_numpy(np.stack((x.real, x.imag)))[None].float()
            with torch.inference_mode():
                img = frontend(t)[0, 0].numpy()
            ax = axes[i, j]
            ax.imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=-1.5, vmax=3.5,
                      extent=(0, 10, 2390, 2490))
            ax.set_title(f"{name} · {cname}", fontsize=9, loc="left")
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("频率 (MHz)")
            if i == len(PICKS) - 1:
                ax.set_xlabel("时间 (ms)")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print("written", args.output)


if __name__ == "__main__":
    main()
