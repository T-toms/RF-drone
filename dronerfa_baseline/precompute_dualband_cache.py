"""One pass over DroneRFa .mat files: dual-band window cache + per-ms spectra.

For every file this writes <cache>/<stem>/:
  win10.npy / win15.npy / win20.npy  float16 [n, 2, 256, 256]
      clean log-power spectrograms of non-overlapping 10/15/20 ms windows;
      channel 0 = RF0 (2.4 GHz band), channel 1 = RF1 (5.8 GHz band).
      Each channel: RF0/RF1 IQ -> remove mean -> STFT (Hamming, n_fft 1024,
      hop 512) -> 10*log10 power -> fftshift (row 0 = lowest frequency) ->
      per-window per-channel standardisation -> area-resize to 256x256.
      Windows never cross a 10,000,000-sample (0.1 s) recording block.
  ms_mean.npy / ms_max.npy  float16 [n_ms, 2, 512]
      per 1 ms (100,000 samples): mean / max over STFT frames of the raw
      log power (dB, not normalised), frequency pooled 1024 -> 512 bins.
      Used for whole-file spectrogram figures and window screening.
  meta.json  sample counts, window start samples, frequency axes.

Area resizing replaces the bilinear resize of model.SpectrogramFrontEnd:
bilinear 1953 -> 256 frames only reads ~2 of every 7.6 frames, so short
hopping bursts can vanish from the image.

Usage:
  python precompute_dualband_cache.py --files-from list.txt --cache D:/dronerfa_cache
Already finished files (meta.json present) are skipped, so reruns resume.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import h5py
import numpy as np

BLOCK = 10_000_000
MS = 100_000
SAMPLE_RATE = 100e6
N_FFT = 1024
HOP = 512
IMAGE = 256
WINDOW_MS = (10, 15, 20)
MS_BINS = 512
CHANNELS = (("RF0_I", "RF0_Q"), ("RF1_I", "RF1_Q"))


def center_freqs_mhz(stem: str) -> tuple[float, float]:
    # FrSky X20 (T10010) and Taranis Plus (T10100) were recorded at 915 / 2440 MHz.
    target = stem.split("_")[0]
    if target in {"T10010", "T10100"}:
        return 915.0, 2440.0
    return 2440.0, 5800.0


def read_block(handle: h5py.File, start: int, stop: int) -> np.ndarray:
    """Return complex64 [2, stop-start] for RF0 and RF1."""
    out = np.empty((2, stop - start), dtype=np.complex64)
    for c, (i_name, q_name) in enumerate(CHANNELS):
        i_ds, q_ds = handle[i_name], handle[q_name]
        if i_ds.ndim == 2:
            i_val = i_ds[0, start:stop]
            q_val = q_ds[0, start:stop]
        else:
            i_val = i_ds[start:stop]
            q_val = q_ds[start:stop]
        out[c].real = i_val
        out[c].imag = q_val
    return out


def log_power_frames(torch, x, window):
    """x: complex [batch, samples] on device -> dB [batch, freq(shifted), frames]."""
    spec = torch.stft(
        x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window=window,
        center=False, normalized=True, onesided=False, return_complex=True,
    )
    power = 10.0 * torch.log10(spec.abs().square() + 1e-10)
    return torch.fft.fftshift(power, dim=1)


def window_images(torch, F, block_iq, length, window):
    """block_iq: complex [2, n] on device -> float16 [k, 2, 256, 256] (k windows)."""
    n = block_iq.shape[1]
    k = n // length
    if k == 0:
        return None
    x = block_iq[:, : k * length].reshape(2, k, length).transpose(0, 1).reshape(2 * k, length)
    x = x - x.mean(dim=1, keepdim=True)
    db = log_power_frames(torch, x, window)
    mean = db.mean(dim=(1, 2), keepdim=True)
    std = db.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
    norm = (db - mean) / std
    img = F.interpolate(norm.unsqueeze(1), size=(IMAGE, IMAGE), mode="area").squeeze(1)
    return img.reshape(k, 2, IMAGE, IMAGE).to(torch.float16).cpu().numpy()


def ms_spectra(torch, F, block_iq, window):
    """complex [2, n] -> (mean, max) float16 [n_ms, 2, MS_BINS] in raw dB."""
    n = block_iq.shape[1]
    k = n // MS
    if k == 0:
        return None, None
    x = block_iq[:, : k * MS].reshape(2 * k, MS)
    db = log_power_frames(torch, x, window)  # [2k, 1024, frames]
    db = F.avg_pool1d(db.transpose(1, 2), kernel_size=N_FFT // MS_BINS).transpose(1, 2)
    mean = db.mean(dim=2).reshape(2, k, MS_BINS).transpose(0, 1)
    peak = db.amax(dim=2).reshape(2, k, MS_BINS).transpose(0, 1)
    return mean.to(torch.float16).cpu().numpy(), peak.to(torch.float16).cpu().numpy()


def process_file(path: str, cache_root: str, device: str) -> str:
    import torch
    from torch.nn import functional as F

    src = Path(path)
    out = Path(cache_root) / src.stem
    if (out / "meta.json").exists():
        return f"skip {src.name}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    window = torch.hamming_window(N_FFT, device=device)
    with h5py.File(src, "r") as handle:
        ds = handle["RF0_I"]
        total = int(max(ds.shape))
        blocks = [(b, min(b + BLOCK, total)) for b in range(0, total, BLOCK)]
        counts = {L: sum((e - s) // (L * MS) for s, e in blocks) for L in WINDOW_MS}
        n_ms = sum((e - s) // MS for s, e in blocks)
        arrays = {
            L: np.lib.format.open_memmap(
                out / f"win{L}.npy.tmp", mode="w+", dtype=np.float16,
                shape=(counts[L], 2, IMAGE, IMAGE),
            )
            for L in WINDOW_MS
        }
        ms_mean = np.empty((n_ms, 2, MS_BINS), dtype=np.float16)
        ms_max = np.empty((n_ms, 2, MS_BINS), dtype=np.float16)
        starts = {L: [] for L in WINDOW_MS}
        cursor = {L: 0 for L in WINDOW_MS}
        ms_cursor = 0
        for block_index, (s, e) in enumerate(blocks):
            iq = torch.from_numpy(read_block(handle, s, e)).to(device)
            with torch.no_grad():
                for L in WINDOW_MS:
                    length = L * MS
                    imgs = window_images(torch, F, iq, length, window)
                    if imgs is None:
                        continue
                    c = cursor[L]
                    arrays[L][c : c + len(imgs)] = imgs
                    cursor[L] += len(imgs)
                    starts[L].extend(s + j * length for j in range(len(imgs)))
                mean, peak = ms_spectra(torch, F, iq, window)
                if mean is not None:
                    ms_mean[ms_cursor : ms_cursor + len(mean)] = mean
                    ms_max[ms_cursor : ms_cursor + len(mean)] = peak
                    ms_cursor += len(mean)
            del iq
    for L, arr in arrays.items():
        arr.flush()
        del arr
    arrays.clear()
    for L in WINDOW_MS:
        os.replace(out / f"win{L}.npy.tmp", out / f"win{L}.npy")
    np.save(out / "ms_mean.npy", ms_mean)
    np.save(out / "ms_max.npy", ms_max)
    rf0, rf1 = center_freqs_mhz(src.stem)
    half = SAMPLE_RATE / 2e6
    meta = {
        "schema": "dronerfa-dualband-cache/v1",
        "source": str(src),
        "source_bytes": src.stat().st_size,
        "total_samples": total,
        "block_samples": BLOCK,
        "n_blocks": len(blocks),
        "n_ms": n_ms,
        "window_ms": list(WINDOW_MS),
        "window_counts": {str(L): counts[L] for L in WINDOW_MS},
        "window_starts": {str(L): starts[L] for L in WINDOW_MS},
        "channels": ["RF0", "RF1"],
        "center_mhz": [rf0, rf1],
        "freq_range_mhz": [[rf0 - half, rf0 + half], [rf1 - half, rf1 + half]],
        "freq_row0": "lowest frequency (fftshift)",
        "stft": {"n_fft": N_FFT, "hop": HOP, "window": "hamming", "normalized": True},
        "image": {"size": IMAGE, "resize": "area", "norm": "per-window per-channel z-score"},
        "ms_bins": MS_BINS,
        "seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return f"done {src.name} {meta['seconds']}s windows={meta['window_counts']}"


def _worker(args):
    path, cache_root, device = args
    try:
        return process_file(path, cache_root, device)
    except Exception:
        return f"FAIL {path}\n{traceback.format_exc()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files-from", required=True, type=Path)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    files = [
        line.strip() for line in args.files_from.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    Path(args.cache).mkdir(parents=True, exist_ok=True)
    jobs = [(f, args.cache, args.device) for f in files]
    print(f"{len(jobs)} files -> {args.cache}, workers={args.workers}", flush=True)
    failed = 0
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(_worker, jobs), 1):
            print(f"[{i}/{len(jobs)}] {msg}", flush=True)
            failed += msg.startswith("FAIL")
    print(f"finished, failed={failed}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
