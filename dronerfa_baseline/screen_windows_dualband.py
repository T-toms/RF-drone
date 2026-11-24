"""Screen windows that do not look like the rest of their own file.

Teacher's rule (2026-10-13): a 10/20 ms window that differs strongly from the
other windows cut from the same file, or lacks the features the file should
have, is discarded (or a longer window is used).

From the dual-band cache (1 ms spectra in raw dB, ms_max = max over the STFT
frames of that ms, ms_mean = mean over them):
  floor[ch, bin]  receiver noise floor = median over background files of the
                  per-file median of ms_max (all files agree within 0.3 dB).
  sustained cell  ms_mean > floor + 6 dB: the bin is on for most of the 1 ms
                  (long bursts, video link, WiFi frames).
  transient cell  ms_max > floor + 6 dB but not sustained (short hop blips).
  6 dB: on pure-noise RF1 of background no cell crosses it.
Band-edge bins (filter roll-off) are ignored.

Per window of L ms there are 4 components: transient / sustained cells in
RF0 and in RF1. A component is "characteristic" of the file when its median
count over the file's windows is >= MIN_CELLS.
  missing  every characteristic component of the file is below
           MISSING_RATIO x its file median -> the window carries none of the
           signal this file normally has. Candidate for removal.
  differ   some component is far above the file's usual level (robust z of
           log1p(count) > Z): extra interference, e.g. a WiFi burst. Kept,
           reported only.
Screening is meant for drone files; background is intermittent by nature,
so background windows are reported but never removed.

Usage:
  python screen_windows_dualband.py --cache DIR --out DIR
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

THRESH_DB = 6.0
EDGE_BINS = 30  # of 512: ignore ~5.9 MHz at each band edge (roll-off)
MIN_CELLS = 5.0
MISSING_RATIO = 0.25
Z = 3.5
WINDOWS = ("10", "15", "20")
COMPONENTS = ("RF0_transient", "RF0_sustained", "RF1_transient", "RF1_sustained")


def noise_floor(cache: Path) -> np.ndarray:
    meds = [
        np.median(np.load(d / "ms_max.npy").astype(np.float32), axis=0)
        for d in sorted(cache.glob("T0000_*"))
        if (d / "meta.json").exists()
    ]
    if not meds:
        raise SystemExit("no background files in cache")
    return np.median(np.stack(meds), axis=0)


def ms_components(cache: Path, stem: str, floor: np.ndarray) -> np.ndarray:
    """[n_ms, 4] cell counts per 1 ms, in COMPONENTS order."""
    ms_max = np.load(cache / stem / "ms_max.npy").astype(np.float32)
    ms_mean = np.load(cache / stem / "ms_mean.npy").astype(np.float32)
    sl = slice(EDGE_BINS, -EDGE_BINS)
    peak = (ms_max > floor + THRESH_DB)[:, :, sl]
    sust = (ms_mean > floor + THRESH_DB)[:, :, sl]
    trans = peak & ~sust
    return np.stack(
        [trans[:, 0].sum(1), sust[:, 0].sum(1), trans[:, 1].sum(1), sust[:, 1].sum(1)], axis=1
    ).astype(np.float32)


def screen_file(cache: Path, stem: str, floor: np.ndarray) -> dict[str, list[dict]]:
    meta = json.loads((cache / stem / "meta.json").read_text(encoding="utf-8"))
    per_ms = ms_components(cache, stem, floor)
    cum = np.vstack([np.zeros((1, 4), np.float32), np.cumsum(per_ms, axis=0)])
    out: dict[str, list[dict]] = {}
    for L in WINDOWS:
        starts = meta["window_starts"][L]
        s_ms = np.array([s // 100_000 for s in starts])
        feats = cum[s_ms + int(L)] - cum[s_ms]  # [n, 4]
        med = np.median(feats, axis=0)
        char = med >= MIN_CELLS
        logf = np.log1p(feats)
        lmed = np.median(logf, axis=0)
        lmad = np.median(np.abs(logf - lmed), axis=0) * 1.4826
        z = (logf - lmed) / np.maximum(lmad, 0.1)
        rows = []
        for i, s in enumerate(starts):
            ratios = feats[i] / np.maximum(med, 1e-9)
            missing = bool(char.any() and (ratios[char] < MISSING_RATIO).all())
            differ = bool((z[i] > Z).any())
            row = {"stem": stem, "window_ms": L, "window_index": i, "start": s,
                   "block_index": s // 10_000_000}
            for k, name in enumerate(COMPONENTS):
                row[name] = int(feats[i, k])
            row["characteristic"] = "+".join(n for n, c in zip(COMPONENTS, char) if c)
            row["max_ratio_char"] = round(float(ratios[char].max()) if char.any() else 1.0, 3)
            row["missing"] = int(missing)
            row["differ"] = int(differ)
            rows.append(row)
        out[L] = rows
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    floor = noise_floor(args.cache)
    np.save(args.out / "noise_floor.npy", floor)
    stems = sorted(p.name for p in args.cache.iterdir() if (p / "meta.json").exists())
    all_rows: list[dict] = []
    summary: list[dict] = []
    for stem in stems:
        res = screen_file(args.cache, stem, floor)
        rec = {"stem": stem, "target": stem.split("_")[0]}
        for L, rows in res.items():
            all_rows.extend(rows)
            n = len(rows)
            rec[f"n{L}"] = n
            rec[f"missing{L}"] = sum(r["missing"] for r in rows)
            rec[f"differ{L}"] = sum(r["differ"] for r in rows)
            rec[f"characteristic{L}"] = rows[0]["characteristic"]
        summary.append(rec)
        print(stem, {L: f"missing {rec[f'missing{L}']}/{rec[f'n{L}']}" for L in WINDOWS}, flush=True)
    with (args.out / "window_screening.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    with (args.out / "file_summary.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    (args.out / "screening_config.json").write_text(json.dumps({
        "thresh_db": THRESH_DB, "edge_bins": EDGE_BINS, "min_cells": MIN_CELLS,
        "missing_ratio": MISSING_RATIO, "z": Z, "components": COMPONENTS,
        "floor": "median over background files of per-file median ms_max",
    }, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
