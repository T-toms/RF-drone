"""Screen every cached file and write per-window decisions and a summary.

Outputs in --out:
  screening_windows.csv  one row per (file, window length, window)
  screening_files.csv    per file and window length: windows, discarded,
                         share of windows containing a long burst
  overview_flagged/<stem>.png  whole-file plot, discarded 10 ms windows in red
                               (files with any discarded 10 ms window)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from plot_file_overview import plot_overview
from window_screening import MAX_SCORE, Screener

BACKGROUND_DEV = [f"T0000_D00_S{s}" for s in ("0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.name for p in args.cache.iterdir() if (p / "meta.json").exists())
    screener = Screener(args.cache, [s for s in BACKGROUND_DEV if s in stems])
    np.save(args.out / "floor_mean.npy", screener.floor_mean)
    np.save(args.out / "floor_max.npy", screener.floor_max)

    with (args.out / "screening_windows.csv").open("w", encoding="utf-8-sig", newline="") as fw, \
         (args.out / "screening_files.csv").open("w", encoding="utf-8-sig", newline="") as ff:
        ww = csv.writer(fw)
        ww.writerow(["stem", "target", "window_ms", "window_index", "start", "score", "long_burst", "discard"])
        wf = csv.writer(ff)
        wf.writerow(["stem", "target", "window_ms", "windows", "discarded", "discard_pct", "long_burst_pct",
                     "score_max"])
        for stem in stems:
            r = screener.screen_file(stem)
            target = stem.split("_")[0]
            for L, w in r["windows"].items():
                n = len(w["start"])
                for i in range(n):
                    ww.writerow([stem, target, L, i, w["start"][i], w["score"][i], int(w["long_burst"][i]),
                                 int(w["discard"][i])])
                d = int(np.sum(w["discard"]))
                wf.writerow([stem, target, L, n, d, round(100 * d / n, 1),
                             round(100 * float(np.mean(w["long_burst"])), 1), max(w["score"])])
            flagged = {L: [i for i, x in enumerate(w["discard"]) if x] for L, w in r["windows"].items()}
            if not args.no_plots and flagged["10"]:
                plot_overview(args.cache, stem, args.out / "overview_flagged", {"10": flagged["10"]})
            print(stem, {L: len(v) for L, v in flagged.items()}, flush=True)
    print(f"rule: knn score > {MAX_SCORE}")


if __name__ == "__main__":
    main()
