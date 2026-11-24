"""Screen windows that differ strongly from the other windows of their file.

Teacher's rule (2026-10-13): a window cut from a file that differs a lot from
the file's other windows is not representative of that file and should be
dropped, or a longer window should be used.

Built on the per-ms spectra of precompute_dualband_cache.py (ms_mean/ms_max,
raw dB, [n_ms, 2 channels, 512 bins]).

  noise floor   per channel and bin: median over every ms of the background
                (T0000) files, separately for ms_mean and ms_max. The fixed
                2400 MHz receiver spur and the band-pass shape are part of it.
  weak cell     (ms, channel, bin) whose 1 ms max-hold exceeds the floor by
                WEAK_DB: any burst, however short.
  strong cell   (ms, channel, bin) whose 1 ms mean exceeds the floor by
                STRONG_DB: the bin is busy for most of that ms.

Window descriptor (per channel, BANDS sub-bands of 100/BANDS MHz): fraction
of the window's ms with any weak cell in the sub-band, and the mean fraction
of strong cells in the sub-band. Sub-bands rather than single bins so that
hopping signals, whose exact frequency changes from window to window, give
similar descriptors.

Outlier score: mean Euclidean distance from a window to its K nearest other
windows of the same file, divided by the file median of that distance. A file
with two recurring states (e.g. M100 dense hopping / long bursts only) forms
two groups of mutually close windows, neither of which is flagged; only
windows unlike any others are. Discard when score > MAX_SCORE.
Background windows are never discarded (their content is random by nature).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WEAK_DB = 6.0
STRONG_DB = 10.0
LONG_MS = 2
BANDS = 16
K = 5
MAX_SCORE = 2.0
MS = 100_000


def load_file(cache: Path, stem: str):
    d = cache / stem
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    return (
        meta,
        np.load(d / "ms_mean.npy").astype(np.float32),
        np.load(d / "ms_max.npy").astype(np.float32),
    )


def cells(mn: np.ndarray, mx: np.ndarray, floor_mean: np.ndarray, floor_max: np.ndarray):
    return mx > floor_max + WEAK_DB, mn > floor_mean + STRONG_DB


def long_burst_ms(strong: np.ndarray) -> np.ndarray:
    """[n_ms] True for ms inside a strong run lasting >= LONG_MS ms in some bin."""
    s = strong | np.roll(strong, 1, axis=2) | np.roll(strong, -1, axis=2)
    n = s.shape[0]
    flat = s.reshape(n, -1)
    run = np.zeros(flat.shape[1], dtype=np.int32)
    hit = np.zeros(n, dtype=bool)
    for t in range(n):
        run = (run + 1) * flat[t]
        if (run >= LONG_MS).any():
            hit[max(0, t - LONG_MS + 1) : t + 1] = True
    return hit


def descriptors(weak: np.ndarray, strong: np.ndarray, meta: dict, window_ms: int) -> np.ndarray:
    n_ms, ch, bins = weak.shape
    wb = weak.reshape(n_ms, ch, BANDS, bins // BANDS).any(axis=3)
    sb = strong.reshape(n_ms, ch, BANDS, bins // BANDS).mean(axis=3)
    out = []
    for s in meta["window_starts"][str(window_ms)]:
        m0 = s // MS
        out.append(np.concatenate([wb[m0 : m0 + window_ms].mean(0).ravel(),
                                   sb[m0 : m0 + window_ms].mean(0).ravel()]))
    return np.asarray(out, dtype=np.float32)


def knn_score(desc: np.ndarray, k: int = K) -> np.ndarray:
    d = np.sqrt(((desc[:, None, :] - desc[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    k = min(k, len(desc) - 1)
    knn = np.sort(d, axis=1)[:, :k].mean(axis=1)
    return knn / max(float(np.median(knn)), 1e-9)


class Screener:
    def __init__(self, cache: Path, background_stems: list[str]) -> None:
        self.cache = cache
        means, maxes = [], []
        for stem in background_stems:
            _, mn, mx = load_file(cache, stem)
            means.append(mn)
            maxes.append(mx)
        self.floor_mean = np.median(np.concatenate(means), axis=0)
        self.floor_max = np.median(np.concatenate(maxes), axis=0)

    def screen_file(self, stem: str, window_lengths=(10, 15, 20)) -> dict:
        meta, mn, mx = load_file(self.cache, stem)
        weak, strong = cells(mn, mx, self.floor_mean, self.floor_max)
        longms = long_burst_ms(strong)
        background = stem.startswith("T0000")
        result = {"stem": stem, "windows": {}}
        for L in window_lengths:
            starts = meta["window_starts"][str(L)]
            score = knn_score(descriptors(weak, strong, meta, L))
            has_long = np.array([longms[s // MS : s // MS + L].any() for s in starts])
            discard = (score > MAX_SCORE) & (not background)
            result["windows"][str(L)] = {
                "start": starts,
                "score": np.round(score, 3).tolist(),
                "long_burst": has_long.tolist(),
                "discard": discard.tolist(),
            }
        return result
