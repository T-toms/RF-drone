"""Stage 1, step 1: background vs drone on clean data, from the dual-band cache.

Inputs are the precomputed clean spectrograms (precompute_dualband_cache.py),
so no noise is added and nothing is read from the external drive.

  --split CSV      columns stem,target,role (role: train / val / test_*).
  --window-ms      10 / 15 / 20.
  --bands          both (2 channels: RF0 2.4 GHz + RF1 5.8 GHz) / rf0 / rf1.
  --screening CSV  window_screening.csv; with --drop-missing, drone windows
                   flagged "missing" (no characteristic signal of their file)
                   are removed from train. Validation is always scored both on
                   all windows and on the screened subset.
Sampling: each epoch draws len(train) windows, half background and half drone,
drone windows spread equally over drone types (inverse-frequency weights).
Selection (validation only, clean): epoch with the highest balanced score
  score = 0.5 * (1 - background false-alarm rate) + 0.5 * mean over drone
          types of (1 - miss rate),
then the decision threshold on the logit margin is the smallest value that
keeps background false alarms <= --max-fa on validation.

Outputs in --out: run_config.json, history.csv, val_scores_epochNN.csv,
best_model.pt, selection.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ResNet18  # noqa: E402

BAND_INDEX = {"both": [0, 1], "rf0": [0], "rf1": [1]}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_split(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_missing(path: Path | None, window_ms: int) -> set[tuple[str, int]]:
    if path is None:
        return set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {
            (r["stem"], int(r["window_index"]))
            for r in csv.DictReader(fh)
            if int(r["window_ms"]) == window_ms and r["missing"] == "1"
        }


class CachedWindows(Dataset):
    def __init__(self, cache: Path, files: list[dict], window_ms: int, bands: str,
                 missing: set[tuple[str, int]], drop_missing: bool) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.items: list[tuple[str, int]] = []
        self.meta: list[dict] = []
        self.channels = BAND_INDEX[bands]
        for f in files:
            stem = f["stem"]
            arr = np.load(cache / stem / f"win{window_ms}.npy", mmap_mode="r")
            self.arrays[stem] = arr
            label = 0 if f["target"] == "T0000" else 1
            for i in range(arr.shape[0]):
                flagged = label == 1 and (stem, i) in missing
                if drop_missing and flagged:
                    continue
                self.items.append((stem, i))
                self.meta.append({"stem": stem, "target": f["target"], "window_index": i,
                                  "label": label, "missing": int(flagged)})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        stem, i = self.items[idx]
        x = np.asarray(self.arrays[stem][i, self.channels], dtype=np.float32)
        return torch.from_numpy(x), self.meta[idx]["label"]


def sampler_weights(meta: list[dict]) -> list[float]:
    bg = [m for m in meta if m["label"] == 0]
    types: dict[str, int] = {}
    for m in meta:
        if m["label"] == 1:
            types[m["target"]] = types.get(m["target"], 0) + 1
    w = []
    for m in meta:
        if m["label"] == 0:
            w.append(0.5 / len(bg))
        else:
            w.append(0.5 / len(types) / types[m["target"]])
    return w


@torch.no_grad()
def score(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader:
        logits = model(x.to(device, non_blocking=True)).float()
        out.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(out)


def metrics(meta: list[dict], margin: np.ndarray, threshold: float, subset=None) -> dict:
    idx = [i for i, m in enumerate(meta) if subset is None or subset(m)]
    labels = np.array([meta[i]["label"] for i in idx])
    pred = (margin[idx] > threshold).astype(int)
    bg = labels == 0
    fa = int(pred[bg].sum())
    res = {"bg_windows": int(bg.sum()), "bg_false_alarms": fa,
           "fa_rate": fa / max(int(bg.sum()), 1), "types": {}}
    targets = sorted({meta[i]["target"] for i in idx if meta[i]["label"] == 1})
    for t in targets:
        sel = np.array([meta[i]["target"] == t for i in idx])
        n = int(sel.sum())
        miss = int((pred[sel] == 0).sum())
        res["types"][t] = {"windows": n, "misses": miss, "miss_rate": miss / n}
    miss_rates = [v["miss_rate"] for v in res["types"].values()]
    res["mean_miss_rate"] = float(np.mean(miss_rates)) if miss_rates else 0.0
    res["worst_miss_rate"] = float(np.max(miss_rates)) if miss_rates else 0.0
    res["balanced_score"] = 0.5 * (1 - res["fa_rate"]) + 0.5 * (1 - res["mean_miss_rate"])
    return res


def pick_threshold(meta: list[dict], margin: np.ndarray, max_fa: float) -> float:
    bg = np.sort(np.array([margin[i] for i, m in enumerate(meta) if m["label"] == 0]))
    allowed = int(np.floor(max_fa * len(bg)))
    if allowed >= len(bg):
        return float("-inf")
    # predict drone when margin > t: at most `allowed` background windows lie above
    # the (allowed+1)-th largest background margin
    return float(bg[len(bg) - 1 - allowed])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--split", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--window-ms", type=int, default=10, choices=(10, 15, 20))
    p.add_argument("--bands", default="both", choices=tuple(BAND_INDEX))
    p.add_argument("--screening", type=Path)
    p.add_argument("--drop-missing", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-fa", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    split = load_split(args.split)
    missing = load_missing(args.screening, args.window_ms)
    train_files = [r for r in split if r["role"] == "train"]
    val_files = [r for r in split if r["role"] == "val"]
    train = CachedWindows(args.cache, train_files, args.window_ms, args.bands, missing, args.drop_missing)
    val = CachedWindows(args.cache, val_files, args.window_ms, args.bands, missing, False)
    gen = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(sampler_weights(train.meta), num_samples=len(train),
                                    replacement=True, generator=gen)
    train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=True)

    model = ResNet18(input_channels=len(BAND_INDEX[args.bands]), num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    config = {
        "schema": "dronerfa-stage1-cached/v1", "args": {k: str(v) for k, v in vars(args).items()},
        "split_sha256": sha256(args.split),
        "screening_sha256": sha256(args.screening) if args.screening else None,
        "train_windows": len(train), "val_windows": len(val),
        "train_bg": sum(m["label"] == 0 for m in train.meta),
        "train_dropped_missing": sum(1 for f in train_files if f["target"] != "T0000"
                                     for (s, _) in missing if s == f["stem"]) if args.drop_missing else 0,
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "selection": "max clean val balanced_score at threshold 0; then threshold for FA<=max_fa",
    }
    (args.out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: config[k] for k in ("train_windows", "val_windows", "train_bg", "train_dropped_missing")}), flush=True)

    history = []
    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(1) == y).sum())
            total += len(y)
        margin = score(model, val_loader, device)
        m_all = metrics(val.meta, margin, 0.0)
        m_scr = metrics(val.meta, margin, 0.0, subset=lambda m: not m["missing"])
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total,
               "val_score": m_all["balanced_score"], "val_fa": m_all["fa_rate"],
               "val_mean_miss": m_all["mean_miss_rate"], "val_worst_miss": m_all["worst_miss_rate"],
               "val_score_screened": m_scr["balanced_score"],
               "val_worst_miss_screened": m_scr["worst_miss_rate"], "seconds": time.time() - t0}
        history.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        with (args.out / f"val_scores_epoch{epoch:02d}.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["stem", "target", "window_index", "label", "missing", "margin"])
            for m, s in zip(val.meta, margin):
                w.writerow([m["stem"], m["target"], m["window_index"], m["label"], m["missing"], f"{s:.5f}"])
        if best is None or row["val_score"] > best[0]:
            best = (row["val_score"], epoch, margin)
            torch.save(model.state_dict(), args.out / "best_model.pt")
    with (args.out / "history.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    _, best_epoch, margin = best
    thr = pick_threshold(val.meta, margin, args.max_fa)
    sel = {"selected_epoch": best_epoch, "threshold": thr, "max_fa": args.max_fa,
           "val_at_0": metrics(val.meta, margin, 0.0),
           "val_at_threshold": metrics(val.meta, margin, thr),
           "val_at_threshold_screened": metrics(val.meta, margin, thr, subset=lambda m: not m["missing"])}
    (args.out / "selection.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    print("selected", best_epoch, "threshold", round(thr, 4),
          "fa", sel["val_at_threshold"]["bg_false_alarms"], "/", sel["val_at_threshold"]["bg_windows"],
          "worst miss", round(sel["val_at_threshold"]["worst_miss_rate"], 4), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
