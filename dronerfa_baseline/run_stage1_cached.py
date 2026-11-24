"""Stage 1 (background vs drone), clean data only, trained from the dual-band cache.

Protocol (report 2026-10-12, step 1-2):
  * input: cached clean spectrograms (precompute_dualband_cache.py), window
    length --window-ms 10/15/20; --bands both (RF0+RF1, 2 channels) or rf0
    (2.4 GHz only, 1 channel);
  * split: split_20261013.build_split (per class 6:2:2 by file, fixed seed);
  * screening (data cleaning, before training): windows flagged by
    run_window_screening.py (differ strongly from the rest of their file) are
    dropped from train, val and test alike; --no-screening keeps them;
  * model: model.ResNet18, Adam lr 1e-3, batch 32, --epochs epochs;
    class-balanced sampling (each batch half background, half drone; drone
    windows drawn evenly over the 12 drone classes);
  * selection: epoch with the highest validation balanced accuracy at
    threshold 0 (earliest wins ties); the decision threshold on
    p(drone) - p(background) logit margin is then set on validation as the
    midpoint that keeps background false alarms <= 1% with the fewest misses;
  * test is evaluated once, for the selected epoch at that threshold.
Outputs: run_config.json, history.csv, val_scores.csv, test_scores.csv,
summary.json, best_model.pt.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from model import ResNet18
from split_20261013 import NAMES, write_split

BG = "T0000"


def load_windows(cache: Path, screening: Path, window_ms: int, split: dict[str, str]):
    discard: dict[tuple[str, int], bool] = {}
    with screening.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if int(r["window_ms"]) == window_ms:
                discard[(r["stem"], int(r["window_index"]))] = r["discard"] == "1"
    rows = []
    arrays = {}
    for stem, role in sorted(split.items()):
        arr = np.load(cache / stem / f"win{window_ms}.npy", mmap_mode="r")
        arrays[stem] = arr
        for i in range(len(arr)):
            rows.append({"stem": stem, "target": stem.split("_")[0], "index": i, "role": role,
                         "label": 0 if stem.startswith(BG) else 1, "discard": discard[(stem, i)]})
    return rows, arrays


def stack(rows, arrays, channels):
    x = np.empty((len(rows), len(channels), 256, 256), dtype=np.float16)
    for j, r in enumerate(rows):
        x[j] = arrays[r["stem"]][r["index"], channels]
    return torch.from_numpy(x)


def balanced_batches(rows, batch, rng):
    bg = [i for i, r in enumerate(rows) if r["label"] == 0]
    by_cls: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r["label"] == 1:
            by_cls.setdefault(r["target"], []).append(i)
    classes = sorted(by_cls)
    n_batches = len(rows) // batch
    for _ in range(n_batches):
        half = batch // 2
        b = list(rng.choice(bg, half))
        cls = rng.choice(classes, batch - half)
        b += [int(rng.choice(by_cls[c])) for c in cls]
        yield b


@torch.no_grad()
def margins(model, x, device, bs=128):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        xb = x[i : i + bs].to(device).float()
        with torch.autocast("cuda", dtype=torch.float16):
            logit = model(xb).float()
        out.append((logit[:, 1] - logit[:, 0]).cpu().numpy())
    return np.concatenate(out)


def metrics(rows, m, thr):
    lab = np.array([r["label"] for r in rows])
    pred = (m > thr).astype(int)
    fa = int(((pred == 1) & (lab == 0)).sum())
    n_bg = int((lab == 0).sum())
    per_cls = {}
    for t in sorted({r["target"] for r in rows if r["label"] == 1}):
        idx = np.array([r["target"] == t for r in rows])
        per_cls[t] = {"windows": int(idx.sum()), "missed": int((pred[idx] == 0).sum())}
    per_file = {}
    for s in sorted({r["stem"] for r in rows}):
        idx = np.array([r["stem"] == s for r in rows])
        wrong = int((pred[idx] != lab[idx]).sum())
        per_file[s] = {"windows": int(idx.sum()), "wrong": wrong}
    recalls = [1 - v["missed"] / v["windows"] for v in per_cls.values()]
    spec = 1 - fa / max(n_bg, 1)
    return {
        "threshold": float(thr),
        "background_windows": n_bg, "false_alarms": fa, "false_alarm_rate": fa / max(n_bg, 1),
        "drone_windows": int((lab == 1).sum()), "missed": int(((pred == 0) & (lab == 1)).sum()),
        "worst_class_miss_rate": max(v["missed"] / v["windows"] for v in per_cls.values()),
        "balanced_accuracy": 0.5 * (spec + float(np.mean(recalls))),
        "per_class": per_cls, "per_file": per_file,
    }


def pick_threshold(rows, m, max_fa=0.01):
    lab = np.array([r["label"] for r in rows])
    bgm = np.sort(m[lab == 0])
    allowed = int(np.floor(max_fa * len(bgm)))
    # threshold above the (allowed+1)-th highest background margin
    hi = bgm[len(bgm) - allowed - 1] if allowed < len(bgm) else -np.inf
    above = m[(lab == 1) & (m > hi)]
    nxt = above.min() if len(above) else hi + 1.0
    return float((hi + nxt) / 2) if np.isfinite(hi) else 0.0


def write_scores(path, rows, m):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "target", "window_index", "label", "discard", "margin"])
        for r, v in zip(rows, m):
            w.writerow([r["stem"], r["target"], r["index"], r["label"], int(r["discard"]), f"{v:.4f}"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--screening", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window-ms", type=int, default=10, choices=(10, 15, 20))
    p.add_argument("--bands", default="both", choices=("both", "rf0"))
    p.add_argument("--no-screening", action="store_true")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda"
    channels = [0, 1] if args.bands == "both" else [0]

    stems = sorted(p.name for p in args.cache.iterdir() if (p / "meta.json").exists())
    split = write_split(stems)
    rows, arrays = load_windows(args.cache, args.screening, args.window_ms, split)
    use = lambda r: args.no_screening or not r["discard"]
    train = [r for r in rows if r["role"] == "train" and use(r)]
    val = [r for r in rows if r["role"] == "val" and use(r)]
    test = [r for r in rows if r["role"] == "test" and use(r)]
    t0 = time.time()
    xtr, xva, xte = stack(train, arrays, channels), stack(val, arrays, channels), stack(test, arrays, channels)
    ytr = torch.tensor([r["label"] for r in train])
    print(f"loaded train {len(train)} val {len(val)} test {len(test)} in {time.time()-t0:.0f}s", flush=True)

    config = {
        "window_ms": args.window_ms, "bands": args.bands, "screening": not args.no_screening,
        "seed": args.seed, "epochs": args.epochs, "batch": args.batch, "lr": 1e-3,
        "split": split, "counts": {
            role: {"files": sum(1 for v in split.values() if v == role),
                   "windows": len(xs), "background": sum(1 for r in xs if r["label"] == 0)}
            for role, xs in (("train", train), ("val", val), ("test", test))},
        "dropped_by_screening": {role: sum(1 for r in rows if r["role"] == role and r["discard"])
                                 for role in ("train", "val", "test")},
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
    }
    (args.out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")

    model = ResNet18(input_channels=len(channels), num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()
    best = (-1.0, 0)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        for b in balanced_batches(train, args.batch, rng):
            xb = xtr[b].to(device).float()
            yb = ytr[b].to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        mv = margins(model, xva, device)
        vm = metrics(val, mv, 0.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_ba": vm["balanced_accuracy"],
                        "val_false_alarms": vm["false_alarms"], "val_missed": vm["missed"],
                        "val_worst_class_miss": vm["worst_class_miss_rate"], "seconds": round(time.time() - t0, 1)})
        print(json.dumps(history[-1]), flush=True)
        if vm["balanced_accuracy"] > best[0]:
            best = (vm["balanced_accuracy"], epoch)
            torch.save(model.state_dict(), args.out / "best_model.pt")
    with (args.out / "history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)

    model.load_state_dict(torch.load(args.out / "best_model.pt", map_location=device))
    mv = margins(model, xva, device)
    thr = pick_threshold(val, mv)
    mt = margins(model, xte, device)
    write_scores(args.out / "val_scores.csv", val, mv)
    write_scores(args.out / "test_scores.csv", test, mt)
    summary = {
        "selected_epoch": best[1],
        "val": metrics(val, mv, thr),
        "test": metrics(test, mt, thr),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("val", "test"):
        s = summary[k]
        worst = max(s["per_class"].items(), key=lambda kv: kv[1]["missed"] / kv[1]["windows"])
        print(f"{k}: FA {s['false_alarms']}/{s['background_windows']}  miss {s['missed']}/{s['drone_windows']}  "
              f"worst {NAMES[worst[0]]} {worst[1]['missed']}/{worst[1]['windows']}", flush=True)


if __name__ == "__main__":
    main()
