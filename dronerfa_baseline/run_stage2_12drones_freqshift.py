"""Single-factor Stage2 experiment: random circular frequency shift of training spectrograms.

2026-10-10 probe (probe_stage2_frequency_shortcut.py) showed the frozen Stage2
model keys on the absolute frequency of the video block: moving a T1011
validation block down 20-40 MHz drops it from 100% to 0-4%. DJI links pick
channels adaptively, so this run rolls every training spectrogram along the
frequency axis by a uniform integer number of rows in [-SHIFT_ROWS, SHIFT_ROWS]
(256 rows = 100 MHz; +-102 rows = +-40 MHz). Validation is not shifted.
Everything else (noise mix, seed, files, model, optimiser, epochs, selection)
matches the frozen baseline run_stage2_12drones.py.

The original baseline code and outputs remain frozen. This run uses the same
seed, model, optimizer, files and validation conditions as the baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, TensorDataset

from model import ResNet18, SpectrogramFrontEnd
from run_stage1_12drones_training import rng_state_payload, restore_rng_state, state_hash
from stage2_12class_data import (CODE_TO_LABEL, FULL_COUNTS, INDEX_SHA256, NOISE_SEEDS,
                                 OUT as CONTRACT_DIR, ROLE_COUNTS, SPLIT_SHA256,
                                 Stage2WindowDataset, load_contract_rows, sha256,
                                 write_contract)
from train_resnet18 import extract_features, seed_everything

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/stage2_freqshift_20261010/seed_2027"
CONDITIONS = {"clean": None, "-10dB": -10.0, "-12dB": -12.0, "-15dB": -15.0}
TRAIN_SNRS = (None, 10.0, 5.0, 0.0, -5.0, -10.0, -12.0)
SHIFT_ROWS = 102          # 40 MHz of the 100 MHz / 256-row spectrogram
SEED, EPOCHS, BATCH = 2027, 5, 4


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def csv_write(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class Positions(Sampler[int]):
    def __init__(self, positions): self.positions = positions
    def __iter__(self): return iter(self.positions)
    def __len__(self): return len(self.positions)


def order_for_epoch(epoch: int, count: int) -> list[int]:
    return torch.randperm(count, generator=torch.Generator().manual_seed(SEED + epoch)).tolist()


def metrics(rows: list[dict]) -> dict:
    file_groups = defaultdict(list)
    by_class = defaultdict(list)
    for r in rows:
        file_groups[(r["target_code"], r["filename"])].append(r)
        by_class[r["label"]].append(r)
    file_accuracy = {(code, name): sum(r["prediction"] == r["label"] for r in group) / len(group)
                     for (code, name), group in file_groups.items()}
    per_class_file = {code: float(np.mean([score for (c, _), score in file_accuracy.items() if c == code]))
                      for code in CODE_TO_LABEL}
    per_class_recall = {int(label): sum(r["prediction"] == r["label"] for r in group) / len(group)
                        for label, group in by_class.items()}
    precision = {}
    f1 = {}
    for label in range(12):
        tp = sum(r["label"] == label and r["prediction"] == label for r in rows)
        predicted = sum(r["prediction"] == label for r in rows)
        p = tp / predicted if predicted else 0.0
        recall = per_class_recall.get(label, 0.0)
        precision[label] = p
        f1[label] = 2*p*recall/(p+recall) if p+recall else 0.0
    return {"samples": len(rows), "pooled_accuracy": sum(r["prediction"] == r["label"] for r in rows)/len(rows),
            "macro_recall": float(np.mean(list(per_class_recall.values()))),
            "macro_f1": float(np.mean(list(f1.values()))),
            "class_file_equal_recall": float(np.mean(list(per_class_file.values()))),
            "per_class_file_equal_recall": per_class_file, "per_class_recall": per_class_recall,
            "per_file_accuracy": [{"target_code": c, "filename": n, "accuracy": v} for (c,n),v in file_accuracy.items()]}


def validation_cache(drones, frontend, device, *, limit: int | None = None):
    cache = {}
    audit = []
    frontend.eval()
    for condition, snr in CONDITIONS.items():
        ds = Stage2WindowDataset(drones, "val", seed=NOISE_SEEDS["val"], fixed_snr=snr)
        if limit:
            per_class = max(1, limit // 12)
            counts = Counter()
            selected = []
            for row in ds.rows:
                label = int(row["stage2_label"])
                if counts[label] < per_class:
                    selected.append(row)
                    counts[label] += 1
            ds.rows = selected
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=0,
                            generator=torch.Generator().manual_seed(62027), pin_memory=device.type == "cuda")
        features, labels = [], []
        started = time.perf_counter()
        with torch.inference_mode():
            for batch, (iq, y) in enumerate(loader, 1):
                feat = extract_features(frontend, iq.to(device)).cpu()
                if not bool(torch.isfinite(feat).all()): raise RuntimeError("Nonfinite validation feature")
                features.append(feat)
                labels.append(y)
                if batch % 100 == 0 or batch == len(loader):
                    print(f"cache {condition} {batch}/{len(loader)}", flush=True)
        xx, yy = torch.cat(features), torch.cat(labels)
        cache[condition] = (xx, yy, ds.rows)
        audit.append({"condition": condition, "windows_read": len(ds), "files_opened": len({r["file_path"] for r in ds.rows}),
                      "feature_sha256": hashlib.sha256(xx.numpy().tobytes()).hexdigest(),
                      "label_sha256": hashlib.sha256(yy.numpy().tobytes()).hexdigest(), "seconds": time.perf_counter()-started})
    return cache, audit


@torch.inference_mode()
def evaluate(model, cache, device, amp):
    model.eval()
    condition_metrics = []
    all_records = []
    for condition, (xx, yy, source) in cache.items():
        records = []
        loader = DataLoader(TensorDataset(xx, yy), batch_size=BATCH, shuffle=False, num_workers=0)
        cursor = 0
        for features, labels in loader:
            with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
                logits = model(features.to(device))
            values = logits.float().cpu().numpy()
            preds = values.argmax(axis=1)
            for i in range(len(labels)):
                row = source[cursor+i]
                records.append({"condition": condition, "row_index": int(row["row_index"]),
                                "full_split_position": int(row["full_split_position"]), "filename": row["filename"],
                                "target_code": row["target_code"], "label": int(labels[i]),
                                "prediction": int(preds[i]), **{f"logit_{j:02d}": float(values[i,j]) for j in range(12)}})
            cursor += len(labels)
        result = metrics(records)
        condition_metrics.append({"condition": condition, **{k:v for k,v in result.items() if k not in ("per_file_accuracy", "per_class_file_equal_recall", "per_class_recall")}})
        all_records.extend(records)
    return condition_metrics, all_records


def shifts_for_epoch(epoch: int, count: int) -> torch.Tensor:
    """Per-window shift indexed by position in the epoch order; deterministic for resume."""
    rng = np.random.default_rng(SEED * 1000 + epoch)
    return torch.from_numpy(rng.integers(-SHIFT_ROWS, SHIFT_ROWS + 1, size=count))


def roll_frequency(features: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """Circularly roll each [1, F, T] spectrogram along F by its own shift."""
    rows = features.shape[2]
    index = (torch.arange(rows, device=features.device)[None, :] - shifts.to(features.device)[:, None]) % rows
    return torch.gather(features, 2, index[:, None, :, None].expand_as(features))


def train_one_epoch(model, frontend, dataset, order, criterion, optimizer, scaler, device, amp, shifts):
    dataset.epoch += 1
    model.train(); frontend.eval()
    loader = DataLoader(dataset, batch_size=BATCH, sampler=Positions(order), num_workers=0,
                        generator=torch.Generator().manual_seed(SEED+50000), pin_memory=device.type == "cuda")
    loss_sum, correct, seen = 0.0, 0, 0
    started = time.perf_counter()
    for batch, (iq, target) in enumerate(loader, 1):
        iq, target = iq.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        features = roll_frequency(extract_features(frontend, iq), shifts[(batch-1)*BATCH:(batch-1)*BATCH+len(target)])
        with torch.amp.autocast(device_type=device.type, enabled=amp, dtype=torch.float16):
            logits = model(features)
            loss = criterion(logits, target)
        if not bool(torch.isfinite(loss)): raise RuntimeError(f"Nonfinite training loss at batch {batch}")
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        n = len(target); seen += n
        loss_sum += float(loss.detach())*n
        correct += int((logits.detach().argmax(1) == target).sum())
        if batch % 100 == 0 or batch == len(loader):
            print(f"train {batch}/{len(loader)} loss={loss_sum/seen:.5f} samples/s={seen/max(time.perf_counter()-started,1e-6):.2f}", flush=True)
    if seen != len(dataset): raise RuntimeError("Training loader did not cover all windows")
    return {"train_loss": loss_sum/seen, "train_accuracy": correct/seen, "train_windows_read": seen,
            "seconds": time.perf_counter()-started}


def run(out: Path, *, smoke: bool = False, resume: bool = False):
    os.environ.setdefault("WINDIR", r"C:\Windows")
    if not torch.cuda.is_available() and not smoke: raise RuntimeError("Full run requires CUDA")
    if (out / "best_model.pt").exists() and not resume: raise FileExistsError("Existing model; refusing overwrite")
    if out.exists() and not resume and any(out.iterdir()): raise FileExistsError("Nonempty run directory; refusing overwrite")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(10)
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    # Existing frozen contract is read-only for this experiment.
    if not (CONTRACT_DIR / "stage2_index.csv").is_file() or not (CONTRACT_DIR / "mapping.json").is_file():
        raise FileNotFoundError("Frozen Stage2 contract is missing")
    _, drones = load_contract_rows()
    train_dataset = Stage2WindowDataset(drones, "train", seed=SEED, train_snrs=TRAIN_SNRS)
    if smoke: train_dataset.rows = train_dataset.rows[:24]
    model = ResNet18(input_channels=1, num_classes=12).to(device)
    frontend = SpectrogramFrontEnd(n_fft=512, hop_length=256, image_size=256).to(device)
    initial_hash = state_hash(model.state_dict())
    class_counts = Counter(int(r["stage2_label"]) for r in [x for x in drones if x["split"] == "train"])
    weights = torch.tensor([ROLE_COUNTS["train"]/(12*class_counts[i]) for i in range(12)], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    source_names = ["run_stage2_12drones_freqshift.py", "stage2_12class_data.py", "model.py", "dronerfa_data.py"]
    identity = {"index_sha256": INDEX_SHA256, "split_sha256": SPLIT_SHA256,
                "view_sha256": sha256(CONTRACT_DIR / "stage2_index.csv"),
                "mapping_sha256": sha256(CONTRACT_DIR / "mapping.json"),
                "source_sha256": {name: sha256(HERE/name) for name in source_names},
                "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH, "model_initial_sha256": initial_hash,
                "train_count": len(train_dataset), "validation_count": (24 if smoke else ROLE_COUNTS["val"])}
    state_file = out / "last_training_state.pt"
    if resume:
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        if state["identity"] != identity: raise RuntimeError("Resume identity mismatch")
        model.load_state_dict(state["model_state"]); frontend.load_state_dict(state["frontend_state"])
        optimizer.load_state_dict(state["optimizer_state"]); scaler.load_state_dict(state["scaler_state"])
        restore_rng_state(state["rng"])
        history = state["history"]
        start_epoch = state["epoch"]+1
    else:
        history, start_epoch = [], 1
    config = {**identity, "schema": "dronerfa-stage2-12class-freqshift/v1", "created_at": now(), "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if amp else None, "python": sys.version,
              "torch": torch.__version__, "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
              "frontend": {"n_fft":512, "hop":256, "hamming":True, "image_size":256},
              "optimizer": {"name":"Adam", "lr":0.001, "weight_decay":0},
              "loss": "CrossEntropyLoss(weight=8990/(12*n_train_class), reduction=mean: batch_weight_sum)",
              "class_counts": {str(k):v for k,v in sorted(class_counts.items())}, "class_weights":weights.cpu().tolist(),
              "train_noise": list(TRAIN_SNRS), "single_factor_change": "random circular roll of training spectrogram along frequency, uniform integer in [-SHIFT_ROWS, SHIFT_ROWS] rows", "shift_rows": SHIFT_ROWS, "shift_rng": "default_rng(2027*1000 + epoch), indexed by position in epoch order",
              "validation_conditions": list(CONDITIONS), "validation_seed":12026,
              "selection": "mean over four conditions of mean over 12 classes of mean source-file accuracy within each class; tie earliest epoch",
              "num_workers":0, "smoke":smoke, "test_iq_opened":False}
    if not resume: dump(out/"run_config.json", config)
    cache, cache_audit = validation_cache(drones, frontend, device, limit=24 if smoke else None)
    dump(out/"validation_cache_manifest.json", cache_audit)
    audit = {"train_windows_read": sum(int(x.get("train_windows_read",0)) for x in history),
             "validation_windows_read": sum(x["windows_read"] for x in cache_audit), "test_files_touched":[],
             "train_files":len({r["file_path"] for r in train_dataset.rows}),
             "validation_files":len({r["file_path"] for r in drones if r["split"] == "val"})}
    for epoch in range(start_epoch, (1 if smoke else EPOCHS)+1):
        train_dataset.epoch = epoch-1
        order = order_for_epoch(epoch, len(train_dataset))
        order_hash = hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest()
        shifts = shifts_for_epoch(epoch, len(train_dataset))
        train_result = train_one_epoch(model, frontend, train_dataset, order, criterion, optimizer, scaler, device, amp, shifts)
        audit["train_windows_read"] += train_result["train_windows_read"]
        condition_metrics, records = evaluate(model, cache, device, amp)
        by_condition = {c:metrics([r for r in records if r["condition"]==c]) for c in CONDITIONS}
        score = float(np.mean([by_condition[c]["class_file_equal_recall"] for c in CONDITIONS]))
        history.append({"epoch":epoch, **train_result, "selection_score":score, "order_sha256":order_hash})
        csv_write(out/f"validation_scores_epoch_{epoch:02d}.csv", records)
        csv_write(out/"training_history.csv", history)
        csv_write(out/"validation_metrics.csv", [{"epoch":epoch, **r} for r in condition_metrics] if epoch==1 else
                  list(csv.DictReader((out/"validation_metrics.csv").open(encoding="utf-8",newline="")))+
                  [{"epoch":epoch, **r} for r in condition_metrics])
        selected = max(history, key=lambda h:(h["selection_score"],-h["epoch"]))
        selection = {"rule":config["selection"], "selected_epoch":selected["epoch"], "selected_score":selected["selection_score"],
                     "score_by_epoch":{str(h["epoch"]):h["selection_score"] for h in history}}
        dump(out/"selection.json", selection)
        if selected["epoch"] == epoch:
            torch.save({"epoch":epoch, "frontend_state":frontend.state_dict(), "model_state":model.state_dict(),
                        "n_fft":512, "hop_length":256, "image_size":256, "label_mapping":CODE_TO_LABEL,
                        "index_sha256":INDEX_SHA256, "split_sha256":SPLIT_SHA256, "selection_score":score}, out/"best_model.pt")
        torch.save({"identity":identity, "epoch":epoch, "frontend_state":frontend.state_dict(),
                    "model_state":model.state_dict(), "optimizer_state":optimizer.state_dict(),
                    "scaler_state":scaler.state_dict(), "rng":rng_state_payload(), "history":history}, state_file)
        dump(out/"read_audit.json",audit)
        dump(out/"progress.json",{"status":"running", "epoch":epoch, "selected_epoch":selected["epoch"],
                                  "score":selected["selection_score"],"updated_at":now()})
        print(f"epoch {epoch} selected={selected['epoch']} score={selected['selection_score']:.6f}",flush=True)
    best = torch.load(out/"best_model.pt",map_location="cpu",weights_only=False)
    if best["epoch"] != selected["epoch"]: raise RuntimeError("Checkpoint/selection mismatch")
    summary = {"status":"completed", "selected_epoch":selected["epoch"], "validation_score":selected["selection_score"],
               "best_model_sha256":sha256(out/"best_model.pt"), "train_windows_read":audit["train_windows_read"],
               "validation_windows_read":audit["validation_windows_read"], "test_files_touched":[], "finished_at":now()}
    dump(out/"summary.json",summary)
    dump(out/"progress.json",summary)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--resume",action="store_true")
    a=parser.parse_args()
    print(json.dumps(run(a.output,smoke=a.smoke,resume=a.resume),ensure_ascii=False,indent=2),flush=True)
