"""Paired validation-only pilot: original AWGN schedule versus adding -15 dB.

Reuses the original dataset, frontend, network and training step. Validation
features are computed once in float32 and shared by both arms and all epochs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from dronerfa_data import DroneRFaWindowDataset, parse_snr_list, snr_name, stratified_limit
from model import ResNet18, SpectrogramFrontEnd
from train_resnet18 import binary_metrics, extract_features, seed_everything, train_epoch

ROOT = Path(__file__).resolve().parent
CONDITIONS = [None, -10.0, -12.0, -15.0]
ARMS = {
    "A_original": "clean,10,5,0,-5,-10,-12",
    "B_plus_minus15": "clean,10,5,0,-5,-10,-12,-15",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_hash(state: dict) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_predictions(rows: list[dict]) -> dict:
    metrics = binary_metrics(
        [r["label"] for r in rows],
        [r["prediction"] for r in rows],
        # The margin preserves ranking when softmax probabilities saturate.
        [r["logit_margin"] for r in rows],
    )
    metrics["fpr"] = metrics["fp"] / (metrics["fp"] + metrics["tn"])
    for code, name in [("0010", "phantom"), ("0101", "air2s")]:
        group = [r for r in rows if r["target_code"] == code]
        metrics[f"{name}_samples"] = len(group)
        metrics[f"{name}_detected"] = sum(r["prediction"] for r in group)
        metrics[f"{name}_recall"] = (
            metrics[f"{name}_detected"] / len(group) if group else float("nan")
        )
    return metrics


def select_epochs(history: list[dict]) -> dict[str, int]:
    # max returns the first tied row: predeclared earliest-epoch tie break.
    return {
        "clean_only": int(max(history, key=lambda r: r["clean_balanced_accuracy"])["epoch"]),
        "multi_noise": int(max(history, key=lambda r: r["mean_balanced_accuracy"])["epoch"]),
    }


def resolve_validation_seed(training_seed: int, explicit_seed: int | None) -> int:
    return training_seed + 10_000 if explicit_seed is None else explicit_seed


def validate_reference_cache(metadata: list[dict], reference: list[dict]) -> None:
    fields = ("samples", "full_dataset_indices", "feature_sha256")
    current = {r["condition"]: r for r in metadata}
    previous = {r["condition"]: r for r in reference}
    if current.keys() != previous.keys():
        raise ValueError("Validation conditions differ from reference")
    for condition in current:
        for field in fields:
            if current[condition][field] != previous[condition][field]:
                raise ValueError(f"Validation differs from reference: {condition}/{field}")


@torch.inference_mode()
def cache_validation(args, frontend, device, out):
    cached = {}
    metadata = []
    for snr in CONDITIONS:
        dataset = DroneRFaWindowDataset(
            args.index, split="val", fixed_snr=snr,
            apply_fixed_snr=snr is not None, seed=args.validation_seed,
        )
        # Subset the full dataset so AWGN RNG retains the full validation index.
        selected = stratified_limit(dataset.rows, args.max_val_samples)
        positions = {id(row): i for i, row in enumerate(dataset.rows)}
        indices = [positions[id(row)] for row in selected]
        loader = DataLoader(
            Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed + 30_000),
        )
        tensors, labels = [], []
        started = time.perf_counter()
        for batch, (signals, yy) in enumerate(loader, 1):
            tensors.append(extract_features(frontend, signals.to(device)).cpu())
            labels.append(yy)
            if batch % 25 == 0 or batch == len(loader):
                print(f"Cache {snr_name(snr)} {batch}/{len(loader)}", flush=True)
        xx, yy = torch.cat(tensors), torch.cat(labels)
        name = snr_name(snr)
        cached[name] = (xx, yy, selected)
        metadata.append({
            "condition": name, "samples": len(selected), "full_dataset_indices": indices,
            "feature_sha256": hashlib.sha256(xx.numpy().tobytes()).hexdigest(),
            "seconds": time.perf_counter() - started,
        })
        dataset.close()
    write_json(out / "validation_cache_manifest.json", metadata)
    return cached, metadata


@torch.inference_mode()
def evaluate_cached(model, cached, criterion, device, amp_enabled, batch_size, seed):
    model.eval()
    metrics_all, records_all = [], []
    for condition, (xx, yy, rows) in cached.items():
        loader = DataLoader(
            TensorDataset(xx, yy), batch_size=batch_size, shuffle=False,
            generator=torch.Generator().manual_seed(seed + 40_000),
        )
        records, total_loss, cursor = [], 0.0, 0
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled, dtype=torch.float16):
                logits = model(features)
                loss = criterion(logits, labels)
            total_loss += float(loss) * len(labels)
            # Convert BEFORE subtraction, matching the original diagnosis.
            logits = logits.float()
            margins = (logits[:, 1] - logits[:, 0]).cpu().tolist()
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            for margin, prob in zip(margins, probs):
                source = rows[cursor]
                records.append({
                    "condition": condition, "validation_index_in_selection": cursor,
                    "file_path": source["file_path"], "start": source["start"],
                    "target_code": source["target_code"], "label": source["label"],
                    "logit_margin": margin, "drone_score": prob,
                    "prediction": int(margin > 0.0),
                })
                cursor += 1
        metrics_all.append({"condition": condition, "loss": total_loss / len(rows),
                            **summarize_predictions(records)})
        records_all.extend(records)
    return metrics_all, records_all


def run_arm(args, name, snrs, cached, device, out):
    arm_dir = out / name
    arm_dir.mkdir()
    seed_everything(args.seed)
    frontend = SpectrogramFrontEnd(n_fft=1024, hop_length=512, image_size=256).to(device)
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    dataset = DroneRFaWindowDataset(
        args.index, split="train", train_snrs=parse_snr_list(snrs),
        max_samples=args.max_train_samples, seed=args.seed,
    )
    counts = {label: sum(r["label"] == label for r in dataset.rows) for label in [0, 1]}
    if not all(counts.values()):
        raise ValueError("Both training classes are required")
    weights = torch.tensor([len(dataset) / (2 * counts[k]) for k in [0, 1]], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    scaler = torch.amp.GradScaler(device.type, enabled=True)
    config = {
        "arm": name, "train_snrs": snrs, "seed": args.seed, "epochs": args.epochs,
        "validation_seed": args.validation_seed,
        "batch_size": args.batch_size, "train_samples": len(dataset), "class_counts": counts,
        "initial_model_sha256": state_hash(model.state_dict()),
        "validation_conditions": list(cached), "selection": "mean_balanced_accuracy",
        "prediction": "float32(logit_drone) - float32(logit_background) > 0",
        "tie_break": "earliest epoch", "index_sha256": sha256_file(args.index),
        "frontend": {"n_fft": 1024, "hop_length": 512, "image_size": 256},
        "optimizer": {"type": "Adam", "learning_rate": 0.001, "weight_decay": 0.0},
        "amp": True, "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    write_json(arm_dir / "run_config.json", config)
    history, all_metrics, order_hashes = [], [], []
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        dataset.set_epoch(epoch)
        # Explicit paired order, independent of model/validation RNG consumption.
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        order_hashes.append(hashlib.sha256(np.asarray(order, dtype=np.int64).tobytes()).hexdigest())
        loader = DataLoader(
            Subset(dataset, order), batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
            generator=torch.Generator().manual_seed(args.seed + epoch + 50_000),
        )
        print(f"{name} epoch {epoch}/{args.epochs}", flush=True)
        train_loss, train_metrics = train_epoch(
            frontend, model, loader, criterion, optimizer, scaler, device, True,
        )
        metrics, records = evaluate_cached(
            model, cached, criterion, device, True, args.batch_size, args.seed,
        )
        mean_score = float(np.mean([m["balanced_accuracy"] for m in metrics]))
        clean = next(m for m in metrics if m["condition"] == "clean")
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "clean_balanced_accuracy": clean["balanced_accuracy"],
            "mean_balanced_accuracy": mean_score, "seconds": time.perf_counter() - started,
        })
        all_metrics.extend({"epoch": epoch, **m} for m in metrics)
        write_csv(arm_dir / f"validation_scores_epoch_{epoch:02d}.csv", records)
        write_csv(arm_dir / "validation_metrics.csv", all_metrics)
        write_csv(arm_dir / "training_history.csv", history)
        torch.save({
            "epoch": epoch, "frontend_state": frontend.state_dict(), "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scaler_state": scaler.state_dict(),
            "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "numpy_rng_state": np.random.get_state(), "python_rng_state": random.getstate(),
            "val_metrics": metrics, "run_config": config,
        }, arm_dir / f"epoch_{epoch:02d}.pt")
        write_json(arm_dir / "selection.json", select_epochs(history))
        print(f"{name} epoch {epoch}: mean BA={mean_score:.6f}; seconds={history[-1]['seconds']:.1f}", flush=True)
        for m in metrics:
            print(f"  {m['condition']}: Air={m['air2s_detected']}/{m['air2s_samples']} "
                  f"Phantom={m['phantom_detected']}/{m['phantom_samples']} FP={m['fp']}/{m['fp']+m['tn']}", flush=True)
    config["epoch_order_sha256"] = order_hashes
    write_json(arm_dir / "run_config.json", config)
    dataset.close()
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return config, history, all_metrics


def write_comparison(out, runs):
    configs = [v[0] for v in runs.values()]
    paired = {
        "same_initial_weights": len({c["initial_model_sha256"] for c in configs}) == 1,
        "same_epoch_orders": configs[0]["epoch_order_sha256"] == configs[1]["epoch_order_sha256"],
        "same_validation_inputs": "shared immutable float32 feature tensors",
        "test_set_evaluated": False,
    }
    assert paired["same_initial_weights"] and paired["same_epoch_orders"]
    write_json(out / "pairing_verification.json", paired)
    rows = []
    for arm, (_, history, metrics) in runs.items():
        for policy, epoch in select_epochs(history).items():
            rows.extend({"arm": arm, "selection_policy": policy, **r} for r in metrics if r["epoch"] == epoch)
    write_csv(out / "selected_model_comparison.csv", rows)
    lines = ["# 原噪声范围与增加 −15 dB：配对试验", "",
             "本轮只使用训练集和验证集。选模规则在运行前固定；本报告为探索性验证结果。", "",
             "A：clean,10,5,0,−5,−10,−12；B：上述范围加 −15。各档等概率抽样，因此 B 同时降低了原各档的抽样概率。",
             f"两组初始权重及每轮训练窗口顺序一致；同一份 float32 验证特征用于所有模型。验证 AWGN seed={configs[0]['validation_seed']}，保留完整验证索引。",
             "模型选择：clean、−10、−12、−15 四档平衡准确率的算术平均，分数相同时取最早轮次；同时记录只按 clean 选择的结果。",
             "默认门槛保持 logit 差 > 0（数学上等价于 drone 概率 > 0.5）。", "",
             "| 组别 | 选模方式 | 轮次 | 附加噪声 | 平衡准确率 | Air 检出 | Phantom 检出 | 背景误报 |",
             "|---|---|---:|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['selection_policy']} | {r['epoch']} | {r['condition']} | "
                     f"{r['balanced_accuracy']:.2%} | {r['air2s_detected']}/{r['air2s_samples']} | "
                     f"{r['phantom_detected']}/{r['phantom_samples']} | {r['fp']}/{r['fp']+r['tn']} |")
    lines += ["", "## 解释边界", "",
              "- 这是单随机种子的配对试验，不能据此声称稳定增益或统计显著。",
              "- 验证集每类仅一个源文件；同一文件的窗口及同窗口的各噪声档互相关联。",
              "- 用于选模的数据也用于本报告描述，结果可能偏乐观；需新的独立采集检验泛化。",
              "- −15 dB 指额外加入的 AWGN 条件，不是现场绝对信噪比。",
              "- 本轮显式固定了配对训练顺序，不能把 A 与历史 pilot 的差异仅归因于选模方式。",
              "- 每轮检查点含模型、优化器、AMP 和随机数状态；selection.json 指向所选轮次。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args, shared_cache=None):
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    args.validation_seed = resolve_validation_seed(args.seed, args.validation_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("This paired pilot requires CUDA")
    out = args.output_dir or ROOT / "outputs" / ("noise_ab_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "status": "running", "started_at": datetime.now().isoformat(),
        "selection_rule": "mean balanced accuracy across clean,-10,-12,-15; earliest tie",
        "arms": ARMS, "source_sha256": {
            p.name: sha256_file(p) for p in [Path(__file__), ROOT / "train_resnet18.py",
                                           ROOT / "dronerfa_data.py", ROOT / "model.py", args.index]
        },
    }
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for name in manifest["source_sha256"]:
        shutil.copy2(args.index if name == args.index.name else ROOT / name, snapshot / name)
    write_json(out / "experiment_manifest.json", manifest)
    print(f"Output: {out.resolve()}", flush=True)
    frontend = SpectrogramFrontEnd().to(device).eval()
    if shared_cache is None:
        cached, metadata = cache_validation(args, frontend, device, out)
    else:
        cached, metadata = shared_cache
        for entry in metadata:
            actual = hashlib.sha256(cached[entry["condition"]][0].numpy().tobytes()).hexdigest()
            if actual != entry["feature_sha256"]:
                raise ValueError("Shared validation cache changed during training")
        write_json(out / "validation_cache_manifest.json", metadata)
    reference_run = args.reference_validation_run
    if reference_run is not None:
        reference_manifest = json.loads((reference_run / "experiment_manifest.json").read_text(encoding="utf-8"))
        reference_args = reference_manifest["arguments"]
        reference_seed = resolve_validation_seed(reference_args["seed"], reference_args.get("validation_seed"))
        if reference_seed != args.validation_seed:
            raise ValueError("Validation seed differs from reference")
        for name in ["train_resnet18.py", "model.py", "dronerfa_data.py", args.index.name]:
            if reference_manifest["source_sha256"][name] != manifest["source_sha256"][name]:
                raise ValueError(f"Experimental implementation differs from reference: {name}")
        reference_metadata = json.loads((reference_run / "validation_cache_manifest.json").read_text(encoding="utf-8"))
        validate_reference_cache(metadata, reference_metadata)
        write_json(out / "reference_validation_verification.json", {
            "reference_run": str(reference_run.resolve()), "validation_seed": args.validation_seed,
            "identical_features_and_indices": True, "shared_cache_reused": shared_cache is not None,
        })
        print("Validation features match reference exactly", flush=True)
    runs = {}
    for name, snrs in ARMS.items():
        runs[name] = run_arm(args, name, snrs, cached, device, out)
    write_comparison(out, runs)
    manifest.update(status="completed", completed_at=datetime.now().isoformat())
    write_json(out / "experiment_manifest.json", manifest)
    print(f"Completed: {out.resolve()}", flush=True)
    return cached, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "outputs/window_index.csv")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-seed", type=int,
                        help="Fixed validation AWGN seed; default is training seed + 10000.")
    parser.add_argument("--reference-validation-run", type=Path)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    run_experiment(parser.parse_args())


if __name__ == "__main__":
    main()
