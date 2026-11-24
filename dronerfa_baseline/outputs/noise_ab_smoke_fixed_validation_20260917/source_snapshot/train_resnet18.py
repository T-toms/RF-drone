from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dronerfa_data import (
    DroneRFaWindowDataset,
    class_counts,
    load_index,
    parse_snr_list,
    snr_name,
)
from model import ResNet18, SpectrogramFrontEnd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leakage-resistant DroneRFa RF0 binary detector."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=PROJECT_DIR / "outputs" / "window_index.csv",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--train-snrs",
        default="clean,10,5,0,-5,-10,-12",
        help="Comma-separated added-AWGN SNRs. Use 'clean' for no added noise.",
    )
    parser.add_argument(
        "--eval-snrs",
        default="clean,0,-5,-10,-12,-15,-20",
        help="Comma-separated fixed added-AWGN SNRs.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed precision for the ResNet (STFT remains float32).",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loader(
    dataset: DroneRFaWindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def binary_metrics(
    labels: list[int], predictions: list[int], probabilities: list[float]
) -> dict[str, float | int]:
    y = np.asarray(labels, dtype=np.int64)
    pred = np.asarray(predictions, dtype=np.int64)
    prob = np.asarray(probabilities, dtype=np.float64)

    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    total = max(len(y), 1)
    accuracy = (tp + tn) / total
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    positive = prob[y == 1]
    negative = prob[y == 0]
    if len(positive) and len(negative):
        comparisons = (
            (positive[:, None] > negative[None, :]).sum()
            + 0.5 * (positive[:, None] == negative[None, :]).sum()
        )
        roc_auc = float(comparisons / (len(positive) * len(negative)))
    else:
        roc_auc = float("nan")

    return {
        "samples": len(y),
        "accuracy": accuracy,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def extract_features(
    frontend: SpectrogramFrontEnd, signals: torch.Tensor
) -> torch.Tensor:
    with torch.amp.autocast(device_type=signals.device.type, enabled=False):
        return frontend(signals.float())


def train_epoch(
    frontend: SpectrogramFrontEnd,
    model: ResNet18,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, float | int]]:
    model.train()
    frontend.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[float] = []
    started = time.perf_counter()

    for batch_index, (signals, labels) in enumerate(loader, start=1):
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        features = extract_features(frontend, signals)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            logits = model(features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        predictions = logits.detach().argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())

        if batch_index % 10 == 0 or batch_index == len(loader):
            elapsed = time.perf_counter() - started
            rate = len(labels_all) / max(elapsed, 1e-9)
            print(
                f"  train batch {batch_index:4d}/{len(loader):4d} "
                f"loss={total_loss / len(labels_all):.4f} "
                f"rate={rate:.1f} samples/s",
                flush=True,
            )

    return (
        total_loss / max(len(labels_all), 1),
        binary_metrics(labels_all, predictions_all, probabilities_all),
    )


@torch.inference_mode()
def evaluate(
    frontend: SpectrogramFrontEnd,
    model: ResNet18,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, float | int]]:
    model.eval()
    frontend.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[float] = []

    for signals, labels in loader:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = extract_features(frontend, signals)
        with torch.amp.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            logits = model(features)
            loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        predictions = logits.argmax(dim=1)
        batch_size = labels.shape[0]
        total_loss += float(loss) * batch_size
        labels_all.extend(labels.cpu().tolist())
        predictions_all.extend(predictions.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())

    return (
        total_loss / max(len(labels_all), 1),
        binary_metrics(labels_all, predictions_all, probabilities_all),
    )


def save_confusion_matrix(
    metrics: dict[str, float | int], title: str, output_path: Path
) -> None:
    matrix = np.array(
        [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
        dtype=np.int64,
    )
    fig, axis = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks((0, 1), labels=("Background", "Drone"))
    axis.set_yticks((0, 1), labels=("Background", "Drone"))
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(title)
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.colorbar(image, ax=axis)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_history(history: list[dict[str, Any]], output_dir: Path) -> None:
    csv_path = output_dir / "training_history.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    epochs = [int(row["epoch"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row["train_balanced_accuracy"] for row in history],
        label="Train",
    )
    axes[1].plot(
        epochs,
        [row["val_balanced_accuracy"] for row in history],
        label="Validation",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def save_snr_curve(results: list[dict[str, Any]], output_path: Path) -> None:
    numeric = [row for row in results if row["snr_db"] != "clean"]
    if not numeric:
        return
    numeric.sort(key=lambda row: float(row["snr_db"]))
    x_values = [float(row["snr_db"]) for row in numeric]
    fig, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    axis.plot(
        x_values,
        [row["balanced_accuracy"] for row in numeric],
        marker="o",
        label="Balanced accuracy",
    )
    axis.plot(
        x_values,
        [row["f1"] for row in numeric],
        marker="s",
        label="F1",
    )
    axis.set_xlabel("Added complex-AWGN SNR (dB)")
    axis.set_ylabel("Metric")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_DIR / "outputs" / f"run_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_snrs = parse_snr_list(args.train_snrs)
    eval_snrs = parse_snr_list(args.eval_snrs)
    train_dataset = DroneRFaWindowDataset(
        args.index,
        split="train",
        train_snrs=train_snrs,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_dataset = DroneRFaWindowDataset(
        args.index,
        split="val",
        max_samples=args.max_eval_samples,
        seed=args.seed + 10_000,
    )
    train_loader = make_loader(
        train_dataset, args.batch_size, True, args.num_workers, device
    )
    val_loader = make_loader(
        val_dataset, args.batch_size, False, args.num_workers, device
    )

    counts = class_counts(train_dataset.rows)
    if set(counts) != {0, 1}:
        raise RuntimeError(f"Training data must contain labels 0 and 1, got {counts}")
    total = sum(counts.values())
    class_weights = torch.tensor(
        [total / (2 * counts[0]), total / (2 * counts[1])],
        dtype=torch.float32,
        device=device,
    )

    frontend = SpectrogramFrontEnd(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        image_size=args.image_size,
    ).to(device)
    model = ResNet18(input_channels=1, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    run_config = {
        "arguments": serializable_args(args),
        "resolved_output_dir": str(output_dir.resolve()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_class_counts": counts,
        "class_weights": class_weights.detach().cpu().tolist(),
        "train_snrs": [snr_name(value) for value in train_snrs],
        "eval_snrs": [snr_name(value) for value in eval_snrs],
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(run_config, ensure_ascii=False, indent=2), flush=True)
    history: list[dict[str, Any]] = []
    best_score = -1.0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        train_dataset.set_epoch(epoch)
        train_loss, train_metrics = train_epoch(
            frontend,
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            amp_enabled,
        )
        val_loss, val_metrics = evaluate(
            frontend,
            model,
            val_loader,
            criterion,
            device,
            amp_enabled,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)
        print(
            f"  epoch={epoch} train_loss={train_loss:.4f} "
            f"train_bal_acc={train_metrics['balanced_accuracy']:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f}",
            flush=True,
        )
        score = float(val_metrics["balanced_accuracy"])
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "frontend_state": frontend.state_dict(),
                    "model_state": model.state_dict(),
                    "val_metrics": val_metrics,
                    "run_config": run_config,
                },
                checkpoint_path,
            )

    save_history(history, output_dir)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    frontend.load_state_dict(checkpoint["frontend_state"])
    model.load_state_dict(checkpoint["model_state"])

    test_results: list[dict[str, Any]] = []
    for snr_db in eval_snrs:
        test_dataset = DroneRFaWindowDataset(
            args.index,
            split="test",
            fixed_snr=snr_db,
            apply_fixed_snr=snr_db is not None,
            max_samples=args.max_eval_samples,
            seed=args.seed + 20_000,
        )
        test_loader = make_loader(
            test_dataset, args.batch_size, False, args.num_workers, device
        )
        test_loss, test_metrics = evaluate(
            frontend,
            model,
            test_loader,
            criterion,
            device,
            amp_enabled,
        )
        result = {
            "snr_db": "clean" if snr_db is None else snr_db,
            "loss": test_loss,
            **test_metrics,
        }
        test_results.append(result)
        name = snr_name(snr_db)
        print(
            f"Test {name}: loss={test_loss:.4f} "
            f"balanced_accuracy={test_metrics['balanced_accuracy']:.4f} "
            f"f1={test_metrics['f1']:.4f}",
            flush=True,
        )
        save_confusion_matrix(
            test_metrics,
            f"Test confusion matrix: {name}",
            output_dir / f"confusion_{name}.png",
        )
        test_dataset.close()

    with (output_dir / "test_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(test_results[0]))
        writer.writeheader()
        writer.writerows(test_results)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_snr_curve(test_results, output_dir / "snr_robustness.png")
    train_dataset.close()
    val_dataset.close()
    print(f"Completed. Results: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
