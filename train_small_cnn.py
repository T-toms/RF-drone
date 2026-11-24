from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "figure" / "signal_aware_dataset_small"
IMAGE_DIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"
OUT_DIR = ROOT / "figure" / "signal_aware_cnn_baseline"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-3
SEED = 20260613


def read_rows() -> list[dict[str, str]]:
    with LABELS_CSV.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["status"] == "ok"]
    for r in rows:
        r["image_path"] = str(IMAGE_DIR / r["filename"])
        r["group_id"] = f"{r['scenario']}__slice{int(r['slice_index']):03d}"
    return rows


def split_by_group(rows: list[dict[str, str]], train_ratio: float = 0.7) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rng = random.Random(SEED)
    groups_by_class: dict[str, list[str]] = defaultdict(list)
    for gid, class_name in sorted({(r["group_id"], r["class_name"]) for r in rows}):
        groups_by_class[class_name].append(gid)

    train_groups: set[str] = set()
    val_groups: set[str] = set()
    for class_name, groups in groups_by_class.items():
        rng.shuffle(groups)
        n_train = max(1, int(round(len(groups) * train_ratio)))
        n_train = min(n_train, len(groups) - 1)
        train_groups.update(groups[:n_train])
        val_groups.update(groups[n_train:])

    train = [r for r in rows if r["group_id"] in train_groups]
    val = [r for r in rows if r["group_id"] in val_groups]
    return train, val


def audit(rows: list[dict[str, str]], train: list[dict[str, str]], val: list[dict[str, str]]) -> dict:
    def group_counts(subset: list[dict[str, str]], key: str) -> dict[str, int]:
        return dict(sorted(Counter(r[key] for r in subset).items()))

    train_groups = {r["group_id"] for r in train}
    val_groups = {r["group_id"] for r in val}
    overlap = sorted(train_groups & val_groups)
    return {
        "total_rows_status_ok": len(rows),
        "train_rows": len(train),
        "val_rows": len(val),
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "group_overlap_count": len(overlap),
        "group_overlap_examples": overlap[:5],
        "class_counts_all": group_counts(rows, "class_name"),
        "class_counts_train": group_counts(train, "class_name"),
        "class_counts_val": group_counts(val, "class_name"),
        "snr_counts_val": group_counts(val, "target_snr_db"),
        "nfft_counts_val": group_counts(val, "nfft"),
    }


def try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset

        return torch, nn, F, DataLoader, Dataset
    except ModuleNotFoundError:
        return None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    train_rows, val_rows = split_by_group(rows)
    audit_info = audit(rows, train_rows, val_rows)
    (OUT_DIR / "dataset_audit.json").write_text(
        json.dumps(audit_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    torch_import = try_import_torch()
    if torch_import is None:
        print("Dataset audit completed, but PyTorch is not installed in this Python environment.")
        print(f"audit: {OUT_DIR / 'dataset_audit.json'}")
        print("Install PyTorch, then rerun this script to train the CNN.")
        return

    torch, nn, F, DataLoader, Dataset = torch_import
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names = sorted({r["class_name"] for r in rows})
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    class SpectrogramDataset(Dataset):
        def __init__(self, items: list[dict[str, str]]):
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int):
            r = self.items[idx]
            img = Image.open(r["image_path"]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr.transpose(2, 0, 1))
            y = torch.tensor(class_to_idx[r["class_name"]], dtype=torch.long)
            meta = {
                "class_name": r["class_name"],
                "target_snr_db": r["target_snr_db"],
                "nfft": r["nfft"],
                "filename": r["filename"],
            }
            return x, y, meta

    class SmallCNN(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    train_loader = DataLoader(SpectrogramDataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SpectrogramDataset(val_rows), batch_size=BATCH_SIZE, shuffle=False)
    model = SmallCNN(len(class_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(y.size(0))

        val_metrics = evaluate(model, val_loader, device, class_names)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_acc": correct / max(total, 1),
            "val_acc": val_metrics["overall"]["accuracy"],
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_acc']:.3f} val_acc={row['val_acc']:.3f}"
        )

    metrics = evaluate(model, val_loader, device, class_names)
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_to_idx": class_to_idx,
            "image_size": IMAGE_SIZE,
        },
        OUT_DIR / "small_cnn.pt",
    )
    write_csv(OUT_DIR / "training_history.csv", history)
    write_metrics(metrics)
    print(f"outputs: {OUT_DIR}")


def evaluate(model, loader, device, class_names: list[str]) -> dict:
    model.eval()
    records = []
    with __import__("torch").no_grad():
        for x, y, metas in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            for i in range(y.size(0)):
                records.append(
                    {
                        "true": class_names[int(y[i].item())],
                        "pred": class_names[int(pred[i].item())],
                        "target_snr_db": str(metas["target_snr_db"][i]),
                        "nfft": str(metas["nfft"][i]),
                        "filename": metas["filename"][i],
                    }
                )

    def acc(items):
        return sum(r["true"] == r["pred"] for r in items) / max(len(items), 1)

    metrics = {"overall": {"count": len(records), "accuracy": acc(records)}}
    for key in ["target_snr_db", "nfft"]:
        grouped = defaultdict(list)
        for r in records:
            grouped[r[key]].append(r)
        metrics[key] = {
            k: {"count": len(v), "accuracy": acc(v)}
            for k, v in sorted(grouped.items(), key=lambda item: float(item[0]))
        }
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["target_snr_db"], r["nfft"])].append(r)
    metrics["snr_x_nfft"] = {
        f"snr{snr}_nfft{nfft}": {"count": len(v), "accuracy": acc(v)}
        for (snr, nfft), v in sorted(grouped.items(), key=lambda item: (float(item[0][0]), int(item[0][1])))
    }
    metrics["predictions"] = records
    return metrics


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics(metrics: dict) -> None:
    serializable = {k: v for k, v in metrics.items() if k != "predictions"}
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(OUT_DIR / "predictions.csv", metrics["predictions"])
    write_confusion_matrix(metrics["predictions"])


def write_confusion_matrix(records: list[dict]) -> None:
    classes = sorted({r["true"] for r in records} | {r["pred"] for r in records})
    idx = {name: i for i, name in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for r in records:
        matrix[idx[r["true"]], idx[r["pred"]]] += 1

    csv_path = OUT_DIR / "confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name, *row.tolist()])

    cell = 120
    label_w = 190
    title_h = 44
    width = label_w + cell * len(classes) + 20
    height = title_h + cell * len(classes) + 90
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((label_w, 12), "Predicted class", fill=(0, 0, 0))
    draw.text((10, title_h + cell * len(classes) // 2), "True class", fill=(0, 0, 0))

    max_count = max(int(matrix.max()), 1)
    for j, class_name in enumerate(classes):
        x = label_w + j * cell
        draw.text((x + 8, title_h + 8), class_name, fill=(0, 0, 0))

    for i, true_name in enumerate(classes):
        y = title_h + 40 + i * cell
        draw.text((10, y + cell // 2 - 8), true_name, fill=(0, 0, 0))
        row_sum = max(int(matrix[i].sum()), 1)
        for j in range(len(classes)):
            x = label_w + j * cell
            count = int(matrix[i, j])
            intensity = int(255 - 180 * (count / max_count))
            color = (intensity, intensity, 255)
            draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=color, outline=(255, 255, 255))
            acc_text = f"{count}\n{count / row_sum:.1%}"
            draw.multiline_text((x + 35, y + 42), acc_text, fill=(0, 0, 0), spacing=4)

    img.save(OUT_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
