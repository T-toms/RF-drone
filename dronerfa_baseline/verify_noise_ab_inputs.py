"""Re-run selected checkpoints on raw-IQ batches to audit cached evaluation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from dronerfa_data import DroneRFaWindowDataset
from model import ResNet18, SpectrogramFrontEnd
from run_noise_ab import resolve_validation_seed, sha256_file, state_hash
from train_resnet18 import extract_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir
    manifest = json.loads((out / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    index = Path(manifest["arguments"]["index"])
    seed = manifest["arguments"]["seed"]
    validation_seed = resolve_validation_seed(seed, manifest["arguments"].get("validation_seed"))
    batch_size = manifest["arguments"]["batch_size"]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    model = ResNet18().to(device).eval()
    frontend = SpectrogramFrontEnd().to(device).eval()
    results = []
    for arm in manifest["arms"]:
        directory = out / arm
        selection = json.loads((directory / "selection.json").read_text())
        for epoch in sorted(set(selection.values())):
            checkpoint_path = directory / f"epoch_{epoch:02d}.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            # Export compact inference checkpoints using the original detector's
            # run_config schema. Optimizer/RNG objects stay in epoch_XX.pt.
            config = dict(checkpoint["run_config"])
            config["torch_version"] = str(config["torch_version"])
            config["arguments"] = {
                **config["frontend"], "seed": seed, "amp": True,
                "validation_seed": validation_seed,
                "index": str(index), "batch_size": batch_size,
                "train_snrs": config["train_snrs"], "epochs": config["epochs"],
            }
            for policy, chosen_epoch in selection.items():
                if epoch != chosen_epoch:
                    continue
                export_path = directory / ("best_model.pt" if policy == "multi_noise" else "best_model_clean.pt")
                torch.save({"epoch": epoch, "frontend_state": checkpoint["frontend_state"],
                            "model_state": checkpoint["model_state"], "val_metrics": checkpoint["val_metrics"],
                            "run_config": config, "selection_policy": policy}, export_path)
                exported = torch.load(export_path, map_location="cpu", weights_only=True)
                assert state_hash(exported["model_state"]) == state_hash(checkpoint["model_state"])
                del exported
            model.load_state_dict(checkpoint["model_state"])
            frontend.load_state_dict(checkpoint["frontend_state"])
            assert checkpoint["epoch"] == epoch
            with (directory / f"validation_scores_epoch_{epoch:02d}.csv").open(encoding="utf-8-sig") as handle:
                stored = {(r["condition"], r["file_path"], int(r["start"])): r for r in csv.DictReader(handle)}
            max_difference, mismatches, checked = 0.0, 0, 0
            for condition, snr in [("clean", None), ("-10dB", -10), ("-12dB", -12), ("-15dB", -15)]:
                dataset = DroneRFaWindowDataset(index, "val", fixed_snr=snr,
                    apply_fixed_snr=snr is not None, seed=validation_seed)
                # Use a middle batch from each target; preserve full dataset
                # positions and the same batch shape as the original inference.
                starts = set()
                for code in ["0000", "0010", "0101"]:
                    indices = [i for i, row in enumerate(dataset.rows) if row["target_code"] == code]
                    starts.add(indices[len(indices) // 2] // batch_size * batch_size)
                for start in sorted(starts):
                    positions = list(range(start, min(start + batch_size, len(dataset))))
                    signals = torch.stack([dataset[i][0] for i in positions]).to(device)
                    with torch.inference_mode():
                        features = extract_features(frontend, signals)
                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            logits = model(features).float()
                    margins = (logits[:, 1] - logits[:, 0]).cpu().tolist()
                    for i, margin in zip(positions, margins):
                        row = dataset.rows[i]
                        saved = stored[condition, row["file_path"], row["start"]]
                        max_difference = max(max_difference, abs(margin - float(saved["logit_margin"])))
                        mismatches += int(int(margin > 0) != int(saved["prediction"]))
                        checked += 1
                dataset.close()
            result = dict(arm=arm, epoch=epoch, checked_predictions=checked,
                          max_margin_difference=max_difference, prediction_mismatches=mismatches,
                          checkpoint_sha256=sha256_file(checkpoint_path))
            print(json.dumps(result), flush=True)
            results.append(result)
            del checkpoint
    root = Path(__file__).resolve().parent
    snapshot = out / "source_snapshot"
    code_root = snapshot if snapshot.exists() else root
    source_hashes_match = all(sha256_file(index if name == index.name else code_root / name) == digest
                             for name, digest in manifest["source_sha256"].items())
    original_files = {
        root / "outputs/pilot_5epochs/best_model.pt": "52b93241e3c79d816977eebafc22c3662ff651d499438af77cf6eb0312c45aa2",
        root / "outputs/validation_diagnosis_20260917_161707/validation_scores.csv":
            "b06357b54faa52f17bcfa9ac3c511e5567e455eb39db9499891c549642485a2d",
    }
    originals_unchanged = all(sha256_file(path) == expected for path, expected in original_files.items())
    verification = dict(results=results, source_hashes_match=source_hashes_match,
                        validation_seed=validation_seed,
                        original_checkpoint_and_diagnosis_unchanged=originals_unchanged)
    (out / "raw_input_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    assert source_hashes_match and originals_unchanged
    assert all(r["prediction_mismatches"] == 0 and r["max_margin_difference"] <= 0.01 for r in results)


if __name__ == "__main__":
    main()
