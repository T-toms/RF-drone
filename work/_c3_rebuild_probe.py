"""CASCADE3 rework-1 probe: rebuild target windows on a chosen device / interpreter.

Diagnostic only.  It recomputes the raw+noise input tensors, the float32 STFT
features and the stage-1/stage-2 logits for the verifier's target windows and
compares them with the published ``window_predictions.csv`` rows.  Run it with two
different interpreters (project ``.venv`` and the bare PATH ``python``) and with
``--device cpu`` to separate *device* effects from *torch build* effects.

    python work/_c3_rebuild_probe.py --dataset heldout --condition clean --seed 2026
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
BASE_DIR = PROJECT_ROOT / "dronerfa_baseline"
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "outputs" / ".matplotlib"))
sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import verify_cascade3_evaluation as V  # noqa: E402


def environment_record(device: torch.device) -> dict:
    record = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": device.type,
    }
    if device.type == "cuda":
        record["device_name"] = torch.cuda.get_device_name(0)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="heldout")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    config = V.read_json(V.CONFIG_PATH)
    split = "heldout" if args.dataset == "heldout" else "test"
    index_key = "heldout_index" if args.dataset == "heldout" else "old_test_index"
    index_path = PROJECT_ROOT / str(config["datasets"][index_key])

    records = V.parse_prediction_rows(V.OUTPUT_ROOT / "window_predictions.csv")
    group = [
        one
        for one in records
        if str(one["dataset"]) == args.dataset
        and str(one["condition"]) == args.condition
        and int(one["training_seed"]) == args.seed
    ]
    if not group:
        raise SystemExit("no rows for the requested group")
    by_position = {int(one["dataset_index_position"]): one for one in group}
    targets = V.choose_targets(group, 3)
    if args.max_targets:
        targets = targets[: args.max_targets]

    selection = [
        int(one["dataset_index_position"])
        for one in sorted(group, key=lambda item: int(item["dataset_index_position"]))
        if bool(one["stage2_executed"])
    ]

    networks = V.load_networks(device)
    cache = V.FeatureChunkCache(index_path, split, args.condition, device)
    rows: list[dict] = []
    started = time.perf_counter()
    digest_matches = 0
    max_stage1 = 0.0
    max_stage2 = 0.0
    gate_flips = 0
    final_flips = 0
    try:
        for position in targets:
            one = by_position[position]
            chunk_start = position - position % V.BATCH_SIZE
            chunk = cache.chunk(chunk_start)
            own = chunk[position - chunk_start]
            digest = V.sha256_bytes(np.ascontiguousarray(own).tobytes())
            digest_match = digest == str(one["stage1_feature_digest"])
            digest_matches += int(digest_match)
            stage1 = V.forward(networks[args.seed]["stage1"], chunk, device)[
                position - chunk_start
            ]
            stage1_diff = max(
                abs(float(stage1[0]) - float(one["stage1_logit_background"])),
                abs(float(stage1[1]) - float(one["stage1_logit_drone"])),
            )
            max_stage1 = max(max_stage1, stage1_diff)
            margin = float(np.float32(stage1[1] - stage1[0]))
            executed = margin > 0.0
            gate_flip = executed != bool(one["stage2_executed"])
            gate_flips += int(gate_flip)
            entry = {
                "position": position,
                "true_class": int(one["true_class"]),
                "digest_match": digest_match,
                "stage1_max_abs_diff": stage1_diff,
                "margin_diff": abs(margin - float(one["stage1_margin"])),
                "gate_flip": gate_flip,
                "csv_executed": bool(one["stage2_executed"]),
                "probe_executed": executed,
            }
            if executed and bool(one["stage2_executed"]):
                batch_index = selection.index(position) // V.BATCH_SIZE
                batch = selection[
                    batch_index * V.BATCH_SIZE : (batch_index + 1) * V.BATCH_SIZE
                ]
                batch_features = np.stack([cache.row(item) for item in batch])
                stage2 = V.forward(networks[args.seed]["stage2"], batch_features, device)
                offset = batch.index(position)
                stage2_diff = max(
                    abs(float(stage2[offset][0]) - float(one["stage2_logit_phantom"])),
                    abs(float(stage2[offset][1]) - float(one["stage2_logit_air2s"])),
                )
                max_stage2 = max(max_stage2, stage2_diff)
                stage2_margin = float(np.float32(stage2[offset][1] - stage2[offset][0]))
                final = 2 if stage2_margin > 0.0 else 1
                final_flip = final != int(one["final_prediction"])
                final_flips += int(final_flip)
                stage2_digest = V.sha256_bytes(
                    np.ascontiguousarray(batch_features[offset]).tobytes()
                )
                entry.update(
                    {
                        "stage2_max_abs_diff": stage2_diff,
                        "stage2_margin_diff": abs(
                            stage2_margin - float(one["stage2_margin"])
                        ),
                        "final_flip": final_flip,
                        "stage2_digest_match": stage2_digest
                        == str(one["stage2_feature_digest"]),
                        "csv_final": int(one["final_prediction"]),
                        "probe_final": final,
                    }
                )
            rows.append(entry)
    finally:
        cache.close()

    result = {
        "environment": environment_record(device),
        "formal_environment": V.read_json(V.OUTPUT_ROOT / "environment.json"),
        "dataset": args.dataset,
        "condition": args.condition,
        "training_seed": args.seed,
        "targets": len(targets),
        "digest_matches": digest_matches,
        "digest_mismatches": len(targets) - digest_matches,
        "max_abs_stage1_logit_diff": max_stage1,
        "max_abs_stage2_logit_diff": max_stage2,
        "gate_flips": gate_flips,
        "final_class_flips": final_flips,
        "logit_tolerance": V.LOGIT_TOLERANCE,
        "seconds": time.perf_counter() - started,
        "rows": rows,
    }
    text = json.dumps(result, indent=2)
    if args.out is not None:
        out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(
        f"env torch={result['environment']['torch']} device={device.type} "
        f"cuda={result['environment']['cuda_available']}",
        flush=True,
    )
    print(
        f"targets={len(targets)} digest_mismatches={result['digest_mismatches']} "
        f"max_stage1={max_stage1:.9f} max_stage2={max_stage2:.9f} "
        f"gate_flips={gate_flips} final_flips={final_flips} "
        f"({result['seconds']:.1f}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
