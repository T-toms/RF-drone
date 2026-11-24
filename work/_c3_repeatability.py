"""Compare the target-window recomputations of several fresh verifier processes.

Produces the difference distribution required by the rework task: two runs in the
formal (CUDA) environment must agree bit for bit, and the cross-environment (CPU)
runs must agree with each other while differing from the formal run only by bounded
numeric drift with unchanged discrete decisions.

    python work/_c3_repeatability.py --out work/cascade3-rework1-repeatability.json ^
        formal=<json> formal=<json> cross=<json> cross=<json>
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")


def load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = payload["checks"]["target_recomputation"]
    rows = {
        (
            row["dataset"],
            row["condition"],
            int(row["training_seed"]),
            int(row["position"]),
        ): row
        for row in target.get("targets", [])
    }
    return {
        "path": str(path),
        "tier": payload.get("numeric_comparison_tier"),
        "passed": payload.get("passed"),
        "environment": payload["checks"]["environment"].get("recomputation"),
        "seconds": payload["checks"].get("seconds"),
        "summary": {
            key: value
            for key, value in target.items()
            if key not in {"targets", "groups"}
        },
        "rows": rows,
    }


def compare(left: dict, right: dict) -> dict:
    shared = sorted(set(left["rows"]) & set(right["rows"]))
    digest_equal = 0
    feature_digest_equal = 0
    stage2_feature_digest_equal = 0
    input_equal = 0
    gate_flips = 0
    class_flips = 0
    stage1 = []
    stage2 = []
    stage1_margin = []
    for key in shared:
        one, other = left["rows"][key], right["rows"][key]
        digest_equal += int(
            one.get("input_full_digest") == other.get("input_full_digest")
        )
        feature_digest_equal += int(
            one.get("recomputed_digest") == other.get("recomputed_digest")
        )
        stage2_feature_digest_equal += int(
            one.get("recomputed_stage2_digest") == other.get("recomputed_stage2_digest")
        )
        input_equal += int(
            one.get("stage2_digest_match") == other.get("stage2_digest_match")
        )
        gate_flips += int(bool(one.get("gate_flip")) != bool(other.get("gate_flip")))
        class_flips += int(bool(one.get("class_flip")) != bool(other.get("class_flip")))
        stage1.append(
            abs(
                float(one.get("stage1_max_abs_logit_diff", 0.0))
                - float(other.get("stage1_max_abs_logit_diff", 0.0))
            )
        )
        stage1_margin.append(
            abs(
                float(one.get("stage1_margin_diff", 0.0))
                - float(other.get("stage1_margin_diff", 0.0))
            )
        )
        stage2.append(
            abs(
                float(one.get("stage2_max_abs_logit_diff", 0.0))
                - float(other.get("stage2_max_abs_logit_diff", 0.0))
            )
        )
    return {
        "left": Path(left["path"]).name,
        "right": Path(right["path"]).name,
        "shared_windows": len(shared),
        "input_full_digest_equal": digest_equal,
        "recomputed_feature_digest_equal": feature_digest_equal,
        "recomputed_stage2_feature_digest_equal": stage2_feature_digest_equal,
        "stage2_digest_status_equal": input_equal,
        "gate_flag_disagreements": gate_flips,
        "class_flag_disagreements": class_flips,
        "left_gate_flips": sum(
            1 for key in shared if left["rows"][key].get("gate_flip")
        ),
        "right_gate_flips": sum(
            1 for key in shared if right["rows"][key].get("gate_flip")
        ),
        "left_class_flips": sum(
            1 for key in shared if left["rows"][key].get("class_flip")
        ),
        "right_class_flips": sum(
            1 for key in shared if right["rows"][key].get("class_flip")
        ),
        "left_digest_matches_csv": sum(
            1 for key in shared if left["rows"][key].get("digest_match")
        ),
        "right_digest_matches_csv": sum(
            1 for key in shared if right["rows"][key].get("digest_match")
        ),
        "max_abs_difference_of_stage1_logit_drift": max(stage1) if stage1 else 0.0,
        "max_abs_difference_of_stage1_margin_drift": max(stage1_margin) if stage1_margin else 0.0,
        "max_abs_difference_of_stage2_logit_drift": max(stage2) if stage2 else 0.0,
        "bit_identical_recomputation": (
            len(shared) > 0
            and digest_equal == len(shared)
            and feature_digest_equal == len(shared)
            and stage2_feature_digest_equal == len(shared)
            and max(stage1 or [0.0]) == 0.0
            and max(stage2 or [0.0]) == 0.0
        ),
    }


def distribution(run: dict) -> dict:
    rows = list(run["rows"].values())
    stage1 = [float(row.get("stage1_max_abs_logit_diff", 0.0)) for row in rows]
    stage2 = [
        float(row["stage2_max_abs_logit_diff"])
        for row in rows
        if "stage2_max_abs_logit_diff" in row
    ]
    margins = [float(row.get("stage1_margin_diff", 0.0)) for row in rows]
    return {
        "windows": len(rows),
        "digest_mismatches": int(run["summary"].get("digest_mismatches", 0)),
        "stage2_digest_mismatches": int(run["summary"].get("stage2_digest_mismatches", 0)),
        "stage1_abs_logit_diff": {
            "max": max(stage1) if stage1 else 0.0,
            "mean": statistics.fmean(stage1) if stage1 else 0.0,
            "nonzero": sum(1 for value in stage1 if value > 0),
        },
        "stage1_abs_margin_diff": {
            "max": max(margins) if margins else 0.0,
            "mean": statistics.fmean(margins) if margins else 0.0,
            "nonzero": sum(1 for value in margins if value > 0),
        },
        "stage2_abs_logit_diff": {
            "max": max(stage2) if stage2 else 0.0,
            "mean": statistics.fmean(stage2) if stage2 else 0.0,
            "nonzero": sum(1 for value in stage2 if value > 0),
        },
        "input_readback_mismatches": int(run["summary"].get("input_readback_mismatches", 0)),
        "stage1_gate_changes": int(run["summary"].get("stage1_gate_changes", 0)),
        "stage2_class_changes": int(run["summary"].get("stage2_class_changes", 0)),
        "stage2_batch_id_errors": int(run["summary"].get("stage2_batch_id_errors", 0)),
        "tolerance": run["summary"].get("tolerance"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("specs", nargs="+", help="label=path, label in {formal, cross}")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[str, list[dict]] = {"formal": [], "cross": []}
    for spec in args.specs:
        label, _, raw = spec.partition("=")
        runs[label].append(load(Path(raw)))

    payload = {
        "task": "CASCADE3 rework 1: cross-process recomputation stability",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": {
            label: [
                {
                    "path": run["path"],
                    "tier": run["tier"],
                    "passed": run["passed"],
                    "environment": run["environment"],
                    "seconds": run["seconds"],
                    "distribution": distribution(run),
                }
                for run in entries
            ]
            for label, entries in runs.items()
        },
        "comparisons": [],
    }
    for label, entries in runs.items():
        for index in range(len(entries) - 1):
            payload["comparisons"].append(
                {"pair": label, **compare(entries[index], entries[index + 1])}
            )
    if runs["formal"] and runs["cross"]:
        payload["comparisons"].append(
            {"pair": "formal_vs_cross", **compare(runs["formal"][0], runs["cross"][0])}
        )
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for comparison in payload["comparisons"]:
        print(json.dumps(comparison, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
