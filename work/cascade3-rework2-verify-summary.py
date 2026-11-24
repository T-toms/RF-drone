"""Summarise the rework-2 --no-write verifier report into a small evidence JSON."""
import json
from pathlib import Path

import sys

BASE = Path(r"C:\Users\Administrator\Desktop\数据集处理")
sys.path.insert(0, str(BASE / "dronerfa_baseline"))
import verify_cascade3_evaluation as verifier  # noqa: E402

report_path = BASE / "work" / "cascade3-rework2-verify-nowrite.json"
report = json.loads(report_path.read_text(encoding="utf-8"))
checks = report["checks"]
root = BASE / "dronerfa_baseline" / "outputs" / "cascade3_evaluation_20260919"
figures = checks["figures"]["figures"]
targets = checks["target_recomputation"]

summary = {
    "passed": report["passed"],
    "failures": report["failures"],
    "failure_count": report["failure_count"],
    "tier": report.get("numeric_comparison_tier")
    or checks["environment"].get("numeric_comparison_tier"),
    "device": checks["environment"].get("recomputation", {}).get("device")
    or checks["environment"].get("device"),
    "seconds": checks["seconds"],
    "manifest": {
        "published_files": checks["manifest"]["published_files"],
        "mismatches": checks["manifest"]["mismatches"],
    },
    "figures": {
        name: {
            "hash_ok": entry["hash_ok"],
            "dimensions": entry["dimensions"],
            "sha256": entry["sha256"],
        }
        for name, entry in figures.items()
    },
    "targets": {
        "checked": targets.get("checked_windows"),
        "input_checks": targets.get("input_checks"),
        "input_readback_mismatches": targets.get("input_readback_mismatches"),
        "max_abs_stage1_logit_diff": targets.get("max_abs_stage1_logit_diff"),
        "max_abs_stage2_logit_diff": targets.get("max_abs_stage2_logit_diff"),
        "digest_mismatches": targets.get("digest_mismatches"),
        "stage2_digest_mismatches": targets.get("stage2_digest_mismatches"),
        "stage1_gate_changes": targets.get("stage1_gate_changes"),
        "stage1_margin_out_of_tolerance": targets.get("stage1_margin_out_of_tolerance"),
        "stage2_class_changes": targets.get("stage2_class_changes"),
        "stage2_margin_out_of_tolerance": targets.get("stage2_margin_out_of_tolerance"),
        "stage2_batch_id_errors": targets.get("stage2_batch_id_errors"),
        "rule_mismatches": targets.get("rule_mismatches"),
    },
}
manifest = json.loads((root / "experiment_manifest.json").read_text(encoding="utf-8"))
files = manifest["published_files"]
summary["manifest"]["published_file_names"] = sorted(files)
summary["manifest"]["figure_evidence_sha256_listed"] = files["figure_evidence.json"]["sha256"]
summary["manifest"]["figure_evidence_sha256_actual"] = verifier.sha256_file(
    root / "figure_evidence.json"
)
summary["manifest"]["predictions_sha256_listed"] = files["window_predictions.csv"]["sha256"]
summary["manifest"]["predictions_sha256_actual"] = verifier.sha256_file(
    root / "window_predictions.csv"
)
summary["manifest"]["digests"] = {
    name: {
        "listed": entry["sha256"],
        "actual": verifier.sha256_file(root / "feature_digests" / name),
    }
    for name, entry in files["feature_digests/"].items()
}
summary["manifest"]["figures_listed_in_manifest"] = [
    name for name in files if name.lower().endswith(".png")
]
evidence = json.loads((root / "figure_evidence.json").read_text(encoding="utf-8"))
summary["figures_recorded"] = {
    name: {
        "recorded": entry["sha256"],
        "actual": verifier.sha256_file(root / name),
    }
    for name, entry in evidence["figures"].items()
}
summary["recall_axis"] = evidence["recall_axis"]

out = BASE / "work" / "cascade3-rework2-verify-summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", out)
print(
    "passed=%s failures=%d tier=%s figures_hash_ok=%s mismatches=%d targets=%s"
    % (
        summary["passed"],
        summary["failure_count"],
        summary["tier"],
        all(entry["hash_ok"] for entry in summary["figures"].values()),
        len(summary["manifest"]["mismatches"]),
        summary["targets"]["checked"],
    )
)
print(
    "figure_evidence manifest==actual:",
    summary["manifest"]["figure_evidence_sha256_listed"]
    == summary["manifest"]["figure_evidence_sha256_actual"],
)
print(
    "predictions manifest==actual:",
    summary["manifest"]["predictions_sha256_listed"]
    == summary["manifest"]["predictions_sha256_actual"],
)
print(
    "all digest listed==actual:",
    all(item["listed"] == item["actual"] for item in summary["manifest"]["digests"].values()),
)
print(
    "all figure evidence listed==actual:",
    all(
        item["recorded"] == item["actual"]
        for item in summary["figures_recorded"].values()
    ),
)
print(
    "rule_mismatches:",
    summary["targets"]["rule_mismatches"],
    "gate_changes:",
    summary["targets"]["stage1_gate_changes"],
    "class_changes:",
    summary["targets"]["stage2_class_changes"],
    "stage2_batch_id_errors:",
    summary["targets"]["stage2_batch_id_errors"],
)
print("png files in manifest:", summary["manifest"]["figures_listed_in_manifest"])
