"""Record that the rework did not change any formal CASCADE3 artefact.

Compares the current SHA256 of the prediction CSV, the metric tables and the five
per-window digest binaries with the values recorded in Codex's first acceptance
evidence (work/cascade3-codex-verification.json) and in the published
experiment_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
ROOT = PROJECT_ROOT / "dronerfa_baseline" / "outputs" / "cascade3_evaluation_20260919"
CODEX_EVIDENCE = PROJECT_ROOT / "work" / "cascade3-codex-verification.json"
OUT = PROJECT_ROOT / "work" / "cascade3-rework1-formal-artifacts.json"

FILES = [
    "window_predictions.csv",
    "metrics_by_condition.csv",
    "metrics_by_condition.json",
    "confusion_matrices.csv",
    "per_file_metrics.csv",
    "error_decomposition.csv",
    "cross_seed_summary.csv",
    "prediction_cache_manifest.json",
    "stage1_reference_parity.json",
    "verification.json",
    "evaluation_run.json",
    "report.md",
    "experiment_manifest.json",
    "feature_digests/window_digests_heldout_clean.bin",
    "feature_digests/window_digests_heldout_-10.bin",
    "feature_digests/window_digests_heldout_-12.bin",
    "feature_digests/window_digests_heldout_-15.bin",
    "feature_digests/window_digests_old_test_clean.bin",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    codex = json.loads(CODEX_EVIDENCE.read_text(encoding="utf-8"))
    codex_checks = codex["checks"]
    manifest = json.loads((ROOT / "experiment_manifest.json").read_text(encoding="utf-8"))
    published = manifest["published_files"]
    recorded: dict[str, str] = {}
    recorded["window_predictions.csv"] = codex_checks["prediction_file"]["sha256"]
    for entry in codex_checks["digests"]["cache_entries"]:
        recorded[str(entry["file"]).split("cascade3_evaluation_20260919\\")[-1]] = entry[
            "file_sha256"
        ]
    for name, entry in published.items():
        if name.endswith("/"):
            for inner, inner_entry in entry.items():
                recorded[f"{name.rstrip('/')}/{inner}"] = inner_entry["sha256"]
        else:
            recorded[name] = entry["sha256"]

    rows = []
    for name in FILES:
        path = ROOT / name
        entry = {"file": name, "exists": path.is_file()}
        if path.is_file():
            stat = path.stat()
            entry["bytes"] = stat.st_size
            entry["mtime_utc"] = datetime.fromtimestamp(
                stat.st_mtime, timezone.utc
            ).isoformat()
            entry["sha256"] = sha256_file(path)
            entry["recorded_sha256"] = recorded.get(name)
            if recorded.get(name) is None:
                entry["match"] = None
                entry["note"] = "no recorded hash available for this file"
            else:
                entry["match"] = entry["sha256"] == recorded[name]
        rows.append(entry)

    payload = {
        "task": "CASCADE3 rework 1: formal artefacts unchanged",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "codex_first_failure_evidence": str(CODEX_EVIDENCE),
        "codex_first_failure_passed": codex["passed"],
        "files": rows,
        "checked": len(rows),
        "compared_against_codex_evidence": sum(
            1 for entry in rows if entry.get("recorded_sha256")
        ),
        "mismatches": [entry["file"] for entry in rows if entry.get("match") is False],
        "unmatched": [entry["file"] for entry in rows if entry.get("match") is None],
        "formal_inference_rerun": False,
        "note": (
            "every verification run of this rework used --no-write, so the published "
            "output root was never written to; experiment_manifest.json cannot list its "
            "own hash and is therefore reported without a recorded comparison"
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"checked={payload['checked']} compared={payload['compared_against_codex_evidence']} "
        f"mismatches={payload['mismatches']} unmatched={payload['unmatched']}"
    )
    print(f"prediction csv sha256={rows[0]['sha256']}")
    return 0 if not payload["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
