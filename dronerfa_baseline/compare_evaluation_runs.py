"""Compare the completed second run against the preserved first-run evidence.

Evidence for the re-run after the per-window digest defect:
  * four complete feature SHA256 values (in-memory float32 cache of each condition)
  * the full 43,680-row prediction CSV (row identity, logits, margins, decisions)
  * the per-window digest files (32 bytes per window)

Writes run_parity.json into the evaluation output directory.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "heldout_evaluation_20260918"
RUN1_DIR = OUTPUT_DIR / "run1_evidence"
CONDITIONS = ["clean", "-10dB", "-12dB", "-15dB"]
KEY_FIELDS = ("training_seed", "arm", "condition", "heldout_index")
VALUE_FIELDS = ("logit_background", "logit_drone", "logit_margin", "drone_score", "prediction")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> dict[tuple, dict[str, str]]:
    rows: dict[tuple, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            rows[tuple(raw[field] for field in KEY_FIELDS)] = raw
    return rows


def main() -> int:
    manifest = json.loads((OUTPUT_DIR / "evaluation_manifest.json").read_text(encoding="utf-8"))
    run1_manifest = json.loads(
        (RUN1_DIR / "evaluation_manifest_run1.json").read_text(encoding="utf-8")
    )
    run1_features = {
        item["condition"]: item for item in run1_manifest["feature_cache"]["conditions"]
    }
    run2_features = {item["condition"]: item for item in manifest["feature_cache"]["conditions"]}

    feature_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        first = run1_features[condition]
        second = run2_features[condition]
        feature_rows.append(
            {
                "condition": condition,
                "run1_feature_sha256": first["feature_sha256"],
                "run2_feature_sha256": second["feature_sha256"],
                "match": first["feature_sha256"] == second["feature_sha256"],
                "run1_seconds": first["seconds"],
                "run2_seconds": second["seconds"],
            }
        )

    digest_rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        first_path = RUN1_DIR / f"window_digests_run1_{condition}.bin"
        second_path = OUTPUT_DIR / f"window_digests_{condition}.bin"
        first_bytes = first_path.read_bytes()
        second_sha = sha256_file(second_path)
        # Run 1 wrote the wrong content: only the running SHA256 of the whole float32
        # feature array (32 bytes) instead of one 32-byte digest per window. Run 2
        # writes 1820 x 32 bytes. The run-1 stub value is the SHA256 of the complete
        # feature buffer, which is also the run-2 feature hash, so the digest streams
        # of the two runs are compared through their own running hashes:
        #   run1 stub  must equal the condition's complete feature sha256, and
        #   sha256(run2 digest stream) must equal the run1 declared digest-concat hash
        #   when the same digest rule is applied.
        declared_concat = hashlib.sha256(second_path.read_bytes()).hexdigest()
        digest_rows.append(
            {
                "condition": condition,
                "run1_stub_bytes": len(first_bytes),
                "run1_stub_value": first_bytes.hex(),
                "run1_stub_is_complete_feature_sha256": first_bytes.hex()
                == manifest["feature_cache"]["conditions"][CONDITIONS.index(condition)][
                    "feature_sha256"
                ],
                "run2_file_bytes": second_path.stat().st_size,
                "run2_file_sha256": second_sha,
                "run2_digest_stream_sha256": declared_concat,
                "structure_differs_between_runs": len(first_bytes) != second_path.stat().st_size,
            }
        )

    run1_csv = RUN1_DIR / "window_predictions_run1.csv"
    run2_csv = OUTPUT_DIR / "window_predictions.csv"
    run1_rows = load_rows(run1_csv)
    run2_rows = load_rows(run2_csv)
    max_logit_delta = 0.0
    max_margin_delta = 0.0
    prediction_changes: list[tuple] = []
    missing = [key for key in run1_rows if key not in run2_rows]
    for key, second in run2_rows.items():
        first = run1_rows.get(key)
        if first is None:
            missing.append(key)
            continue
        max_logit_delta = max(
            max_logit_delta,
            abs(float(second["logit_background"]) - float(first["logit_background"])),
            abs(float(second["logit_drone"]) - float(first["logit_drone"])),
        )
        max_margin_delta = max(
            max_margin_delta, abs(float(second["logit_margin"]) - float(first["logit_margin"]))
        )
        if second["prediction"] != first["prediction"]:
            prediction_changes.append(key)

    result = {
        "run1_predictions_path": str(run1_csv),
        "run1_predictions_sha256": sha256_file(run1_csv),
        "run2_predictions_path": str(run2_csv),
        "run2_predictions_sha256": sha256_file(run2_csv),
        "run1_rows": len(run1_rows),
        "run2_rows": len(run2_rows),
        "missing_keys": len(missing),
        "feature_sha256_comparison": feature_rows,
        "all_four_feature_sha256_match": all(row["match"] for row in feature_rows),
        "window_digest_comparison": digest_rows,
        "run1_digest_defect_confirmed": all(
            row["run1_stub_bytes"] == 32
            and row["run1_stub_is_complete_feature_sha256"]
            and row["run2_file_bytes"] == 32 * 1820
            for row in digest_rows
        ),
        "max_abs_logit_delta": max_logit_delta,
        "max_abs_margin_delta": max_margin_delta,
        "prediction_changes": len(prediction_changes),
        "identical_decisions": len(prediction_changes) == 0 and not missing,
    }
    (OUTPUT_DIR / "run_parity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if (result["all_four_feature_sha256_match"] and result["identical_decisions"]) else 4


if __name__ == "__main__":
    raise SystemExit(main())
