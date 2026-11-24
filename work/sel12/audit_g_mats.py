#!/usr/bin/env python3
"""SELECT12-F1 step 2: read-only audit of the source .mat files on G:.

Checks every ``*.mat`` under ``G:\\DroneRFa数据集\\original`` against the frozen
104-file target table:

  * file name is in the target table (no extra / no unknown file),
  * real byte size equals the archive index ``unpacked_size``,
  * HDF5 opens and exposes the four RF0/RF1 I/Q vectors with equal, non-empty
    lengths and a finite read probe.

SHA256 is computed for every file so the later G -> F copy can be byte-verified.
Nothing on G: or F: is written or modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK = PROJECT_ROOT / "work" / "sel12"
CONFIG = WORK / "sel12-config-20260924.json"
G_ORIGINAL = Path("G:/DroneRFa数据集/original")

REQUIRED_KEYS = ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q")


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_mat(path: Path) -> dict:
    import h5py
    import numpy as np

    info: dict = {"path": str(path)}
    try:
        with h5py.File(path, "r") as handle:
            keys = sorted(handle.keys())
            info["keys"] = keys
            info["missing_required"] = [key for key in REQUIRED_KEYS if key not in handle]
            if info["missing_required"]:
                info["ok"] = False
                return info
            lengths: dict[str, int] = {}
            dtypes: dict[str, str] = {}
            shapes: dict[str, list[int]] = {}
            for key in REQUIRED_KEYS:
                dataset = handle[key]
                shapes[key] = list(dataset.shape)
                dtypes[key] = str(dataset.dtype)
                lengths[key] = int(dataset.shape[0]) if dataset.ndim == 1 else int(max(dataset.shape))
            info["shapes"] = shapes
            info["dtypes"] = dtypes
            info["channel_lengths"] = lengths
            info["lengths_consistent"] = len(set(lengths.values())) == 1
            info["source_samples"] = next(iter(lengths.values()))
            info["all_float"] = all(value.startswith("float") for value in dtypes.values())
            probe_start = max(info["source_samples"] // 2 - 500_000, 0)
            probe_end = min(probe_start + 1000, info["source_samples"])
            values = handle["RF0_I"][probe_start:probe_end]
            info["read_probe"] = {
                "start": probe_start,
                "samples": int(values.shape[0]),
                "finite": bool(np.isfinite(values).all()),
            }
            info["complete_1M_windows"] = info["source_samples"] // 1_000_000
            info["ok"] = bool(
                info["lengths_consistent"]
                and info["source_samples"] > 0
                and info["all_float"]
                and info["read_probe"]["samples"] == probe_end - probe_start
                and info["read_probe"]["finite"]
            )
    except Exception as exc:  # noqa: BLE001
        info["ok"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK / "g-mat-audit-20260924.json")
    parser.add_argument("--no-sha256", action="store_true", help="跳过SHA256（快速结构核对）")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    targets = {item["filename"]: item for item in config["files"]}

    on_disk = sorted(G_ORIGINAL.glob("*.mat"))
    actual_names = {path.name for path in on_disk}
    unknown = sorted(actual_names - set(targets))
    missing = sorted(name for name in targets if name not in actual_names)

    records: list[dict] = []
    problems: list[str] = []
    started = time.monotonic()
    for index, path in enumerate(on_disk, start=1):
        target = targets.get(path.name)
        record: dict = {
            "filename": path.name,
            "actual_bytes": path.stat().st_size,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        if target is None:
            record["status"] = "unknown_file"
            problems.append(f"G盘存在目标表之外的文件：{path.name}")
        else:
            record["expected_bytes"] = target["expected_mat_bytes"]
            record["origin"] = target["origin"]
            record["former_partition"] = target["former_partition"]
            size_ok = record["actual_bytes"] == target["expected_mat_bytes"]
            record["size_ok"] = size_ok
            structure = inspect_mat(path)
            record["hdf5"] = structure
            if not size_ok:
                problems.append(
                    f"{path.name}: 字节数{record['actual_bytes']}与归档索引"
                    f"{target['expected_mat_bytes']}不符"
                )
            if not structure.get("ok"):
                problems.append(f"{path.name}: HDF5结构检查未通过：{structure.get('error') or structure.get('missing_required')}")
            record["status"] = "valid" if (size_ok and structure.get("ok")) else "invalid"
        if not args.no_sha256:
            record["sha256"] = sha256_file(path)
        records.append(record)
        print(
            f"[{index}/{len(on_disk)}] {path.name} status={record['status']} "
            f"bytes={record['actual_bytes']}"
            + (f" samples={record.get('hdf5', {}).get('source_samples')}" if record.get("hdf5") else ""),
            flush=True,
        )

    valid = [item for item in records if item["status"] == "valid"]
    report = {
        "schema": "dronerfa-select12-g-audit/v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "g_original": str(G_ORIGINAL),
        "target_table": str(CONFIG),
        "counts": {
            "mat_on_g": len(records),
            "valid": len(valid),
            "invalid": len(records) - len(valid),
            "unknown": len(unknown),
            "target_total": len(targets),
            "target_missing_on_g": len(missing),
            "target_present_on_g": len(actual_names & set(targets)),
        },
        "bytes_on_g": sum(item["actual_bytes"] for item in records),
        "bytes_valid": sum(item["actual_bytes"] for item in valid),
        "missing_target_files": missing,
        "unknown_files_on_g": unknown,
        "problems": problems,
        "per_target_valid": {},
        "sha256_computed": not args.no_sha256,
        "files": records,
    }
    for item in valid:
        key = item["filename"].split("_")[0]
        bucket = report["per_target_valid"].setdefault(key, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += item["actual_bytes"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("counts", "bytes_on_g", "bytes_valid", "problems")},
                     ensure_ascii=False, indent=2))
    print(json.dumps({"per_target_valid": report["per_target_valid"]}, ensure_ascii=False, indent=2))
    print(f"report -> {args.out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
