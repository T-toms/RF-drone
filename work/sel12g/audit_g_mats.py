#!/usr/bin/env python3
"""SELECT12-G3 step 1: in-place read-only audit of the 104 MAT files on G:.

Checks every ``*.mat`` under ``G:\\DroneRFa数据集\\original`` against the frozen
104-file G target table (``work/sel12g/sel12g-config-20260924.json``):

  * the file name is a target-table name (unknown files are reported, never touched),
  * the real byte size equals the official archive-index ``unpacked_size``,
  * HDF5 opens and exposes the four RF0/RF1 I/Q vectors with equal, non-empty
    lengths, float dtypes and a finite read probe,
  * SHA256 is recorded so a later copy / re-check is byte-comparable.

The finite-value probe reads **only** the requested ~1000 samples
(:func:`read_channel_probe`).  A DroneRFa channel is stored as ``(1, 150e6)`` float64,
i.e. ~1.2 GiB per channel: materialising a whole channel just to slice 1000 samples
(the previous ``np.asarray(probe).reshape(-1)[a:b]``) is what this module must never
do again.

A bad file is recorded with its exact path and reason and is *never* overwritten or
deleted: it is simply not usable as training data.  Nothing on G: is written.
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
WORK = PROJECT_ROOT / "work" / "sel12g"
CONFIG = WORK / "sel12g-config-20260924.json"
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


def read_channel_probe(dataset, start: int, count: int) -> tuple:
    """Read exactly ``count`` samples starting at ``start`` from one channel.

    Returns ``(values, layout, error)``:

    * ``layout`` is ``"flat"`` for a 1-D ``(N,)`` channel and ``"row_1xN"`` for the
      real DroneRFa ``(1, N)`` layout;
    * ``error`` is a non-empty string and ``values`` is ``None`` for any other
      shape (for example ``(N, 1)`` or 3-D), which is rejected instead of guessed.

    Only the bounded slice is requested from HDF5 -- h5py reads just those samples,
    so no full-channel array is ever allocated.  ``count`` must be small (the caller
    uses ~1000).
    """
    import numpy as np

    if count <= 0:
        return None, None, f"invalid probe length {count}"
    shape = [int(size) for size in getattr(dataset, "shape", ())]
    ndim = int(getattr(dataset, "ndim", len(shape)))
    if ndim == 1:
        layout = "flat"
        values = dataset[start:start + count]
    elif ndim == 2 and shape and shape[0] == 1:
        layout = "row_1xN"
        values = dataset[0, start:start + count]
    else:
        return None, None, f"unsupported channel shape {shape}"
    values = np.asarray(values).reshape(-1)
    if values.shape[0] != count:
        return None, layout, f"probe read {values.shape[0]} samples, expected {count}"
    return values, layout, None


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
            shapes: dict[str, list[int]] = {}
            dtypes: dict[str, str] = {}
            lengths: dict[str, int] = {}
            for key in REQUIRED_KEYS:
                dataset = handle[key]
                shapes[key] = list(dataset.shape)
                dtypes[key] = str(dataset.dtype)
                lengths[key] = int(np.prod(dataset.shape))
            info["shapes"] = shapes
            info["dtypes"] = dtypes
            info["channel_lengths"] = lengths
            info["channel_ndim"] = {key: len(shapes[key]) for key in REQUIRED_KEYS}
            info["lengths_consistent"] = len(set(lengths.values())) == 1
            info["source_samples"] = next(iter(lengths.values()))
            info["all_float"] = all(value.startswith("float") for value in dtypes.values())
            probe_start = max(info["source_samples"] // 2 - 500_000, 0)
            probe_end = min(probe_start + 1000, info["source_samples"])
            probe_count = probe_end - probe_start
            values, layout, probe_error = read_channel_probe(
                handle["RF0_I"], probe_start, probe_count
            )
            info["read_probe"] = {
                "start": probe_start,
                "requested": probe_count,
                "layout": layout,
                "samples": 0 if values is None else int(values.shape[0]),
                "finite": bool(values is not None and np.isfinite(values).all()),
            }
            if probe_error:
                info["read_probe"]["error"] = probe_error
            info["complete_1M_windows"] = info["source_samples"] // 1_000_000
            info["ok"] = bool(
                info["lengths_consistent"]
                and info["source_samples"] > 0
                and info["all_float"]
                and probe_error is None
                and info["read_probe"]["samples"] == probe_count
                and info["read_probe"]["finite"]
            )
    except Exception as exc:  # noqa: BLE001
        info["ok"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK / "sel12g-mat-audit-20260924.json")
    parser.add_argument("--no-sha256", action="store_true")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    targets = {item["filename"]: item for item in config["files"]}
    g_original = Path(config["paths"]["g_original"])

    on_disk = sorted(g_original.glob("*.mat"))
    actual_names = {path.name for path in on_disk}
    unknown = sorted(actual_names - set(targets))
    missing = sorted(name for name in targets if name not in actual_names)

    records: list[dict] = []
    problems: list[str] = []
    started = time.monotonic()
    for index, path in enumerate(on_disk, start=1):
        stat = path.stat()
        target = targets.get(path.name)
        record: dict = {
            "filename": path.name,
            "path": str(path),
            "actual_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        }
        if target is None:
            record["status"] = "unknown_file"
            problems.append(f"G盘存在目标表之外的文件：{path}")
        else:
            record["expected_bytes"] = target["expected_mat_bytes"]
            record["origin"] = target["origin"]
            record["former_partition"] = target["former_partition"]
            record["split_status"] = target["split_status"]
            size_ok = stat.st_size == target["expected_mat_bytes"]
            record["size_ok"] = size_ok
            structure = inspect_mat(path)
            record["hdf5"] = structure
            if not size_ok:
                problems.append(
                    f"{path.name}: 实到{stat.st_size}字节 != 归档索引"
                    f"{target['expected_mat_bytes']}字节"
                )
            if not structure.get("ok"):
                problems.append(
                    f"{path.name}: HDF5结构检查未通过："
                    f"{structure.get('error') or structure.get('missing_required')}"
                )
            record["status"] = "valid" if (size_ok and structure.get("ok")) else "invalid"
        if not args.no_sha256:
            digest_started = time.monotonic()
            record["sha256"] = sha256_file(path)
            record["sha256_seconds"] = round(time.monotonic() - digest_started, 1)
        records.append(record)
        print(
            f"[{index}/{len(on_disk)}] {path.name} status={record['status']} "
            f"bytes={record['actual_bytes']}"
            + (f" samples={record.get('hdf5', {}).get('source_samples')}"
               if record.get("hdf5") else ""),
            flush=True,
        )

    valid = [item for item in records if item["status"] == "valid"]
    invalid = [item for item in records if item["status"] == "invalid"]
    per_target: dict[str, dict] = {}
    for item in records:
        key = item["filename"].split("_")[0]
        bucket = per_target.setdefault(key, {"files": 0, "valid": 0, "invalid": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["valid" if item["status"] == "valid" else "invalid"] += 1
        if item["status"] == "valid":
            bucket["bytes"] += item["actual_bytes"]

    report = {
        "schema": "dronerfa-select12g-mat-audit/v1",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "g_original": str(g_original),
        "target_table": str(CONFIG),
        "counts": {
            "mat_on_g": len(records),
            "valid": len(valid),
            "invalid": len(invalid),
            "unknown": len(unknown),
            "target_total": len(targets),
            "target_present_on_g": len(actual_names & set(targets)),
            "target_missing_on_g": len(missing),
        },
        "bytes_on_g": sum(item["actual_bytes"] for item in records),
        "bytes_valid": sum(item["actual_bytes"] for item in valid),
        "missing_target_files": missing,
        "unknown_files_on_g": unknown,
        "problems": problems,
        "per_target": per_target,
        "sha256_computed": not args.no_sha256,
        "files": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("counts", "bytes_on_g", "bytes_valid", "problems")},
                     ensure_ascii=False, indent=2))
    print(json.dumps({"per_target": per_target}, ensure_ascii=False, indent=2))
    print(f"report -> {args.out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
