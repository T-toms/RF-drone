#!/usr/bin/env python3
"""SELECT12-G1 helper: SHA256 every target MAT present on G: (read-only).

Kept separate from the structural audit so a long hash pass never delays the
structural result.  Nothing on G: is written.
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


def sha256_file(path: Path, chunk: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK / "sel12g-mat-sha256-20260924.json")
    args = parser.parse_args()

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    targets = {item["filename"]: item for item in config["files"]}
    g_original = Path(config["paths"]["g_original"])

    records: list[dict] = []
    started = time.monotonic()
    files = sorted(g_original.glob("*.mat"))
    for index, path in enumerate(files, start=1):
        digest_started = time.monotonic()
        digest = sha256_file(path)
        target = targets.get(path.name, {})
        records.append({
            "filename": path.name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "expected_bytes": target.get("expected_mat_bytes"),
            "origin": target.get("origin"),
            "former_partition": target.get("former_partition"),
            "split_status": target.get("split_status"),
            "sha256": digest,
            "seconds": round(time.monotonic() - digest_started, 2),
        })
        print(f"[{index}/{len(files)}] {path.name} {digest}", flush=True)

    report = {
        "schema": "dronerfa-select12g-mat-sha256/v1",
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "g_original": str(g_original),
        "count": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "files": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {args.out}  ({len(records)} files, {report['bytes']} bytes, "
          f"{report['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
