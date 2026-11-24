"""Publish the reviewed 104-row manifest to G: with a verified legacy backup."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "work/sel12g/dataset_manifest_candidate_104_20260925.csv"
BACKUP = ROOT / "work/sel12g/dataset_manifest_legacy24_backup_20260925.csv"
TARGET = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
TEMP = TARGET.with_name("dataset_manifest.csv.select12g-publishing")
EXPECTED_CANDIDATE = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"
EXPECTED_LEGACY = "009e27e62a7232b19825f0121812ae881a4329fcddad669278f28956c78e3246"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if digest(CANDIDATE) != EXPECTED_CANDIDATE:
        raise RuntimeError("Candidate hash changed; refusing publication")
    if not TARGET.is_file() or digest(TARGET) != EXPECTED_LEGACY:
        raise RuntimeError("G legacy manifest changed; refusing publication")
    if TEMP.exists():
        raise RuntimeError("G temporary manifest already exists; inspect before retry")
    if BACKUP.exists() and digest(BACKUP) != EXPECTED_LEGACY:
        raise RuntimeError("Legacy backup differs; refusing publication")
    if not BACKUP.exists():
        with TARGET.open("rb") as source, BACKUP.open("xb") as backup:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                backup.write(chunk)
            backup.flush()
            os.fsync(backup.fileno())
    if digest(BACKUP) != EXPECTED_LEGACY:
        raise RuntimeError("Legacy backup verification failed")
    try:
        with CANDIDATE.open("rb") as source, TEMP.open("xb") as staged:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                staged.write(chunk)
            staged.flush()
            os.fsync(staged.fileno())
        if digest(TEMP) != EXPECTED_CANDIDATE:
            raise RuntimeError("G staged manifest hash mismatch")
        os.replace(TEMP, TARGET)
    finally:
        TEMP.unlink(missing_ok=True)
    if digest(TARGET) != EXPECTED_CANDIDATE:
        raise RuntimeError("Published G manifest hash mismatch")
    print(f"published={TARGET}")
    print("rows=104 sha256=" + EXPECTED_CANDIDATE)
    print(f"legacy_backup={BACKUP}")


if __name__ == "__main__":
    main()
