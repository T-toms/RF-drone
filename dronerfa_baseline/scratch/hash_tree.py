"""Hash every file of a directory tree (rework R5 evidence freeze).

Writes a manifest with per-file size + SHA256 and a single combined digest, so two
snapshots can be compared without keeping the trees themselves.

Usage::

    python scratch/hash_tree.py <directory> <output.json>
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root = Path(argv[0])
    out = Path(argv[1])
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    entries = []
    combined = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        entries.append({"path": relative, "bytes": size, "sha256": digest})
        combined.update(relative.encode("utf-8"))
        combined.update(digest.encode("ascii"))
        combined.update(str(size).encode("ascii"))
    payload = {
        "root": str(root),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "combined_sha256": combined.hexdigest(),
        "files": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "root": payload["root"],
                "file_count": payload["file_count"],
                "total_bytes": payload["total_bytes"],
                "combined_sha256": payload["combined_sha256"],
                "output": str(out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
