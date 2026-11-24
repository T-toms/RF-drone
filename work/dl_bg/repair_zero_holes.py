"""Find and patch all-zero 4 KiB pages in a selective DroneRFa RAR (or .rar.part).

Layout of the local archive (download_dronerfa_subset.py): 16 header bytes,
then the selected entries' RAR blocks back to back in the order of
<archive>.selection.json (or the selection given by --names for a .part),
then the end header. A zero page at local offset p inside entry k maps to
remote offset block_offset[k] + (p - local_start[k]). Each run of zero pages
is re-fetched with curl (plus one page either side) until two downloads
agree, and written back in place (genuine zero pages are simply rewritten).

Usage:
  python repair_zero_holes.py ARCHIVE.rar [--names a.mat,b.mat ...] [--only a.mat,...] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
URL = "https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d"
HEADER = 16
PAGE = 4096
READ = 64 * 1024 * 1024


def fetch(start: int, end: int) -> bytes:
    """Fetch a range until two downloads agree (genuine data may contain zero pages)."""
    seen = []
    for _ in range(10):
        r = subprocess.run(["curl", "-sS", "-L", "--fail", "--max-time", "300", "-r", f"{start}-{end}", "-o", "-", URL],
                           capture_output=True, check=False)
        data = r.stdout
        if r.returncode != 0 or len(data) != end - start + 1:
            continue
        if any(data == s for s in seen):
            return data
        seen.append(data)
    raise RuntimeError(f"cannot fetch a consistent range {start}-{end}")


def zero_pages(buf: bytes, base: int) -> np.ndarray:
    """Absolute offsets of all-zero pages fully inside buf (page grid relative to base)."""
    a = np.frombuffer(buf, dtype=np.uint8)
    lead = (-base) % PAGE
    a = a[lead:]
    n = len(a) // PAGE * PAGE
    pages = a[:n].reshape(-1, PAGE)
    return base + lead + np.flatnonzero(~pages.any(axis=1)) * PAGE


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--names", help="comma-separated entry names, required for a .part")
    p.add_argument("--only", help="comma-separated entry names to repair (default: all)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    manifest = {e["name"].split("/")[-1]: e for e in
                json.loads((ROOT / "dronerfa_manifest.json").read_text(encoding="utf-8"))["entries"]}
    if args.names:
        names = args.names.split(",")
    else:
        sel = json.loads(args.archive.with_name(args.archive.name + ".selection.json").read_text(encoding="utf-8"))
        names = [e["name"].split("/")[-1] for e in sel["entries"]]
    spans = []  # (local_start, local_end, remote_start, name)
    pos = HEADER
    for n in names:
        e = manifest[n]
        spans.append((pos, pos + e["block_size"], e["block_offset"], n))
        pos += e["block_size"]

    size = args.archive.stat().st_size
    holes = []
    with args.archive.open("rb") as f:
        only = set(args.only.split(",")) if args.only else None
        for ls, le, rs, n in spans:
            if only and n not in only:
                continue
            off = ls
            while off < min(le, size):
                f.seek(off)
                buf = f.read(min(READ, min(le, size) - off))
                for z in zero_pages(buf, off):
                    holes.append((int(z), ls, rs, n))
                off += len(buf)
    # merge consecutive zero pages into runs
    runs = []
    for z, ls, rs, n in holes:
        if runs and runs[-1][3] == n and z == runs[-1][1]:
            runs[-1][1] = z + PAGE
        else:
            runs.append([z, z + PAGE, (ls, rs), n])
    print(f"{args.archive.name}: {len(holes)} zero pages in {len(runs)} runs")
    by_file = {}
    for a, b, _, n in runs:
        by_file[n] = by_file.get(n, 0) + (b - a)
    for n, v in by_file.items():
        print(f"  {n}: {v / 2**20:.2f} MiB zero")
    if args.dry_run or not runs:
        return 0
    with args.archive.open("r+b") as f:
        for a, b, (ls, rs), n in runs:
            le = next(s[1] for s in spans if s[0] == ls)
            a2, b2 = max(ls, a - PAGE), min(le, b + PAGE)
            data = fetch(rs + (a2 - ls), rs + (b2 - ls) - 1)
            f.seek(a2)
            f.write(data)
        f.flush()
    print("patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
