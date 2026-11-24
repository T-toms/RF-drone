"""One-off environment probe for the heldout preparation task (read-only)."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)

for name in ("h5py", "numpy", "requests", "urllib3", "torch", "scipy", "matplotlib", "certifi"):
    spec = importlib.util.find_spec(name)
    if spec is None:
        print(f"module {name}: MISSING")
        continue
    module = __import__(name)
    print(f"module {name}: {getattr(module, '__version__', 'unknown')}")

for drive in ("C:\\", "D:\\"):
    try:
        total, used, free = shutil.disk_usage(drive)
    except OSError as exc:
        print(f"disk {drive}: unavailable ({exc})")
        continue
    print(
        f"disk {drive}: total {total / 1024**3:.2f} GiB  used {used / 1024**3:.2f} GiB  "
        f"free {free / 1024**3:.2f} GiB"
    )

root = Path(__file__).resolve().parents[2]
print("workspace:", root)
print("workspace writable:", root.is_dir())
