"""Find a Python runtime that already has h5py + numpy + requests."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CANDIDATES = [
    r"D:\anaconda\python.exe",
    r"D:\anaconda3\python.exe",
    r"C:\ProgramData\anaconda3\python.exe",
    r"C:\ProgramData\Anaconda3\python.exe",
    r"C:\Users\Administrator\anaconda3\python.exe",
    r"C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe",
    r"C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe",
    r"C:\Python313\python.exe",
    r"C:\Python312\python.exe",
    r"C:\Python311\python.exe",
]
PROBE = (
    "import importlib.util as u, sys;"
    "mods=['h5py','numpy','requests'];"
    "print(sys.version.split()[0], sys.executable, {m: u.find_spec(m) is not None for m in mods})"
)

print("current venv:", sys.executable)
for candidate in CANDIDATES:
    path = Path(candidate)
    if not path.is_file():
        print(f"{candidate}: absent")
        continue
    try:
        result = subprocess.run(
            [str(path), "-c", PROBE], capture_output=True, timeout=120, check=False
        )
    except Exception as exc:  # noqa: BLE001 - probe
        print(f"{candidate}: run failed {type(exc).__name__}: {exc}")
        continue
    out = result.stdout.decode("utf-8", errors="replace").strip()
    err = result.stderr.decode("utf-8", errors="replace").strip()[:160]
    print(f"{candidate}: rc={result.returncode} {out} {err}")

# Is there a requests package anywhere already on disk we could point at?
import glob

print("\nrequests packages found on disk:")
for pattern in (
    r"D:\anaconda\Lib\site-packages\requests\__init__.py",
    r"C:\ProgramData\anaconda3\Lib\site-packages\requests\__init__.py",
    r"C:\Users\Administrator\AppData\Roaming\Python\Python3*\site-packages\requests\__init__.py",
    r"C:\Users\Administrator\AppData\Local\Programs\Python\Python3*\Lib\site-packages\requests\__init__.py",
):
    for hit in glob.glob(pattern):
        print("  ", hit)
