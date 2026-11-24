"""Confirm the venv's own key packages are untouched after the requests bootstrap."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline")

import download_heldout_files as heldout  # noqa: E402
import h5py  # noqa: E402
import numpy  # noqa: E402
import torch  # noqa: E402

print("requests version used:", heldout.REQUESTS_VERSION)
print("requests file:", sys.modules["requests"].__file__)
print("h5py:", h5py.__version__, h5py.__file__)
print("numpy:", numpy.__version__, numpy.__file__)
print("torch:", torch.__version__)
base_entry = str(Path(sys.base_prefix) / "Lib" / "site-packages")
print("base site-packages still on sys.path:", base_entry in sys.path)
print("base module path:", sys.base_prefix)
print("h5py is the venv copy:", "dronerfa_baseline" not in h5py.__file__ and ".venv" in h5py.__file__)
