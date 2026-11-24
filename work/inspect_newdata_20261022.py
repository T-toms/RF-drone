"""Inspect the structure of one lab-collected .mat file (MAT v7, scipy needed).

Usage: python -I inspect_newdata_20261022.py <file.mat> [more.mat ...]
"""
import sys

import numpy as np
from scipy.io import loadmat, whosmat


def describe(path):
    print("=" * 100)
    print(path)
    try:
        print("whosmat:", whosmat(path))
    except Exception as exc:  # noqa: BLE001
        print("whosmat failed:", type(exc).__name__, exc)
    md = loadmat(path, squeeze_me=False, struct_as_record=False)
    for key, val in md.items():
        if key.startswith("__"):
            print(f"  {key}: {val!r}"[:200])
            continue
        if isinstance(val, np.ndarray):
            print(f"  {key}: shape={val.shape} dtype={val.dtype}")
            flat = val.ravel()
            if flat.size and np.issubdtype(val.dtype, np.number):
                sl = flat[: min(flat.size, 1000)]
                print(
                    f"      first5={flat[:5]}  min={np.min(flat):.6g} max={np.max(flat):.6g} "
                    f"mean={np.mean(sl):.6g} rms={np.sqrt(np.mean(np.abs(sl.astype(np.float64)) ** 2)):.6g}"
                )
            elif flat.size:
                print(f"      value={flat[:3]}")
        else:
            print(f"  {key}: {type(val)} {val!r}"[:300])


if __name__ == "__main__":
    for p in sys.argv[1:]:
        describe(p)
