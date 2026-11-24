"""Check what metadata the heldout MAT containers actually carry."""
from __future__ import annotations

from pathlib import Path

import h5py

ROOT = Path(r"F:\DroneRFa数据集\heldout_20260917")

for path in sorted(ROOT.glob("*.mat"))[:3]:
    print("=" * 70)
    print(path.name)
    with h5py.File(path, "r") as handle:
        print("  root attrs:", dict(handle.attrs))
        print("  keys:", sorted(handle.keys()))
        for key in sorted(handle.keys()):
            item = handle[key]
            if isinstance(item, h5py.Dataset):
                print(
                    f"    {key}: shape={item.shape} dtype={item.dtype} "
                    f"attrs={dict(item.attrs)} compression={item.compression}"
                )
            else:
                print(f"    {key}: group attrs={dict(item.attrs)}")
        for key in sorted(handle.keys()):
            item = handle[key]
            if isinstance(item, h5py.Dataset) and key not in (
                "RF0_I",
                "RF0_Q",
                "RF1_I",
                "RF1_Q",
            ):
                print(f"    other dataset {key}: {item[()]!r}")
