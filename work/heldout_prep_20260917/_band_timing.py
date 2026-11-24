"""Time the activity-band analysis on a single file to validate the read path."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline")

import check_heldout_activity_band as band  # noqa: E402

path = Path(r"F:\DroneRFa数据集\heldout_20260917\T0000_D00_S0100.mat")
started = time.monotonic()
record = band.analyse_file(path)
elapsed = time.monotonic() - started
print(f"文件：{record['filename']}  {record['source_samples']:,} 样本  {record['complete_blocks']} 块")
print(f"耗时：{elapsed:.1f} 秒")
for window in record["windows"]:
    rf0 = window["RF0"]
    rf1 = window["RF1"]
    print(
        f"  {window['label']:<28} start={window['start']:>11,}"
        f"  RF0 act={rf0['active']} peak_off={rf0['peak_freq_offset_hz']/1e6:+.2f}MHz"
        f"  RF1 act={rf1['active']} peak_off={rf1['peak_freq_offset_hz']/1e6:+.2f}MHz"
    )
print(f"\n推算 24 文件总耗时约 {elapsed * 24 / 60:.1f} 分钟")
