"""Final check: all 12 heldout MAT files vs the candidate manifest; clean leftovers.

Downloads are done. This verifies byte sizes against the frozen candidate list and
removes only the one temporary RAR left behind by the pre-fix run (whose MAT is
already placed and validated). No analysis, no model work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
HELDOUT = Path(r"F:\DroneRFa数据集\heldout_20260917")
DOWNLOADS = HELDOUT / "downloads"
CANDIDATES = (
    ROOT / "dronerfa_baseline/outputs/noise_ab_replicates_20260917/candidate_heldout_files.json"
)

candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))["files"]
ok = 0
problems: list[str] = []
total = 0

print(f"{'file':<26} {'expected':>13} {'actual':>13}  match")
for item in candidates:
    name = Path(item["name"]).name
    expected = item["unpacked_size"]
    path = HELDOUT / name
    if not path.is_file():
        problems.append(f"{name}: 缺失")
        print(f"{name:<26} {expected:>13} {'MISSING':>13}  NO")
        continue
    actual = path.stat().st_size
    total += actual
    match = actual == expected
    ok += int(match)
    if not match:
        problems.append(f"{name}: 大小 {actual} != manifest {expected}")
    print(f"{name:<26} {expected:>13} {actual:>13}  {'OK' if match else 'MISMATCH'}")

print(f"\n大小匹配 {ok}/{len(candidates)}，合计 {total / 1024**3:.2f} GiB")

leftover = DOWNLOADS / "T0000_D00_S0100.mat.rar"
if leftover.is_file():
    size = leftover.stat().st_size
    mat = HELDOUT / "T0000_D00_S0100.mat"
    if mat.is_file() and mat.stat().st_size == 908097315:
        leftover.unlink()
        print(f"已清理修复前遗留的临时压缩包：{leftover.name}（{size / 1024**2:.1f} MiB），其MAT已落位校验通过")
    else:
        print(f"保留 {leftover.name}：对应MAT未通过校验")
else:
    print("无遗留临时压缩包")

remaining = sorted(
    p.name for p in DOWNLOADS.iterdir() if p.is_file() and p.suffix not in {".log", ".json"}
)
print("downloads 中剩余非日志文件：", remaining or "（无）")

records = ROOT / "dronerfa_baseline/outputs/heldout_preparation_20260917/download_records.json"
if records.is_file():
    data = json.loads(records.read_text(encoding="utf-8"))
    files = data.get("files", {})
    statuses: dict[str, int] = {}
    for record in files.values():
        status = record.get("status", "?")
        statuses[status] = statuses.get(status, 0) + 1
    print(f"download_records.json: {len(files)} 条记录，状态统计 {statuses}")

if problems:
    print("\n问题：")
    for problem in problems:
        print("  -", problem)
    sys.exit(1)
print("\n全部 12 个文件大小与候选清单严格一致。")
