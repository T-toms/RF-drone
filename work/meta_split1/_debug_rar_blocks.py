# -*- coding: utf-8 -*-
"""META-SPLIT-1 debug: dump the first RAR5 blocks of the pilot archive and of the
two leftover .part downloads."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scan_rar_headers import (RAR5_SIG, read_block_header, dump_partial,  # noqa: E402
                              WORKSPACE)

TARGETS = [
    os.path.join(WORKSPACE, "DroneRFa_pilot.rar"),
    r"G:\DroneRFa数据集\download_records_12models_20260922\downloads\T0011_D00_S0111.mat.rar.part",
    r"G:\DroneRFa数据集\download_records_12models_20260922\downloads\T1000_D00_S0111.mat.rar.part",
]

import json  # noqa: E402

for path in TARGETS:
    print("=" * 30)
    print(path)
    if not os.path.exists(path):
        print("  MISSING")
        continue
    size = os.path.getsize(path)
    print("  size", size)
    with open(path, "rb") as fh:
        sig = fh.read(8)
        print("  sig ok", sig == RAR5_SIG)
        offset = 8
        for i in range(6):
            if offset + 8 >= size:
                print("  stop: offset beyond file", offset)
                break
            try:
                info, head_size, data_size = read_block_header(fh, offset)
            except Exception as exc:
                print("  parse error at", offset, repr(exc))
                break
            brief = {k: v for k, v in info.items() if k != "extra_records"}
            print("  ", json.dumps(brief, ensure_ascii=False))
            if info.get("extra_records"):
                print("      extra:", json.dumps(info["extra_records"], ensure_ascii=False))
            if info["head_type"] == 5:
                break
            offset += 4 + head_size + data_size
