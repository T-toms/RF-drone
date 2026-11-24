"""Summarise the stage-1 clean grid (outputs/stage1_clean_20261013).

For every setting and seed, at the validation-chosen threshold, writes:
  grid_summary.csv   setting, seed, background false alarms, misses, worst
                     class excluding Matrice 300, Matrice 300 miss rate
  grid_per_class.csv setting, seed, class, windows, missed (test)
and prints mean / worst over seeds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from split_20261013 import NAMES

ROOT = Path(__file__).resolve().parent / "outputs" / "stage1_clean_newsplit_20261013"
M300 = "T1011"


def main() -> None:
    rows, per_class = [], []
    for setting in sorted(p.name for p in ROOT.iterdir() if p.is_dir()):
        for seed_dir in sorted((ROOT / setting).glob("seed_*")):
            f = seed_dir / "summary.json"
            if not f.exists():
                continue
            s = json.loads(f.read_text(encoding="utf-8"))
            for split_key, label in (("val", "val"), ("test", "test")):
                m = s[split_key]
                pc = m["per_class"]
                others = {k: v["missed"] / v["windows"] for k, v in pc.items() if k != M300}
                worst = max(others, key=others.get)
                rows.append({
                    "setting": setting, "seed": seed_dir.name[5:], "split": label, "epoch": s["selected_epoch"],
                    "bg_fa": m["false_alarms"], "bg_windows": m["background_windows"],
                    "fa_rate_pct": round(100 * m["false_alarm_rate"], 2),
                    "missed_excl_m300": sum(v["missed"] for k, v in pc.items() if k != M300),
                    "drone_windows_excl_m300": sum(v["windows"] for k, v in pc.items() if k != M300),
                    "worst_excl_m300": NAMES[worst], "worst_excl_m300_pct": round(100 * others[worst], 2),
                    "m300_miss_pct": round(100 * pc[M300]["missed"] / pc[M300]["windows"], 1),
                    "classes_over_1pct": sum(1 for v in others.values() if v > 0.01),
                })
                if label == "test":
                    for k, v in pc.items():
                        per_class.append({"setting": setting, "seed": seed_dir.name[5:], "class": NAMES[k],
                                          "windows": v["windows"], "missed": v["missed"]})
    with (ROOT / "grid_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (ROOT / "grid_per_class.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0]))
        w.writeheader()
        w.writerows(per_class)
    print(f"{'setting':24s} split  n  FA%mean FA%max  miss%(excl M300) worst-class%max  >1%classes(max)  M300miss%mean")
    for setting in sorted({r["setting"] for r in rows}):
        for split in ("val", "test"):
            rs = [r for r in rows if r["setting"] == setting and r["split"] == split]
            fa = [r["fa_rate_pct"] for r in rs]
            ms = [100 * r["missed_excl_m300"] / r["drone_windows_excl_m300"] for r in rs]
            print(f"{setting:24s} {split:5s} {len(rs)}  {np.mean(fa):6.2f} {max(fa):6.2f}   {np.mean(ms):6.2f}"
                  f"           {max(r['worst_excl_m300_pct'] for r in rs):6.2f}"
                  f"          {max(r['classes_over_1pct'] for r in rs)}"
                  f"               {np.mean([r['m300_miss_pct'] for r in rs]):5.1f}")


if __name__ == "__main__":
    main()
