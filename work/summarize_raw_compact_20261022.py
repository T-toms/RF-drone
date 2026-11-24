"""Compact grouped summary of the JSON sidecars under drone/raw.

Usage: python -I summarize_raw_compact_20261022.py <raw_dir>
"""
import collections
import json
import os
import sys


def walk(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in sorted(filenames):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                        yield dirpath, f, json.load(fh)
                except Exception:  # noqa: BLE001, S112
                    continue


def main(root):
    per_file = []
    for dirpath, fname, m in walk(root):
        cap, dec, ses = m.get("capture", {}), m.get("declared", {}), m.get("session", {})
        binp = os.path.join(dirpath, fname[:-5] + ".bin")
        per_file.append(
            {
                "top": os.path.relpath(dirpath, root).split(os.sep)[0],
                "campaign": ses.get("campaign"),
                "class": dec.get("class"),
                "model": "|".join(str(x) for x in dec.get("model", [])) or "-",
                "state": ",".join(str(x) for x in dec.get("state", [])) or "-",
                "links": ",".join(str(x) for x in dec.get("links", [])) or "-",
                "bw_MHz": ",".join("?" if x is None else f"{x / 1e6:g}" for x in dec.get("bw", [])) or "-",
                "fc_MHz": int((cap.get("center_freq_hz") or 0) / 1e6),
                "fs_MHz": int((cap.get("sample_rate_hz") or 0) / 1e6),
                "nsamp": cap.get("sample_count"),
                "rf_env": dec.get("rf_env"),
                "session": ses.get("id"),
                "size": os.path.getsize(binp) if os.path.exists(binp) else 0,
                "has_bin": os.path.exists(binp),
            }
        )

    print(f"json sidecars = {len(per_file)}, with .bin = {sum(r['has_bin'] for r in per_file)}, "
          f"total .bin = {sum(r['size'] for r in per_file) / 2**30:.1f} GB")
    print()
    print("--- by campaign / top-level dir ---")
    c = collections.Counter((r["campaign"], r["top"]) for r in per_file)
    for (camp, top), n in sorted(c.items()):
        print(f"  {str(camp):<20} {top:<30} files={n}")
    print()
    print("--- by class / model / state / band ---")
    c = collections.Counter(
        (r["class"], r["model"], r["state"], r["bw_MHz"], r["fc_MHz"], r["fs_MHz"]) for r in per_file
    )
    print(f"  {'class':<8} {'model':<14} {'state':<6} {'decl_bw':<10} {'fc':>5} {'fs':>4} {'files':>6} {'sessions':>8}")
    sess = collections.defaultdict(set)
    for r in per_file:
        sess[(r["class"], r["model"], r["state"], r["bw_MHz"], r["fc_MHz"], r["fs_MHz"])].add(r["session"])
    for k, n in sorted(c.items()):
        print(f"  {k[0]:<8} {k[1]:<14} {k[2]:<6} {k[3]:<10} {k[4]:>5} {k[5]:>4} {n:>6} {len(sess[k]):>8}")
    print()
    print("--- sample_count values ---", collections.Counter(r["nsamp"] for r in per_file))
    print("--- rf_env ---", collections.Counter(str(r["rf_env"]) for r in per_file))
    print("--- links ---", collections.Counter(r["links"] for r in per_file))


if __name__ == "__main__":
    main(sys.argv[1])
