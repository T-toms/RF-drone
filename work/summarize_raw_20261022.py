"""Aggregate the JSON sidecars under drone/raw into a per-session table.

Usage: python -I summarize_raw_20261022.py <raw_dir>
"""
import collections
import json
import os
import sys


def main(root):
    rows = []
    for dirpath, _dirnames, filenames in os.walk(root):
        js = sorted(f for f in filenames if f.endswith(".json"))
        bins = [f for f in filenames if f.endswith(".bin")]
        if not js:
            continue
        metas = []
        for f in js:
            try:
                with open(os.path.join(dirpath, f), encoding="utf-8") as fh:
                    metas.append(json.load(fh))
            except Exception as exc:  # noqa: BLE001
                print("  !! bad json", os.path.join(dirpath, f), type(exc).__name__, exc)
        if not metas:
            continue
        m0 = metas[0]
        cap, dec, ses = m0.get("capture", {}), m0.get("declared", {}), m0.get("session", {})
        fcs = sorted({mm.get("capture", {}).get("center_freq_hz") for mm in metas})
        states = sorted({str(s) for mm in metas for s in mm.get("declared", {}).get("state", [])})
        links = sorted({str(s) for mm in metas for s in mm.get("declared", {}).get("links", [])})
        bws = sorted({str(s) for mm in metas for s in mm.get("declared", {}).get("bw", [])})
        size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in bins)
        rows.append(
            {
                "rel": os.path.relpath(dirpath, root),
                "campaign": ses.get("campaign"),
                "class": dec.get("class"),
                "model": "|".join(str(x) for x in dec.get("model", [])),
                "state": ",".join(states),
                "links": ",".join(links),
                "bw": ",".join(bws),
                "rf_env": dec.get("rf_env"),
                "fs": cap.get("sample_rate_hz"),
                "fc": ",".join(f"{(x or 0) / 1e6:.0f}" for x in fcs),
                "dtype": cap.get("datatype"),
                "nsamp": cap.get("sample_count"),
                "njson": len(js),
                "nbin": len(bins),
                "GB": round(size / 2**30, 2),
                "paired_bg": ses.get("paired_bg"),
                "notes": m0.get("notes"),
                "t0": cap.get("t_start_utc"),
            }
        )

    rows.sort(key=lambda r: (r["rel"]))
    hdr = ["rel", "class", "model", "state", "links", "bw", "rf_env", "fs", "fc", "nbin", "GB", "notes"]
    widths = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in hdr}
    print(" ".join(h.ljust(widths[h]) for h in hdr))
    print(" ".join("-" * widths[h] for h in hdr))
    for r in rows:
        print(" ".join(str(r[h]).ljust(widths[h]) for h in hdr))

    print()
    print(f"sessions={len(rows)}  bin_files={sum(r['nbin'] for r in rows)}  total={sum(r['GB'] for r in rows):.1f} GB")
    for key in ("class", "model", "state", "dtype", "fs", "campaign"):
        c = collections.Counter(str(r[key]) for r in rows)
        print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main(sys.argv[1])
