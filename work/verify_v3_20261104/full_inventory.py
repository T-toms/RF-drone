# -*- coding: utf-8 -*-
"""Full inventory of G:/drone: classify every capture as drone/background,
compute duration, map datasets_v3 files to raw .bin captures, report per (label, band-relevance)."""
import collections, csv, glob, json, os, re
import scipy.io as sio

ROOT = r"G:/drone"
OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- raw/ ----------
raw = []
for j in glob.glob(os.path.join(ROOT, "raw", "**", "*.json"), recursive=True):
    d = json.load(open(j, encoding="utf-8"))
    c = d["capture"]; dec = d.get("declared", {}); ses = d.get("session", {})
    b = j[:-5] + ".bin"
    raw.append(dict(
        json=os.path.relpath(j, ROOT), bin_exists=os.path.exists(b),
        bin_bytes=os.path.getsize(b) if os.path.exists(b) else 0,
        cls=dec.get("class"), model=";".join(dec.get("model") or []), state=";".join(dec.get("state") or []),
        fs=c["sample_rate_hz"], fc=c["center_freq_hz"], n=c["sample_count"], sha=c["sha256"],
        session=ses.get("id"), campaign=ses.get("campaign"), rf_env=dec.get("rf_env"),
        gain=d.get("receiver", {}).get("gain_db")))
uniq = {}
for r in raw:
    uniq.setdefault(r["sha"], r)
print(f"raw json={len(raw)} unique captures={len(uniq)}  (duplicates={len(raw)-len(uniq)})")
print("raw class counts (unique):", collections.Counter(r["cls"] for r in uniq.values()))
ms = lambda r: r["n"] / r["fs"] * 1e3
agg = collections.defaultdict(lambda: [0, 0.0])
for r in uniq.values():
    k = (r["cls"], r["fs"] / 1e6, r["fc"] / 1e6 if r["fc"] / 1e6 in (2440, 2450, 2500, 2600, 5800) else "sweep/other")
    agg[k][0] += 1; agg[k][1] += ms(r)
print("\nraw unique captures: (class, fs MHz, fc MHz) -> n, total seconds")
for k in sorted(agg, key=str):
    print("  ", k, agg[k][0], round(agg[k][1] / 1e3, 2))
print("raw .mat files:", [os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "raw", "**", "*.mat"), recursive=True)])

# ---------- datasets_v3 -> raw mapping ----------
v3 = json.load(open(os.path.join(OUT, "v3_meta.json"), encoding="utf-8"))
key2raw = {}
for r in uniq.values():
    # session id like 20260709-021044_ambient ; file iq_..._000000.bin
    m = re.search(r"_(\d{6})\.json$", r["json"])
    key2raw[(r["session"].split("-93ce")[0] if r["session"] else None, int(m.group(1)) if m else None)] = r
v3cap = collections.defaultdict(list)
for f in v3:
    m = re.search(r"(\d{8}-\d{6})_(\w+?)_iq_.*_(\d{6})_s(\d{4})_new\.mat$", f["name"])
    if m:
        v3cap[(m.group(1) + "_" + m.group(2), int(m.group(3)))].append(f)
    else:
        v3cap[("noise:" + f["name"][:16], f["name"])].append(f)
matched = sum(1 for k in v3cap if k in key2raw)
print(f"\ndatasets_v3 captures (session, NN) = {len(v3cap)}; matched to a raw capture = {matched}")
v3_ms = collections.Counter(); v3_lab = collections.Counter()
for k, fs in v3cap.items():
    lab = "background" if fs[0]["sub"] == "background" else "drone"
    v3_lab[lab] += 1; v3_ms[lab] += sum(x["n"] for x in fs) / 100e6 * 1e3
print("datasets_v3 by label: captures", dict(v3_lab), " seconds", {k: round(v / 1e3, 2) for k, v in v3_ms.items()})
unmatched_raw = [r for r in uniq.values() if (r["session"].split("-93ce")[0] if r["session"] else None, int(re.search(r"_(\d{6})\.json$", r["json"]).group(1))) not in v3cap]
print("raw captures NOT present in datasets_v3:", len(unmatched_raw), collections.Counter((r["cls"], r["fc"] / 1e6) for r in unmatched_raw).most_common(12))

# ---------- data/ ----------
man = list(csv.DictReader(open(os.path.join(os.path.dirname(OUT), "audit_drone_20261023", "data_preprocessing_manifest.csv"), encoding="utf-8-sig")))
nd = [m for m in man if m["duplicate"] == "False"]
print(f"\ndata/: files={len(man)} non-duplicate={len(nd)} -> seconds={len(nd)*0.05:.1f}; all drone (no ambient folders)")
print("data/ folder names containing bg/ambient/noise:", sorted({m['folder'] for m in man if re.search('ambient|bg|noise|background', m['folder'], re.I)}))

with open(os.path.join(OUT, "raw_inventory.csv"), "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(raw[0].keys())); w.writeheader(); w.writerows(raw)
