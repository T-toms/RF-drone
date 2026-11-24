# -*- coding: utf-8 -*-
"""Verify datasets_v3 metadata vs filenames, v3<->raw correspondence/overlap, raw duplicates.
Read-only on G:/drone."""
import glob, json, os, re, sys, collections
import numpy as np, scipy.io as sio
from scipy.signal import resample_poly, fftconvolve

ROOT = r"G:/drone"
V3 = os.path.join(ROOT, "datasets_v3")
RAW = os.path.join(ROOT, "raw")
OUT = os.path.dirname(os.path.abspath(__file__))
step = sys.argv[1] if len(sys.argv) > 1 else "all"

def log(*a):
    print(*a, flush=True)

# ---------- A: metadata of every v3 file vs filename ----------
if step in ("all", "A"):
    rows = []
    pat = re.compile(r"_iq_([\d.]+)M_([\d.]+)M_(\d+)dB_")
    for p in sorted(glob.glob(os.path.join(V3, "**", "*.mat"), recursive=True)):
        m = sio.loadmat(p, variable_names=["fs", "centfreq"])
        w = sio.whosmat(p)
        n = [s for (nm, s, t) in w if nm == "IQ"][0][0]
        g = pat.search(os.path.basename(p))
        rows.append(dict(sub=os.path.relpath(os.path.dirname(p), V3), name=os.path.basename(p),
                         fs_file=float(m["fs"].ravel()[0]), fc_file=float(m["centfreq"].ravel()[0]),
                         fs_name=float(g.group(1)) * 1e6 if g else None,
                         fc_name=float(g.group(2)) * 1e6 if g else None, n=int(n)))
    json.dump(rows, open(os.path.join(OUT, "v3_meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    c = collections.Counter((r["fs_file"], r["fs_name"], r["n"]) for r in rows)
    log("fs_file, fs_name, n_samples -> count:", dict(c))
    bad = [r for r in rows if r["fc_name"] is None or abs(r["fc_file"] - r["fc_name"]) > 1]
    log("fc mismatch count:", len(bad))
    for r in bad[:20]:
        log("  ", r["sub"], r["name"], r["fc_file"], r["fc_name"])
    pre = collections.Counter(re.sub(r"_\d{6}_s\d{4}_new\.mat$", "", r["name"]) for r in rows if r["sub"] == "background")
    log("background session groups:")
    for k, v in sorted(pre.items()):
        log("  ", v, k)

# ---------- B: which fs is right + overlap between s0001/s0002 ----------
def load_raw(binp):
    a = np.fromfile(binp, dtype="<i2").astype(np.float32)
    return (a[0::2] + 1j * a[1::2]).astype(np.complex64)

def best_match(y, probe):
    c = fftconvolve(y, np.conj(probe[::-1]), mode="valid")
    e = np.cumsum(np.abs(y.astype(np.complex128)) ** 2)
    e = np.concatenate([[0], e])
    L = len(probe)
    win = e[L:] - e[:-L]
    nc = np.abs(c) / (np.sqrt(win[:len(c)]) * np.linalg.norm(probe) + 1e-30)
    k = int(np.argmax(nc))
    return k, float(nc[k])

if step in ("all", "B"):
    cases = [
        ("2026-07-09_urban/ambient/20260709-020956_ambient", "2026-07-09_urban_20260709-020956_ambient", "background", "000000"),
        ("2026-07-09_urban/ambient/20260709-021044_ambient", "2026-07-09_urban_20260709-021044_ambient", "background", "000000"),
        ("2026-07-09_urban/20260709-040508_mavic2", "2026-07-09_urban_20260709-040508_mavic2", "consumer_drone/mavic2", "000000"),
    ]
    PL = 100_000
    for rawdir, v3pre, sub, idx in cases:
        binp = glob.glob(os.path.join(RAW, rawdir, f"*_{idx}.bin"))[0]
        y250 = load_raw(binp)
        y100 = resample_poly(y250, 2, 5).astype(np.complex64)
        log(f"\n[{v3pre}] raw {os.path.basename(binp)} n={len(y250)} ({len(y250)/250e6*1e3:.1f} ms @250M)")
        offs = {}
        for s in ("s0001", "s0002"):
            vp = glob.glob(os.path.join(V3, sub, f"{v3pre}_*_{idx}_{s}_new.mat"))
            if not vp:
                log("  missing", s); continue
            x = sio.loadmat(vp[0], variable_names=["IQ"])["IQ"].ravel().astype(np.complex64)
            for hyp, y in (("100M", y100), ("250M", y250)):
                k, r = best_match(y, x[:PL])
                log(f"  {s} hyp fs={hyp}: start={k} ({k/(100e6 if hyp=='100M' else 250e6)*1e3:.2f} ms) corr={r:.4f}")
                if hyp == "100M":
                    offs[s] = (k, r, len(x))
            # end-of-file probe to confirm whole segment is a contiguous slice
            k2, r2 = best_match(y100, x[-PL:])
            log(f"  {s} tail probe @100M: start={k2} expected={offs[s][0]+len(x)-PL} corr={r2:.4f}")
        if "s0001" in offs and "s0002" in offs:
            a, b = offs["s0001"], offs["s0002"]
            log(f"  s0001 [{a[0]}, {a[0]+a[2]}), s0002 [{b[0]}, {b[0]+b[2]}) -> overlap samples = {max(0, a[0]+a[2]-b[0])}")

# ---------- C: raw duplicates via JSON sha256 ----------
if step in ("all", "C"):
    by = collections.defaultdict(list)
    for j in glob.glob(os.path.join(RAW, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(j, encoding="utf-8"))
            by[d["capture"]["sha256"]].append(os.path.relpath(j, RAW))
        except Exception as e:
            log("json err", j, e)
    dup = {k: v for k, v in by.items() if len(v) > 1}
    log("\nraw json total", sum(len(v) for v in by.values()), "unique sha", len(by), "dup groups", len(dup))
    dd = collections.Counter(tuple(sorted(os.path.dirname(x) for x in v)) for v in dup.values())
    for k, v in dd.items():
        log("  ", v, "x", k)
    ce = glob.glob(os.path.join(RAW, "clean_export", "**", "*"), recursive=True)
    log("clean_export ext:", collections.Counter(os.path.splitext(x)[1] for x in ce if os.path.isfile(x)))
    sweep = glob.glob(os.path.join(RAW, "2026-07-09_urban", "ambient", "2026-07-16_lab", "**", "*.json"), recursive=True)
    log("07-16 sweep json:", len(sweep))
    fcs = collections.Counter()
    for j in sweep:
        d = json.load(open(j, encoding="utf-8"))
        fcs[(d["capture"]["sample_rate_hz"], d["capture"]["center_freq_hz"])] += 1
    log("  (fs, fc) distinct:", len(fcs), "fc range MHz:", min(f for _, f in fcs) / 1e6, max(f for _, f in fcs) / 1e6)
    if sweep:
        d = json.load(open(sorted(sweep)[0], encoding="utf-8"))
        log("  sample json:", json.dumps(d, ensure_ascii=False)[:900])
