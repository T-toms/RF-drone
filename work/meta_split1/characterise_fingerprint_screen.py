# -*- coding: utf-8 -*-
"""META-SPLIT-1: characterise the pre-registered spectral screen and the
secondary block-RMS profiles, so the report can state honestly how
discriminative each descriptor actually is."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

nd = json.load(open(os.path.join(HERE, "near_duplicate_screen.json"), encoding="utf-8"))
fp = json.load(open(os.path.join(HERE, "fragment_fingerprints_104.json"), encoding="utf-8"))
bp = json.load(open(os.path.join(HERE, "block_rms_profiles_104.json"), encoding="utf-8"))

# ---- rebuild the full pairwise L1 matrix per cell from stored descriptors ----
recs = fp["records"]
frags = []
for r in recs:
    for fr in r["fragments"]:
        frags.append((r["file_name"], r["class_id"], fr["channel"], fr["offset_index"],
                      fr["spectral_shape_64"]))

cells = {}
for name, cid, ch, oi, shape in frags:
    cells.setdefault((ch, oi), []).append((name, cid, np.asarray(shape, dtype=float)))

stats = {}
all_d = []
for cell, items in sorted(cells.items()):
    v = np.asarray([it[2] for it in items])
    n = v.shape[0]
    d = np.abs(v[:, None, :] - v[None, :, :]).sum(axis=2)
    iu = np.triu_indices(n, k=1)
    dists = d[iu]
    all_d.append(dists)
    same = np.array([items[iu[0][i]][1] == items[iu[1][i]][1] for i in range(len(dists))])
    j = int(np.argmin(dists))
    stats["%s|%d" % cell] = {
        "n_pairs": int(len(dists)),
        "min": round(float(dists.min()), 6),
        "min_pair": [items[iu[0][j]][0], items[iu[1][j]][0]],
        "min_pair_same_class": bool(same[j]),
        "min_same_class": round(float(dists[same].min()), 6) if same.any() else None,
        "min_cross_class": round(float(dists[~same].min()), 6) if (~same).any() else None,
        "median": round(float(np.median(dists)), 6),
        "p01": round(float(np.percentile(dists, 1)), 6),
        "frac_below_0.05": round(float((dists < 0.05).mean()), 4),
        "frac_below_0.01": round(float((dists < 0.01).mean()), 4),
    }
all_d = np.concatenate(all_d)

# ---- secondary: block-RMS profile similarity, same channel, all alignments ----
prof = {}
for r in bp["records"]:
    for ch in ("RF0_I", "RF0_Q"):
        p = r["profile"].get(ch)
        if p:
            prof.setdefault(ch, []).append((r["file_name"], r["class_id"], np.asarray(p["rms"])))

def best_r(a, b, max_shift=2):
    best = -2.0
    best_shift = None
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            x, y = a[s:], b[:len(b) - s] if s else b
        else:
            x, y = a[:len(a) + s], b[-s:]
        m = min(len(x), len(y))
        if m < 6:
            continue
        x, y = x[:m], y[:m]
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        r = float(np.corrcoef(x, y)[0, 1])
        if r > best:
            best, best_shift = r, s
    return best, best_shift

pair_rows = []
for ch, items in prof.items():
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            r, s = best_r(items[i][2], items[j][2])
            pair_rows.append({"channel": ch, "a": items[i][0], "b": items[j][0],
                              "a_class": items[i][1], "b_class": items[j][1],
                              "best_r": round(r, 6), "shift_blocks": s,
                              "same_class": items[i][1] == items[j][1]})
pair_rows.sort(key=lambda x: -x["best_r"])

out = {
    "spectral_screen_characterisation": {
        "cells": stats,
        "pairs_total": int(all_d.size),
        "overall": {
            "min": round(float(all_d.min()), 6),
            "p01": round(float(np.percentile(all_d, 1)), 6),
            "p05": round(float(np.percentile(all_d, 5)), 6),
            "median": round(float(np.median(all_d)), 6),
            "max": round(float(all_d.max()), 6),
            "frac_below_preregistered_0.05": round(float((all_d < 0.05).mean()), 4),
            "frac_below_0.01": round(float((all_d < 0.01).mean()), 4),
        },
        "verdict": (
            "The pre-registered L1<0.05 flag fires on 42.6% of all fragment pairs, so the 64-bin "
            "normalised spectral-shape descriptor is NOT discriminative at that threshold and cannot "
            "by itself support or refute a near-duplicate claim. The operative evidence from the frozen "
            "screen is the exact-digest result (0 duplicate groups at both float64 and float32)."
        ),
    },
    "secondary_block_rms_profile_similarity": {
        "status": "exploratory post-hoc, computed from the probes already collected; not part of the frozen screen",
        "n_pairs": len(pair_rows),
        "top20": pair_rows[:20],
        "pairs_with_best_r_ge_0.99": sum(1 for p in pair_rows if p["best_r"] >= 0.99),
        "pairs_with_best_r_ge_0.999": sum(1 for p in pair_rows if p["best_r"] >= 0.999),
        "same_class_top20": sum(1 for p in pair_rows[:20] if p["same_class"]),
    },
}
p = os.path.join(HERE, "fingerprint_screen_characterisation.json")
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out["spectral_screen_characterisation"]["overall"], ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in list(out["spectral_screen_characterisation"]["cells"].items())[:4]},
                 ensure_ascii=False, indent=1))
print(json.dumps(out["secondary_block_rms_profile_similarity"], ensure_ascii=False, indent=1)[:2500])
print("wrote", p)
