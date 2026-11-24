# -*- coding: utf-8 -*-
"""META-SPLIT-1 / step 2: bounded, pre-registered fragment fingerprint and
near-duplicate screen over the 104 G-drive MAT files.

The plan is frozen in fingerprint-preregistration-20260925.json and must not be
changed after the first run.  Only short fixed fragments are read; no full
waveform is scanned.

Outputs (work/meta_split1/):
  fragment_fingerprints_104.json   per-file, per-fragment digests and descriptors
  near_duplicate_screen.json       cross-file comparison result
  block_rms_profiles_104.json      secondary 10M-block RF0 activity profile
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- frozen plan (mirrors fingerprint-preregistration-20260925.json) --------
OFFSETS = (0, 20_000_000, 50_000_000, 90_000_000)
FRAG_LEN = 65_536
CHANNELS = ("RF0_I", "RF0_Q", "RF1_I", "RF1_Q")
SPECTRAL_BINS = 64
SPECTRAL_FLAG_L1 = 0.05
BLOCK = 10_000_000
BLOCK_PROBE = 4096


def sha256_of(arr: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=dtype).tobytes()).hexdigest()


def spectral_shape(x: np.ndarray) -> list:
    y = np.asarray(x, dtype=np.float64)
    y = y - y.mean()
    mag = np.abs(np.fft.rfft(y))
    if mag.size < SPECTRAL_BINS:
        return []
    edges = np.linspace(0, mag.size, SPECTRAL_BINS + 1).astype(int)
    bins = np.array([mag[edges[i]:max(edges[i] + 1, edges[i + 1])].mean()
                     for i in range(SPECTRAL_BINS)])
    total = bins.sum()
    if total <= 0:
        return [0.0] * SPECTRAL_BINS
    return np.round(bins / total, 9).tolist()


def read_fragment(ds: h5py.Dataset, start: int, length: int) -> np.ndarray:
    n_axis = len(ds.shape) - 1
    sel = tuple(slice(start, start + length) if i == n_axis else slice(0, 1)
                for i in range(len(ds.shape)))
    return np.asarray(ds[sel]).reshape(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"G:\DroneRFa数据集\dataset_manifest.csv")
    ap.add_argument("--outdir", default=HERE)
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    records = []
    block_profiles = []
    problems = []
    for idx, row in enumerate(rows, 1):
        name = row["file_name"]
        path = row["unified_path"]
        rec = {"file_name": name, "class_id": row.get("class_id"),
               "class_name": row.get("class_name"), "origin": row.get("origin"),
               "fragments": [], "channel_lengths": {}}
        try:
            with h5py.File(path, "r") as f:
                profile = {}
                for ch in CHANNELS:
                    if ch not in f:
                        problems.append({"file_name": name, "problem": "channel_missing", "channel": ch})
                        continue
                    ds = f[ch]
                    n = int(np.prod(ds.shape))
                    rec["channel_lengths"][ch] = n
                    for oi, off in enumerate(OFFSETS):
                        if off + FRAG_LEN > n:
                            problems.append({"file_name": name, "problem": "offset_out_of_range",
                                             "channel": ch, "offset": off, "n": n})
                            continue
                        x = read_fragment(ds, off, FRAG_LEN)
                        xf = np.asarray(x, dtype=np.float64)
                        rec["fragments"].append({
                            "channel": ch,
                            "offset_index": oi,
                            "offset": off,
                            "length": FRAG_LEN,
                            "sha256_float64_raw": sha256_of(xf, "<f8"),
                            "sha256_float32_cast": sha256_of(xf, "<f4"),
                            "spectral_shape_64": spectral_shape(xf),
                            "mean": float(xf.mean()),
                            "rms": float(np.sqrt(np.mean(np.square(xf)))),
                            "min": float(xf.min()),
                            "max": float(xf.max()),
                            "n_nonfinite": int(xf.size - np.isfinite(xf).sum()),
                        })
                    # secondary: 10M-block start probe RMS on this channel
                    if ch in ("RF0_I", "RF0_Q"):
                        rms_list = []
                        k = 0
                        while k * BLOCK + BLOCK_PROBE <= n:
                            p = read_fragment(ds, k * BLOCK, BLOCK_PROBE).astype(np.float64)
                            rms_list.append(float(np.sqrt(np.mean(np.square(p)))))
                            k += 1
                        profile[ch] = {"n_blocks": k, "rms": [float("%.6g" % v) for v in rms_list]}
                block_profiles.append({"file_name": name, "class_id": row.get("class_id"),
                                       "channel_lengths": rec["channel_lengths"], "profile": profile})
        except Exception as exc:
            problems.append({"file_name": name, "problem": "fingerprint_failed", "error": repr(exc)})
            rec["fingerprint_ok"] = False
            records.append(rec)
            continue
        rec["fingerprint_ok"] = True
        records.append(rec)
        if idx % 10 == 0:
            print("processed %d/%d" % (idx, len(rows)), flush=True)

    # ---------------- cross-file comparison ----------------
    frags = []
    for r in records:
        for fr in r.get("fragments", []):
            frags.append({
                "file_name": r["file_name"],
                "class_id": r.get("class_id"),
                "origin": r.get("origin"),
                "channel": fr["channel"],
                "offset_index": fr["offset_index"],
                "sha256_float64_raw": fr["sha256_float64_raw"],
                "sha256_float32_cast": fr["sha256_float32_cast"],
                "spectral_shape_64": fr["spectral_shape_64"],
            })

    def dup_groups(key):
        groups = {}
        for fr in frags:
            groups.setdefault(fr[key], []).append("%s|%s|%d" % (fr["file_name"], fr["channel"],
                                                                fr["offset_index"]))
        return {k: sorted(v) for k, v in groups.items() if len(v) > 1}

    exact_dups = dup_groups("sha256_float64_raw")
    f32_dups = dup_groups("sha256_float32_cast")

    cells = {}
    for fr in frags:
        if len(fr["spectral_shape_64"]) != SPECTRAL_BINS:
            continue
        cells.setdefault((fr["channel"], fr["offset_index"]), []).append(fr)

    nearest = {}
    flagged = []
    for cell, items in sorted(cells.items()):
        v = np.asarray([it["spectral_shape_64"] for it in items], dtype=np.float64)
        n = v.shape[0]
        d = np.abs(v[:, None, :] - v[None, :, :]).sum(axis=2)
        iu = np.triu_indices(n, k=1)
        dists = d[iu]
        order = np.argsort(dists)[:20]
        nearest["%s|%d" % cell] = [
            {"a": items[iu[0][i]]["file_name"], "b": items[iu[1][i]]["file_name"],
             "a_class": items[iu[0][i]]["class_id"], "b_class": items[iu[1][i]]["class_id"],
             "l1_spectral": round(float(dists[i]), 9),
             "same_class": items[iu[0][i]]["class_id"] == items[iu[1][i]]["class_id"]}
            for i in order]
        for i in np.where(dists < SPECTRAL_FLAG_L1)[0]:
            flagged.append({"cell": "%s|%d" % cell, "a": items[iu[0][i]]["file_name"],
                            "b": items[iu[1][i]]["file_name"],
                            "l1_spectral": round(float(dists[i]), 9)})

    files_with_dup_fragment = sorted({f.split("|")[0] for g in exact_dups.values() for f in g})
    candidate_pairs = set()
    for g in list(exact_dups.values()) + list(f32_dups.values()):
        names = sorted({f.split("|")[0] for f in g})
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                candidate_pairs.add((names[i], names[j]))
    for fl in flagged:
        candidate_pairs.add(tuple(sorted((fl["a"], fl["b"]))))

    out = {
        "schema": "dronerfa-meta-split-near-duplicate-screen/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration": "fingerprint-preregistration-20260925.json",
        "plan": {"offsets": list(OFFSETS), "fragment_length": FRAG_LEN, "channels": list(CHANNELS),
                 "spectral_bins": SPECTRAL_BINS, "spectral_flag_l1": SPECTRAL_FLAG_L1,
                 "block_size": BLOCK, "block_probe": BLOCK_PROBE},
        "fragments_compared": len(frags),
        "files_with_fragments": len({f["file_name"] for f in frags}),
        "exact_duplicate_fragment_groups": exact_dups,
        "exact_duplicate_group_count": len(exact_dups),
        "float32_cast_duplicate_group_count": len(f32_dups),
        "files_with_duplicate_fragment": files_with_dup_fragment,
        "spectral_flag_l1_pairs": flagged,
        "candidate_near_duplicate_file_pairs": sorted(candidate_pairs),
        "closest_nonidentical_pairs_per_cell": nearest,
        "problems": problems,
        "limitations": [
            "Bounded screen: four fixed 65536-sample offsets per channel, at most 0.24% of the "
            "shortest channel. No match here does not prove the files are disjoint.",
            "A shifted overlap aligns with a fixed offset only by chance and would be missed.",
            "All eight files of one class were written within ~13 minutes of each other, so even a "
            "positive near-duplicate finding would be expected rather than surprising.",
        ],
    }
    nd_path = os.path.join(args.outdir, "near_duplicate_screen.json")
    with open(nd_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    fp_path = os.path.join(args.outdir, "fragment_fingerprints_104.json")
    with open(fp_path, "w", encoding="utf-8") as fh:
        json.dump({"schema": "dronerfa-meta-split-fragment-fingerprints/v1",
                   "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "plan": out["plan"], "records": records}, fh, ensure_ascii=False, indent=1)

    bp_path = os.path.join(args.outdir, "block_rms_profiles_104.json")
    with open(bp_path, "w", encoding="utf-8") as fh:
        json.dump({"schema": "dronerfa-meta-split-block-rms-profiles/v1",
                   "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "block_size": BLOCK, "probe": BLOCK_PROBE,
                   "records": block_profiles}, fh, ensure_ascii=False, indent=1)

    print(json.dumps({
        "files": len(records),
        "fingerprint_ok": sum(1 for r in records if r.get("fingerprint_ok")),
        "fragments": len(frags),
        "exact_duplicate_group_count": len(exact_dups),
        "float32_cast_duplicate_group_count": len(f32_dups),
        "spectral_flag_l1_pairs": len(flagged),
        "candidate_near_duplicate_file_pairs": len(candidate_pairs),
        "problems": problems,
        "wrote": [fp_path, nd_path, bp_path],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
