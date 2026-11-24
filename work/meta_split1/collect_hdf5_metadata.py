# -*- coding: utf-8 -*-
"""META-SPLIT-1 / step 1b: bounded HDF5 metadata probe for every manifest MAT.

For each file this reads only structural HDF5 metadata plus a small, fixed number
of samples per channel.  No full-waveform scan.

Pre-registered (fixed before running):
  * probe offsets  : first 4096 samples of every dataset
  * finite probe   : same 4096 samples
  * dataset attrs  : all attributes present on the root group and on each dataset
  * HDF5 libver / userblock / compression / chunking reported by h5py

Outputs (work/meta_split1/):
  hdf5_metadata_104.json
  hdf5_metadata_104.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys

import h5py
import numpy as np

PROBE_SAMPLES = 4096


def jsonable(v):
    """Best-effort conversion of an HDF5 attribute value to plain JSON types."""
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray):
        return [jsonable(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    return str(v)


def describe_dataset(ds: h5py.Dataset) -> dict:
    info = {
        "name": ds.name,
        "shape": list(ds.shape),
        "dtype": str(ds.dtype),
        "chunks": list(ds.chunks) if ds.chunks else None,
        "compression": ds.compression,
        "compression_opts": jsonable(ds.compression_opts),
        "shuffle": bool(ds.shuffle),
        "fletcher32": bool(ds.fletcher32),
        "fillvalue": None if ds.fillvalue is None else str(ds.fillvalue),
        "attrs": {k: jsonable(v) for k, v in ds.attrs.items()},
    }
    total = int(np.prod(ds.shape)) if ds.shape else 0
    info["n_samples"] = total
    n = min(PROBE_SAMPLES, total)
    if n:
        # MATLAB row vectors are stored as (1, N); read along the last axis.
        sel = tuple(slice(0, n) if i == len(ds.shape) - 1 else slice(0, 1)
                    for i in range(len(ds.shape)))
        probe = np.asarray(ds[sel]).reshape(-1)[:n]
        info["probe_n"] = int(probe.size)
        info["probe_finite"] = bool(np.all(np.isfinite(probe)))
        info["probe_min"] = float(np.min(probe))
        info["probe_max"] = float(np.max(probe))
        info["probe_mean"] = float(np.mean(probe.astype(np.float64)))
        info["probe_rms"] = float(np.sqrt(np.mean(np.square(probe.astype(np.float64)))))
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"G:\DroneRFa数据集\dataset_manifest.csv")
    ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()

    with open(args.manifest, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    records = []
    problems = []
    for row in rows:
        name = row["file_name"]
        path = row["unified_path"]
        rec = {"file_name": name, "class_id": row.get("class_id"), "class_name": row.get("class_name"),
               "origin": row.get("origin"), "former_partition": row.get("former_partition"),
               "split_status": row.get("split_status"), "evaluation_role": row.get("evaluation_role")}
        try:
            with h5py.File(path, "r") as f:
                rec["hdf5_userblock"] = f.userblock_size
                rec["hdf5_libver"] = f.libver
                rec["root_groups"] = sorted(f.keys())
                rec["root_attrs"] = {k: jsonable(v) for k, v in f.attrs.items()}
                datasets = {}
                for key in f.keys():
                    obj = f[key]
                    if isinstance(obj, h5py.Dataset):
                        datasets[key] = describe_dataset(obj)
                    else:
                        datasets[key] = {"kind": "group", "attrs": dict(obj.attrs)}
                rec["datasets"] = datasets
                rec["channel_names"] = sorted(datasets.keys())
                lengths = {k: v.get("n_samples") for k, v in datasets.items()}
                rec["channel_lengths"] = lengths
                rec["channel_lengths_uniform"] = len(set(lengths.values())) == 1
        except Exception as exc:
            problems.append({"file_name": name, "problem": "hdf5_read_failed", "error": repr(exc)})
            rec["hdf5_read_ok"] = False
            records.append(rec)
            continue
        rec["hdf5_read_ok"] = True
        records.append(rec)

    outdir = args.outdir
    json_path = os.path.join(outdir, "hdf5_metadata_104.json")
    csv_path = os.path.join(outdir, "hdf5_metadata_104.csv")

    payload = {
        "schema": "dronerfa-meta-split-hdf5-metadata/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "manifest": args.manifest,
        "probe_samples_per_channel": PROBE_SAMPLES,
        "records": records,
        "problems": problems,
        "notes": [
            "Only HDF5 structural metadata and the first %d samples of each dataset were read." % PROBE_SAMPLES,
            "MATLAB v7.3 files carry no acquisition session/date/place/device attributes; the root group "
            "of every file is expected to hold only the four RF0/RF1 I/Q datasets.",
        ],
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)

    cols = ["file_name", "class_id", "class_name", "origin", "former_partition", "split_status",
            "evaluation_role", "hdf5_read_ok", "hdf5_userblock", "hdf5_libver", "channel_names",
            "channel_lengths", "channel_lengths_uniform", "root_attr_count"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([
                r["file_name"], r["class_id"], r["class_name"], r["origin"], r["former_partition"],
                r["split_status"], r["evaluation_role"], r.get("hdf5_read_ok"),
                r.get("hdf5_userblock"), json.dumps(r.get("hdf5_libver")),
                ";".join(r.get("channel_names", [])),
                json.dumps(r.get("channel_lengths", {}), ensure_ascii=False),
                r.get("channel_lengths_uniform"), len(r.get("root_attrs", {})),
            ])

    lengths = {}
    for r in records:
        for k, v in (r.get("channel_lengths") or {}).items():
            lengths.setdefault(k, set()).add(v)
    root_attr_names = set()
    ds_attr_names = set()
    for r in records:
        root_attr_names.update((r.get("root_attrs") or {}).keys())
        for d in (r.get("datasets") or {}).values():
            ds_attr_names.update((d.get("attrs") or {}).keys())

    print(json.dumps({
        "files": len(records),
        "hdf5_ok": sum(1 for r in records if r.get("hdf5_read_ok")),
        "problems": problems,
        "userblocks": sorted({r.get("hdf5_userblock") for r in records}),
        "channel_length_distribution": {k: sorted(v) for k, v in lengths.items()},
        "root_attr_names": sorted(root_attr_names),
        "dataset_attr_names": sorted(ds_attr_names),
        "wrote": [json_path, csv_path],
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
