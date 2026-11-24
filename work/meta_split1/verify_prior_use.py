# -*- coding: utf-8 -*-
"""META-SPLIT-1 / step 4a: verify the 'never used in selection' claim instead of
asserting it.

Scans every data/model artefact in the project for DroneRFa MAT file names and
reports which of the 104 manifest files have ever appeared in a window index,
prediction table, metric table or checkpoint manifest.

Output: work/meta_split1/prior_use_scan.json
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))

NAME_RE = re.compile(r"T\d{3,5}(?:_D\d{2})?_S\d{4,5}\.mat")
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "meta_split1"}
SKIP_EXT = {".mat", ".rar", ".part", ".restart", ".png", ".jpg", ".docx", ".pdf", ".zip",
            ".pt", ".pth", ".ckpt", ".npy", ".npz"}
MAX_BYTES = 8 * 1024 * 1024

# A "data/model artefact" is anything produced under dronerfa_baseline/outputs/ or
# reports/ by a run that read data or trained/served a model.  Everything under
# work/ is governance, planning or download plumbing, and *.py is source code that
# may legitimately name targets.  Remote-archive size manifests are called out
# separately because they list all 630 archive members for integrity checks.
REMOTE_INDEX_HINTS = ("frozen_inputs_verification", "dronerfa_manifest.json",
                      "sel12g-config", "sel12-config", "download12", "dronerfa-12model",
                      "12model-download")
PLUMBING_HINTS = ("download_records", "select12g-state", "scripts\\", "scripts/",
                  "cleanup-audit", "restart-cleanup")


def classify(rel: str) -> str:
    low = rel.replace("/", "\\").lower()
    if any(h.lower() in low for h in REMOTE_INDEX_HINTS):
        return "remote_archive_index_or_size_manifest"
    if any(h.replace("/", "\\").lower() in low for h in PLUMBING_HINTS):
        return "download_or_workflow_plumbing"
    if low.startswith("dronerfa_baseline\\outputs\\"):
        return "data_or_model_artifact"
    if low.startswith("work\\"):
        return "work_governance"
    if low.endswith(".py"):
        return "source_code"
    return "other_docs"


def main() -> int:
    manifest_names = set()
    import csv
    with open(r"G:\DroneRFa数据集\dataset_manifest.csv", "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            manifest_names.add(row["file_name"])
    new80 = set()
    with open(r"G:\DroneRFa数据集\dataset_manifest.csv", "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("origin") == "selected_80":
                new80.add(row["file_name"])

    hits = collections.defaultdict(lambda: collections.defaultdict(set))
    scanned = 0
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT:
                continue
            path = os.path.join(root, fn)
            try:
                if os.path.getsize(path) > MAX_BYTES:
                    continue
                text = open(path, "r", encoding="utf-8", errors="ignore").read()
            except Exception:
                continue
            scanned += 1
            rel = os.path.relpath(path, WORKSPACE)
            for m in set(NAME_RE.findall(text)):
                if m in manifest_names:
                    hits[m][classify(rel)].add(rel)

    per_file = {}
    for name in sorted(manifest_names):
        cats = {k: sorted(v) for k, v in hits.get(name, {}).items()}
        per_file[name] = {
            "categories": {k: v for k, v in cats.items()},
            "in_data_or_model_artifact": bool(cats.get("data_or_model_artifact")),
            "n_referencing_files": sum(len(v) for v in cats.values()),
        }

    artefacts_only = {k: v for k, v in per_file.items() if v["in_data_or_model_artifact"]}
    new80_in_artefacts = sorted(n for n in new80 if per_file[n]["in_data_or_model_artifact"])

    # Which artefacts mention which of the 24 legacy files?
    legacy_refs = collections.Counter()
    for name, v in per_file.items():
        for f in v["categories"].get("data_or_model_artifact", []):
            legacy_refs[f] += 1

    result = {
        "schema": "dronerfa-meta-split-prior-use-scan/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files_scanned": scanned,
        "manifest_files": len(manifest_names),
        "selected80_files": len(new80),
        "files_appearing_in_data_or_model_artifacts": sorted(artefacts_only),
        "selected80_appearing_in_data_or_model_artifacts": new80_in_artefacts,
        "verdict": {
            "selected80_used_in_any_data_or_model_artifact": bool(new80_in_artefacts),
            "legacy24_used_in_data_or_model_artifacts": len(artefacts_only),
        },
        "artifacts_referencing_manifest_files": dict(legacy_refs.most_common()),
        "per_file": per_file,
        "notes": [
            "Remote-archive index / size manifests are classified separately: they list all 630 archive "
            "members for integrity checks and are not model or data selections.",
            "Download plumbing, task files and handoffs are also separate: they name targets by necessity.",
        ],
    }
    out = os.path.join(HERE, "prior_use_scan.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: result[k] for k in
                      ("files_scanned", "manifest_files", "selected80_files",
                       "files_appearing_in_data_or_model_artifacts",
                       "selected80_appearing_in_data_or_model_artifacts", "verdict")},
                     ensure_ascii=False, indent=1))
    print("top artifacts:", json.dumps(dict(legacy_refs.most_common(12)), ensure_ascii=False, indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
