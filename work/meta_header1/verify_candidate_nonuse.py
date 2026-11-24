# -*- coding: utf-8 -*-
"""META-HEADER1 / step 5: verify (do not assert) that the 88 un-downloaded
legacy-class candidates have never taken part in this project's fitting or
selection.

Reuses the exact scanner and classification rules of META-SPLIT-1's
``verify_prior_use.py`` so the two tasks cannot disagree about what counts as a
"data or model artefact".

Output: work/meta_header1/candidate_nonuse_scan.json
"""

from __future__ import annotations

import collections
import csv
import datetime as dt
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(WORKSPACE, "work", "meta_split1"))

import verify_prior_use as VPU  # noqa: E402

FROZEN = os.path.join(WORKSPACE, "work", "meta_split1",
                      "archive_unused_for_legacy3_20260925.csv")
MANIFEST = r"G:\DroneRFa数据集\dataset_manifest.csv"
OUT = os.path.join(HERE, "candidate_nonuse_scan.json")

# META-SPLIT-1's rule maps everything under dronerfa_baseline\outputs\ to
# "data_or_model_artifact".  That bucket also contains the *extracted text of the
# published papers*, which is literature, not a project data/model product.  Any
# path matching these hints is re-labelled so the "used in fitting/selection"
# claim cannot silently absorb a citation.
LITERATURE_HINTS = ("literature_review", "\\text\\", "/text/", "\\pdfs\\", "/pdfs/")


def classify(rel: str) -> str:
    base = VPU.classify(rel)
    low = rel.replace("/", "\\").lower()
    if base == "data_or_model_artifact" and any(h.lower() in low for h in LITERATURE_HINTS):
        return "literature_text"
    return base


def main() -> int:
    with open(FROZEN, "r", encoding="utf-8-sig", newline="") as fh:
        frozen_rows = list(csv.DictReader(fh))
    candidates = {row["file_name"]: row["t"] for row in frozen_rows}

    with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as fh:
        manifest_names = {row["file_name"] for row in csv.DictReader(fh)}

    hits = collections.defaultdict(lambda: collections.defaultdict(set))
    scanned = 0
    for root, dirs, files in os.walk(WORKSPACE):
        dirs[:] = [d for d in dirs if d not in VPU.SKIP_DIRS]
        for fn in files:
            if os.path.splitext(fn)[1].lower() in VPU.SKIP_EXT:
                continue
            path = os.path.join(root, fn)
            try:
                if os.path.getsize(path) > VPU.MAX_BYTES:
                    continue
                text = open(path, "r", encoding="utf-8", errors="ignore").read()
            except Exception:                                   # noqa: BLE001
                continue
            scanned += 1
            rel = os.path.relpath(path, WORKSPACE)
            for name in set(VPU.NAME_RE.findall(text)):
                if name in candidates:
                    hits[name][classify(rel)].add(rel)

    per_file = {}
    for name in sorted(candidates):
        categories = {k: sorted(v) for k, v in hits.get(name, {}).items()}
        per_file[name] = {
            "t_code": "T" + candidates[name],
            "in_manifest_104": name in manifest_names,
            "in_data_or_model_artifact": bool(categories.get("data_or_model_artifact")),
            "categories": categories,
            "n_referencing_files": sum(len(v) for v in categories.values()),
        }

    in_artefacts = sorted(n for n, v in per_file.items() if v["in_data_or_model_artifact"])
    in_manifest = sorted(n for n, v in per_file.items() if v["in_manifest_104"])
    category_counts = collections.Counter()
    for value in per_file.values():
        for cat in value["categories"]:
            category_counts[cat] += 1

    payload = {
        "schema": "dronerfa-meta-header1-candidate-nonuse-scan/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scanner": {
            "reused_from": os.path.join(WORKSPACE, "work", "meta_split1",
                                        "verify_prior_use.py"),
            "name_regex": VPU.NAME_RE.pattern,
            "skip_dirs": sorted(VPU.SKIP_DIRS),
            "max_bytes": VPU.MAX_BYTES,
            "local_deviation": ("paths under dronerfa_baseline\\outputs\\ that are extracted "
                                "paper text are re-labelled 'literature_text' instead of "
                                "'data_or_model_artifact' (hints: "
                                + ", ".join(LITERATURE_HINTS) + ")"),
        },
        "files_scanned": scanned,
        "candidates": len(candidates),
        "candidates_present_in_104_manifest": in_manifest,
        "candidates_in_data_or_model_artifacts": in_artefacts,
        "candidates_referenced_only_in_literature_text": sorted(
            n for n, v in per_file.items()
            if v["categories"].get("literature_text") and not v["in_data_or_model_artifact"]),
        "per_file": per_file,
        "category_file_counts": dict(category_counts.most_common()),
        "verdict": {
            "any_candidate_in_a_data_or_model_artifact": bool(in_artefacts),
            "any_candidate_in_the_104_manifest": bool(in_manifest),
        },
        "notes": [
            "Reuses META-SPLIT-1's scanner unchanged; only the literature-text override is local.",
            "Only a claim about this project's prior artefacts can be made; it says nothing "
            "about whether other researchers have used these archive members.",
        ],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: payload[k] for k in
                      ("files_scanned", "candidates", "candidates_present_in_104_manifest",
                       "candidates_in_data_or_model_artifacts",
                       "candidates_referenced_only_in_literature_text",
                       "category_file_counts", "verdict")}, ensure_ascii=False, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
