# -*- coding: utf-8 -*-
"""Session-pause check: is META-HEADER1 in a complete, restable state?"""
import json
import os

ROOT = r"C:\Users\Administrator\Desktop\数据集处理"
HERE = os.path.join(ROOT, "work", "meta_header1")

probe = json.load(open(os.path.join(HERE, "member_header_probe_88.json"),
                       encoding="utf-8"))
verify = json.load(open(os.path.join(HERE, "verification_meta_header1.json"),
                        encoding="utf-8"))
handoff = json.load(open(os.path.join(ROOT, "work", "deepseek-handoffs",
                                      "20260925-203917-190ce0d1.json"),
                         encoding="utf-8"))

artifacts = sorted(os.listdir(HERE))
leftovers = [n for n in artifacts
             if n.startswith("_partial") or n == "__pycache__"
             or n == "probe_run_console.log" or "_smoke" in n]

print(json.dumps({
    "probe_is_not_partial": not probe.get("partial"),
    "probe_candidates": len(probe["candidates"]),
    "probe_aborted": probe.get("aborted"),
    "verification_verdict": verify["verdict"],
    "verification_checks_failed": verify["checks_failed"],
    "handoff_kind": handoff["kind"],
    "leftover_temp_files": leftovers,
    "artifact_count": len(artifacts),
    "artifacts": artifacts,
}, ensure_ascii=False, indent=1))
