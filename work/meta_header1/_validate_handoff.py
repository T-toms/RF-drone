# -*- coding: utf-8 -*-
"""Final gate: validate the handoff JSON and the report's required answers."""
import json
import os
import sys

ROOT = r"C:\Users\Administrator\Desktop\数据集处理"
HANDOFF = os.path.join(ROOT, "work", "deepseek-handoffs",
                       "20260925-203917-190ce0d1.json")
REPORT = os.path.join(ROOT, "work", "meta_header1", "META-HEADER1-report.md")

required = ["schemaVersion", "runId", "sessionId", "kind", "summary",
            "changedFiles", "checks", "problem", "attempts",
            "suggestedNextStep", "reportedAt"]
handoff = json.load(open(HANDOFF, encoding="utf-8"))
problems = []
for key in required:
    if key not in handoff:
        problems.append("missing key: " + key)
if handoff.get("schemaVersion") != 1:
    problems.append("schemaVersion != 1")
if handoff.get("runId") != "20260925-203917-190ce0d1":
    problems.append("runId mismatch")
if handoff.get("sessionId") != "session-f7e71d98-9be8-4146-ad21-687511b63324":
    problems.append("sessionId mismatch")
if handoff.get("kind") != "completed":
    problems.append("kind != completed")
if not isinstance(handoff.get("summary"), str) or len(handoff["summary"]) < 100:
    problems.append("summary missing or too short")
if not isinstance(handoff.get("changedFiles"), list) or not handoff["changedFiles"]:
    problems.append("changedFiles empty")
missing_files = [p for p in handoff["changedFiles"]
                 if not os.path.exists(os.path.join(ROOT, p))]
if missing_files:
    problems.append("changedFiles not on disk: %s" % missing_files)
if not isinstance(handoff.get("checks"), list) or len(handoff["checks"]) < 5:
    problems.append("checks too few")
if not all({"check", "command", "result"} <= set(c) for c in handoff["checks"]):
    problems.append("a check entry is missing check/command/result")
if not isinstance(handoff.get("attempts"), list) or not handoff["attempts"]:
    problems.append("attempts empty")

report = open(REPORT, encoding="utf-8").read()
for needle in ("9.1", "9.2", "9.3", "T0000", "T0010", "T0101",
               "file-time proxy", "不得声称"):
    if needle not in report:
        problems.append("report missing %r" % needle)
# Overclaims that the report must only ever name inside its prohibition list.
forbidden = ("13 类都有了未参与选型的 final test",
             "三个旧类都能补到跨批次数据",
             "未被任何研究使用")
restriction_at = report.find("## 10. 不得声称")
if restriction_at < 0:
    problems.append("report has no '不得声称' restriction section")
else:
    outside = report[:restriction_at]
    for phrase in forbidden:
        if phrase in outside:
            problems.append("report asserts forbidden claim %r outside the "
                            "prohibition list" % phrase)
    bullets = [line for line in report[restriction_at:].splitlines()
               if line.startswith("- ")]
    if len(bullets) < 5:
        problems.append("prohibition list has only %d bullets" % len(bullets))

print(json.dumps({
    "handoff_keys": sorted(handoff.keys()),
    "changedFiles": len(handoff["changedFiles"]),
    "checks": len(handoff["checks"]),
    "attempts": len(handoff["attempts"]),
    "summary_chars": len(handoff["summary"]),
    "report_chars": len(report),
    "problems": problems,
    "verdict": "PASS" if not problems else "FAIL",
}, ensure_ascii=False, indent=1))
sys.exit(0 if not problems else 1)
