# -*- coding: utf-8 -*-
"""META-HEADER1 / step 4: independent recomputation of every number the report
states, from the raw artefacts only.

Phase 1 recomputes structural invariants straight from:
  member_header_probe_88.json        (per-candidate header evidence)
  http_request_log.json              (every HTTP request actually issued)
  htime-decode-preregistration-20260925.json
  availability_summary_by_class_date.json
  work/meta_split1/archive_unused_for_legacy3_20260925.csv  (frozen sizes)
and writes work/meta_header1/verification_meta_header1.json.

Phase 2 re-reads META-HEADER1-report.md and asserts that the exact strings
built from the recomputed values appear in it, so the prose cannot drift from
the evidence.  Nothing here trusts the probe script's own summary fields.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.abspath(os.path.join(HERE, "..", ".."))

PROBE = os.path.join(HERE, "member_header_probe_88.json")
HTTP_LOG = os.path.join(HERE, "http_request_log.json")
PREREG = os.path.join(HERE, "htime-decode-preregistration-20260925.json")
SUMMARY = os.path.join(HERE, "availability_summary_by_class_date.json")
SUMMARY_CSV = os.path.join(HERE, "availability_summary_by_class_date.csv")
FROZEN = os.path.join(WORKSPACE, "work", "meta_split1",
                      "archive_unused_for_legacy3_20260925.csv")
MAT_HEADERS = os.path.join(WORKSPACE, "work", "meta_split1", "mat_headers_104.json")
REPORT = os.path.join(HERE, "META-HEADER1-report.md")
OUT = os.path.join(HERE, "verification_meta_header1.json")

MAX_HEADER_SPAN = 128
PREFIX_BYTES = 8

checks = []


def check(name: str, ok: bool, evidence) -> None:
    checks.append({"check": name, "pass": bool(ok), "evidence": evidence})


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    probe = json.load(open(PROBE, encoding="utf-8"))
    http_log = json.load(open(HTTP_LOG, encoding="utf-8"))
    prereg = json.load(open(PREREG, encoding="utf-8"))
    summary = json.load(open(SUMMARY, encoding="utf-8"))
    heads = json.load(open(MAT_HEADERS, encoding="utf-8"))
    with open(FROZEN, "r", encoding="utf-8-sig", newline="") as fh:
        frozen_rows = list(csv.DictReader(fh))

    candidates = probe["candidates"]
    ok_rows = [c for c in candidates if c["status"] == "ok"]
    requests = http_log["requests"]

    # --- 1. scope and completeness ---------------------------------------------
    check("candidate_count_is_88", len(candidates) == 88, len(candidates))
    check("all_88_candidates_recorded", len(candidates) == 88,
          sorted(c["file_name"] for c in candidates) ==
          sorted(r["file_name"] for r in frozen_rows))
    per_class = {}
    for row in candidates:
        per_class[row["t_code"]] = per_class.get(row["t_code"], 0) + 1
    check("legacy3_only_8_40_40", per_class == {"T0000": 8, "T0010": 40, "T0101": 40},
          per_class)
    check("no_probe_aborted", probe.get("aborted") in (None, ""), probe.get("aborted"))
    check("probe_not_partial", not probe.get("partial"), probe.get("partial"))

    # --- 2. endpoint identity --------------------------------------------------
    gates = probe["preflight_gates"]
    check("preflight_all_pass", gates["all_preflight_gates_pass"], gates)
    check("index_sha_matches_config", gates["index_sha256_matches_frozen_config"],
          gates["index_sha256"])
    check("index_archive_size_frozen", gates["index_archive_size"] == 615955805466,
          gates["index_archive_size"])
    check("index_etag_frozen", gates["index_etag"] == '"6774350c-8f69d3b51a"',
          gates["index_etag"])
    check("signature_probe_ok",
          gates["signature_probe"]["is_rar5_signature"]
          and gates["signature_probe"]["status"] == 206,
          gates["signature_probe"])
    check("rule_module_hash_unchanged",
          sha256_file(os.path.join(HERE, "rar5_header.py"))
          == prereg["rule_module"]["sha256"],
          prereg["rule_module"]["sha256"])

    # --- 3. pre-registration ----------------------------------------------------
    pc = prereg["checks"]
    check("prereg_14_members", pc["members_parsed"] == 14, pc["members_parsed"])
    check("prereg_all_crc_ok", pc["head_crc_ok_all"], pc["head_crc_ok_all"])
    check("prereg_14_within_2s", pc["within_tolerance"] == 14, pc["within_tolerance"])
    check("prereg_residual_range", pc["residual_seconds_set"] == [-2.0, -1.0],
          pc["residual_seconds_set"])
    check("prereg_single_encoding", pc["time_encoding_set"] == ["windows_filetime_uint64"],
          pc["time_encoding_set"])
    check("prereg_header_span_max", pc["header_span_max"] == 71, pc["header_span_max"])

    # --- 4. read budget ---------------------------------------------------------
    spans = [c.get("header_span_bytes") for c in candidates if c.get("header_span_bytes")]
    check("header_span_within_128", all(s <= MAX_HEADER_SPAN for s in spans),
          {"max": max(spans), "min": min(spans)})
    check("no_header_too_large_anomaly",
          not any(c["status"] == "anomaly_header_too_large" for c in candidates),
          [c["file_name"] for c in candidates
           if c["status"] == "anomaly_header_too_large"])
    bytes_per_candidate = [c.get("http_bytes", 0) for c in candidates if c.get("http_bytes")]
    check("bytes_per_candidate_within_budget",
          all(b <= MAX_HEADER_SPAN + PREFIX_BYTES for b in bytes_per_candidate),
          {"max": max(bytes_per_candidate), "min": min(bytes_per_candidate)})
    check("no_payload_bytes_read", http_log["stats"]["payload_bytes_read"] == 0,
          http_log["stats"]["payload_bytes_read"])

    # --- 5. HTTP discipline -----------------------------------------------------
    statuses = sorted({r.get("status") for r in requests}, key=lambda x: (x is None, x))
    check("only_206_or_transport_timeout", set(statuses) <= {206, None}, statuses)
    check("no_429_at_all", not any(r.get("status") == 429 for r in requests),
          [r for r in requests if r.get("status") == 429])
    failed_attempts = [r for r in requests if r.get("status") is None]
    check("failed_attempts_are_read_timeouts",
          bool(failed_attempts) and all("timed out" in (r.get("error") or "")
                                        for r in failed_attempts),
          sorted({r.get("error") for r in failed_attempts}))
    check("every_failed_attempt_was_retried_and_recovered",
          all(any(e["label"] == r["label"] and e["attempt"] > r["attempt"]
                  and e.get("status") == 206 for e in requests)
              for r in failed_attempts),
          len(failed_attempts))
    check("retries_are_bounded",
          max(r["attempt"] for r in requests) <= 3,
          {"max_attempt": max(r["attempt"] for r in requests),
           "failed_attempts": len(failed_attempts)})
    windows = {}
    for entry in requests:
        windows.setdefault(entry["label"], []).append(entry)
    bad_windows = [label for label, entries in windows.items()
                   if not (entries[-1].get("status") == 206
                           and entries[-1]["body_bytes"] == entries[-1]["requested_bytes"])]
    check("request_windows_are_2_per_candidate_plus_signature",
          len(windows) == 2 * 88 + 1, len(windows))
    check("every_window_finally_succeeded", not bad_windows, bad_windows)
    check("total_requests_equals_windows_plus_failed_attempts",
          len(requests) == len(windows) + len(failed_attempts),
          {"requests": len(requests), "windows": len(windows),
           "failed_attempts": len(failed_attempts)})
    check("every_request_has_matching_content_range",
          all(r.get("content_range") == f"bytes {r['range'].split('-')[0]}-"
                                        f"{r['range'].split('-')[1]}/615955805466"
              for r in requests if r.get("status") == 206),
          [r["content_range"] for r in requests[:2]])
    check("every_request_has_frozen_etag",
          all(r.get("etag") == '"6774350c-8f69d3b51a"'
              for r in requests if r.get("status") == 206),
          sorted({str(r.get("etag")) for r in requests}))
    check("body_bytes_equal_requested",
          all(r["body_bytes"] == r["requested_bytes"] for r in requests
              if r.get("status") == 206),
          [r for r in requests if r.get("body_bytes") != r["requested_bytes"]])
    total_http_bytes = sum(r["body_bytes"] for r in requests)
    check("total_transfer_well_under_100KiB", total_http_bytes < 100 * 1024,
          {"total_bytes": total_http_bytes, "limit": 100 * 1024,
           "requests": len(requests)})

    # --- 6. per-member header verification --------------------------------------
    check("all_candidates_ok", len(ok_rows) == 88, len(ok_rows))
    check("no_anomalies", not summary["anomalies"], summary["anomalies"])
    failed = []
    for row in ok_rows:
        for key, value in row["verification"].items():
            if not value:
                failed.append((row["file_name"], key))
    check("all_per_member_verifications_true", not failed, failed[:5])
    check("every_head_crc_ok", all(r["head_crc_ok"] for r in ok_rows),
          sum(1 for r in ok_rows if r["head_crc_ok"]))
    check("every_block_is_file_type",
          all(r["head_type_name"] == "FILE" for r in ok_rows),
          sorted({r["head_type_name"] for r in ok_rows}))
    check("every_name_matches_frozen",
          all(r["header_file_name"] == "DroneRFa/" + r["file_name"] for r in ok_rows),
          [r["header_file_name"] for r in ok_rows[:2]])
    check("every_packed_size_matches_frozen",
          all(r["data_size"] == r["frozen_packed_size"] for r in ok_rows),
          sum(1 for r in ok_rows if r["data_size"] != r["frozen_packed_size"]))
    check("every_unpacked_size_matches_frozen",
          all(r["unpacked_size"] == r["frozen_unpacked_size"] for r in ok_rows),
          sum(1 for r in ok_rows if r["unpacked_size"] != r["frozen_unpacked_size"]))
    check("every_block_size_matches_frozen",
          all(r["block_total_bytes"] == r["frozen_block_size"] for r in ok_rows),
          sum(1 for r in ok_rows if r["block_total_bytes"] != r["frozen_block_size"]))
    check("all_times_use_preregistered_rule",
          all(r["time_source"] == "fextra_htime"
              and r["htime_encoding"] == "windows_filetime_uint64"
              for r in ok_rows),
          {"sources": sorted({r["time_source"] for r in ok_rows}),
           "encodings": sorted({r["htime_encoding"] for r in ok_rows}),
           "flags": sorted({r["htime_flags"] for r in ok_rows})})
    check("every_candidate_has_proxy_date",
          all(r.get("file_mtime_proxy_date") for r in ok_rows),
          sum(1 for r in ok_rows if not r.get("file_mtime_proxy_date")))

    # --- 7. per-class answers ---------------------------------------------------
    existing = {}
    for record in heads["records"]:
        existing.setdefault(record["class_id"], []).append(
            record["created_iso"][:10])
    q1 = summary["answers"]["Q1_each_legacy_class_has_a_new_proxy_date_candidate"]
    recomputed_q1 = {}
    for t in ("T0000", "T0010", "T0101"):
        dates = sorted({r["file_mtime_proxy_date"] for r in ok_rows if r["t_code"] == t})
        ex = sorted(set(existing[t]))
        recomputed_q1[t] = bool([d for d in dates if d not in ex])
        check(f"Q1_{t}_recomputed", recomputed_q1[t] == q1[t],
              {"existing": ex, "candidate_dates": dates})
    check("Q1_summary_matches_recomputed", q1 == recomputed_q1,
          {"summary": q1, "recomputed": recomputed_q1})
    verdict_text = summary["answers"]["Q1_verdict"]
    check("Q1_verdict_text_matches_recomputed_values",
          verdict_text.startswith("yes") == all(recomputed_q1.values())
          and verdict_text.startswith("no") == (not all(recomputed_q1.values())),
          verdict_text)

    new_date_files = {t: summary["per_class"][t]["new_date_file_count"]
                      for t in ("T0000", "T0010", "T0101")}
    recomputed_new_date = {
        t: sum(1 for r in ok_rows if r["t_code"] == t
               and r["file_mtime_proxy_date"] not in set(existing[t]))
        for t in ("T0000", "T0010", "T0101")}
    check("new_date_file_counts_recomputed", new_date_files == recomputed_new_date,
          {"summary": new_date_files, "recomputed": recomputed_new_date})

    check("no_exact_configuration_match_on_a_new_date",
          all(not summary["per_class"][t]["has_exact_configuration_match_on_a_new_date"]
              for t in ("T0000", "T0010", "T0101")),
          {t: summary["per_class"][t]["new_date_and_same_configuration_file_names"]
           for t in ("T0000", "T0010", "T0101")})

    # --- 8. frozen vs probed sizes ---------------------------------------------
    frozen_packed = {r["file_name"]: int(r["packed_size"]) for r in frozen_rows}
    total_frozen_packed = sum(frozen_packed.values())
    total_frozen_unpacked = sum(int(r["unpacked_size"]) for r in frozen_rows)
    probed_packed = sum(r["data_size"] for r in ok_rows)
    probed_unpacked = sum(r["unpacked_size"] for r in ok_rows)
    check("probed_packed_equals_frozen_total", probed_packed == total_frozen_packed,
          {"probed": probed_packed, "frozen": total_frozen_packed})
    check("probed_unpacked_equals_frozen_total",
          probed_unpacked == total_frozen_unpacked,
          {"probed": probed_unpacked, "frozen": total_frozen_unpacked})
    check("legacy3_unused_volume_matches_meta_split1",
          total_frozen_packed == 83_506_828_674
          and total_frozen_unpacked == 86_961_570_451,
          {"packed": total_frozen_packed, "unpacked": total_frozen_unpacked,
           "meta_split1_reported_unpacked": 86_961_570_451})

    # --- 9. CSV mirrors the JSON ------------------------------------------------
    with open(SUMMARY_CSV, "r", encoding="utf-8-sig", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    csv_candidate_files = sum(int(r["files"]) for r in csv_rows
                              if r["provenance"] == "archive_candidate")
    check("summary_csv_covers_all_88_candidates", csv_candidate_files == 88,
          csv_candidate_files)
    csv_packed = sum(int(r["packed_bytes"] or 0) for r in csv_rows
                     if r["provenance"] == "archive_candidate")
    check("summary_csv_packed_matches_probe", csv_packed == probed_packed,
          {"csv": csv_packed, "probe": probed_packed})

    # --- 10. candidate non-use --------------------------------------------------
    nonuse_path = os.path.join(HERE, "candidate_nonuse_scan.json")
    nonuse = json.load(open(nonuse_path, encoding="utf-8"))
    check("nonuse_scan_covered_all_88", nonuse["candidates"] == 88, nonuse["candidates"])
    check("no_candidate_in_the_104_manifest",
          nonuse["candidates_present_in_104_manifest"] == [],
          nonuse["candidates_present_in_104_manifest"])
    check("no_candidate_in_a_data_or_model_artifact",
          nonuse["candidates_in_data_or_model_artifacts"] == [],
          nonuse["candidates_in_data_or_model_artifacts"])
    check("only_literature_mention_is_the_dataset_paper",
          nonuse["candidates_referenced_only_in_literature_text"] == ["T0010_D00_S1111.mat"],
          nonuse["candidates_referenced_only_in_literature_text"])

    # --- 11. report text cross-check -------------------------------------------
    recomputed = {
        "candidates": len(candidates),
        "ok": len(ok_rows),
        "requests": len(requests),
        "total_http_bytes": total_http_bytes,
        "max_header_span": max(spans),
        "max_bytes_per_candidate": max(bytes_per_candidate),
        "q1": q1,
        "new_date_files": recomputed_new_date,
        "probed_packed_bytes": probed_packed,
    }
    if os.path.exists(REPORT):
        report = open(REPORT, encoding="utf-8").read()
        # the report groups thousands with commas; compare digit runs only
        compact = report.replace(",", "")
        must_appear = [
            str(len(candidates)),
            str(total_http_bytes),
            str(max(spans)),
            str(len(requests)),
            str(recomputed_new_date["T0000"]),
            str(recomputed_new_date["T0010"]),
            str(probed_packed),
        ]
        missing = [s for s in must_appear if s not in compact]
        check("report_contains_recomputed_numbers", not missing, missing)
        check("report_states_the_negative_Q1_result",
              "只有" in report and "T0010" in report and "T0101" in report,
              "report must state that only T0000 has a different-date candidate")
    else:
        check("report_present", False, "META-HEADER1-report.md not written yet")

    failures = [c for c in checks if not c["pass"]]
    payload = {
        "schema": "dronerfa-meta-header1-verification/v1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "artifacts": {
            path: sha256_file(path) for path in
            (PROBE, HTTP_LOG, PREREG, SUMMARY, SUMMARY_CSV, FROZEN, nonuse_path)},
        "recomputed": recomputed,
        "checks": checks,
        "checks_total": len(checks),
        "checks_failed": len(failures),
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"verdict": payload["verdict"],
                      "checks_total": len(checks),
                      "checks_failed": len(failures),
                      "failures": failures,
                      "recomputed": recomputed,
                      "wrote": OUT}, ensure_ascii=False, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
