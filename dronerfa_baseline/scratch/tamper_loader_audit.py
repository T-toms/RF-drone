"""Reproducible tamper tool for the loader identity audit (rework R2 evidence).

Writes a mutated copy of ``loader_read_audit.json`` so the verifier can be shown to
reject a tampered run through its normal command line.  The mutation is deliberately
subtle: one training window disappears from one epoch *and* the corresponding digest
is rewritten to a plausible-looking value, so a checker that only compares totals
would still pass it.

Usage::

    python scratch/tamper_loader_audit.py <run-dir> [--kind drop_window|swap_epochs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))


def tamper_drop_window(payload: dict) -> dict:
    keys = sorted(payload["train"]["keys"])
    entry = payload["train"]["keys"][keys[0]]
    entry["reads"]["windows_read"] -= 1
    entry["consumed"]["windows_completed"] -= 1
    entry["consumed"]["batches_completed"] -= 1
    # Rewrite the orders to something self-consistent but wrong.
    entry["consumed"]["completed_order_sha256"] = "0" * 64
    entry["consumed"]["delivered_order_sha256"] = "0" * 64
    entry["reads"]["position_order_sha256"] = "0" * 64
    entry["reads"]["position_multiset_sha256"] = "0" * 64
    return payload


def tamper_swap_epochs(payload: dict) -> dict:
    keys = sorted(payload["train"]["keys"])
    if len(keys) < 2:
        raise SystemExit("need at least two training epochs to swap")
    first, second = keys[0], keys[1]
    payload["train"]["keys"][second]["consumed"]["completed_order_sha256"] = payload[
        "train"
    ]["keys"][first]["consumed"]["completed_order_sha256"]
    return payload


KINDS = {"drop_window": tamper_drop_window, "swap_epochs": tamper_swap_epochs}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Tamper with loader_read_audit.json")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--kind", choices=sorted(KINDS), default="drop_window")
    args = parser.parse_args(argv)
    path = args.run_dir / "loader_read_audit.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    payload = KINDS[args.kind](payload)
    # No BOM: the artefact must stay a normal UTF-8 JSON document.
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"tampered {path} with kind={args.kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
