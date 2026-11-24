"""Re-generate the reviewable UI rendering evidence with a headless browser.

This is a thin wrapper around ``verify_signal_analyzer.py --render-only``: it boots the
real server with the six frozen checkpoints, performs one real bounded analysis through
the HTTP API (a heldout Phantom file, 6 windows), writes the offline snapshot plus the
copied review artefacts, and then asks headless Chrome/Edge for a screenshot.

Use it on a normal desktop session: the DSH file sandbox denies Chrome the named pipes
its crash reporter and mojo channel need, so screenshots cannot be produced from inside
that sandbox (see ``outputs/signal_analyzer_validation_20260920/verification_report.md``).

    .venv\\Scripts\\python.exe dronerfa_baseline\\render_ui_evidence.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import verify_signal_analyzer as verification


def main() -> int:
    return verification.main(["--render-only", "--device", "auto"])


if __name__ == "__main__":
    raise SystemExit(main())
