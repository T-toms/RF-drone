"""Targeted tests for ``download_full_archive.py`` (FULL-ARCHIVE1).

Every test runs against a local stub HTTP server that mimics the official
endpoint, so no real data is downloaded.  The tests cover the safety contract
that matters for an unattended multi-day run:

* a non-206 answer (server ignoring ``Range``) is refused;
* a ``Content-Range`` that disagrees with the request is refused;
* a changed total size or ETag is refused;
* an over-long body is refused *without* being written;
* a short body is retried and, when it keeps failing, stops the run;
* a valid existing ``.part`` resumes and the bytes still line up;
* a crashed tail is truncated back to the verified prefix;
* the single-instance lock stops a second download process;
* the volume marker pins the download to one volume;
* an under-sized free-space budget stops the run before any write;
* paths escaping the authorized root are refused;
* a clean run renames ``.part`` to ``.rar`` with the exact byte count.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BASELINE = Path(__file__).resolve().parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import download_full_archive as mod  # noqa: E402

# The sandbox temp directory is not writable here, so the harness works inside
# the project's own scratch tree (which is also where other suites keep state).
SCRATCH = BASELINE / "scratch" / "_fullarchive_tests"
SCRATCH.mkdir(parents=True, exist_ok=True)

PAYLOAD = bytes((i * 37 + 11) % 256 for i in range(65536))
ETAG = '"test-etag-0001"'


class StubArchive:
    """Synthetic archive plus injectable faults."""

    def __init__(self, payload: bytes = PAYLOAD, etag: str = ETAG) -> None:
        self.payload = payload
        self.etag = etag
        self.fault = None
        self.fault_remaining = 0
        self.requests = 0
        self.lock = threading.Lock()

    def arm(self, fault: str, times: int = 10_000) -> None:
        with self.lock:
            self.fault = fault
            self.fault_remaining = times

    def clear(self) -> None:
        with self.lock:
            self.fault = None
            self.fault_remaining = 0

    def take_fault(self) -> str | None:
        with self.lock:
            self.requests += 1
            if self.fault and self.fault_remaining > 0:
                self.fault_remaining -= 1
                fault, self.fault = self.fault, None if self.fault_remaining == 0 else self.fault
                return fault
            return None


def make_handler(stub: StubArchive):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):  # silence stderr noise
            pass

        def do_GET(self):  # noqa: N802 - http.server API
            total = len(stub.payload)
            fault = stub.take_fault()

            rng = self.headers.get("Range", "")
            if not rng.startswith("bytes="):
                self.send_error(400, "Range required")
                return
            first, _, last = rng[6:].partition("-")
            start = int(first)
            end = int(last) if last else total - 1
            end = min(end, total - 1)
            want = end - start + 1

            if fault == "ignore_range":
                body = stub.payload
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("ETag", stub.etag)
                self.end_headers()
                self.wfile.write(body)
                return

            if fault == "http_503":
                body = b"busy"
                self.send_response(503)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            cr_start, cr_end, cr_total = start, end, total
            if fault == "range_mismatch":
                cr_start, cr_end = start + 1, end + 1
            elif fault == "wrong_total":
                cr_total = total + 4096
            etag = '"other-etag"' if fault == "wrong_etag" else stub.etag
            declared = want + 7 if fault == "wrong_length_header" else want

            body = stub.payload[start:end + 1]

            if fault == "overlong":
                # A body longer than the requested range is only observable when
                # the response is delimited by connection close: with a correct
                # Content-Length the client stops reading at the declared end.
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
                self.send_header("ETag", stub.etag)
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body + b"\x00" * 128)
                self.close_connection = True
                return

            self.send_response(206)
            self.send_header("Content-Range", f"bytes {cr_start}-{cr_end}/{cr_total}")
            self.send_header("Content-Length", str(declared))
            self.send_header("ETag", etag)
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

            if fault == "short":
                self.wfile.write(body[: max(1, len(body) // 2)])
                self.wfile.flush()
                self.close_connection = True
            else:
                self.wfile.write(body)

    return Handler


class Harness:
    """Boots a stub server plus a temp target tree with a frozen config."""

    def __init__(self, payload: bytes = PAYLOAD) -> None:
        self.stub = StubArchive(payload)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.stub))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.tmp = SCRATCH / f"case-{uuid.uuid4().hex[:12]}"
        self.tmp.mkdir(parents=True, exist_ok=False)
        self.root = self.tmp / "target"
        self.root.mkdir(parents=True)
        self.uuid = str(uuid.uuid4())
        self.config_path = self.tmp / "config.json"

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/download?fileId=test"

    def write_config(self, **overrides) -> Path:
        data = {
            "schemaVersion": 1,
            "url": self.url,
            "expectedSizeBytes": len(self.stub.payload),
            "expectedEtag": self.stub.etag,
            "expectedExtractedBytes": 0,
            "spaceMarginBytes": 0,
            "rootDirectory": str(self.root),
            "partFile": str(self.root / "DroneRFa.rar.part"),
            "completedArchive": str(self.root / "DroneRFa.rar"),
            "extractDirectory": str(self.root / "extracted"),
            "logsDirectory": str(self.root / "logs"),
            "volumeMarker": str(self.root / "volume-marker.json"),
            "volumeMarkerUuid": self.uuid,
        }
        data.update(overrides)
        self.config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
        return self.config_path

    def init_marker(self) -> None:
        """Create the volume marker through the real --init-volume path."""
        self.write_config()
        code = mod.main(["--config", str(self.config_path), "--init-volume",
                         "--allow-insecure-http", "--no-echo"])
        assert code == mod.EXIT_OK, f"--init-volume failed: {code}"
        marker = json.loads((self.root / "volume-marker.json").read_text(encoding="utf-8"))
        self.uuid = marker["uuid"]
        self.write_config()

    def run(self, *extra: str, expect: int | None = None) -> int:
        args = ["--config", str(self.config_path), "--download",
                "--allow-insecure-http", "--no-echo"]
        args.extend(extra)
        code = mod.main(args)
        if expect is not None:
            assert code == expect, f"expected exit {expect}, got {code}"
        return code

    @property
    def part(self) -> Path:
        return self.root / "DroneRFa.rar.part"

    @property
    def archive(self) -> Path:
        return self.root / "DroneRFa.rar"

    def login(self) -> list[str]:
        log = self.root / "logs" / "download.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)


class BaseCase(unittest.TestCase):
    """Blocks are 4 KiB so every scenario runs in milliseconds."""

    BLOCK_BYTES = 4096

    def setUp(self) -> None:
        self.h = Harness()
        self.addCleanup(self.h.close)
        self._orig = (mod.MAX_BLOCK_ATTEMPTS, mod.MAX_CONSECUTIVE_TRANSIENT,
                      mod.RETRY_BACKOFF_BASE, mod.MAX_PROBE_ATTEMPTS)
        mod.MAX_BLOCK_ATTEMPTS = 2
        mod.MAX_CONSECUTIVE_TRANSIENT = 2
        mod.RETRY_BACKOFF_BASE = 0.05
        mod.MAX_PROBE_ATTEMPTS = 3

        def restore() -> None:
            (mod.MAX_BLOCK_ATTEMPTS, mod.MAX_CONSECUTIVE_TRANSIENT,
             mod.RETRY_BACKOFF_BASE, mod.MAX_PROBE_ATTEMPTS) = self._orig
        self.addCleanup(restore)

    def run_dl(self, *extra: str) -> int:
        args = ["--config", str(self.h.config_path), "--download", "--allow-insecure-http",
                "--no-echo", "--workers", "3",
                "--block-bytes", str(self.BLOCK_BYTES)]
        args.extend(extra)
        return mod.main(args)


class TestSafetyRefusals(BaseCase):
    def test_probe_rejects_range_mismatch(self) -> None:
        self.h.init_marker()
        self.h.stub.arm("range_mismatch")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists(), ".part must not be created")
        self.assertFalse(self.h.archive.exists())

    def test_probe_rejects_changed_etag(self) -> None:
        self.h.init_marker()
        self.h.stub.arm("wrong_etag")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists())

    def test_probe_rejects_changed_total(self) -> None:
        self.h.init_marker()
        self.h.stub.arm("wrong_total")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists())

    def test_non_206_is_refused(self) -> None:
        self.h.init_marker()
        self.h.stub.arm("ignore_range")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists(), "no bytes may be written when Range is ignored")
        self.assertTrue(any("206" in line for line in self.h.login()))

    def test_overlong_body_is_refused_without_writing(self) -> None:
        self.h.init_marker()
        self.h.stub.arm("overlong")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        written = self.h.part.stat().st_size if self.h.part.exists() else 0
        self.assertEqual(written, 0, "over-long body must never reach the .part")

    def test_wrong_content_length_header_is_refused(self) -> None:
        self.h.init_marker()
        self.h.stub.arm("wrong_length_header")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists())

    def test_short_body_retries_then_stops(self) -> None:
        self.h.init_marker()
        self.h.stub.arm("short")
        code = self.run_dl()
        self.assertNotEqual(code, mod.EXIT_OK)
        written = self.h.part.stat().st_size if self.h.part.exists() else 0
        self.assertEqual(written, 0, "a short block must not be partially appended")
        self.assertTrue(any("短读" in line or "重试" in line for line in self.h.login()))

    def test_transient_http_is_retried_then_succeeds(self) -> None:
        self.h.init_marker()
        self.h.stub.arm("http_503", times=2)
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_OK)
        self.assertTrue(self.h.archive.exists())


class TestVolumeMarker(BaseCase):
    def test_missing_marker_is_refused(self) -> None:
        self.h.write_config()  # never created the marker
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists())

    def test_changed_marker_uuid_is_refused(self) -> None:
        self.h.init_marker()
        marker_path = self.h.root / "volume-marker.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["uuid"] = str(uuid.uuid4())
        marker_path.write_text(json.dumps(marker, ensure_ascii=False), encoding="utf-8")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists())

    def test_changed_volume_serial_is_refused(self) -> None:
        self.h.init_marker()
        marker_path = self.h.root / "volume-marker.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["serialNumber"] = "DEADBEEF"
        marker_path.write_text(json.dumps(marker, ensure_ascii=False), encoding="utf-8")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists())

    def test_init_volume_does_not_overwrite(self) -> None:
        self.h.init_marker()
        first = json.loads((self.h.root / "volume-marker.json").read_text(encoding="utf-8"))
        code = mod.main(["--config", str(self.h.config_path), "--init-volume",
                         "--allow-insecure-http", "--no-echo"])
        self.assertEqual(code, mod.EXIT_OK)
        second = json.loads((self.h.root / "volume-marker.json").read_text(encoding="utf-8"))
        self.assertEqual(first["uuid"], second["uuid"])
        self.assertEqual(first["createdAt"], second["createdAt"])


class TestDownloadAndResume(BaseCase):
    def test_clean_download_renames_to_archive(self) -> None:
        self.h.init_marker()
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_OK)
        self.assertFalse(self.h.part.exists(), ".part must be renamed away")
        self.assertTrue(self.h.archive.exists())
        self.assertEqual(self.h.archive.stat().st_size, len(PAYLOAD))
        self.assertEqual(hashlib.sha256(self.h.archive.read_bytes()).hexdigest(),
                         hashlib.sha256(PAYLOAD).hexdigest())

    def test_bounded_run_then_resume_is_byte_identical(self) -> None:
        self.h.init_marker()
        # First run stops after ~16 KiB, leaving a legal checkpoint.
        code = self.run_dl("--max-bytes", "16384")
        self.assertEqual(code, mod.EXIT_OK)
        self.assertTrue(self.h.part.exists())
        part_size = self.h.part.stat().st_size
        self.assertGreater(part_size, 0)
        self.assertLess(part_size, len(PAYLOAD))
        self.assertEqual(self.h.part.read_bytes(), PAYLOAD[:part_size],
                         "prefix must match the archive bytes exactly")
        state = json.loads((self.h.root / "logs" / "download-state.json").read_text("utf-8"))
        self.assertEqual(state["verifiedPrefixBytes"], part_size)

        # Second run resumes and finishes.
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_OK)
        self.assertEqual(self.h.archive.stat().st_size, len(PAYLOAD))
        self.assertEqual(self.h.archive.read_bytes(), PAYLOAD)
        self.assertTrue(any("起点" in line for line in self.h.login()))

    def test_resume_after_multiple_bounded_runs(self) -> None:
        self.h.init_marker()
        for _ in range(3):
            self.assertEqual(self.run_dl("--max-bytes", "8192"), mod.EXIT_OK)
        self.assertEqual(self.run_dl(), mod.EXIT_OK)
        self.assertEqual(self.h.archive.read_bytes(), PAYLOAD)

    def test_crashed_tail_is_truncated_back_to_prefix(self) -> None:
        self.h.init_marker()
        self.assertEqual(self.run_dl("--max-bytes", "16384"), mod.EXIT_OK)
        committed = self.h.part.stat().st_size
        # Simulate a crash between writing a block and persisting the checkpoint.
        with open(self.h.part, "ab") as fh:
            fh.write(b"\xAA" * 5000)
        self.assertEqual(self.h.part.stat().st_size, committed + 5000)
        self.assertEqual(self.run_dl(), mod.EXIT_OK)
        self.assertEqual(self.h.archive.read_bytes(), PAYLOAD)
        self.assertTrue(any("截断回断点" in line for line in self.h.login()))

    def test_part_shorter_than_declared_prefix_is_refused(self) -> None:
        """A .part shorter than the checkpoint claims is a hard stop."""
        self.h.init_marker()
        self.assertEqual(self.run_dl("--max-bytes", "16384"), mod.EXIT_OK)
        state_path = self.h.root / "logs" / "download-state.json"
        state = json.loads(state_path.read_text("utf-8"))
        state["verifiedPrefixBytes"] = state["verifiedPrefixBytes"] + 4096
        state_path.write_text(json.dumps(state), encoding="utf-8")
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_PROTOCOL)

    def test_already_complete_is_idempotent(self) -> None:
        self.h.init_marker()
        self.assertEqual(self.run_dl(), mod.EXIT_OK)
        digest = hashlib.sha256(self.h.archive.read_bytes()).hexdigest()
        self.assertEqual(self.run_dl(), mod.EXIT_OK)
        self.assertEqual(hashlib.sha256(self.h.archive.read_bytes()).hexdigest(), digest)


class TestLockingAndSpace(BaseCase):
    def test_lock_prevents_second_instance(self) -> None:
        self.h.init_marker()
        lock_path = self.h.root / "logs" / "download.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with mod.InstanceLock(lock_path):
            code = self.run_dl()
            self.assertEqual(code, mod.EXIT_LOCKED)
        self.assertFalse(self.h.part.exists(), "a locked run must not write")
        # Once released, the download proceeds normally.
        self.assertEqual(self.run_dl(), mod.EXIT_OK)

    def test_space_shortfall_stops_before_writing(self) -> None:
        self.h.init_marker()
        self.h.write_config(expectedExtractedBytes=10 ** 15)
        code = self.run_dl()
        self.assertEqual(code, mod.EXIT_ERROR)
        self.assertFalse(self.h.part.exists())
        self.assertTrue(any("可用空间不足" in line for line in self.h.login()))


class TestConfigGuards(BaseCase):
    def test_path_escape_is_refused(self) -> None:
        self.h.init_marker()
        self.h.write_config(partFile=str(self.h.root.parent / "escaped.part"))
        with self.assertRaises(mod.ConfigError):
            mod.Config.load(self.h.config_path)

    def test_non_https_requires_explicit_override(self) -> None:
        self.h.init_marker()
        code = mod.main(["--config", str(self.h.config_path), "--download", "--no-echo"])
        self.assertEqual(code, mod.EXIT_PROTOCOL)
        self.assertFalse(self.h.part.exists())

    def test_status_mode_reports_checkpoint(self) -> None:
        self.h.init_marker()
        self.assertEqual(self.run_dl("--max-bytes", "8192"), mod.EXIT_OK)
        code = mod.main(["--config", str(self.h.config_path), "--status",
                         "--allow-insecure-http", "--no-echo"])
        self.assertEqual(code, mod.EXIT_OK)


class TestUnits(unittest.TestCase):
    def test_parse_content_range(self) -> None:
        self.assertEqual(mod.parse_content_range("bytes 0-7/100"), (0, 7, 100))
        self.assertEqual(mod.parse_content_range("bytes 10-19/100"), (10, 19, 100))
        self.assertIsNone(mod.parse_content_range("bytes 0-7/*"))
        self.assertIsNone(mod.parse_content_range("0-7/100"))
        self.assertIsNone(mod.parse_content_range(""))
        self.assertIsNone(mod.parse_content_range(None))

    def test_normalize_etag(self) -> None:
        self.assertEqual(mod.normalize_etag('"ABC123"'), "abc123")
        self.assertEqual(mod.normalize_etag('W/"ABC123"'), "abc123")
        self.assertEqual(mod.normalize_etag("abc123"), "abc123")
        self.assertIsNone(mod.normalize_etag(None))

    def test_assert_under_rejects_escape(self) -> None:
        root = SCRATCH / "fa-root"
        with self.assertRaises(mod.ConfigError):
            mod.assert_under(root / ".." / "outside", root)
        self.assertEqual(mod.assert_under(root / "inside", root),
                         Path(os.path.abspath(str(root / "inside"))))


if __name__ == "__main__":
    unittest.main(verbosity=2)
