#!/usr/bin/env python3
"""Required failure cases and small live-response checks for the heldout tooling.

Four kinds of check, all chosen for risk in the new code:

1. ``RangeValidationTests`` – a range response is rejected when the status is not
   206, when ``Content-Range`` does not match the requested start/end, and when
   the body would exceed the requested length.  This guards the failure mode
   where a 200 answer to a Range request streams the whole ~615 GB archive.
2. ``SafetyTests`` – destructive operations outside the authorised F: roots, and
   any deletion of a file that is not part of this round, are refused.
3. ``VectorReadTests`` – ``dronerfa_data.dataset_length``/``read_vector_slice``
   handle ``(1, N)`` and ``(N, 1)`` stored vectors, so the index/window checks do
   not silently rely on ``shape[0]``.
4. ``LiveRangeTests`` – the real ScienceDB endpoint, two tiny requests, verifying
   that the *same GET response* carries a matching ``Content-Range`` for a
   candidate's own block offset (needs network; skipped without it).

Run:  .venv\\Scripts\\python.exe dronerfa_baseline\\test_heldout_prep.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _path in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

# Import the wrapper first: it bootstraps `requests`, which
# download_dronerfa_subset imports at module level.
import download_heldout_files as heldout  # noqa: E402
import download_dronerfa_subset as base  # noqa: E402

from dronerfa_data import dataset_length, read_vector_slice  # noqa: E402

ROOT = PROJECT_ROOT
CANDIDATES = (
    ROOT / "dronerfa_baseline/outputs/noise_ab_replicates_20260917/candidate_heldout_files.json"
)

# Body blocks under test must be staged on the authorised data volume, not in C:\TEMP.
SCRATCH = heldout.DEFAULT_STAGING / "scratch"


class _FakeCompleted:
    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class RangeValidationTests(unittest.TestCase):
    """Reject responses whose status, Content-Range or length cannot be trusted."""

    def _client_with_result(self, status_code: str, header_text: str, body: bytes):
        client = heldout.CurlRangeClient(base.SOURCE_URL, scratch_dir=SCRATCH)

        def fake_invoke(extra, timeout):  # noqa: ANN001, ANN202
            # Reproduce curl's contract: --dump-header <path> --output <path>.
            output_path = Path(extra[extra.index("--output") + 1])
            dump_path = Path(extra[extra.index("--dump-header") + 1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(body)
            dump_path.write_text(header_text, encoding="utf-8")
            return _FakeCompleted(stdout=status_code.encode())

        client._invoke = fake_invoke  # type: ignore[method-assign]
        return client

    def test_scratch_dir_is_used_and_never_on_c(self) -> None:
        client = heldout.CurlRangeClient(base.SOURCE_URL, scratch_dir=SCRATCH)
        self.assertEqual(client.scratch_dir.drive.upper(), "F:")
        self.assertEqual(client.stats["scratch_dir"], str(client.scratch_dir))
        body_path, headers_path = client._new_scratch()
        self.assertEqual(body_path.parent, client.scratch_dir)
        self.assertEqual(headers_path.parent, client.scratch_dir)
        self.assertNotEqual(body_path.drive.upper(), "C:")

    def test_scratch_dir_is_required(self) -> None:
        with self.assertRaises(base.DroneRFaError):
            heldout.CurlRangeClient(base.SOURCE_URL)

    def test_http_200_is_rejected_without_streaming_the_archive(self) -> None:
        """A 200 answer to a Range request must abort, not read the full archive."""
        client = self._client_with_result(
            "200",
            "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
            "Content-Length: 615955805466\r\n",
            b"x" * 8,
        )
        with self.assertRaises(base.DroneRFaError) as ctx:
            client.probe_range(0, 7)
        self.assertIn("期望206", str(ctx.exception))
        self.assertEqual(client.stats["protocol_violations"], [])

    def test_content_range_mismatch_is_rejected(self) -> None:
        """Headers from the same GET that do not match the request must fail."""
        client = self._client_with_result(
            "206",
            "HTTP/1.1 206 Partial Content\r\n"
            "Content-Range: bytes 4096-4351/615955805466\r\n"
            "Content-Length: 256\r\n",
            b"y" * 256,
        )
        with self.assertRaises(heldout.RangeProtocolError) as ctx:
            client.probe_range(1024, 1279)
        self.assertIn("Content-Range区间不符", str(ctx.exception))
        self.assertEqual(len(client.stats["protocol_violations"]), 1)

    def test_oversized_body_is_rejected(self) -> None:
        """A body longer than requested is a protocol violation, not data."""
        client = self._client_with_result(
            "206",
            "HTTP/1.1 206 Partial Content\r\n"
            "Content-Range: bytes 0-7/615955805466\r\n"
            "Content-Length: 8\r\n",
            b"z" * 64,
        )
        with self.assertRaises(heldout.RangeProtocolError) as ctx:
            client.probe_range(0, 7)
        self.assertIn("超过请求长度", str(ctx.exception))

    def test_content_encoding_is_rejected(self) -> None:
        """A compressed body would break offset arithmetic."""
        client = self._client_with_result(
            "206",
            "HTTP/1.1 206 Partial Content\r\n"
            "Content-Range: bytes 0-7/615955805466\r\n"
            "Content-Encoding: gzip\r\n"
            "Content-Length: 8\r\n",
            b"w" * 8,
        )
        with self.assertRaises(heldout.RangeProtocolError) as ctx:
            client.probe_range(0, 7)
        self.assertIn("Content-Encoding", str(ctx.exception))


class SafetyTests(unittest.TestCase):
    """Destructive paths must stay inside the authorised roots and file list."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.staging = root / "downloads"
        self.extract = root / "heldout"
        self.staging.mkdir()
        self.extract.mkdir()
        self.name = "T0000_D00_S0100.mat"
        self.safety = heldout.SafePaths(
            roots=(self.staging, self.extract),
            deletable=frozenset(
                {
                    self.staging / f"{self.name}.rar",
                    self.staging / f"{self.name}.rar.part",
                    self.staging / f"{self.name}.rar.state.json",
                    self.extract / self.name,
                }
            ),
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_delete_outside_authorised_roots_is_refused(self) -> None:
        outsider = Path(tempfile.gettempdir()) / "not_ours.mat"
        outsider.write_bytes(b"keep me")
        try:
            with self.assertRaises(heldout.SafetyError):
                self.safety.safe_unlink(outsider, what="越界文件")
            self.assertTrue(outsider.exists(), "越界文件不应被删除")
        finally:
            outsider.unlink(missing_ok=True)

    def test_delete_of_unknown_mat_in_root_is_refused(self) -> None:
        """An existing .mat that is not part of this round must never be removed."""
        unknown = self.extract / "T0010_D00_S0000.mat"
        unknown.write_bytes(b"pre-existing")
        with self.assertRaises(heldout.SafetyError):
            self.safety.safe_unlink(unknown, what="非本轮文件")
        self.assertTrue(unknown.exists())

    def test_recursive_delete_of_root_itself_is_refused(self) -> None:
        with self.assertRaises(heldout.SafetyError):
            self.safety.safe_rmtree(self.staging, what="授权根目录")
        self.assertTrue(self.staging.exists())

    def test_write_target_outside_roots_is_refused(self) -> None:
        with self.assertRaises(heldout.SafetyError):
            self.safety.safe_replace(
                self.staging / "src.bin",
                Path(r"C:\Users\Administrator\Documents\DroneRFa数据集\evil.mat"),
                what="越界落位",
            )

    def test_listed_artefacts_are_deletable(self) -> None:
        archive = self.staging / f"{self.name}.rar"
        archive.write_bytes(b"temp")
        self.safety.safe_unlink(archive, what="本轮临时压缩包")
        self.assertFalse(archive.exists())


class VectorReadTests(unittest.TestCase):
    """Stored vectors must be measured with the project's own helpers."""

    def setUp(self) -> None:
        import h5py

        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "vectors.h5"
        with h5py.File(self.path, "w") as handle:
            values = np.arange(1000, dtype=np.float32)
            handle.create_dataset("row_vector", data=values.reshape(1, 1000))
            handle.create_dataset("column_vector", data=values.reshape(1000, 1))
            handle.create_dataset("flat_vector", data=values)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_dataset_length_handles_2d_vectors(self) -> None:
        import h5py

        with h5py.File(self.path, "r") as handle:
            for key in ("row_vector", "column_vector", "flat_vector"):
                with self.subTest(key=key):
                    self.assertEqual(dataset_length(handle[key]), 1000)

    def test_read_vector_slice_handles_2d_vectors(self) -> None:
        import h5py

        with h5py.File(self.path, "r") as handle:
            for key in ("row_vector", "column_vector", "flat_vector"):
                with self.subTest(key=key):
                    values = read_vector_slice(handle[key], 10, 20)
                    self.assertEqual(values.shape, (20,))
                    self.assertEqual(values.dtype, np.float32)
                    np.testing.assert_allclose(values, np.arange(10, 30, dtype=np.float32))


class LiveRangeTests(unittest.TestCase):
    """Small real requests: the same GET response must carry a matching range."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.candidates = json.loads(CANDIDATES.read_text(encoding="utf-8"))["files"]
        cls.manifest = {
            Path(entry["name"]).name: entry
            for entry in json.loads(
                (ROOT / "dronerfa_manifest.json").read_text(encoding="utf-8")
            )["entries"]
        }

    def test_signature_and_total_size(self) -> None:
        client = heldout.CurlRangeClient(base.SOURCE_URL, scratch_dir=SCRATCH)
        try:
            total, _etag = client.inspect()
            self.assertEqual(total, base.EXPECTED_ARCHIVE_SIZE)
        finally:
            client.cleanup_scratch()

    def test_candidate_offset_returns_its_own_block_header(self) -> None:
        """Probe the first and last candidate at their manifest offset."""
        client = heldout.CurlRangeClient(base.SOURCE_URL, scratch_dir=SCRATCH)
        try:
            for item in (self.candidates[0], self.candidates[-1]):
                with self.subTest(name=item["name"]):
                    short = Path(item["name"]).name
                    entry = base.ArchiveEntry(**self.manifest[short])
                    self.assertEqual(entry.block_offset, item["block_offset"])
                    self.assertEqual(entry.packed_size, item["packed_size"])
                    _status, headers, body = client.probe_range(
                        entry.block_offset, entry.block_offset + heldout.SAFETY_PROBE_SPAN - 1
                    )
                    content_range = heldout.parse_content_range(headers["content-range"])
                    self.assertIsNotNone(content_range)
                    assert content_range is not None
                    self.assertEqual(content_range[0], entry.block_offset)
                    self.assertEqual(content_range[2], base.EXPECTED_ARCHIVE_SIZE)
                    header = base.parse_block(
                        _MemoryClient(body), 0, len(body)  # type: ignore[arg-type]
                    )
                    self.assertEqual(header["name"], item["name"])
                    self.assertEqual(header["data_size"], entry.packed_size)
                    self.assertEqual(
                        header["header_total_size"] + header["data_size"], entry.block_size
                    )
        finally:
            client.cleanup_scratch()


class _MemoryClient:
    """Minimal get_range() client so base.parse_block can read a probe buffer."""

    def __init__(self, blob: bytes) -> None:
        self._blob = blob

    def get_range(self, start: int, end: int) -> bytes:
        if end >= len(self._blob):
            raise AssertionError(
                f"parse_block requested {start}-{end} beyond the {len(self._blob)}-byte probe"
            )
        return self._blob[start : end + 1]


class EnvironmentTests(unittest.TestCase):
    def test_target_directories_are_on_the_authorised_volume(self) -> None:
        for path in (heldout.DEFAULT_STAGING, heldout.DEFAULT_EXTRACT):
            with self.subTest(path=str(path)):
                self.assertEqual(Path(os.path.abspath(str(path))).drive.upper(), "F:")
        self.assertEqual(heldout.DEFAULT_STAGING.name, "downloads")

    def test_curl_is_available_for_the_download_backend(self) -> None:
        self.assertTrue(heldout.find_curl())
        result = subprocess.run(
            [heldout.find_curl(), "--version"], capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
