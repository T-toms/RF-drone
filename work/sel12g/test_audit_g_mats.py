#!/usr/bin/env python3
"""Targeted tests for the SELECT12-G3 MAT audit probe in ``audit_g_mats.py``.

The regression this suite exists for (2026-09-25, SELECT12-G3):

    values = np.asarray(probe).reshape(-1)[probe_start:probe_end]

A DroneRFa channel is stored as ``(1, 150e6)`` float64 (~1.2 GiB).  The old line
materialised the whole channel and then sliced 1000 samples out of it, so auditing
104 files meant allocating ~1.2 GiB per file for nothing.  The replacement
(:func:`audit_g_mats.read_channel_probe`) must request *only* the bounded slice.

Because "it returns 1000 samples" alone cannot distinguish the two implementations,
this suite pins the property three ways:

1. a recording stand-in dataset proves the exact index handed to h5py is a bounded
   slice -- no ``[:]``, no whole-array access;
2. a real, sparse, 150e6-sample HDF5 channel proves the peak working set does not
   grow by anything like 1.2 GiB;
3. real small HDF5 fixtures pin the accepted layouts ``(1, N)`` / ``(N,)``, the
   rejected layouts ``(N, 1)`` / 3-D, the float-dtype rule and the finite-value probe.

Nothing here touches ``G:``: every fixture is created under
``work/sel12g/_probe_tests`` and removed afterwards.

Run: .venv\\Scripts\\python.exe work\\sel12g\\test_audit_g_mats.py
"""

from __future__ import annotations

import ctypes
import shutil
import sys
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

import audit_g_mats as audit  # noqa: E402

FIXTURE_DIR = WORK / "_probe_tests"
BIG_SAMPLES = 150_000_000  # one real DroneRFa channel: 150e6 float64 == 1.2 GiB
# Probe window for this length is [1_002_000 // 2 - 500_000, +1000) == [1000, 2000).
PROBE_WINDOW_SAMPLES = 1_002_000


class ProcessMemoryCounters(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS from ``psapi.h`` (needs explicit argtypes to marshal)."""

    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def peak_working_set_bytes() -> int | None:
    """Peak (high-water) working set of this process, or None if unreadable."""
    if sys.platform != "win32":
        return None
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        query = kernel32.K32GetProcessMemoryInfo
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        query.restype = ctypes.c_int
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001 - measurement is optional evidence
        return None


class RecordingDataset:
    """Stand-in for an h5py dataset that records exactly which index is requested.

    ``__getitem__`` returns an array of the size the requested selection would have,
    so the audit code cannot tell the difference -- but ``requests`` shows whether the
    caller asked for the whole channel or for one bounded slice.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.requests: list = []

    def __getitem__(self, key):  # noqa: ANN001 - mirrors h5py
        self.requests.append(key)
        if isinstance(key, tuple):
            _, column = key
            assert isinstance(column, slice), f"unbounded column index {column!r}"
            assert column.step in (None, 1), f"strided read {column!r}"
            length = column.stop - column.start
        else:
            assert isinstance(key, slice), f"unbounded index {key!r}"
            length = key.stop - key.start
        return np.zeros(length, dtype=np.float64)


def write_channels(path: Path, *, samples: int, dtype: str = "float64",
                   shapes: dict | None = None) -> None:
    """Write the four required channels into a small HDF5 fixture."""
    shapes = shapes or {key: (1, samples) for key in audit.REQUIRED_KEYS}
    with h5py.File(path, "w") as handle:
        for key in audit.REQUIRED_KEYS:
            shape = shapes[key]
            data = np.zeros(shape, dtype=dtype)
            handle.create_dataset(key, data=data)


class ProbeIndexTests(unittest.TestCase):
    """The probe must request one bounded slice and nothing else."""

    def test_row_layout_requests_only_the_bounded_slice(self) -> None:
        dataset = RecordingDataset((1, BIG_SAMPLES))
        values, layout, error = audit.read_channel_probe(dataset, 74_500_000, 1000)
        self.assertIsNone(error)
        self.assertEqual(layout, "row_1xN")
        self.assertEqual(values.shape, (1000,))
        self.assertEqual(dataset.requests, [(0, slice(74_500_000, 74_501_000))])

    def test_flat_layout_requests_only_the_bounded_slice(self) -> None:
        dataset = RecordingDataset((BIG_SAMPLES,))
        values, layout, error = audit.read_channel_probe(dataset, 74_500_000, 1000)
        self.assertIsNone(error)
        self.assertEqual(layout, "flat")
        self.assertEqual(values.shape, (1000,))
        self.assertEqual(dataset.requests, [slice(74_500_000, 74_501_000)])

    def test_unsupported_shapes_are_rejected_not_read(self) -> None:
        for shape in ((BIG_SAMPLES, 1), (2, 3, 4), ()):
            with self.subTest(shape=shape):
                dataset = RecordingDataset(shape)
                values, layout, error = audit.read_channel_probe(dataset, 0, 1000)
                self.assertIsNone(values)
                self.assertIsNone(layout)
                self.assertIn("unsupported channel shape", error)
                self.assertEqual(dataset.requests, [], "rejected shape must not be read")

    def test_non_positive_probe_length_is_rejected(self) -> None:
        dataset = RecordingDataset((1, BIG_SAMPLES))
        values, layout, error = audit.read_channel_probe(dataset, 10, 0)
        self.assertIsNone(values)
        self.assertIn("invalid probe length", error)
        self.assertEqual(dataset.requests, [])


class InspectMatTests(unittest.TestCase):
    """End-to-end behaviour of :func:`audit_g_mats.inspect_mat` on real HDF5 files."""

    @classmethod
    def setUpClass(cls) -> None:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(FIXTURE_DIR, ignore_errors=True)

    def test_row_layout_file_is_valid(self) -> None:
        path = FIXTURE_DIR / "row-1xN.mat"
        write_channels(path, samples=20_000)
        info = audit.inspect_mat(path)
        self.assertTrue(info["ok"], info)
        self.assertEqual(info["read_probe"]["layout"], "row_1xN")
        self.assertEqual(info["read_probe"]["samples"], 1000)
        self.assertEqual(info["read_probe"]["requested"], 1000)
        self.assertTrue(info["read_probe"]["finite"])
        self.assertTrue(info["lengths_consistent"])
        self.assertTrue(info["all_float"])
        self.assertEqual(info["source_samples"], 20_000)

    def test_flat_layout_file_is_valid(self) -> None:
        path = FIXTURE_DIR / "flat-N.mat"
        write_channels(path, samples=20_000,
                       shapes={key: (20_000,) for key in audit.REQUIRED_KEYS})
        info = audit.inspect_mat(path)
        self.assertTrue(info["ok"], info)
        self.assertEqual(info["read_probe"]["layout"], "flat")

    def test_column_vector_layout_is_rejected(self) -> None:
        path = FIXTURE_DIR / "column-Nx1.mat"
        write_channels(path, samples=20_000,
                       shapes={key: (20_000, 1) for key in audit.REQUIRED_KEYS})
        info = audit.inspect_mat(path)
        self.assertFalse(info["ok"])
        self.assertIn("unsupported channel shape", info["read_probe"]["error"])

    def test_integer_channels_are_rejected(self) -> None:
        path = FIXTURE_DIR / "int-channels.mat"
        write_channels(path, samples=20_000, dtype="int32")
        info = audit.inspect_mat(path)
        self.assertFalse(info["ok"])
        self.assertFalse(info["all_float"])

    def test_non_finite_sample_inside_probe_window_is_detected(self) -> None:
        path = FIXTURE_DIR / "nan-in-probe.mat"
        # 1_002_000 samples put the probe window at [1000, 2000).
        write_channels(path, samples=PROBE_WINDOW_SAMPLES)
        with h5py.File(path, "r+") as handle:
            handle["RF0_I"][0, 1500] = np.nan  # inside [1000, 2000)
        info = audit.inspect_mat(path)
        self.assertEqual(info["read_probe"]["start"], 1000)
        self.assertFalse(info["ok"])
        self.assertFalse(info["read_probe"]["finite"])

    def test_non_finite_sample_outside_probe_window_is_not_probed(self) -> None:
        path = FIXTURE_DIR / "nan-outside-probe.mat"
        write_channels(path, samples=PROBE_WINDOW_SAMPLES)
        with h5py.File(path, "r+") as handle:
            handle["RF0_I"][0, 0] = np.nan  # outside [1000, 2000)
        info = audit.inspect_mat(path)
        self.assertEqual(info["read_probe"]["start"], 1000)
        self.assertTrue(info["ok"], info)
        self.assertTrue(info["read_probe"]["finite"])

    def test_probe_windows_are_inside_the_channel(self) -> None:
        path = FIXTURE_DIR / "tiny-channel.mat"
        write_channels(path, samples=400)  # shorter than the 1000-sample probe
        info = audit.inspect_mat(path)
        self.assertTrue(info["ok"], info)
        self.assertEqual(info["read_probe"]["start"], 0)
        self.assertEqual(info["read_probe"]["requested"], 400)
        self.assertEqual(info["read_probe"]["samples"], 400)

    def test_missing_channel_is_reported(self) -> None:
        path = FIXTURE_DIR / "missing-channel.mat"
        write_channels(path, samples=20_000)
        with h5py.File(path, "r+") as handle:
            del handle["RF1_Q"]
        info = audit.inspect_mat(path)
        self.assertFalse(info["ok"])
        self.assertEqual(info["missing_required"], ["RF1_Q"])

    def test_big_sparse_channel_is_probed_without_materialising_it(self) -> None:
        """Peak working set must not grow by anything like one 1.2 GiB channel."""
        path = FIXTURE_DIR / "big-sparse.mat"
        with h5py.File(path, "w") as handle:
            for key in audit.REQUIRED_KEYS:
                # Chunked and never written: the file stays tiny, but the dataspace is
                # the real 150e6-sample size, so a whole-channel read would allocate
                # 1.2 GiB of fill values.
                handle.create_dataset(
                    key, shape=(1, BIG_SAMPLES), dtype="float64", chunks=(1, 500_000)
                )
        before = peak_working_set_bytes()
        if before is None:
            self.skipTest("peak working set is not readable on this platform")
        info = audit.inspect_mat(path)
        after = peak_working_set_bytes()
        self.assertTrue(info["ok"], info)
        self.assertEqual(info["read_probe"]["samples"], 1000)
        self.assertEqual(info["source_samples"], BIG_SAMPLES)
        growth = after - before
        self.assertLess(
            growth,
            300 * 1024**2,
            f"probe grew peak working set by {growth / 1024**2:.0f} MiB; the "
            "whole-channel read is back",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
