"""INFER-UI1 unit + server tests (bounded fixtures, no GPU-heavy work).

Run with the project interpreter::

    .venv\\Scripts\\python.exe dronerfa_baseline\\test_signal_analyzer.py

The suite builds its own small MAT/NPY/NPZ/RAW fixtures under
``dronerfa_baseline/scratch/signal_analyzer_tests`` so the F: analysis-output root
and the frozen CASCADE3 artefacts are never touched.  Model weights are stubbed
with freshly initialised ``ResNet18`` instances of the correct shape: these tests
check plumbing, formats, planning, aggregation, cancellation and HTTP behaviour,
while the real frozen checkpoints are exercised by ``verify_signal_analyzer.py``.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import h5py
import numpy as np
import torch

import signal_analyzer_core as core
import signal_analyzer_server as server
from model import ResNet18, SpectrogramFrontEnd

SCRATCH = PROJECT_DIR / "scratch" / "signal_analyzer_tests"
WINDOW = core.WINDOW_SAMPLES

# the six frozen CASCADE3 checkpoint hashes, read from the frozen config itself
FROZEN_WEIGHTS: dict[tuple[int, int], str] = {
    (int(pairing["seed"]), stage): str(pairing[f"stage{stage}_sha256"])
    for pairing in core.load_json(core.CASCADE_CONFIG_PATH)["seed_pairing"]
    for stage in (1, 2)
}


def tone(samples: int, seed: int = 0, frequency: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(seed)
    index = np.arange(samples, dtype=np.float32)
    return (
        0.7 * np.exp(2j * np.pi * frequency * index) + 0.05 * rng.standard_normal(samples)
    ).astype(np.complex64)


def write_mat(path: Path, iq: np.ndarray) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("RF0_I", data=np.asarray(iq.real, dtype=np.float64).reshape(1, -1))
        handle.create_dataset("RF0_Q", data=np.asarray(iq.imag, dtype=np.float64).reshape(1, -1))
        handle.create_dataset("RF1_I", data=np.zeros((1, iq.shape[0]), dtype=np.float64))
        handle.create_dataset("RF1_Q", data=np.zeros((1, iq.shape[0]), dtype=np.float64))


def make_fake_bundle(device: str = "cpu") -> core.ModelBundle:
    torch.manual_seed(0)
    device_object = torch.device(device)
    models: dict[int, dict] = {}
    for seed in core.TRAINING_SEEDS:
        frontend = SpectrogramFrontEnd(
            n_fft=core.N_FFT, hop_length=core.HOP_LENGTH, image_size=core.IMAGE_SIZE
        ).to(device_object)
        frontend.eval()
        models[seed] = {"frontend": frontend, "stage1_epoch": 5, "stage2_epoch": 3}
        for stage in (1, 2):
            network = ResNet18(input_channels=1, num_classes=2).to(device_object)
            network.eval()
            for parameter in network.parameters():
                parameter.requires_grad_(False)
            models[seed][f"stage{stage}_network"] = network
            models[seed][f"stage{stage}_network_checkpoint"] = "stub"
            models[seed][f"stage{stage}_network_sha256"] = "stub"
            models[seed][f"stage{stage}_network_epoch"] = 5 if stage == 1 else 3
    return core.ModelBundle(
        device=device_object,
        models=models,
        frontend=models[core.TRAINING_SEEDS[0]]["frontend"],
        config={},
        config_sha256="stub",
        expected_config_sha256="stub",
        weights=[
            {
                "training_seed": seed,
                "stage": stage,
                "checkpoint": "stub",
                "sha256": FROZEN_WEIGHTS[(seed, stage)],
                "epoch": 5 if stage == 1 else 3,
                "bytes": 0,
            }
            for seed in core.TRAINING_SEEDS
            for stage in (1, 2)
        ],
        weights_verified=True,
        errors=[],
        load_seconds=0.0,
        environment=core.environment_report(device_object),
    )


class FixtureCase(unittest.TestCase):
    """Shared fixture directory (project scratch, never F:)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = SCRATCH
        if cls.root.exists():
            shutil.rmtree(cls.root, ignore_errors=True)
        cls.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.root, ignore_errors=True)


class TestFormatsAndSources(FixtureCase):
    def test_detects_mat_npy_npz_raw(self) -> None:
        iq = tone(4000, seed=1)
        mat = self.root / "fixture.mat"
        write_mat(mat, iq)
        npy = self.root / "fixture.npy"
        np.save(npy, iq)
        npy_2n = self.root / "fixture_2n.npy"
        np.save(npy_2n, np.stack((iq.real, iq.imag), axis=0).astype(np.float32))
        npz = self.root / "fixture.npz"
        np.savez(npz, iq=iq)
        npz_pair = self.root / "fixture_pair.npz"
        np.savez(npz_pair, i=iq.real.astype(np.float32), q=iq.imag.astype(np.float32))
        raw = self.root / "fixture.c64.raw"
        iq.astype(np.complex64).tofile(raw)
        raw_interleaved = self.root / "fixture.iq.raw"
        np.stack((iq.real, iq.imag), axis=1).astype(np.float32).tofile(raw_interleaved)

        self.assertEqual(core.detect_format(mat)["magic_detected"], "mat_v73")
        self.assertEqual(core.detect_format(npy)["magic_detected"], "npy")
        self.assertEqual(core.detect_format(npz)["magic_detected"], "npz")
        self.assertEqual(core.detect_format(raw)["magic_detected"], "raw")

        mat_info = core.inspect_path(mat)
        self.assertEqual(mat_info["samples"], 4000)
        self.assertEqual(mat_info["default_candidate"]["label"], "RF0_I/RF0_Q")
        self.assertTrue(mat_info["short_file"])

        for path in (npy, npy_2n, npz, npz_pair):
            info = core.inspect_path(path)
            self.assertEqual(info["samples"], 4000, path.name)
        raw_info = core.inspect_path(raw, {"dtype": "complex64", "complex": True})
        self.assertEqual(raw_info["samples"], 4000)
        raw_iq_info = core.inspect_path(
            raw_interleaved, {"dtype": "float32", "order": "IQ"}
        )
        self.assertEqual(raw_iq_info["samples"], 4000)

        # unknown/failed layouts must be reported, not silently accepted
        bad_npy = self.root / "bad.npy"
        np.save(bad_npy, np.zeros((3, 7), dtype=np.float32))
        self.assertFalse(core.inspect_path(bad_npy)["layout_supported"])
        with self.assertRaises(core.SignalAnalyzerError):
            core.open_source(bad_npy)

    def test_mat_missing_keys_reports_available(self) -> None:
        path = self.root / "other_keys.mat"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("RF1_I", data=np.zeros(4000, dtype=np.float64))
            handle.create_dataset("RF1_Q", data=np.zeros(4000, dtype=np.float64))
        with self.assertRaises(core.SignalAnalyzerError) as context:
            core.open_source(path)
        self.assertIn("RF1_I", str(context.exception))

    def test_streaming_reads_are_chunked(self) -> None:
        """A window larger than the read chunk must be fetched in bounded slices."""
        calls: list[tuple[int, int]] = []
        iq = tone(2 * WINDOW + 123, seed=3)

        class Counting(core.IQSource):
            def __init__(self) -> None:
                super().__init__(int(iq.shape[0]), "counting")

            def _read_i(self, start: int, stop: int) -> np.ndarray:
                calls.append((int(start), int(stop)))
                return np.asarray(iq[start:stop].real, dtype=np.float32)

            def _read_q(self, start: int, stop: int) -> np.ndarray:
                return np.asarray(iq[start:stop].imag, dtype=np.float32)

        source = Counting()
        original = core.READ_CHUNK_SAMPLES
        try:
            core.READ_CHUNK_SAMPLES = 400_000
            data = source.read_iq(core.READ_CHUNK_SAMPLES - 10, WINDOW)
        finally:
            core.READ_CHUNK_SAMPLES = original
        self.assertEqual(data.shape[0], WINDOW)
        self.assertTrue(np.allclose(data[:5], iq[400_000 - 10 : 400_000 - 5]))
        self.assertTrue(len(calls) >= 3, calls)
        for start, stop in calls:
            self.assertLessEqual(stop - start, 400_000)
        # the first read starts exactly at the requested offset
        self.assertEqual(calls[0][0], 400_000 - 10)
        self.assertEqual(calls[-1][1], 400_000 - 10 + WINDOW)

    def test_iq_order_iq_vs_qi(self) -> None:
        iq = tone(3000, seed=5)
        interleaved = self.root / "order.raw"
        np.stack((iq.real, iq.imag), axis=1).astype(np.float32).tofile(interleaved)
        iq_source = core.open_source(interleaved, {"dtype": "float32", "order": "IQ"})
        qi_source = core.open_source(interleaved, {"dtype": "float32", "order": "QI"})
        try:
            read_iq = iq_source.read_iq(0, 100)
            read_qi = qi_source.read_iq(0, 100)
        finally:
            iq_source.close()
            qi_source.close()
        self.assertTrue(np.allclose(read_iq.real, iq[:100].real, atol=1e-6))
        self.assertTrue(np.allclose(read_iq.imag, iq[:100].imag, atol=1e-6))
        self.assertTrue(np.allclose(read_qi.real, iq[:100].imag, atol=1e-6))
        self.assertTrue(np.allclose(read_qi.imag, iq[:100].real, atol=1e-6))

    def test_short_signal_and_bad_window_requests(self) -> None:
        iq = tone(5000, seed=7)
        mat = self.root / "short.mat"
        write_mat(mat, iq)
        info = core.inspect_path(mat)
        self.assertTrue(info["short_file"])
        with self.assertRaises(core.SignalAnalyzerError):
            core.plan_windows(info["samples"])
        with self.assertRaises(core.SignalAnalyzerError):
            core.plan_windows(3 * WINDOW, start_sample=3 * WINDOW + 1)
        with self.assertRaises(core.SignalAnalyzerError):
            core.plan_windows(3 * WINDOW, stride_samples=0)
        with self.assertRaises(core.SignalAnalyzerError):
            core.plan_windows(3 * WINDOW, max_windows=0)
        long_enough = self.root / "long.npy"
        np.save(long_enough, tone(2 * WINDOW + 10, seed=8))
        source = core.open_source(long_enough)
        try:
            self.assertEqual(source.samples, 2 * WINDOW + 10)
            with self.assertRaises(core.SignalAnalyzerError):
                source.read_iq(2 * WINDOW + 5, WINDOW)
            with self.assertRaises(core.SignalAnalyzerError):
                source.read_iq(-1, WINDOW)
            with self.assertRaises(core.SignalAnalyzerError):
                source.read_iq(0, 0)
        finally:
            source.close()


class TestPlanningAndAggregation(unittest.TestCase):
    def test_window_plan_tail_and_stride(self) -> None:
        plan = core.plan_windows(2_500_000)
        self.assertEqual(plan.windows, 2)
        self.assertEqual(plan.requested_windows, 2)
        self.assertEqual(plan.dropped_samples, 500_000)
        self.assertEqual(plan.window_bounds(1), (1_000_000, 2_000_000))
        strided = core.plan_windows(5_000_000, start_sample=1_000_000, stride_samples=2_000_000)
        self.assertEqual(strided.windows, 2)
        self.assertEqual(strided.analysed_stop, 4_000_000)
        self.assertEqual(strided.dropped_samples, 1_000_000)
        limited = core.plan_windows(10_000_000, max_windows=3)
        self.assertEqual(limited.windows, 3)
        self.assertEqual(limited.requested_windows, 10)
        self.assertEqual(limited.dropped_samples, 10_000_000 - 3 * WINDOW)

    def test_majority_rule(self) -> None:
        self.assertEqual(core.majority_label([1, 1, 1]), ("DJI Phantom 4 Pro", "3/3", False))
        self.assertEqual(core.majority_label([2, 2, 1]), ("DJI Air 2S", "2/3", False))
        self.assertEqual(core.majority_label([0, 0, 2]), ("Background", "2/3", False))
        label, agreement, uncertain = core.majority_label([0, 1, 2])
        self.assertEqual(label, core.UNCERTAIN_LABEL)
        self.assertTrue(uncertain)
        self.assertEqual(agreement, "1/3")
        # a 1-1-1 split must never be forced into a known class
        self.assertNotIn(label, {"Background", "DJI Phantom 4 Pro", "DJI Air 2S"})

    def test_file_summary_rules(self) -> None:
        def row(label: str, agreement: str = "3/3") -> dict:
            return {
                "ensemble_class_name": label,
                "ensemble_agreement": agreement,
                "is_drone": label in {"DJI Phantom 4 Pro", "DJI Air 2S"},
                "is_uncertain": "true" if label == core.UNCERTAIN_LABEL else "false",
            }

        drone_dominant = core.summarize_file(
            [row("DJI Air 2S")] * 3 + [row("Background")] * 2
        )
        self.assertEqual(drone_dominant["dominant_label"], "drone_dominant")
        self.assertTrue(drone_dominant["any_drone_window"])
        self.assertAlmostEqual(drone_dominant["drone_ratio"], 0.6)
        self.assertEqual(drone_dominant["machine_summary"], "DJI Air 2S")

        background_dominant = core.summarize_file(
            [row("Background")] * 3 + [row("DJI Phantom 4 Pro")] * 2
        )
        self.assertEqual(background_dominant["dominant_label"], "background_dominant")
        self.assertTrue(background_dominant["any_drone_window"])
        self.assertEqual(background_dominant["drone_ratio"], 0.4)

        mixed = core.summarize_file([row("Background"), row("DJI Phantom 4 Pro")])
        self.assertEqual(mixed["dominant_label"], "mixed_or_uncertain")

        split_machines = core.summarize_file(
            [row("DJI Phantom 4 Pro"), row("DJI Air 2S"), row(core.UNCERTAIN_LABEL, "1/3")]
        )
        # 2 of 3 windows are drone windows -> drone dominant (>50%), machine is mixed
        self.assertEqual(split_machines["dominant_label"], "drone_dominant")
        self.assertEqual(split_machines["machine_summary"], "混合或不确定")
        self.assertEqual(split_machines["uncertain_windows"], 1)
        self.assertFalse(split_machines["model_agreement_rate"] == 1.0)

        # exactly 50% drone windows are NOT dominant
        half = core.summarize_file(
            [row("DJI Phantom 4 Pro"), row("Background"), row(core.UNCERTAIN_LABEL, "1/3")]
        )
        self.assertEqual(half["dominant_label"], "mixed_or_uncertain")
        self.assertTrue(half["any_drone_window"])

        no_drone = core.summarize_file([row("Background")] * 4)
        self.assertFalse(no_drone["any_drone_window"])
        self.assertEqual(no_drone["machine_summary"], "无无人机窗")
        self.assertEqual(no_drone["model_agreement_rate"], 1.0)


class TestPathSafety(unittest.TestCase):
    def test_allowed_root_and_traversal(self) -> None:
        roots = (PROJECT_DIR,)
        inside = PROJECT_DIR / "signal_analyzer_core.py"
        self.assertEqual(server.resolve_allowed(inside, roots), inside.resolve())
        with self.assertRaises(server.ApiError):
            server.resolve_allowed(Path("C:/Windows/System32/drivers/etc/hosts"), roots)
        with self.assertRaises(server.ApiError):
            server.resolve_allowed(Path("signal_analyzer_core.py"), roots)
        with self.assertRaises(server.ApiError):
            server.resolve_allowed(PROJECT_DIR / "does_not_exist_12345.bin", roots)
        with self.assertRaises(server.ApiError):
            server.safe_child(PROJECT_DIR, "../../secret.txt")
        with self.assertRaises(server.ApiError):
            server.safe_child(PROJECT_DIR, "..\\secret.txt")
        self.assertEqual(
            server.safe_child(PROJECT_DIR / "scratch", "a/nested.txt"),
            (PROJECT_DIR / "scratch" / "a" / "nested.txt").resolve(),
        )

    def test_non_loopback_host_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            server.create_server(host="0.0.0.0", port=0, preload=False)


class TestStreamAnalysis(unittest.TestCase):
    """Runs the real streaming loop with stub weights on CPU."""

    def test_shared_features_and_per_seed_rows(self) -> None:
        bundle = make_fake_bundle("cpu")
        iq = tone(3 * WINDOW + 17, seed=11)
        source = core.IQSource(0, "memory")

        class Memory(core.IQSource):
            def __init__(self) -> None:
                super().__init__(int(iq.shape[0]), "memory")

            def read_iq(self, start: int, length: int) -> np.ndarray:
                return np.asarray(iq[start : start + length], dtype=np.complex64)

        plan = core.plan_windows(iq.shape[0], max_windows=3)
        seen: list[dict] = []
        run = core.run_stream_analysis(
            source=Memory(), plan=plan, bundle=bundle, progress=seen.append
        )
        self.assertEqual(len(run["ensemble_rows"]), 3)
        for seed in core.TRAINING_SEEDS:
            self.assertEqual(len(run["per_seed_rows"][seed]), 3)
        # stage 1 runs for all three seeds on the SAME feature digest of a window
        for index in range(3):
            digests = {
                run["per_seed_rows"][seed][index]["stage1_feature_digest"]
                for seed in core.TRAINING_SEEDS
            }
            self.assertEqual(len(digests), 1)
        # stage 2, when executed, must reuse the same tensor digest
        for seed in core.TRAINING_SEEDS:
            for row in run["per_seed_rows"][seed]:
                if row["stage2_executed"]:
                    self.assertEqual(row["stage2_feature_digest"], row["stage1_feature_digest"])
                else:
                    self.assertIsNone(row["stage2_logit_phantom"])
                    self.assertEqual(row["final_prediction"], 0)
        # every per-seed intermediate value is preserved (never only the ensemble)
        for row in run["per_seed_rows"][core.TRAINING_SEEDS[0]]:
            for key in (
                "stage1_logit_background",
                "stage1_logit_drone",
                "stage1_margin",
                "final_prediction",
                "stage1_feature_digest",
            ):
                self.assertIn(key, row)
        self.assertIn("stage1", {item["stage"] for item in seen})
        self.assertIn("features", {item["stage"] for item in seen})
        plan_dict = plan.to_dict()
        self.assertEqual(plan_dict["dropped_samples"], 17)

    def test_cancellation_stops_without_publishing(self) -> None:
        bundle = make_fake_bundle("cpu")
        iq = tone(6 * WINDOW, seed=13)

        class Memory(core.IQSource):
            def __init__(self) -> None:
                super().__init__(int(iq.shape[0]), "memory")
                self.closed = False

            def read_iq(self, start: int, length: int) -> np.ndarray:
                return np.asarray(iq[start : start + length], dtype=np.complex64)

            def close(self) -> None:
                self.closed = True

        source = Memory()
        plan = core.plan_windows(iq.shape[0])
        state = {"cancelled": False}

        def progress(payload: dict) -> None:
            if payload["stage"] == "stage1":
                state["cancelled"] = True

        with self.assertRaises(core.SignalsCancelled):
            core.run_stream_analysis(
                source=source,
                plan=plan,
                bundle=bundle,
                progress=progress,
                cancelled=lambda: state["cancelled"],
            )
        self.assertTrue(source.closed, "取消后必须关闭文件句柄")


class ServerCase(FixtureCase):
    """Boots a real HTTP server on a random loopback port."""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.output_root = cls.root / "outputs"
        cls.output_root.mkdir(parents=True, exist_ok=True)
        cls.iq = tone(3 * WINDOW + 5, seed=21)
        cls.mat = cls.root / "server_fixture.mat"
        write_mat(cls.mat, cls.iq)
        cls.bundle = make_fake_bundle("cpu")
        cls.server, cls.manager = server.create_server(
            host="127.0.0.1",
            port=0,
            output_root=cls.output_root,
            allowed_roots=(cls.root, PROJECT_DIR),
            preload=False,
            quiet=True,
        )
        cls.manager._bundle = cls.bundle
        cls.manager._bundle_state = "ready"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.upload_dir = cls.server.RequestHandlerClass.upload_dir
        cls.upload_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.manager.shutdown()
        cls.server.server_close()
        super().tearDownClass()

    # -- helpers -----------------------------------------------------------
    def http(self, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, method=method)
        request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def http_bytes(self, path: str) -> tuple[int, bytes, str]:
        with urllib.request.urlopen(self.base + path, timeout=60) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")

    def wait_job(self, job_id: str, timeout: float = 600.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            _, payload = self.http(f"/api/jobs/{job_id}")
            job = payload["job"]
            if job["status"] in {"completed", "failed", "cancelled"}:
                return job
            time.sleep(0.2)
        raise AssertionError(f"job {job_id} did not settle")

    # -- tests -------------------------------------------------------------
    def setUp(self) -> None:
        # the manager bundle is shared class state; re-install a fresh stub per test
        self.manager._bundle = make_fake_bundle("cpu")
        self.manager._bundle_state = "ready"

    def test_health_and_config(self) -> None:
        status, payload = self.http("/api/health")
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["host"], "127.0.0.1")
        self.assertEqual(payload["limits"]["max_concurrent_gpu_jobs"], 1)
        self.assertEqual(payload["model"]["state"], "ready")
        self.assertTrue(payload["model"]["model"]["weights_verified"])
        status, config = self.http("/api/config")
        self.assertEqual(config["labels"]["0"], "Background")
        self.assertEqual(config["window_samples"], WINDOW)

    def test_static_page_is_offline(self) -> None:
        status, body, content_type = self.http_bytes("/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        text = body.decode("utf-8")
        self.assertIn("任意IQ信号分析界面", text)
        for needle in ("http://cdn", "https://cdn", "//unpkg", "//cdnjs", "googleapis"):
            self.assertNotIn(needle, text)
        for asset in ("/styles.css", "/app.js"):
            status, body, _ = self.http_bytes(asset)
            self.assertEqual(status, 200, asset)
            self.assertGreater(len(body), 200)
        # static traversal must fail (never leak a project source file)
        try:
            status, _, _ = self.http_bytes("/../signal_analyzer_core.py")
        except urllib.error.HTTPError as error:
            status = error.code
        self.assertIn(status, (400, 403, 404))

    def test_files_listing_and_outside_root(self) -> None:
        status, payload = self.http(f"/api/files?root={urllib.parse.quote(str(self.root))}")
        self.assertEqual(status, 200)
        names = {entry["name"] for entry in payload["entries"]}
        self.assertIn("server_fixture.mat", names)
        status, payload = self.http(
            f"/api/files?root={urllib.parse.quote('C:/Windows')}"
        )
        self.assertEqual(status, 403)

    def test_inspect_blocks_unconfirmed_rate_and_short_file(self) -> None:
        status, payload = self.http(
            "/api/inspect",
            "POST",
            {"path": str(self.mat), "sample_rate_confirmed": False},
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["classification_allowed"])
        self.assertTrue(any("100 MHz" in item for item in payload["classification_blockers"]))

        status, payload = self.http(
            "/api/inspect",
            "POST",
            {
                "path": str(self.mat),
                "sample_rate_confirmed": True,
                "sample_rate_hz": 50_000_000,
            },
        )
        self.assertFalse(payload["classification_allowed"])
        self.assertTrue(any("不是 100 MHz" in item for item in payload["classification_blockers"]))

        short = self.root / "short_server.mat"
        write_mat(short, tone(10_000, seed=2))
        status, payload = self.http(
            "/api/inspect",
            "POST",
            {"path": str(short), "sample_rate_confirmed": True},
        )
        self.assertFalse(payload["classification_allowed"])
        self.assertTrue(any("不补零" in item for item in payload["classification_blockers"]))

        status, payload = self.http(
            "/api/inspect", "POST", {"path": "C:/Windows/win.ini"}
        )
        self.assertEqual(status, 403)

    def test_full_http_flow_inspect_analyze_poll_download(self) -> None:
        status, payload = self.http(
            "/api/inspect",
            "POST",
            {
                "path": str(self.mat),
                "sample_rate_confirmed": True,
                "max_windows": 3,
            },
        )
        self.assertTrue(payload["classification_allowed"])
        self.assertEqual(payload["window_plan"]["windows"], 3)
        self.assertEqual(payload["window_plan"]["dropped_samples"], 5)

        status, payload = self.http(
            "/api/analyze",
            "POST",
            {
                "path": str(self.mat),
                "sample_rate_confirmed": True,
                "max_windows": 3,
                "display_name": self.mat.name,
            },
        )
        self.assertEqual(status, 202)
        job = self.wait_job(payload["job"]["job_id"])
        self.assertEqual(job["status"], "completed", job.get("error"))
        result = job["result"]
        summary = result["analysis"]["file_summary"]
        self.assertEqual(summary["analysed_windows"], 3)
        self.assertEqual(summary["dropped_samples"], 5)
        self.assertEqual(len(result["analysis"]["window_series"]), 3)
        self.assertIn("seed_rows", result["analysis"])

        csv_status, csv_body, csv_type = self.http_bytes(result["downloads"]["predictions_csv"])
        self.assertEqual(csv_status, 200)
        text = csv_body.decode("utf-8-sig")
        self.assertIn("stage1_logit_background", text)
        self.assertEqual(len(text.strip().splitlines()), 1 + 3 * 3)  # header + 3 windows x 3 seeds
        ensemble_status, ensemble_body, _ = self.http_bytes(result["downloads"]["ensemble_csv"])
        self.assertEqual(len(ensemble_body.decode("utf-8-sig").strip().splitlines()), 1 + 3)
        for key in ("analysis_json", "summary_md", "model_hashes"):
            status, body, _ = self.http_bytes(result["downloads"][key])
            self.assertEqual(status, 200, key)
            self.assertGreater(len(body), 10, key)
        md = self.http_bytes(result["downloads"]["summary_md"])[1].decode("utf-8")
        self.assertIn("主导结论", md)
        self.assertIn("不是概率", md)
        for key in ("timeline_png", "waveform_png"):
            status, body, content_type = self.http_bytes(result["downloads"][key])
            self.assertEqual(status, 200, key)
            self.assertIn("image/png", content_type)
            self.assertTrue(body.startswith(b"\x89PNG"), key)
        status, blob, content_type = self.http_bytes(result["downloads"]["zip"])
        self.assertEqual(content_type, "application/zip")
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = set(archive.namelist())
        self.assertIn("window_predictions.csv", names)
        self.assertIn("summary.md", names)

        status, payload = self.http(f"/api/jobs/{job['job_id']}/stft?window=2")
        self.assertEqual(status, 200)
        self.assertEqual(payload["window_index"], 2)
        self.assertEqual(payload["start_sample"], 2 * WINDOW)
        self.assertEqual(len(payload["stft_series"]["values"]), core.N_FFT)
        self.assertEqual(len(payload["waveform_series"]["i"]), payload["waveform_series"]["points"])
        status, payload = self.http(f"/api/jobs/{job['job_id']}/stft?window=99")
        self.assertEqual(status, 400)

        analysis = json.loads(
            self.http_bytes(result["downloads"]["analysis_json"])[1].decode("utf-8")
        )
        self.assertEqual(analysis["input"]["rf_channel"], "RF0")
        self.assertFalse(analysis["input"]["added_noise"])
        self.assertEqual(analysis["inspection"]["detected_format"], "mat_v73")
        self.assertTrue(analysis["model"]["cascade_config_sha256_match"])
        # the frozen manifest must always list all six seed-matched checkpoints
        self.assertEqual(len(analysis["model"]["weights"]), 6)
        manifest = {
            (entry["training_seed"], entry["stage"]): entry["sha256"]
            for entry in analysis["model"]["weights"]
        }
        self.assertEqual(len(manifest), 6)
        self.assertEqual(manifest[(2026, 1)], FROZEN_WEIGHTS[(2026, 1)])
        self.assertEqual(manifest[(2028, 2)], FROZEN_WEIGHTS[(2028, 2)])
        self.assertTrue(analysis["model"]["weights_verified"])

    def test_concurrency_limit_and_cancel(self) -> None:
        big = self.root / "server_big.mat"
        if not big.exists():
            write_mat(big, tone(8 * WINDOW, seed=31))
        status, payload = self.http(
            "/api/analyze",
            "POST",
            {"path": str(big), "sample_rate_confirmed": True, "display_name": "cancel.mat"},
        )
        self.assertEqual(status, 202)
        first = payload["job"]["job_id"]
        # a second submission while the first is queued/running must be refused
        status, second = self.http(
            "/api/analyze",
            "POST",
            {"path": str(self.mat), "sample_rate_confirmed": True},
        )
        self.assertEqual(status, 409, second)
        self.assertIn("单 GPU 队列", second["error"])
        # wait until the worker has really started and created its output directory,
        # then cancel while it is provably still running (deterministic, not a race)
        deadline = time.time() + 60
        started = False
        while time.time() < deadline:
            _, self_payload = self.http(f"/api/jobs/{first}")
            current = self_payload["job"]
            if current["status"] == "running" and current.get("output_dir"):
                started = True
                break
            if current["status"] in {"completed", "failed", "cancelled"}:
                break
            time.sleep(0.01)
        self.assertTrue(started, "任务在 60 秒内没有进入 running 状态")
        status, payload = self.http(f"/api/jobs/{first}/cancel", "POST", {})
        self.assertEqual(status, 200)
        status_at_cancel = payload["job"]["status"]
        self.assertEqual(status_at_cancel, "running")
        job = self.wait_job(first)
        self.assertEqual(job["status"], "cancelled", job.get("error"))
        # a cancelled job must publish nothing and clean up its partial directory
        self.assertIsNone(
            job["output_dir"],
            f"取消后仍保留产物目录；日志={job['log'][-4:]}",
        )
        self.assertFalse(any(self.output_root.glob("*cancel*")))
        # the queue accepts a new job afterwards
        status, payload = self.http(
            "/api/analyze",
            "POST",
            {"path": str(self.mat), "sample_rate_confirmed": True, "max_windows": 1},
        )
        self.assertEqual(status, 202)
        job = self.wait_job(payload["job"]["job_id"])
        self.assertEqual(job["status"], "completed", job.get("error"))

    def test_upload_chunking_and_abort_cleanup(self) -> None:
        payload_bytes = self.mat.read_bytes()
        upload_id = "testupload01"
        name = "uploaded_fixture.mat"
        url = f"{self.base}/api/upload/{upload_id}"

        def send(chunk: bytes, stage: str) -> tuple[int, dict]:
            request = urllib.request.Request(url, data=chunk, method="POST")
            request.add_header("Content-Type", "application/octet-stream")
            request.add_header("X-Upload-Stage", stage)
            request.add_header("X-File-Name", urllib.parse.quote(name))
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.status, json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                return error.code, json.loads(error.read().decode("utf-8"))

        send(b"", "start")
        half = len(payload_bytes) // 2
        status, first = send(payload_bytes[:half], "chunk")
        self.assertEqual(status, 200)
        self.assertEqual(first["received_bytes"], half)
        # abort must delete the partial file and leave nothing behind
        status, aborted = send(b"", "abort")
        self.assertTrue(aborted["aborted"])
        partial = Path(first["partial"])
        self.assertFalse(partial.exists())
        self.assertFalse(Path(aborted["path"]).exists())

        send(b"", "start")
        send(payload_bytes[:half], "chunk")
        status, finished = send(payload_bytes[half:], "finish")
        self.assertEqual(status, 200)
        uploaded = Path(finished["path"])
        self.assertTrue(uploaded.is_file())
        self.assertEqual(uploaded.stat().st_size, len(payload_bytes))
        self.assertEqual(core.detect_format(uploaded)["magic_detected"], "mat_v73")
        # the uploaded file is inside the upload root and can be analysed
        status, payload = self.http(
            "/api/inspect",
            "POST",
            {"path": str(uploaded), "sample_rate_confirmed": True, "max_windows": 1},
        )
        self.assertTrue(payload["classification_allowed"])
        uploaded.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
