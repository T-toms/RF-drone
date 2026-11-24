"""INFER-UI1 server: stdlib-only local HTTP service for the Chinese IQ UI.

Only ``127.0.0.1`` is bound.  The UI is served from ``signal_analyzer_web/`` and
the compute work happens on a single background GPU worker (one job at a time) by
calling into ``signal_analyzer_core``.

No third-party web framework is used because the project virtual environment has
no Streamlit/Gradio/Flask/FastAPI/Plotly; the module only needs the standard
library plus the already-installed torch/numpy/h5py/matplotlib.

Endpoints::

    GET  /api/health
    GET  /api/files?root=...&subdir=...
    GET  /api/config
    POST /api/inspect
    POST /api/analyze
    POST /api/upload/<upload_id>
    GET  /api/jobs/<job_id>
    POST /api/jobs/<job_id>/cancel
    GET  /api/jobs/<job_id>/stft?window=N
    GET  /api/jobs/<job_id>/artifact?name=...
    GET  /api/jobs/<job_id>/download.zip
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import posixpath
import queue
import re
import socket
import sys
import threading
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch

import signal_analyzer_core as core

WEB_DIR = PROJECT_DIR / "signal_analyzer_web"
SCRATCH_DIR = PROJECT_DIR / "scratch"
UPLOAD_DIR = SCRATCH_DIR / "signal_analyzer_uploads"
JOB_WORK_DIR = SCRATCH_DIR / "signal_analyzer_jobs"
DEFAULT_OUTPUT_ROOT = Path("F:/DroneRFa数据集/analysis_outputs")
DEFAULT_ALLOWED_ROOTS: tuple[Path, ...] = (
    Path("F:/DroneRFa数据集"),
    PROJECT_ROOT,
)
MAX_JSON_BODY = 1 << 20
UPLOAD_CHUNK = 1 << 22  # 4 MiB per streamed upload chunk
MAX_UPLOAD_BYTES = 64 * 1024**3  # 64 GiB guard

STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".csv": "text/csv; charset=utf-8",
    ".zip": "application/zip",
}

STAGE_ZH = {
    "queued": "排队等待",
    "loading_models": "加载模型",
    "opening": "打开文件",
    "reading": "读取窗口",
    "features": "生成特征",
    "stage1": "第一阶段推理",
    "stage2": "第二阶段推理",
    "windows": "汇总窗口",
    "exporting": "导出结果",
    "done": "完成",
    "failed": "失败",
    "cancelled": "已取消",
}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


# ---------------------------------------------------------------------------
# path safety
# ---------------------------------------------------------------------------
def resolve_allowed(path: Path, roots: tuple[Path, ...], must_exist: bool = True) -> Path:
    """Resolve ``path`` and require the real path to stay inside one allowed root."""
    raw = Path(path)
    if not raw.is_absolute():
        raise ApiError(f"必须使用绝对路径：{raw}", 400)
    try:
        resolved = raw.resolve(strict=must_exist)
    except OSError as error:
        raise ApiError(f"路径不可用：{raw}（{error}）", 400) from error
    for root in roots:
        try:
            root_resolved = Path(root).resolve(strict=False)
        except OSError:  # pragma: no cover - unreadable root
            continue
        if resolved == root_resolved or root_resolved in resolved.parents:
            return resolved
    allowed = "；".join(str(Path(root).resolve(strict=False)) for root in roots)
    raise ApiError(
        f"路径越界：{resolved} 不在允许根目录内。允许根目录：{allowed}",
        403,
    )


def is_within(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolve_allowed(path, roots, must_exist=False)
        return True
    except ApiError:
        return False


def safe_child(base: Path, name: str) -> Path:
    """Join ``name`` onto ``base`` rejecting any traversal component."""
    candidate = str(name or "").replace("\\", "/")
    if candidate in {"", "."}:
        raise ApiError("文件名不能为空。", 400)
    normalized = posixpath.normpath(candidate)
    if normalized.startswith("../") or normalized == ".." or normalized.startswith("/"):
        raise ApiError(f"非法文件名：{name}", 400)
    target = (Path(base) / Path(*normalized.split("/"))).resolve(strict=False)
    root = Path(base).resolve(strict=False)
    if target != root and root not in target.parents:
        raise ApiError(f"非法文件名：{name}", 400)
    return target


# ---------------------------------------------------------------------------
# job records
# ---------------------------------------------------------------------------
@dataclass
class Job:
    job_id: str
    request: dict[str, Any]
    status: str = "queued"
    stage: str = "queued"
    stage_text: str = STAGE_ZH["queued"]
    progress: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested: bool = False
    error: str | None = None
    result: dict[str, Any] | None = None
    log: list[str] = field(default_factory=list)
    source_path: str | None = None
    output_dir: str | None = None
    window_index: int | None = None

    def public(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "stage_text": self.stage_text,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            # None while a cancelled job has had its partial directory removed
            "output_dir": self.output_dir,
            "source_path": self.source_path,
            "log": self.log[-40:],
        }
        if self.result is not None:
            payload["result"] = self.result
        return payload


class JobManager:
    """Single background GPU worker: one queued/running job at a time."""

    def __init__(
        self,
        bundle_loader: Callable[[], core.ModelBundle],
        output_root: Path,
        allowed_roots: tuple[Path, ...],
    ) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._bundle: core.ModelBundle | None = None
        self._bundle_error: str | None = None
        self._bundle_state = "loading"
        self._bundle_seconds: float | None = None
        self._bundle_loader = bundle_loader
        self._output_root = Path(output_root)
        self._allowed_roots = allowed_roots
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._threads: list[threading.Thread] = []
        self._shutdown = False

    # -- model bundle ------------------------------------------------------
    def preload_models(self) -> None:
        started = time.perf_counter()
        try:
            bundle = self._bundle_loader()
        except Exception as error:  # noqa: BLE001 - surfaced through /api/health
            with self._lock:
                self._bundle_state = "failed"
                self._bundle_error = f"{type(error).__name__}: {error}"
            return
        with self._lock:
            self._bundle = bundle
            self._bundle_state = "ready"
            self._bundle_error = None
            self._bundle_seconds = time.perf_counter() - started

    def bundle_status(self) -> dict[str, Any]:
        with self._lock:
            bundle = self._bundle
            payload: dict[str, Any] = {
                "state": self._bundle_state,
                "error": self._bundle_error,
                "load_seconds": None
                if self._bundle_seconds is None
                else round(self._bundle_seconds, 3),
            }
            if bundle is not None:
                payload["model"] = bundle.to_dict()
                payload["environment"] = bundle.environment
            return payload

    def require_bundle(self) -> core.ModelBundle:
        with self._lock:
            if self._bundle is None:
                raise ApiError(
                    "模型尚未加载完成或加载失败：" + str(self._bundle_error or "正在加载，请稍候"),
                    503,
                )
            return self._bundle

    # -- workers -----------------------------------------------------------
    def start(self, workers: int = 1) -> None:
        for index in range(max(1, workers)):
            thread = threading.Thread(
                target=self._worker_loop, name=f"signal-analyzer-worker-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def shutdown(self) -> None:
        self._shutdown = True
        for _ in self._threads:
            self._queue.put("__stop__")

    def _worker_loop(self) -> None:  # pragma: no cover - exercised through HTTP tests
        while not self._shutdown:
            job_id = self._queue.get()
            if job_id == "__stop__":
                return
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                if job.cancel_requested:
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.stage_text = STAGE_ZH["cancelled"]
                    job.finished_at = datetime.now().isoformat(timespec="seconds")
                    continue
                job.status = "running"
                job.started_at = datetime.now().isoformat(timespec="seconds")
                job.stage = "orphan"
            try:
                self._run_job(job)
            except core.SignalsCancelled:
                with self._lock:
                    job.status = "cancelled"
                    job.stage = "cancelled"
                    job.stage_text = STAGE_ZH["cancelled"]
                    job.finished_at = datetime.now().isoformat(timespec="seconds")
                    job.log.append("任务已取消，未发布任何正式结果。")
                self._cleanup_partial_output(job)
            except Exception as error:  # noqa: BLE001 - must never kill the server
                with self._lock:
                    job.status = "failed"
                    job.stage = "failed"
                    job.stage_text = STAGE_ZH["failed"]
                    job.error = f"{type(error).__name__}: {error}"
                    job.finished_at = datetime.now().isoformat(timespec="seconds")
                    job.log.append(
                        "任务失败：" + str(error) + "\n" + traceback.format_exc(limit=6)
                    )
                self._cleanup_partial_output(job)

    def _cleanup_partial_output(self, job: Job) -> None:
        output_dir = job.output_dir
        if not output_dir:
            return
        path = Path(output_dir)
        if not path.exists():
            job.output_dir = None
            return
        try:
            import shutil

            shutil.rmtree(path)
            job.log.append(f"已清理未完成产物目录：{path}")
            job.output_dir = None
        except Exception as error:  # pragma: no cover - best effort
            job.log.append(f"清理未完成产物目录失败：{path}（{error}）")

    # -- public API --------------------------------------------------------
    def submit(self, request: dict[str, Any]) -> Job:
        with self._lock:
            pending = [
                job
                for job in self._jobs.values()
                if job.status in {"queued", "running"}
            ]
            if pending:
                raise ApiError(
                    "已有分析任务在排队或运行（单 GPU 队列，一次只跑一个）。"
                    "请等待完成或取消后再提交。",
                    409,
                )
            job_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{len(self._order) + 1:03d}"
            job = Job(job_id=job_id, request=dict(request))
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise ApiError(f"任务不存在：{job_id}", 404)
        return job

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._jobs[job_id].public() for job_id in reversed(self._order)][:50]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        with self._lock:
            job.cancel_requested = True
            if job.status == "queued":
                job.status = "cancelled"
                job.stage = "cancelled"
                job.stage_text = STAGE_ZH["cancelled"]
                job.finished_at = datetime.now().isoformat(timespec="seconds")
            elif job.status == "running":
                job.log.append("已请求取消，将在当前批次结束后停止。")
        return job

    def _update(self, job: Job, **fields: Any) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(job, key, value)

    # -- the actual work ---------------------------------------------------
    def _run_job(self, job: Job) -> None:
        request = job.request

        def guard() -> None:
            """Honour a cancellation before any expensive or publishing step."""
            if job.cancel_requested:
                raise core.SignalsCancelled("任务已被取消。")

        guard()
        bundle = self.require_bundle()
        guard()
        source_path = Path(str(request["path"]))
        inspection = core.inspect_path(source_path, request.get("params") or {})
        job.source_path = str(source_path)
        guard()

        if request.get("require_classification", True):
            if not request.get("sample_rate_confirmed"):
                raise core.SignalAnalyzerError(
                    "尚未确认采样率为 100 MHz：可以检查文件预览，但本界面在未确认前不做分类。"
                )
            if int(request.get("sample_rate_hz", core.SAMPLE_RATE_HZ)) != core.SAMPLE_RATE_HZ:
                raise core.SignalAnalyzerError(
                    f"采样率 {request['sample_rate_hz']} Hz 与冻结协议 100 MHz 不匹配，"
                    "本界面不对该信号做分类（可先预览）。"
                )

        samples = int(inspection.get("samples") or 0)
        plan = core.plan_windows(
            samples=samples,
            start_sample=int(request.get("start_sample") or 0),
            stride_samples=int(request.get("stride_samples") or core.DEFAULT_STRIDE_SAMPLES),
            max_windows=request.get("max_windows"),
            sample_rate_hz=int(request.get("sample_rate_hz") or core.SAMPLE_RATE_HZ),
        )
        guard()

        display_name = request.get("display_name") or source_path.name
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = self._output_root / f"{stamp}_{core.safe_name(display_name)}"
        job.output_dir = str(output_dir)
        self._update(job, stage="opening", stage_text=STAGE_ZH["opening"])

        source = core.open_source(source_path, request.get("params") or {}, inspection)

        def progress(payload: dict[str, Any]) -> None:
            stage = str(payload.get("stage") or "reading")
            self._update(
                job,
                stage=stage,
                stage_text=STAGE_ZH.get(stage, stage),
                progress={
                    "windows_done": int(payload.get("windows_done") or 0),
                    "windows_total": int(payload.get("windows_total") or plan.windows),
                    "ratio": float(payload.get("ratio") or 0.0),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                },
            )

        run = core.run_stream_analysis(
            source=source,
            plan=plan,
            bundle=bundle,
            progress=progress,
            cancelled=lambda: job.cancel_requested,
            batch_size=core.BATCH_SIZE,
        )

        self._update(job, stage="exporting", stage_text=STAGE_ZH["exporting"])
        summary = core.summarize_file(
            run["ensemble_rows"],
            total_windows_available=plan.requested_windows,
            dropped_samples=plan.dropped_samples,
        )
        payload: dict[str, Any] = {
            "schema_version": 1,
            "task": "INFER-UI1",
            "created_at": core.utc_now(),
            "status": "completed",
            "input": {
                "path": str(source_path),
                "display_name": display_name,
                "sample_rate_hz": int(request.get("sample_rate_hz") or core.SAMPLE_RATE_HZ),
                "sample_rate_confirmed": bool(request.get("sample_rate_confirmed")),
                "rf_channel": "RF0",
                "added_noise": False,
                "params": request.get("params") or {},
            },
            "inspection": {key: value for key, value in inspection.items() if key != "groups"},
            "window_plan": plan.to_dict(),
            "model": bundle.to_dict(),
            "environment": bundle.environment,
            "file_summary": summary,
            "timing": run["timing"],
            "seeds": run["seeds"],
            "ensemble_rule": (
                "三个训练种子的最终三级标签多数决；1-1-1 记为 uncertain，不强行归类。"
            ),
            "score_note": "margin 是 logit 差值（决策分数），不是概率。",
            "window_series": [
                {
                    "window_index": int(row["window_index"]),
                    "start_sample": int(row["start_sample"]),
                    "stop_sample": int(row["stop_sample"]),
                    "start_seconds": float(row["start_seconds"]),
                    "stop_seconds": float(row["stop_seconds"]),
                    "ensemble_class_name": str(row["ensemble_class_name"]),
                    "ensemble_agreement": str(row["ensemble_agreement"]),
                    "is_drone": bool(row["is_drone"]),
                    "is_uncertain": bool(row["is_uncertain"]),
                    "votes_seed_2026": int(row["votes_seed_2026"]),
                    "votes_seed_2027": int(row["votes_seed_2027"]),
                    "votes_seed_2028": int(row["votes_seed_2028"]),
                }
                for row in run["ensemble_rows"]
            ],
            "boundaries": [
                "只支持 Background / DJI Phantom 4 Pro / DJI Air 2S 三类。",
                "只在用户确认 100 MS/s 与 RF0 协议时分类。",
                "未知设备可能被分到已知类别。",
                "文件级结论是窗口多数描述，不是另一个校准模型。",
                "用户信号不加噪声、不重采样。",
            ],
        }
        stft_window = None
        if run["ensemble_rows"]:
            chosen = int(request.get("stft_window_index", 0))
            chosen = max(0, min(chosen, len(run["ensemble_rows"]) - 1))
            row = run["ensemble_rows"][chosen]
            stft_source = core.open_source(source_path, request.get("params") or {}, inspection)
            try:
                iq = stft_source.read_iq(int(row["start_sample"]), core.WINDOW_SAMPLES)
            finally:
                stft_source.close()
            stft_window = {
                "iq": iq,
                "window_index": int(row["window_index"]),
                "start_sample": int(row["start_sample"]),
                "stop_sample": int(row["stop_sample"]),
            }
            payload["stft_series"] = core.compute_stft_image(iq)
            preview_iq = run["preview_iq"]
            if preview_iq is not None:
                payload["waveform_series"] = core.preview_series(preview_iq)

        flat_rows = core.flatten_prediction_rows(run["per_seed_rows"], run["seeds"])
        embedded_limit = int(request.get("embedded_seed_rows", 600))
        payload["seed_rows"] = _json_safe(flat_rows[:embedded_limit])
        payload["seed_rows_hidden"] = max(0, len(flat_rows) - embedded_limit)

        artifacts = core.export_analysis(
            output_dir=output_dir,
            payload=payload,
            per_seed_rows=run["per_seed_rows"],
            ensemble_rows=run["ensemble_rows"],
            preview=run["preview_iq"],
            stft_window=stft_window,
        )
        job.result = {
            "analysis": _public_analysis(payload),
            "artifacts": artifacts,
            "downloads": {
                "predictions_csv": f"/api/jobs/{job.job_id}/artifact?name=window_predictions.csv",
                "ensemble_csv": f"/api/jobs/{job.job_id}/artifact?name=ensemble_predictions.csv",
                "analysis_json": f"/api/jobs/{job.job_id}/artifact?name=analysis.json",
                "summary_md": f"/api/jobs/{job.job_id}/artifact?name=summary.md",
                "model_hashes": f"/api/jobs/{job.job_id}/artifact?name=model_hashes.json",
                "timeline_png": f"/api/jobs/{job.job_id}/artifact?name=timeline.png",
                "waveform_png": f"/api/jobs/{job.job_id}/artifact?name=waveform_preview.png",
                "zip": f"/api/jobs/{job.job_id}/download.zip",
            },
        }
        job.window_index = int(stft_window["window_index"]) if stft_window else None
        self._update(
            job,
            status="completed",
            stage="done",
            stage_text=STAGE_ZH["done"],
            progress={
                "windows_done": len(run["ensemble_rows"]),
                "windows_total": plan.windows,
                "ratio": 1.0,
                "elapsed_seconds": run["timing"]["total_seconds"],
            },
            finished_at=datetime.now().isoformat(timespec="seconds"),
        )


def _json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays so ``json.dumps`` cannot fail."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _public_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """The subset forwarded to the browser (per-window series capped, CSV complete)."""
    seed_rows = list(payload.get("seed_rows") or [])
    hidden = int(payload.get("seed_rows_hidden") or 0)
    return {
        "created_at": payload["created_at"],
        "input": payload["input"],
        "inspection": {
            key: payload["inspection"].get(key)
            for key in ("detected_format", "samples", "size_bytes", "short_file")
        },
        "window_plan": payload["window_plan"],
        "file_summary": payload["file_summary"],
        "timing": payload["timing"],
        "environment": payload["environment"],
        "model": payload["model"],
        "waveform_series": payload.get("waveform_series"),
        "stft_series": payload.get("stft_series"),
        "window_series": payload.get("window_series"),
        "seed_rows": seed_rows,
        "seed_rows_truncated": hidden > 0,
        "artifacts": payload.get("artifacts"),
        "stft_preview": payload.get("stft_preview"),
        "ensemble_rule": payload["ensemble_rule"],
        "score_note": payload["score_note"],
        "boundaries": payload["boundaries"],
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class AnalyzerHandler(BaseHTTPRequestHandler):
    server_version = "DroneRFaSignalAnalyzer/1.0"
    protocol_version = "HTTP/1.1"

    # injected by create_server
    manager: JobManager
    allowed_roots: tuple[Path, ...]
    web_dir: Path
    job_work_dir: Path
    upload_dir: Path

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if getattr(self.server, "quiet", False):
            return
        sys.stderr.write(
            "%s - %s\n" % (self.address_string(), format % args)
        )

    # -- helpers -----------------------------------------------------------
    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"ok": False, "error": message}, status)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_JSON_BODY:
            raise ApiError("请求体过大。", 413)
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ApiError(f"请求 JSON 解析失败：{error}", 400) from error
        if not isinstance(payload, dict):
            raise ApiError("请求体必须是 JSON 对象。", 400)
        return payload

    def _send_file(self, path: Path, download_name: str | None = None) -> None:
        if not path.is_file():
            self._send_error_json(f"文件不存在：{path.name}", 404)
            return
        body = path.read_bytes()
        content_type = STATIC_TYPES.get(
            path.suffix.lower(), mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        )
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if download_name:
            self.send_header(
                "Content-Disposition",
                f"attachment; filename*=UTF-8''{_url_quote(download_name)}",
            )
        self.end_headers()
        self.wfile.write(body)

    # -- routing -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            route = parsed.path
            if route.startswith("/api/"):
                self._handle_api_get(route, parse_qs(parsed.query))
            else:
                self._serve_static(route)
        except ApiError as error:
            self._send_error_json(error.message, error.status)
        except BrokenPipeError:  # pragma: no cover - client went away
            pass
        except Exception as error:  # noqa: BLE001 - server must survive
            self._send_error_json(f"服务器内部错误：{type(error).__name__}: {error}", 500)

    def do_POST(self) -> None:  # noqa: N802
        try:
            parsed = urlparse(self.path)
            route = parsed.path
            if route.startswith("/api/upload/"):
                self._handle_upload(route[len("/api/upload/") :])
                return
            if route == "/api/inspect":
                self._handle_inspect()
                return
            if route == "/api/analyze":
                self._handle_analyze()
                return
            if route.startswith("/api/jobs/") and route.endswith("/cancel"):
                job_id = route[len("/api/jobs/") : -len("/cancel")]
                job = self.manager.cancel(job_id)
                self._send_json({"ok": True, "job": job.public()})
                return
            self._send_error_json(f"未知接口：{route}", 404)
        except ApiError as error:
            self._send_error_json(error.message, error.status)
        except Exception as error:  # noqa: BLE001
            self._send_error_json(f"服务器内部错误：{type(error).__name__}: {error}", 500)

    def _handle_api_get(self, route: str, query: dict[str, list[str]]) -> None:
        if route == "/api/health":
            self._send_json(
                {
                    "ok": True,
                    "service": "DroneRFa 任意IQ信号分析界面",
                    "host": self.server.server_address[0],
                    "port": self.server.server_address[1],
                    "uptime_seconds": round(time.time() - getattr(self.server, "started_at", time.time()), 1),
                    "model": self.manager.bundle_status(),
                    "jobs": self.manager.list_jobs(),
                    "limits": {
                        "window_samples": core.WINDOW_SAMPLES,
                        "default_stride_samples": core.DEFAULT_STRIDE_SAMPLES,
                        "batch_size": core.BATCH_SIZE,
                        "sample_rate_hz": core.SAMPLE_RATE_HZ,
                        "max_concurrent_gpu_jobs": 1,
                    },
                    "allowed_roots": [str(root) for root in self.allowed_roots],
                    "output_root": str(self.manager._output_root),
                }
            )
            return
        if route == "/api/config":
            self._send_json(
                {
                    "ok": True,
                    "labels": {str(key): value for key, value in core.CLASS_NAMES.items()},
                    "labels_zh": {str(key): value for key, value in core.CLASS_NAMES_ZH.items()},
                    "window_samples": core.WINDOW_SAMPLES,
                    "sample_rate_hz": core.SAMPLE_RATE_HZ,
                    "default_stride_samples": core.DEFAULT_STRIDE_SAMPLES,
                    "n_fft": core.N_FFT,
                    "hop_length": core.HOP_LENGTH,
                    "image_size": core.IMAGE_SIZE,
                    "allowed_roots": [str(root) for root in self.allowed_roots],
                }
            )
            return
        if route == "/api/files":
            self._handle_files(query)
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)", route)
        if match:
            job = self.manager.get(match.group(1))
            self._send_json({"ok": True, "job": job.public()})
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/artifact", route)
        if match:
            self._handle_artifact(match.group(1), query)
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/stft", route)
        if match:
            self._handle_stft(match.group(1), query)
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)/download\.zip", route)
        if match:
            self._handle_zip(match.group(1))
            return
        self._send_error_json(f"未知接口：{route}", 404)

    def _handle_files(self, query: dict[str, list[str]]) -> None:
        root_text = (query.get("root") or [""])[0]
        subdir = (query.get("subdir") or [""])[0]
        if not root_text:
            entries = []
            for root in self.allowed_roots:
                entries.append(
                    {
                        "name": str(root),
                        "path": str(root),
                        "is_dir": True,
                        "size_bytes": None,
                    }
                )
            self._send_json({"ok": True, "root": None, "entries": entries})
            return
        root = resolve_allowed(Path(root_text), self.allowed_roots)
        if not root.is_dir():
            raise ApiError(f"不是目录：{root}", 400)
        current = safe_child(root, subdir) if subdir else root
        if not current.is_dir():
            raise ApiError(f"不是目录：{current}", 400)
        entries: list[dict[str, Any]] = []
        if current != root:
            entries.append({"name": "..", "path": str(current.parent), "is_dir": True, "size_bytes": None})
        try:
            children = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        except OSError as error:
            raise ApiError(f"无法读取目录：{current}（{error}）", 400) from error
        for child in children[:500]:
            try:
                size = child.stat().st_size if child.is_file() else None
            except OSError:
                size = None
            entries.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "is_dir": child.is_dir(),
                    "size_bytes": size,
                    "suffix": child.suffix.lower(),
                }
            )
        self._send_json({"ok": True, "root": str(root), "path": str(current), "entries": entries})

    def _handle_artifact(self, job_id: str, query: dict[str, list[str]]) -> None:
        job = self.manager.get(job_id)
        name = (query.get("name") or [""])[0]
        if not job.output_dir:
            raise ApiError("该任务没有可下载的产物。", 404)
        base = Path(job.output_dir)
        target = safe_child(base, name)
        if not target.is_file():
            raise ApiError(f"产物不存在：{name}", 404)
        self._send_file(target, download_name=target.name)

    def _handle_stft(self, job_id: str, query: dict[str, list[str]]) -> None:
        job = self.manager.get(job_id)
        if job.status != "completed":
            raise ApiError("任务尚未完成，无法生成 STFT 预览。", 409)
        if not job.source_path:
            raise ApiError("任务没有记录源文件路径。", 404)
        index = int((query.get("window") or ["0"])[0])
        request = job.request
        source_path = Path(job.source_path)
        inspection = core.inspect_path(source_path, request.get("params") or {})
        plan = core.plan_windows(
            samples=int(inspection.get("samples") or 0),
            start_sample=int(request.get("start_sample") or 0),
            stride_samples=int(request.get("stride_samples") or core.DEFAULT_STRIDE_SAMPLES),
            max_windows=request.get("max_windows"),
            sample_rate_hz=int(request.get("sample_rate_hz") or core.SAMPLE_RATE_HZ),
        )
        if index < 0 or index >= plan.windows:
            raise ApiError(f"窗口序号超出范围：{index}（共 {plan.windows} 窗）", 400)
        start, stop = plan.window_bounds(index)
        source = core.open_source(source_path, request.get("params") or {}, inspection)
        try:
            iq = source.read_iq(start, core.WINDOW_SAMPLES)
        finally:
            source.close()
        window_rows = _load_ensemble_rows(job, index)
        self._send_json(
            {
                "ok": True,
                "window_index": index,
                "start_sample": int(start),
                "stop_sample": int(stop),
                "start_seconds": start / plan.sample_rate_hz,
                "stop_seconds": stop / plan.sample_rate_hz,
                "stft_series": core.compute_stft_image(iq),
                "waveform_series": core.preview_series(iq, points=2000),
                "ensemble_row": window_rows,
            }
        )

    def _handle_zip(self, job_id: str) -> None:
        job = self.manager.get(job_id)
        if not job.output_dir or not Path(job.output_dir).is_dir():
            raise ApiError("该任务没有可打包的产物。", 404)
        base = Path(job.output_dir)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(base.iterdir()):
                if path.is_file():
                    archive.write(path, arcname=path.name)
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header(
            "Content-Disposition",
            f"attachment; filename*=UTF-8''{_url_quote(base.name + '.zip')}",
        )
        self.end_headers()
        self.wfile.write(body)

    # -- inspect / analyze -------------------------------------------------
    def _normalize_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        params = request.get("params") or {}
        if not isinstance(params, dict):
            raise ApiError("params 必须是 JSON 对象。", 400)
        request["params"] = {str(key): value for key, value in params.items() if value not in (None, "")}
        path_text = request.get("path")
        if not path_text:
            raise ApiError("必须提供本地路径 path（或先上传文件）。", 400)
        request["path"] = str(resolve_allowed(Path(str(path_text)), self.allowed_roots))
        request["start_sample"] = int(request.get("start_sample") or 0)
        stride = request.get("stride_samples")
        request["stride_samples"] = int(stride) if stride not in (None, "") else core.DEFAULT_STRIDE_SAMPLES
        max_windows = request.get("max_windows")
        if max_windows in (None, "", "all", "All"):
            request["max_windows"] = None
        else:
            request["max_windows"] = int(max_windows)
        request["sample_rate_hz"] = int(request.get("sample_rate_hz") or core.SAMPLE_RATE_HZ)
        request["sample_rate_confirmed"] = bool(request.get("sample_rate_confirmed"))
        return request

    def _handle_inspect(self) -> None:
        payload = self._read_json_body()
        request = self._normalize_request(payload)
        path = Path(request["path"])
        if not path.is_file():
            raise ApiError(f"不是文件：{path}", 400)
        inspection = core.inspect_path(path, request["params"])
        plan_preview: dict[str, Any]
        samples = int(inspection.get("samples") or 0)
        classify_allowed = (
            request["sample_rate_confirmed"]
            and request["sample_rate_hz"] == core.SAMPLE_RATE_HZ
            and samples >= core.WINDOW_SAMPLES
        )
        reasons: list[str] = []
        if not request["sample_rate_confirmed"]:
            reasons.append("尚未确认采样率为 100 MHz。")
        if request["sample_rate_hz"] != core.SAMPLE_RATE_HZ:
            reasons.append(f"采样率 {request['sample_rate_hz']} Hz 不是 100 MHz。")
        if samples < core.WINDOW_SAMPLES:
            reasons.append(
                f"文件只有 {samples} 点，少于一个窗口 {core.WINDOW_SAMPLES} 点（不补零）。"
            )
        try:
            plan_preview = core.plan_windows(
                samples=samples,
                start_sample=request["start_sample"],
                stride_samples=request["stride_samples"],
                max_windows=request["max_windows"],
                sample_rate_hz=request["sample_rate_hz"],
            ).to_dict()
        except core.SignalAnalyzerError as error:
            plan_preview = {"error": str(error)}
        self._send_json(
            {
                "ok": True,
                "inspection": {key: value for key, value in inspection.items() if key != "groups"},
                "params": request["params"],
                "window_plan": plan_preview,
                "classification_allowed": classify_allowed,
                "classification_blockers": reasons,
                "sample_rate_confirmed": request["sample_rate_confirmed"],
                "sample_rate_hz": request["sample_rate_hz"],
            }
        )

    def _handle_analyze(self) -> None:
        payload = self._read_json_body()
        request = self._normalize_request(payload)
        path = Path(request["path"])
        if not path.is_file():
            raise ApiError(f"不是文件：{path}", 400)
        request["display_name"] = str(request.get("display_name") or path.name)
        job = self.manager.submit(request)
        self._send_json({"ok": True, "job": job.public()}, 202)

    # -- upload ------------------------------------------------------------
    def _handle_upload(self, upload_id: str) -> None:
        upload_id = upload_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", upload_id):
            raise ApiError("上传 ID 非法（只允许字母数字与 -_，4-64 位）。", 400)
        raw_name = self.headers.get("X-File-Name") or ""
        stage = (self.headers.get("X-Upload-Stage") or "chunk").lower()
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        final_path = safe_child(self.upload_dir, f"{upload_id}_{core.safe_name(unquote(raw_name) or upload_id)}")
        partial_path = final_path.parent / (final_path.name + ".partial")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_UPLOAD_BYTES:
            raise ApiError("上传分块过大。", 413)
        body = self.rfile.read(length) if length else b""
        if stage == "abort":
            for path in (partial_path, final_path):
                path.unlink(missing_ok=True)
            self._send_json({"ok": True, "aborted": True, "path": str(final_path)})
            return
        if stage == "start":
            partial_path.unlink(missing_ok=True)
            final_path.unlink(missing_ok=True)
        try:
            with partial_path.open("ab") as handle:
                handle.write(body)
        except OSError as error:
            raise ApiError(f"写入上传临时文件失败：{error}", 500) from error
        if stage == "finish":
            partial_path.replace(final_path)
            self._send_json(
                {
                    "ok": True,
                    "path": str(final_path),
                    "size_bytes": final_path.stat().st_size,
                    "upload_id": upload_id,
                }
            )
            return
        self._send_json(
            {
                "ok": True,
                "received_bytes": length,
                "partial": str(partial_path),
                "size_bytes": partial_path.stat().st_size if partial_path.exists() else 0,
            }
        )

    # -- static ------------------------------------------------------------
    def _serve_static(self, route: str) -> None:
        relative = "index.html" if route in {"", "/"} else unquote(route.lstrip("/"))
        target = safe_child(self.web_dir, relative)
        if not target.is_file():
            self._send_error_json(f"静态资源不存在：{relative}", 404)
            return
        self._send_file(target)


def _url_quote(text: str) -> str:
    from urllib.parse import quote

    return quote(text, safe="")


def _load_ensemble_rows(job: Job, index: int) -> dict[str, Any] | None:
    if not job.output_dir:
        return None
    path = Path(job.output_dir) / "ensemble_predictions.csv"
    if not path.is_file():
        return None
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            if position == index:
                return dict(row)
    return None


# ---------------------------------------------------------------------------
# server factory
# ---------------------------------------------------------------------------
class AnalyzerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def create_server(
    host: str = "127.0.0.1",
    port: int = 8765,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    allowed_roots: tuple[Path, ...] = DEFAULT_ALLOWED_ROOTS,
    device: str = "auto",
    preload: bool = True,
    quiet: bool = False,
) -> tuple[AnalyzerServer, JobManager]:
    if host not in {"127.0.0.1", "localhost"}:
        raise SystemExit(
            "本界面只允许监听 127.0.0.1（本地回环）。拒绝绑定 "
            f"{host}；不会修改防火墙。"
        )
    allowed = tuple(Path(root) for root in allowed_roots)
    manager = JobManager(
        bundle_loader=lambda: core.load_model_bundle(device_name=device),
        output_root=Path(output_root),
        allowed_roots=allowed,
    )
    if preload:
        manager.preload_models()
    else:
        threading.Thread(target=manager.preload_models, daemon=True).start()
    manager.start(workers=1)

    handler = type(
        "BoundAnalyzerHandler",
        (AnalyzerHandler,),
        {
            "manager": manager,
            "allowed_roots": allowed,
            "web_dir": WEB_DIR,
            "job_work_dir": JOB_WORK_DIR,
            "upload_dir": UPLOAD_DIR,
        },
    )
    server = AnalyzerServer((host, port), handler)
    server.quiet = quiet
    server.started_at = time.time()
    return server, manager


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DroneRFa 任意IQ信号分析界面（本地 127.0.0.1）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=None,
        help="允许分析的根目录，可重复；默认 F:/DroneRFa数据集 与项目根目录",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-preload", action="store_true", help="后台加载模型，先启动服务")
    parser.add_argument("--open-browser", action="store_true", help="启动后尝试打开默认浏览器")
    parser.add_argument("--quiet", action="store_true", help="不打印每个 HTTP 请求")
    args = parser.parse_args(argv)

    roots = (
        tuple(Path(item) for item in args.allowed_root)
        if args.allowed_root
        else DEFAULT_ALLOWED_ROOTS
    )
    server, manager = create_server(
        host=args.host,
        port=args.port,
        output_root=Path(args.output_root),
        allowed_roots=roots,
        device=args.device,
        preload=not args.no_preload,
        quiet=args.quiet,
    )
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    url = f"http://{bound_host}:{bound_port}/"
    print(f"服务已启动：{url}")
    print(f"输出根目录：{args.output_root}")
    print("允许根目录：" + "；".join(str(root) for root in roots))
    status = manager.bundle_status()
    if status["state"] == "ready":
        model = status.get("model") or {}
        environment = status.get("environment") or {}
        print(
            f"模型：{environment.get('device')} 上已加载 6 个冻结权重，"
            f"配置 hash 匹配={model.get('cascade_config_sha256_match')}，"
            f"权重校验={model.get('weights_verified')}"
        )
        for warning in environment.get("warnings", []):
            print("警告：" + warning)
    elif status["state"] == "failed":
        print("警告：模型加载失败：" + str(status["error"]))
    else:
        print("模型正在后台加载……")
    if args.open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:  # pragma: no cover - headless环境
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("正在关闭……")
    finally:
        server.shutdown()
        manager.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
