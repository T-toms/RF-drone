"""INFER-UI1 core: arbitrary IQ file probing, streaming windows, cascade inference.

This module is the computation half of the local Chinese signal-analysis UI.  It
reuses the **already accepted** CASCADE3 artefacts:

* ``work/cascade3-config-20260919.json`` (SHA256 pinned by the UI config),
* the six frozen ``best_model.pt`` checkpoints (seed 2026/2027/2028 x stage 1/2),
* ``model.SpectrogramFrontEnd`` / ``model.ResNet18`` (n_fft 512, hop 256, 256x256),
* ``dronerfa_data.rms_normalize``,
* ``run_cascade3_evaluation.load_cascade_models`` (loads + SHA256-verifies weights),
* ``run_cascade3_evaluation.network_logits`` / ``stage2_batches`` /
  ``build_prediction_row`` (exact frozen decision rules and AMP batch protocol).

Nothing is trained, no threshold is tuned and no frozen file is written.  The
user signal is **never** given AWGN: only demean + RMS normalisation.

Frozen rules reproduced here:

* stage 1 runs on the shared float32 feature tensor of a window for all three
  seeds; stage 2 runs for a seed only when that seed's own stage-1 margin > 0;
* stage 2 consumes the **same** feature tensor (no second STFT of the same window);
* a window's three per-seed final labels are combined by majority vote and a
  1-1-1 split is reported as ``uncertain`` instead of being forced into a class.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

PROJECT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROJECT_DIR.parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib"))

import numpy as np
import torch

from dronerfa_data import rms_normalize
from model import ResNet18, SpectrogramFrontEnd
from run_cascade3_evaluation import (
    BATCH_SIZE,
    CLASS_NAMES,
    CONFIG_PATH as CASCADE_CONFIG_PATH,
    HOP_LENGTH,
    IMAGE_SIZE,
    N_FFT,
    build_prediction_row,
    load_cascade_models,
    network_logits,
    stage2_batches,
)
from train_resnet18 import extract_features

# ---------------------------------------------------------------------------
# frozen protocol constants (cross-checked against both configs at startup)
# ---------------------------------------------------------------------------
UI_CONFIG_PATH = PROJECT_ROOT / "work" / "signal-analyzer-ui-config-20260920.json"
SPEC_PATH = PROJECT_ROOT / "work" / "arbitrary-signal-ui-spec.md"

TRAINING_SEEDS: tuple[int, ...] = (2026, 2027, 2028)
WINDOW_SAMPLES = 1_000_000
SAMPLE_RATE_HZ = 100_000_000
DEFAULT_STRIDE_SAMPLES = 1_000_000
FEATURE_DTYPE = np.float32
VERIFY_HASH_ALGORITHM = "sha256"

UNCERTAIN_LABEL = "uncertain"
UNCERTAIN_NAME_ZH = "模型意见不一致"

CLASS_NAMES_ZH: dict[int, str] = {
    0: "背景 Background",
    1: "DJI Phantom 4 Pro",
    2: "DJI Air 2S",
}

# per-window prediction CSV columns (superset of the CASCADE3 CSV + UI columns)
PREDICTION_COLUMNS: tuple[str, ...] = (
    "window_index",
    "start_sample",
    "stop_sample",
    "start_seconds",
    "stop_seconds",
    "training_seed",
    "stage1_epoch",
    "stage2_epoch",
    "stage1_logit_background",
    "stage1_logit_drone",
    "stage1_margin",
    "stage1_prediction",
    "stage1_decision",
    "stage2_executed",
    "stage2_batch_id",
    "stage2_logit_phantom",
    "stage2_logit_air2s",
    "stage2_margin",
    "final_prediction",
    "final_class_name",
    "stage1_feature_digest",
    "stage2_feature_digest",
)
ENSEMBLE_COLUMNS: tuple[str, ...] = (
    "window_index",
    "start_sample",
    "stop_sample",
    "start_seconds",
    "stop_seconds",
    "votes_seed_2026",
    "votes_seed_2027",
    "votes_seed_2028",
    "ensemble_prediction",
    "ensemble_class_name",
    "ensemble_agreement",
    "is_uncertain",
    "is_drone",
    "majority_label_id",
)
TIMELINE_COLORS: dict[str, str] = {
    "Background": "#2f6fdc",
    "DJI Phantom 4 Pro": "#e08b1e",
    "DJI Air 2S": "#d1495b",
    UNCERTAIN_LABEL: "#7a7a7a",
}
TIMELINE_ORDER: tuple[str, ...] = (
    "Background",
    "DJI Phantom 4 Pro",
    "DJI Air 2S",
    UNCERTAIN_LABEL,
)


class SignalAnalyzerError(RuntimeError):
    """Any user-facing failure; the message is already Chinese and actionable."""


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def utc_now() -> str:
    from datetime import timezone

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def class_label(value: Any) -> str:
    """Normalise ``0/1/2``, a frozen class name or ``uncertain`` to a class name."""
    if value is None or value == "":
        return UNCERTAIN_LABEL
    if isinstance(value, str):
        text = value.strip()
        if text in {UNCERTAIN_LABEL, "uncertain", UNCERTAIN_NAME_ZH}:
            return UNCERTAIN_LABEL
        if text in CLASS_NAMES.values():
            return text
        if text in CLASS_NAMES_ZH.values():
            for class_id, name in CLASS_NAMES_ZH.items():
                if name == text:
                    return CLASS_NAMES[class_id]
        try:
            return CLASS_NAMES[int(text)]
        except (KeyError, ValueError) as error:
            raise SignalAnalyzerError(f"未知类别名：{value}") from error
    try:
        return CLASS_NAMES[int(value)]
    except (KeyError, ValueError) as error:
        raise SignalAnalyzerError(f"未知类别名：{value}") from error


def class_label_zh(value: Any) -> str:
    label = class_label(value)
    if label == UNCERTAIN_LABEL:
        return UNCERTAIN_NAME_ZH
    for class_id, name in CLASS_NAMES.items():
        if name == label:
            return CLASS_NAMES_ZH[class_id]
    return label


def safe_name(text: str, fallback: str = "signal") -> str:
    stem = Path(str(text)).stem or fallback
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", stem).strip("._-")
    return (cleaned or fallback)[:64]


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SignalAnalyzerError("已指定 CUDA，但当前 PyTorch 检测不到可用 GPU。")
    return device


def expected_venv_python() -> Path:
    return PROJECT_DIR.parent / ".venv" / "Scripts" / "python.exe"


def environment_report(device: torch.device | None = None) -> dict[str, Any]:
    """Describe the *actual* interpreter/GPU environment used for inference."""
    executable = Path(sys.executable)
    expected = expected_venv_python()
    in_project_venv = False
    try:
        in_project_venv = executable.resolve() == expected.resolve()
    except OSError:
        in_project_venv = False
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    if device is None:
        device = resolve_device("auto")
    warnings: list[str] = []
    if not cuda_available:
        warnings.append(
            "当前没有可用 CUDA，正在用 CPU 推理：速度显著变慢，且数值环境与正式 "
            "CASCADE3（项目 .venv + CUDA float16 AMP）不同，逐窗数值可能只有微小差异。"
        )
    if not in_project_venv:
        warnings.append(
            f"当前解释器不是项目虚拟环境：{executable}。请改用 "
            f"{expected} 启动，避免 CPU PyTorch 静默替换正式 CUDA 环境。"
        )
    return {
        "python_executable": str(executable),
        "python_version": sys.version.split()[0],
        "expected_venv_python": str(expected),
        "in_project_venv": in_project_venv,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": str(device),
        "gpu_name": gpu_name,
        "amp_float16": device.type == "cuda",
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# input format detection + streaming readers
# ---------------------------------------------------------------------------
READ_CHUNK_SAMPLES = WINDOW_SAMPLES


@dataclass(frozen=True)
class IQLayout:
    """How one stored shape is turned into (I, Q) real vectors."""

    kind: str  # "separate" | "complex" | "interleaved"
    shape: tuple[int, ...]
    samples: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "shape": list(self.shape),
            "samples": int(self.samples),
            "detail": self.detail,
        }


def candidate_named_datasets(keys: Sequence[str]) -> list[dict[str, Any]]:
    """Recognised I/Q dataset-name conventions inside one group."""
    key_set = set(keys)
    candidates: list[dict[str, Any]] = []
    for i_key, q_key, label in (
        ("RF0_I", "RF0_Q", "RF0_I/RF0_Q"),
        ("RF1_I", "RF1_Q", "RF1_I/RF1_Q"),
        ("i", "q", "i/q"),
        ("I", "Q", "I/Q"),
        ("real", "imag", "real/imag"),
        ("I_data", "Q_data", "I_data/Q_data"),
    ):
        if i_key in key_set and q_key in key_set:
            candidates.append({"mode": "pair", "label": label, "i_key": i_key, "q_key": q_key})
    for key in ("iq", "IQ", "complex_iq", "data", "signal", "samples"):
        if key in key_set:
            candidates.append({"mode": "complex", "label": key, "key": key})
    return candidates


def describe_dataset(name: str, dataset: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": [int(dim) for dim in dataset.shape],
        "dtype": str(dataset.dtype),
        "chunks": [int(dim) for dim in dataset.chunks] if dataset.chunks else None,
    }


def detect_mat_groups(handle: Any) -> list[dict[str, Any]]:
    import h5py

    groups: list[dict[str, Any]] = []
    for key in handle.keys():
        item = handle[key]
        if isinstance(item, h5py.Group):
            datasets = {name: describe_dataset(name, item[name]) for name in item.keys()}
            groups.append(
                {
                    "group": key,
                    "datasets": datasets,
                    "candidates": candidate_named_datasets(list(datasets)),
                }
            )
    return groups


def detect_npy_layout(array: np.ndarray, allow_load: bool = True) -> dict[str, Any]:
    """NPY may be 1-D complex, or a float array shaped [2, N] / [N, 2]."""
    shape = tuple(int(dim) for dim in array.shape)
    dtype = str(array.dtype)
    if array.ndim == 1:
        if np.issubdtype(array.dtype, np.complexfloating):
            return {
                "supported": True,
                "layout": {"kind": "complex", "shape": shape, "samples": shape[0]},
                "detail": "一维复数数组，直接作为 I+jQ。",
            }
        return {
            "supported": False,
            "layout": None,
            "detail": f"一维数组 dtype={dtype} 不是复数；请用 NPZ 的 i/q 键或 RAW 并提供 I/Q 顺序。",
        }
    if array.ndim == 2 and np.issubdtype(array.dtype, np.complexfloating) and 1 in shape:
        return {
            "supported": True,
            "layout": {"kind": "complex", "shape": shape, "samples": max(shape)},
            "detail": "复数数组且有一个长度为 1 的维度，按最长维度取样点。",
        }
    if array.ndim == 2 and shape[0] == 2 and np.issubdtype(array.dtype, np.floating):
        return {
            "supported": True,
            "layout": {
                "kind": "interleaved",
                "shape": shape,
                "samples": shape[1],
                "order": "I_row_then_Q_row",
            },
            "detail": "[2, N] 浮点数组：第 0 行为 I，第 1 行为 Q。",
        }
    if array.ndim == 2 and shape[1] == 2 and np.issubdtype(array.dtype, np.floating):
        return {
            "supported": True,
            "layout": {
                "kind": "interleaved",
                "shape": shape,
                "samples": shape[0],
                "order": "I_column_then_Q_column",
            },
            "detail": "[N, 2] 浮点数组：第 0 列为 I，第 1 列为 Q。",
        }
    return {
        "supported": False,
        "layout": None,
        "detail": f"暂不支持该 NPY 形状/类型：shape={shape} dtype={dtype}。",
    }


def raw_layout_from_params(params: dict[str, Any], byte_size: int) -> dict[str, Any]:
    dtype_name = str(params.get("dtype") or "float32")
    complex_flag = bool(params.get("complex", False))
    byte_order = str(params.get("byte_order") or "little")
    order = str(params.get("order") or "IQ")
    if byte_order not in {"little", "big"}:
        raise SignalAnalyzerError("RAW byte_order 只能是 little 或 big。")
    try:
        base = np.dtype(dtype_name)
    except TypeError as error:
        raise SignalAnalyzerError(f"RAW dtype 无法解析：{dtype_name}（{error}）") from error
    if base.kind == "c":
        if base.itemsize not in (8, 16):
            raise SignalAnalyzerError(f"RAW 复数类型只支持 complex64/complex128，收到 {base}")
        complex_flag = True
    if base.kind not in {"f", "c"}:
        raise SignalAnalyzerError(
            f"RAW 只支持 float32/float64 或 complex64/complex128，收到 {base}"
        )
    byteorder = "<" if byte_order == "little" else ">"
    base = base.newbyteorder(byteorder)
    if complex_flag:
        if base.kind == "c":
            complex_dtype = base
        else:
            if base.itemsize not in (4, 8):
                raise SignalAnalyzerError("RAW complex 模式需要 float32/float64 基类型。")
            complex_dtype = np.dtype(
                "complex64" if base.itemsize == 4 else "complex128"
            ).newbyteorder(byteorder)
        samples = byte_size // complex_dtype.itemsize
        return {
            "kind": "complex",
            "dtype": complex_dtype.str,
            "samples": int(samples),
            "order": None,
            "detail": f"复数 {complex_dtype.str}，按 I+jQ 解释。",
        }
    if order not in {"IQ", "QI"}:
        raise SignalAnalyzerError("RAW 交织模式的 order 只能是 IQ 或 QI。")
    pair_dtype = np.dtype([("first", base), ("second", base)])
    samples = byte_size // pair_dtype.itemsize
    return {
        "kind": "interleaved",
        "dtype": base.str,
        "samples": int(samples),
        "order": order,
        "detail": f"{base.str} 交织，顺序 {order}。",
    }


class IQSource:
    """Streaming (I, Q) slice provider.  A window is read in bounded chunks so a
    multi-GB file is never materialised in full."""

    kind = "base"

    def __init__(self, samples: int, layout: str) -> None:
        self.samples = int(samples)
        self.layout = str(layout)
        self.path: str | None = None

    # -- to implement in subclasses -----------------------------------------
    def _read_i(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def _read_q(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def read_iq(self, start: int, length: int) -> np.ndarray:
        start = int(start)
        length = int(length)
        if start < 0 or length <= 0:
            raise SignalAnalyzerError(f"非法窗口范围 start={start} length={length}")
        if start + length > self.samples:
            raise SignalAnalyzerError(
                f"窗口超出文件长度：start={start} length={length} samples={self.samples}"
            )
        iq = np.empty(length, dtype=np.complex64)
        for offset in range(0, length, READ_CHUNK_SAMPLES):
            stop_local = min(offset + READ_CHUNK_SAMPLES, length)
            span = slice(start + offset, start + stop_local)
            i_values = self._read_i(span.start, span.stop)
            q_values = self._read_q(span.start, span.stop)
            iq[offset:stop_local] = i_values + 1j * q_values
        return iq

    def close(self) -> None:
        return None


class _H5PairSource(IQSource):
    """I/Q stored as two separate 1-D (or [1, N]) float datasets."""

    def __init__(self, handle: Any, i_key: str, q_key: str, root: str | None = None) -> None:
        self._handle = handle
        prefix = f"{root}/" if root else ""
        i_dataset = handle[f"{prefix}{i_key}"]
        q_dataset = handle[f"{prefix}{q_key}"]
        samples = _vector_length(i_dataset)
        if _vector_length(q_dataset) != samples:
            raise SignalAnalyzerError(
                f"{prefix}{i_key} 与 {prefix}{q_key} 长度不一致，无法作为 I/Q 对读取。"
            )
        super().__init__(samples, "separate")
        self.i_key = f"{prefix}{i_key}"
        self.q_key = f"{prefix}{q_key}"
        self._i = i_dataset
        self._q = q_dataset

    def _slice(self, dataset: Any, start: int, stop: int) -> np.ndarray:
        if dataset.ndim == 1:
            values = dataset[start:stop]
        elif dataset.ndim == 2 and dataset.shape[0] == 1:
            values = dataset[0, start:stop]
        elif dataset.ndim == 2 and dataset.shape[1] == 1:
            values = dataset[start:stop, 0]
        else:
            raise SignalAnalyzerError(f"不支持的 I/Q 数据形状 {dataset.shape}")
        return np.asarray(values, dtype=np.float32)

    def _read_i(self, start: int, stop: int) -> np.ndarray:
        return self._slice(self._i, start, stop)

    def _read_q(self, start: int, stop: int) -> np.ndarray:
        return self._slice(self._q, start, stop)

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


class _H5ComplexSource(IQSource):
    """Complex dataset stored directly in HDF5 (alternative NPZ-like MAT layout)."""

    def __init__(self, handle: Any, key: str) -> None:
        dataset = handle[key]
        if not np.issubdtype(dataset.dtype, np.complexfloating):
            raise SignalAnalyzerError(f"{key} 不是复数数据集，无法作为 I/Q 读取。")
        super().__init__(_vector_length(dataset), "complex")
        self._dataset = dataset
        self.key = key
        self._handle = handle

    def _read(self, start: int, stop: int) -> np.ndarray:
        dataset = self._dataset
        if dataset.ndim == 1:
            values = dataset[start:stop]
        elif dataset.ndim == 2 and dataset.shape[0] == 1:
            values = dataset[0, start:stop]
        elif dataset.ndim == 2 and dataset.shape[1] == 1:
            values = dataset[start:stop, 0]
        else:
            raise SignalAnalyzerError(f"不支持的复数数据形状 {dataset.shape}")
        return np.asarray(values)

    def _read_i(self, start: int, stop: int) -> np.ndarray:
        return np.asarray(self._read(start, stop).real, dtype=np.float32)

    def _read_q(self, start: int, stop: int) -> np.ndarray:
        return np.asarray(self._read(start, stop).imag, dtype=np.float32)

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


class _ArrayIQSource(IQSource):
    """Whole in-memory array source (small NPY/NPZ fixtures and echoes)."""

    def __init__(self, iq: np.ndarray) -> None:
        iq = np.asarray(iq)
        if iq.ndim != 1:
            raise SignalAnalyzerError(f"内部错误：IQ 数组必须是一维，收到 {iq.shape}")
        super().__init__(iq.shape[0], "memory")
        self._iq = iq

    def read_iq(self, start: int, length: int) -> np.ndarray:
        start = int(start)
        length = int(length)
        if start < 0 or length <= 0 or start + length > self.samples:
            raise SignalAnalyzerError(
                f"窗口超出数据长度：start={start} length={length} samples={self.samples}"
            )
        return np.asarray(self._iq[start : start + length], dtype=np.complex64)

    def _read_i(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        return np.asarray(self._iq[start:stop].real, dtype=np.float32)

    def _read_q(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        return np.asarray(self._iq[start:stop].imag, dtype=np.float32)


class _NpySource(IQSource):
    """Memory-mapped NPY source (complex 1-D or [2, N] / [N, 2] float)."""

    def __init__(self, path: Path, layout: dict[str, Any]) -> None:
        array = np.load(Path(path), mmap_mode="r")
        self._array = array
        self._order = layout.get("order")
        super().__init__(int(layout["samples"]), "npy")
        self.path = str(path)

    def _vector(self, start: int, stop: int) -> np.ndarray:
        array = self._array
        if array.ndim == 1:
            return np.asarray(array[start:stop])
        if array.ndim == 2 and 1 in array.shape:
            if array.shape[0] == 1:
                return np.asarray(array[0, start:stop])
            return np.asarray(array[start:stop, 0])
        if array.ndim == 2 and array.shape[0] == 2:
            return np.asarray(array[:, start:stop])
        if array.ndim == 2 and array.shape[1] == 2:
            return np.asarray(array[start:stop, :])
        raise SignalAnalyzerError(f"不支持的 NPY 形状 {array.shape}")

    def read_iq(self, start: int, length: int) -> np.ndarray:
        start = int(start)
        length = int(length)
        if start < 0 or length <= 0 or start + length > self.samples:
            raise SignalAnalyzerError(
                f"窗口超出数据长度：start={start} length={length} samples={self.samples}"
            )
        block = self._vector(start, start + length)
        if np.issubdtype(block.dtype, np.complexfloating):
            return np.asarray(block, dtype=np.complex64)
        if block.ndim == 2:
            first = np.asarray(block[0], dtype=np.float32)
            second = np.asarray(block[1], dtype=np.float32)
            if self._order == "I_row_then_Q_row" or self._array.shape[0] == 2:
                i_values, q_values = first, second
            else:
                i_values, q_values = first, second
            if self._order == "Q_column_then_I_column":
                i_values, q_values = second, first
            return (i_values + 1j * q_values).astype(np.complex64)
        raise SignalAnalyzerError("NPY 数据既不是复数也不是 I/Q 交织，无法读取。")

    def _read_i(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        return self.read_iq(start, stop - start).real.astype(np.float32)

    def _read_q(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        return self.read_iq(start, stop - start).imag.astype(np.float32)


class _NpzSource(IQSource):
    """NPZ source: complex key ``iq`` or paired keys ``i``/``q`` (full load)."""

    def __init__(self, path: Path, layout: dict[str, Any]) -> None:
        with np.load(Path(path), allow_pickle=False) as payload:
            if layout["mode"] == "complex":
                data = np.asarray(payload[layout["key"]])
                if data.ndim == 2 and 1 in data.shape:
                    data = data.reshape(-1)
                if data.ndim != 1 or not np.issubdtype(data.dtype, np.complexfloating):
                    raise SignalAnalyzerError(
                        f"NPZ 键 {layout['key']} 必须是一维复数数组，"
                        f"收到 {data.shape}/{data.dtype}"
                    )
                super().__init__(data.shape[0], "npz")
                self._iq = np.asarray(data, dtype=np.complex64)
                self._pair = None
            else:
                i_values = np.asarray(payload[layout["i_key"]]).reshape(-1)
                q_values = np.asarray(payload[layout["q_key"]]).reshape(-1)
                if i_values.shape != q_values.shape:
                    raise SignalAnalyzerError("NPZ 的 i/q 键长度不一致。")
                super().__init__(i_values.shape[0], "npz")
                self._iq = None
                self._pair = (
                    np.asarray(i_values, dtype=np.float32),
                    np.asarray(q_values, dtype=np.float32),
                )
        self.path = str(path)

    def read_iq(self, start: int, length: int) -> np.ndarray:
        start = int(start)
        length = int(length)
        if start < 0 or length <= 0 or start + length > self.samples:
            raise SignalAnalyzerError(
                f"窗口超出数据长度：start={start} length={length} samples={self.samples}"
            )
        if self._iq is not None:
            return np.asarray(self._iq[start : start + length], dtype=np.complex64)
        i_values, q_values = self._pair  # type: ignore[misc]
        return (
            i_values[start : start + length] + 1j * q_values[start : start + length]
        ).astype(np.complex64)

    def _read_i(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        return self.read_iq(start, stop - start).real.astype(np.float32)

    def _read_q(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        return self.read_iq(start, stop - start).imag.astype(np.float32)


class _RawSource(IQSource):
    """Raw binary source with an explicit dtype/byte-order/IQ-order contract."""

    def __init__(self, path: Path, layout: dict[str, Any]) -> None:
        self._path = Path(path)
        self._layout = dict(layout)
        self._handle = self._path.open("rb")
        super().__init__(int(layout["samples"]), "raw")
        self.path = str(path)

    def _read_block(self, start: int, stop: int) -> np.ndarray:
        layout = self._layout
        if layout["kind"] == "complex":
            dtype = np.dtype(layout["dtype"])
            self._handle.seek(start * dtype.itemsize)
            buffer = self._handle.read((stop - start) * dtype.itemsize)
            return np.frombuffer(buffer, dtype=dtype)
        dtype = np.dtype(layout["dtype"])
        pair = np.dtype([("first", dtype), ("second", dtype)])
        self._handle.seek(start * pair.itemsize)
        buffer = self._handle.read((stop - start) * pair.itemsize)
        return np.frombuffer(buffer, dtype=pair)

    def read_iq(self, start: int, length: int) -> np.ndarray:
        start = int(start)
        length = int(length)
        if start < 0 or length <= 0 or start + length > self.samples:
            raise SignalAnalyzerError(
                f"窗口超出数据长度：start={start} length={length} samples={self.samples}"
            )
        out = np.empty(length, dtype=np.complex64)
        for offset in range(0, length, READ_CHUNK_SAMPLES):
            stop_local = min(offset + READ_CHUNK_SAMPLES, length)
            block = self._read_block(start + offset, start + stop_local)
            if self._layout["kind"] == "complex":
                out[offset:stop_local] = np.asarray(block, dtype=np.complex64)
            else:
                first = np.asarray(block["first"], dtype=np.float32)
                second = np.asarray(block["second"], dtype=np.float32)
                if self._layout["order"] == "IQ":
                    out[offset:stop_local] = first + 1j * second
                else:
                    out[offset:stop_local] = second + 1j * first
        return out

    def _read_i(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        return self.read_iq(start, stop - start).real.astype(np.float32)

    def _read_q(self, start: int, stop: int) -> np.ndarray:  # pragma: no cover
        return self.read_iq(start, stop - start).imag.astype(np.float32)

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:
            pass


def _vector_length(dataset: Any) -> int:
    if dataset.ndim == 1:
        return int(dataset.shape[0])
    if dataset.ndim == 2 and 1 in dataset.shape:
        return int(max(dataset.shape))
    raise SignalAnalyzerError(f"不支持的数据集形状 {dataset.shape}")


# ---------------------------------------------------------------------------
# format probing
# ---------------------------------------------------------------------------
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
HDF5_USERBLOCK_OFFSETS: tuple[int, ...] = (0, 512, 1024, 2048)


def detect_format(path: Path) -> dict[str, Any]:
    """Cheap extension + file-magic detection (never loads the whole file).

    MAT v7.3 files are HDF5 and often carry a 512-byte MATLAB user block, so the
    HDF5 signature is searched at the standard user-block offsets before falling
    back to RAW.
    """
    path = Path(path)
    if not path.exists():
        raise SignalAnalyzerError(f"文件不存在：{path}")
    if not path.is_file():
        raise SignalAnalyzerError(f"不是普通文件：{path}")
    if path.stat().st_size == 0:
        raise SignalAnalyzerError(f"文件为空：{path}")
    suffix = path.suffix.lower()
    with path.open("rb") as handle:
        head = handle.read(max(HDF5_USERBLOCK_OFFSETS) + len(HDF5_SIGNATURE))
    hdf5_offset = next(
        (offset for offset in HDF5_USERBLOCK_OFFSETS if head[offset : offset + 8] == HDF5_SIGNATURE),
        None,
    )
    if hdf5_offset is not None:
        detected = "mat_v73"
    elif head.startswith(b"MATLAB 5"):
        detected = "mat_v5"
    elif head.startswith(b"\x93NUMPY"):
        detected = "npy"
    elif head.startswith(b"PK\x03\x04"):
        detected = "npz"
    else:
        detected = "raw"
    extension_guess = {
        ".mat": "mat_v73",
        ".npy": "npy",
        ".npz": "npz",
        ".raw": "raw",
        ".bin": "raw",
        ".iq": "raw",
        ".dat": "raw",
    }.get(suffix)
    return {
        "path": str(path),
        "suffix": suffix,
        "size_bytes": int(path.stat().st_size),
        "magic_detected": detected,
        "hdf5_signature_offset": hdf5_offset,
        "extension_guess": extension_guess,
        "consistent": extension_guess is None or extension_guess == detected,
    }


def inspect_path(
    path: Path,
    params: dict[str, Any] | None = None,
    max_scan_samples: int = 200_000,
) -> dict[str, Any]:
    """Detect the I/Q layout of one file without reading it end to end."""
    params = dict(params or {})
    path = Path(path)
    detection = detect_format(path)
    kind = detection["magic_detected"]
    payload: dict[str, Any] = {
        "detected_format": kind,
        "detection": detection,
        "size_bytes": detection["size_bytes"],
    }
    if kind == "mat_v5":
        raise SignalAnalyzerError(
            "这是 MATLAB v5/v7 旧格式 MAT，本界面只支持 v7.3(HDF5)；"
            "请另存为 -v7.3 或导出为 NPY/RAW。"
        )
    if kind == "mat_v73":
        import h5py

        with h5py.File(path, "r") as handle:
            root_datasets = {
                name: describe_dataset(name, handle[name])
                for name in handle.keys()
                if isinstance(handle[name], h5py.Dataset)
            }
            candidates = candidate_named_datasets(list(root_datasets))
            for candidate in candidates:
                if candidate["mode"] == "pair":
                    candidate["samples"] = _vector_length(handle[candidate["i_key"]])
                else:
                    candidate["samples"] = _vector_length(handle[candidate["key"]])
            payload.update(
                {
                    "datasets": root_datasets,
                    "groups": detect_mat_groups(handle),
                    "candidates": candidates,
                    "default_candidate": candidates[0] if candidates else None,
                }
            )
    elif kind == "npy":
        array = np.load(path, mmap_mode="r")
        detected = detect_npy_layout(array)
        payload.update(
            {
                "array": {"shape": list(array.shape), "dtype": str(array.dtype)},
                "layout_supported": bool(detected.get("supported")),
                "layout": detected.get("layout"),
                "layout_detail": detected.get("detail"),
            }
        )
    elif kind == "npz":
        with np.load(path, allow_pickle=False) as payload_handle:
            keys = list(payload_handle.keys())
            described: dict[str, Any] = {}
            candidates: list[dict[str, Any]] = []
            for key in keys:
                item = payload_handle[key]
                described[key] = {"shape": list(item.shape), "dtype": str(item.dtype)}
                if np.issubdtype(item.dtype, np.complexfloating) and item.ndim <= 2:
                    candidates.append({"mode": "complex", "label": key, "key": key, "samples": int(np.prod(item.shape))})
            for i_key, q_key, label in (("i", "q", "i/q"), ("I", "Q", "I/Q")):
                if i_key in keys and q_key in keys:
                    candidates.append(
                        {
                            "mode": "pair",
                            "label": label,
                            "i_key": i_key,
                            "q_key": q_key,
                            "samples": int(np.prod(payload_handle[i_key].shape)),
                        }
                    )
            payload.update(
                {
                    "arrays": described,
                    "candidates": candidates,
                    "default_candidate": candidates[0] if candidates else None,
                    "compressed": _npz_is_compressed(path),
                }
            )
    elif kind == "raw":
        layout = raw_layout_from_params(params, detection["size_bytes"])
        payload.update({"layout": layout, "params_required": not params})
    else:  # pragma: no cover - defensive
        raise SignalAnalyzerError(f"无法识别的文件格式：{kind}")

    detected_samples = _probe_samples(payload)
    if detected_samples is not None:
        payload["samples"] = int(detected_samples)
        payload["short_file"] = bool(int(detected_samples) < WINDOW_SAMPLES)
    return payload


def _probe_samples(payload: dict[str, Any]) -> int | None:
    """Total complex sample count of the probed file, if it can be determined."""
    if payload.get("samples") is not None:
        return int(payload["samples"])
    layout = payload.get("layout")
    if isinstance(layout, dict) and layout.get("samples") is not None:
        return int(layout["samples"])
    candidate = payload.get("default_candidate")
    if isinstance(candidate, dict) and candidate.get("samples") is not None:
        return int(candidate["samples"])
    arrays = payload.get("arrays")
    if isinstance(arrays, dict):
        candidates = payload.get("candidates") or []
        if candidates:
            best = candidates[0]
            described = arrays.get(best.get("key") or best.get("i_key"))
            if described and described.get("shape"):
                return int(np.prod(described["shape"]))
    return None


def _npz_is_compressed(path: Path) -> bool:
    """A compressed NPZ cannot be memory-mapped and must be fully decompressed."""
    try:
        with np.load(path, allow_pickle=False) as payload:
            for key in payload.keys():
                if payload[key].nbytes > 0:
                    return True
    except Exception:
        return True
    return False


def open_source(
    path: Path,
    params: dict[str, Any] | None = None,
    inspection: dict[str, Any] | None = None,
) -> IQSource:
    """Open one file as a streaming I/Q source (raises Chinese errors)."""
    params = dict(params or {})
    path = Path(path)
    inspection = inspection or inspect_path(path, params)
    kind = inspection["detected_format"]
    source: IQSource
    if kind == "mat_v73":
        import h5py

        handle = h5py.File(path, "r")
        i_key = str(params.get("i_key") or "RF0_I")
        q_key = str(params.get("q_key") or "RF0_Q")
        group = params.get("group")
        try:
            if group:
                prefix = str(group)
                if prefix not in handle or not hasattr(handle[prefix], "keys"):
                    raise SignalAnalyzerError(f"MAT 中不存在分组 {prefix}。")
                if i_key not in handle[prefix] or q_key not in handle[prefix]:
                    raise SignalAnalyzerError(f"分组 {prefix} 中不存在 {i_key}/{q_key}。")
                source = _H5PairSource(handle, i_key, q_key, root=prefix)
            elif i_key in handle and q_key in handle:
                source = _H5PairSource(handle, i_key, q_key)
            elif i_key in handle:
                source = _H5ComplexSource(handle, i_key)
            else:
                available = ", ".join(sorted(str(key) for key in handle.keys()))
                raise SignalAnalyzerError(
                    f"MAT 中不存在 I/Q 数据键 {i_key}/{q_key}；可用键：{available}"
                )
        except SignalAnalyzerError:
            handle.close()
            raise
        except Exception as error:
            handle.close()
            raise SignalAnalyzerError(f"打开 MAT 文件失败：{error}") from error
    elif kind == "npy":
        layout = inspection.get("layout")
        if layout is None:
            layout = detect_npy_layout(np.load(path, mmap_mode="r")).get("layout")
        if not inspection.get("layout_supported", True) or layout is None:
            raise SignalAnalyzerError(
                str(inspection.get("layout_detail") or "NPY 布局不受支持。")
            )
        source = _NpySource(path, layout)
    elif kind == "npz":
        if params.get("key") or params.get("i_key") or params.get("q_key"):
            candidate = {
                "mode": "complex" if params.get("key") else "pair",
                "key": params.get("key"),
                "i_key": params.get("i_key") or "i",
                "q_key": params.get("q_key") or "q",
            }
        else:
            candidate = inspection.get("default_candidate")
        if not candidate:
            raise SignalAnalyzerError(
                "NPZ 中没有找到复数 iq 键或 i/q 键；请在格式参数中显式指定键名。"
            )
        source = _NpzSource(path, candidate)
    elif kind == "raw":
        layout = inspection.get("layout") or raw_layout_from_params(params, path.stat().st_size)
        source = _RawSource(path, layout)
    else:
        raise SignalAnalyzerError(f"暂不支持的文件格式：{kind}")
    source.path = str(path)
    return source


def handle_keys(path: Path) -> list[str]:
    import h5py

    with h5py.File(path, "r") as handle:
        return [str(key) for key in handle.keys()]


# ---------------------------------------------------------------------------
# window planning
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowPlan:
    start_sample: int
    stride_samples: int
    max_windows: int | None
    requested_windows: int
    windows: int
    analysed_samples: int
    analysed_stop: int
    dropped_samples: int
    sample_rate_hz: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_sample": int(self.start_sample),
            "stride_samples": int(self.stride_samples),
            "max_windows": self.max_windows,
            "requested_windows": int(self.requested_windows),
            "windows": int(self.windows),
            "analysed_samples": int(self.analysed_samples),
            "analysed_stop": int(self.analysed_stop),
            "dropped_samples": int(self.dropped_samples),
            "sample_rate_hz": int(self.sample_rate_hz),
            "window_samples": WINDOW_SAMPLES,
        }

    def window_bounds(self, index: int) -> tuple[int, int]:
        start = self.start_sample + index * self.stride_samples
        return start, start + WINDOW_SAMPLES


def plan_windows(
    samples: int,
    start_sample: int = 0,
    stride_samples: int = DEFAULT_STRIDE_SAMPLES,
    max_windows: int | None = None,
    sample_rate_hz: int = SAMPLE_RATE_HZ,
) -> WindowPlan:
    samples = int(samples)
    start_sample = int(start_sample)
    stride_samples = int(stride_samples)
    if start_sample < 0:
        raise SignalAnalyzerError("start sample 不能为负数。")
    if stride_samples < 1:
        raise SignalAnalyzerError("stride 必须是 >= 1 的整数。")
    if max_windows is not None and int(max_windows) < 1:
        raise SignalAnalyzerError("max windows 必须是 >= 1 的整数（或留空表示全部）。")
    remaining = samples - start_sample
    if remaining < WINDOW_SAMPLES:
        raise SignalAnalyzerError(
            f"从 start={start_sample} 起只剩 {max(remaining, 0)} 点，少于一个窗口 "
            f"{WINDOW_SAMPLES} 点；本界面不做补零，请调整起点或换文件。"
        )
    requested = 1 + (remaining - WINDOW_SAMPLES) // stride_samples
    windows = requested if max_windows is None else min(requested, int(max_windows))
    analysed_stop = start_sample + (windows - 1) * stride_samples + WINDOW_SAMPLES
    return WindowPlan(
        start_sample=start_sample,
        stride_samples=stride_samples,
        max_windows=None if max_windows is None else int(max_windows),
        requested_windows=int(requested),
        windows=int(windows),
        analysed_samples=int(windows * WINDOW_SAMPLES),
        analysed_stop=int(analysed_stop),
        dropped_samples=int(samples - analysed_stop),
        sample_rate_hz=int(sample_rate_hz),
    )


# ---------------------------------------------------------------------------
# frozen model bundle
# ---------------------------------------------------------------------------
@dataclass
class ModelBundle:
    device: torch.device
    models: dict[int, dict[str, Any]]
    frontend: SpectrogramFrontEnd
    config: dict[str, Any]
    config_sha256: str
    expected_config_sha256: str
    weights: list[dict[str, Any]]
    weights_verified: bool
    errors: list[str]
    load_seconds: float
    environment: dict[str, Any]
    loaded_at: str = field(default_factory=utc_now)

    def stage1_network(self, seed: int) -> ResNet18:
        return self.models[seed]["stage1_network"]

    def stage2_network(self, seed: int) -> ResNet18:
        return self.models[seed]["stage2_network"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "cascade_config": str(CASCADE_CONFIG_PATH),
            "cascade_config_sha256": self.config_sha256,
            "cascade_config_sha256_expected": self.expected_config_sha256,
            "cascade_config_sha256_match": self.config_sha256 == self.expected_config_sha256,
            "weights_verified": bool(self.weights_verified),
            "weights": self.weights,
            "seeds": {
                str(seed): {
                    "stage1_epoch": int(self.models[seed]["stage1_epoch"]),
                    "stage2_epoch": int(self.models[seed]["stage2_epoch"]),
                    "stage1_sha256": self.models[seed].get("stage1_network_sha256"),
                    "stage2_sha256": self.models[seed].get("stage2_network_sha256"),
                }
                for seed in sorted(self.models)
            },
            "load_seconds": round(self.load_seconds, 3),
            "environment": self.environment,
            "errors": self.errors,
            "loaded_at": self.loaded_at,
        }


def load_model_bundle(
    device_name: str = "auto",
    ui_config_path: Path | None = None,
    verify_hashes: bool = True,
) -> ModelBundle:
    """Load the six frozen checkpoints once and keep them resident in memory."""
    ui_config_path = Path(ui_config_path or UI_CONFIG_PATH)
    ui_config = load_json(ui_config_path)
    expected_config_sha = str(ui_config["model"]["cascade_config_sha256"])
    started = time.perf_counter()
    actual_config_sha = sha256_file(CASCADE_CONFIG_PATH)
    if verify_hashes and actual_config_sha != expected_config_sha:
        raise SignalAnalyzerError(
            "冻结配置 hash 不匹配："
            f"{CASCADE_CONFIG_PATH} 期望 {expected_config_sha}，实际 {actual_config_sha}。"
            "已停止加载模型。"
        )
    from run_cascade3_evaluation import load_config as load_cascade_config

    config = load_cascade_config()
    device = resolve_device(device_name)
    environment = environment_report(device)
    errors: list[str] = []
    models = load_cascade_models(config, device, TRAINING_SEEDS)
    weights: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        record = models[seed]
        for stage in (1, 2):
            key = f"stage{stage}_network"
            weights.append(
                {
                    "training_seed": int(seed),
                    "stage": stage,
                    "checkpoint": str(record[f"{key}_checkpoint"]),
                    "sha256": str(record[f"{key}_sha256"]),
                    "epoch": int(record[f"{key}_epoch"]),
                    "bytes": Path(record[f"{key}_checkpoint"]).stat().st_size,
                }
            )
    return ModelBundle(
        device=device,
        models=models,
        frontend=models[TRAINING_SEEDS[0]]["frontend"],
        config=config,
        config_sha256=actual_config_sha,
        expected_config_sha256=expected_config_sha,
        weights=weights,
        weights_verified=True,
        errors=errors,
        load_seconds=time.perf_counter() - started,
        environment=environment,
    )


# ---------------------------------------------------------------------------
# streaming inference
# ---------------------------------------------------------------------------
def iq_to_tensor(iq: np.ndarray) -> torch.Tensor:
    """Frozen preprocessing: demean + RMS normalise, then [2, N] float32."""
    normalized = rms_normalize(np.asarray(iq))
    return torch.from_numpy(np.stack((normalized.real, normalized.imag), axis=0))


def features_from_signals(
    frontend: SpectrogramFrontEnd, signals: torch.Tensor, device: torch.device
) -> np.ndarray:
    """float32 1x256x256 features for a ``[batch, 2, N]`` batch (frozen frontend)."""
    batch = signals.to(device, non_blocking=False)
    with torch.inference_mode():
        features = extract_features(frontend, batch)
    array = features.detach().to("cpu", torch.float32).numpy()
    if array.shape != (int(signals.shape[0]), 1, IMAGE_SIZE, IMAGE_SIZE):
        raise SignalAnalyzerError(f"前端输出形状异常：{array.shape}")
    if not np.isfinite(array).all():
        raise SignalAnalyzerError("前端输出包含非有限值，无法继续推理。")
    return array


class SignalsCancelled(RuntimeError):
    """Raised inside the inference loop when the caller cancelled the job."""


def majority_label(labels: Sequence[Any]) -> tuple[str, str, bool]:
    """Three per-seed final labels -> (ensemble label, agreement, is_uncertain).

    ``agreement`` is ``3/3`` or ``2/3``; a 1-1-1 split yields ``uncertain`` and is
    never forced into a known class.
    """
    names = [class_label(value) for value in labels]
    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    best = max(counts.values())
    winners = sorted(name for name, count in counts.items() if count == best)
    if best == 3 or (best == 2 and len(winners) == 1):
        return winners[0], f"{best}/{len(names)}", False
    return UNCERTAIN_LABEL, f"{best}/{len(names)}", True


def summarize_file(
    ensemble_rows: Sequence[dict[str, Any]],
    total_windows_available: int | None = None,
    dropped_samples: int = 0,
) -> dict[str, Any]:
    """File-level descriptive summary (majority description, not a new model)."""
    analysed = len(ensemble_rows)
    counts = {name: 0 for name in TIMELINE_ORDER}
    for row in ensemble_rows:
        counts[str(row["ensemble_class_name"])] += 1
    uncertain = counts[UNCERTAIN_LABEL]
    drone_windows = counts["DJI Phantom 4 Pro"] + counts["DJI Air 2S"]
    background_windows = counts["Background"]
    votes_3 = sum(1 for row in ensemble_rows if str(row["ensemble_agreement"]) == "3/3")
    drone_ratio = (drone_windows / analysed) if analysed else 0.0
    if drone_ratio > 0.5:
        dominant = "无人机占主导"
        dominant_label = "drone_dominant"
    elif analysed and (background_windows / analysed) > 0.5:
        dominant = "背景占主导"
        dominant_label = "background_dominant"
    else:
        dominant = "混合或不确定"
        dominant_label = "mixed_or_uncertain"

    machine_counts = {
        "DJI Phantom 4 Pro": counts["DJI Phantom 4 Pro"],
        "DJI Air 2S": counts["DJI Air 2S"],
    }
    if drone_windows:
        top = sorted(machine_counts.items(), key=lambda item: (-item[1], item[0]))
        if top[0][1] > drone_windows / 2:
            machine_summary = top[0][0]
        else:
            machine_summary = "混合或不确定"
    else:
        machine_summary = "无无人机窗"

    return {
        "analysed_windows": int(analysed),
        "total_windows_available": None if total_windows_available is None else int(total_windows_available),
        "dropped_samples": int(dropped_samples),
        "counts": {name: int(counts[name]) for name in TIMELINE_ORDER},
        "ratios": {
            name: (counts[name] / analysed if analysed else 0.0) for name in TIMELINE_ORDER
        },
        "drone_windows": int(drone_windows),
        "drone_ratio": float(drone_ratio),
        "background_windows": int(background_windows),
        "uncertain_windows": int(uncertain),
        "any_drone_window": bool(drone_windows > 0),
        "dominant_label": dominant_label,
        "dominant_text": dominant,
        "machine_summary": machine_summary,
        "model_agreement_rate": (votes_3 / analysed) if analysed else 0.0,
        "all_three_agree_windows": int(votes_3),
        "note": (
            "文件级结论是已分析窗口的多数描述，不是另一个经过校准的事件检测器；"
            "决策分数(margin)不是概率。"
        ),
    }


def run_stream_analysis(
    source: IQSource,
    plan: WindowPlan,
    bundle: ModelBundle,
    progress: Callable[[dict[str, Any]], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    batch_size: int = BATCH_SIZE,
    keep_preview: int = 2_000_000,
    seeds: Sequence[int] = TRAINING_SEEDS,
) -> dict[str, Any]:
    """Run the frozen three-seed cascade over the planned windows.

    * exactly one float32 feature tensor per window (shared by the three stage-1
      networks) and one stage-1 forward per seed on that tensor;
    * stage 2 for a seed only when that seed's own stage-1 margin > 0, always on the
      *same* tensor of the window;
    * every per-seed intermediate value is returned, plus the majority-vote row.
    """
    seeds = tuple(int(seed) for seed in seeds)
    device = bundle.device
    frontend = bundle.frontend
    windows = plan.windows
    per_seed_rows: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    ensemble_rows: list[dict[str, Any]] = []
    preview: np.ndarray | None = None
    preview_digest = hashlib.sha256()
    started = time.perf_counter()
    features_seconds = 0.0
    stage1_seconds = 0.0
    stage2_seconds = 0.0
    checked = 0

    def emit(stage: str, extra: dict[str, Any] | None = None) -> None:
        if progress is None:
            return
        payload = {
            "stage": stage,
            "windows_done": int(checked),
            "windows_total": int(windows),
            "ratio": (checked / windows) if windows else 0.0,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        if extra:
            payload.update(extra)
        progress(payload)

    def guard() -> None:
        if cancelled is not None and cancelled():
            raise SignalsCancelled("任务已被取消。")

    try:
        emit("reading")
        for batch_start in range(0, windows, batch_size):
            guard()
            batch_stop = min(batch_start + batch_size, windows)
            positions = list(range(batch_start, batch_stop))
            signals: list[torch.Tensor] = []
            for index in positions:
                start, stop = plan.window_bounds(index)
                iq = source.read_iq(start, WINDOW_SAMPLES)
                if preview is None:
                    preview = np.asarray(iq[: min(keep_preview, iq.shape[0])], dtype=np.complex64)
                signals.append(iq_to_tensor(iq))
                del iq
            features_begin = time.perf_counter()
            features = features_from_signals(frontend, torch.stack(signals), device)
            features_seconds += time.perf_counter() - features_begin
            del signals
            emit("features", {"batch_start": batch_start, "batch_stop": batch_stop})
            guard()

            stage1_begin = time.perf_counter()
            stage1_logits = {
                seed: network_logits(bundle.stage1_network(seed), features, device)
                for seed in seeds
            }
            stage1_seconds += time.perf_counter() - stage1_begin
            emit("stage1", {"batch_start": batch_start, "batch_stop": batch_stop})
            guard()

            executed_local: dict[int, dict[int, np.ndarray]] = {}
            batch_ids: dict[int, dict[int, int]] = {}
            stage2_begin = time.perf_counter()
            for seed in seeds:
                margins = (stage1_logits[seed][:, 1] - stage1_logits[seed][:, 0]).astype(np.float32)
                selection = margins > 0.0
                executed_local[seed] = {}
                batch_ids[seed] = {}
                for batch_id, local_positions in enumerate(stage2_batches(selection, batch_size)):
                    guard()
                    batch_array = np.ascontiguousarray(features[local_positions])
                    batch = torch.from_numpy(batch_array).to(device)
                    with torch.inference_mode():
                        with torch.amp.autocast(
                            device_type=device.type,
                            enabled=device.type == "cuda",
                            dtype=torch.float16,
                        ):
                            logits = bundle.stage2_network(seed)(batch)
                    logits = logits.float().cpu().numpy()
                    if not np.isfinite(logits).all():
                        raise SignalAnalyzerError("第二阶段 logits 出现非有限值。")
                    for offset, local in enumerate(local_positions):
                        executed_local[seed][int(local)] = logits[offset]
                        batch_ids[seed][int(local)] = int(batch_id)
            stage2_seconds += time.perf_counter() - stage2_begin
            emit("stage2", {"batch_start": batch_start, "batch_stop": batch_stop})
            guard()

            for local, index in enumerate(positions):
                start, stop = plan.window_bounds(index)
                feature = np.ascontiguousarray(features[local])
                feature_sha = hashlib.sha256(feature.tobytes()).hexdigest()
                final_labels: list[int] = []
                for seed in seeds:
                    stage2_row = executed_local[seed].get(local)
                    row = build_prediction_row(
                        spec={"dataset": "user_file", "condition": "clean"},
                        row={
                            "dataset_index_position": int(index),
                            "file_path": getattr(source, "path", None) or "",
                            "filename": Path(getattr(source, "path", "") or "").name or "user_signal",
                            "target_code": "",
                            "target_name": "unknown",
                            "distance_code": "",
                            "segment_code": "",
                            "split": "user",
                            "stage1_label": -1,
                            "true_class": -1,
                            "true_class_name": "unknown",
                            "block_index": int(index),
                            "window_index": int(index),
                            "start": int(start),
                            "length": int(WINDOW_SAMPLES),
                            "source_samples": int(plan.analysed_stop),
                        },
                        seed=seed,
                        epochs={
                            "stage1_epoch": int(bundle.models[seed]["stage1_epoch"]),
                            "stage2_epoch": int(bundle.models[seed]["stage2_epoch"]),
                        },
                        stage1_logits=stage1_logits[seed][local],
                        stage2_logits=stage2_row,
                        stage2_batch_id=batch_ids[seed].get(local),
                        stage1_digest=feature_sha,
                        stage2_digest=feature_sha if stage2_row is not None else None,
                    )
                    row.update(
                        {
                            "window_index": int(index),
                            "start_sample": int(start),
                            "stop_sample": int(stop),
                            "start_seconds": start / plan.sample_rate_hz,
                            "stop_seconds": stop / plan.sample_rate_hz,
                        }
                    )
                    per_seed_rows[seed].append(row)
                    final_labels.append(int(row["final_prediction"]))
                ensemble_label, agreement, is_uncertain = majority_label(final_labels)
                ensemble_rows.append(
                    {
                        "window_index": int(index),
                        "start_sample": int(start),
                        "stop_sample": int(stop),
                        "start_seconds": start / plan.sample_rate_hz,
                        "stop_seconds": stop / plan.sample_rate_hz,
                        "votes_seed_2026": final_labels[0],
                        "votes_seed_2027": final_labels[1],
                        "votes_seed_2028": final_labels[2],
                        "ensemble_prediction": None
                        if ensemble_label == UNCERTAIN_LABEL
                        else int(_class_id(ensemble_label)),
                        "ensemble_class_name": ensemble_label,
                        "ensemble_agreement": agreement,
                        "is_uncertain": "true" if is_uncertain else "false",
                        "is_drone": ensemble_label in {"DJI Phantom 4 Pro", "DJI Air 2S"},
                        "majority_label_id": None
                        if ensemble_label == UNCERTAIN_LABEL
                        else int(_class_id(ensemble_label)),
                    }
                )
                checked += 1
            del features, stage1_logits, executed_local
            emit("windows", {"batch_start": batch_start, "batch_stop": batch_stop})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except SignalsCancelled:
        raise
    finally:
        source.close()

    if preview is not None:
        preview_digest.update(np.ascontiguousarray(preview.real).tobytes())
        preview_digest.update(np.ascontiguousarray(preview.imag).tobytes())

    return {
        "per_seed_rows": per_seed_rows,
        "ensemble_rows": ensemble_rows,
        "preview_iq": preview,
        "preview_sha256": preview_digest.hexdigest() if preview is not None else None,
        "timing": {
            "total_seconds": round(time.perf_counter() - started, 3),
            "features_seconds": round(features_seconds, 3),
            "stage1_seconds": round(stage1_seconds, 3),
            "stage2_seconds": round(stage2_seconds, 3),
        },
        "seeds": list(seeds),
    }


def _class_id(name: str) -> int:
    for class_id, class_name in CLASS_NAMES.items():
        if class_name == name:
            return class_id
    raise SignalAnalyzerError(f"未知类别名：{name}")


# ---------------------------------------------------------------------------
# export: CSV / JSON / Markdown / figures
# ---------------------------------------------------------------------------
def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.9g}"
    return value


def write_rows_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    """Write a CSV; an empty row list still produces the header (never a corrupt file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in columns})


def flatten_prediction_rows(
    per_seed_rows: dict[int, list[dict[str, Any]]], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(per_seed_rows[int(seed)])
    rows.sort(key=lambda row: (int(row["window_index"]), int(row["training_seed"])))
    return rows


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def build_timeline_figure(
    ensemble_rows: Sequence[dict[str, Any]],
    path: Path,
    stride_seconds: float,
    window_seconds: float,
):
    plt = _setup_matplotlib()
    figure, (axis_top, axis_bottom) = plt.subplots(
        2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    y = 0.0
    for row in ensemble_rows:
        label = str(row["ensemble_class_name"])
        axis_top.broken_barh(
            [(float(row["start_seconds"]), window_seconds)],
            (y, 1.0),
            facecolors=TIMELINE_COLORS.get(label, "#999999"),
            edgecolors="none",
        )
    axis_top.set_ylim(-0.2, 1.4)
    axis_top.set_yticks([])
    axis_top.set_title(
        f"窗口时间轴（比例尺：每窗 {window_seconds:g}s，步长 {stride_seconds:g}s，"
        f"共 {len(ensemble_rows)} 窗）"
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=TIMELINE_COLORS[name]) for name in TIMELINE_ORDER
    ]
    axis_top.legend(
        handles,
        [
            class_label_zh(0),
            class_label_zh(1),
            class_label_zh(2),
            UNCERTAIN_NAME_ZH,
        ],
        ncol=4,
        fontsize=8,
        loc="upper right",
    )
    axis_top.grid(axis="x", linestyle=":", alpha=0.4)

    drone_flags = [1.0 if bool(row["is_drone"]) else 0.0 for row in ensemble_rows]
    strides = [float(row["start_seconds"]) for row in ensemble_rows]
    axis_bottom.step(strides, drone_flags, where="post", color="#b3261e", linewidth=1.2)
    axis_bottom.set_ylim(-0.1, 1.2)
    axis_bottom.set_yticks([0, 1])
    axis_bottom.set_yticklabels(["无无人机窗", "有无人机窗"])
    axis_bottom.set_xlabel("时间（秒，相对文件起点）")
    axis_bottom.grid(linestyle=":", alpha=0.4)
    if len(ensemble_rows) > 1:
        axis_bottom.set_xlim(
            float(ensemble_rows[0]["start_seconds"]),
            float(ensemble_rows[-1]["stop_seconds"]),
        )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def build_waveform_figure(
    preview: np.ndarray,
    path: Path,
    sample_rate_hz: int,
    title: str,
):
    plt = _setup_matplotlib()
    count = int(preview.shape[0])
    stride = max(1, count // 200_000)
    sampled = preview[::stride]
    times = np.arange(sampled.shape[0]) * stride / sample_rate_hz
    figure, (axis_iq, axis_amplitude) = plt.subplots(2, 1, figsize=(12, 5.0), sharex=True)
    axis_iq.plot(times, sampled.real, linewidth=0.5, color="#1f4e9c", label="I")
    axis_iq.plot(times, sampled.imag, linewidth=0.5, color="#c25b1e", label="Q")
    axis_iq.set_ylabel("幅度")
    axis_iq.set_title(f"{title}：I/Q 波形预览（前 {count} 点）")
    axis_iq.legend(fontsize=8, loc="upper right")
    axis_iq.grid(linestyle=":", alpha=0.4)
    axis_amplitude.plot(times, np.abs(sampled), linewidth=0.5, color="#2f6f4f")
    axis_amplitude.set_ylabel("|IQ|")
    axis_amplitude.set_xlabel("时间（秒，相对文件起点）")
    axis_amplitude.grid(linestyle=":", alpha=0.4)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def compute_stft_db(iq: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """fftshift(log power) spectrogram of one window (visual only, no noise)."""
    import torch as _torch

    window = _torch.hamming_window(n_fft)
    signal = _torch.from_numpy(np.asarray(iq, dtype=np.complex64))
    spectra = _torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        normalized=True,
        onesided=False,
        return_complex=True,
    )
    log_power = 10.0 * _torch.log10(spectra.abs().square() + 1e-10)
    return _torch.fft.fftshift(log_power, dim=0).numpy()


def build_stft_figure(iq: np.ndarray, path: Path, title: str):
    plt = _setup_matplotlib()
    spectrogram = compute_stft_db(iq)
    figure, axis = plt.subplots(figsize=(8.4, 5.0))
    extent = [0.0, iq.shape[0] / SAMPLE_RATE_HZ, -SAMPLE_RATE_HZ / 2e6, SAMPLE_RATE_HZ / 2e6]
    image = axis.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="viridis",
    )
    axis.set_title(title)
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("相对通道中心频率（MHz）")
    figure.colorbar(image, ax=axis, label="log 功率 (dB)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    plt.close(figure)


def compute_stft_image(iq: np.ndarray, line_width: int = 1024) -> dict[str, Any]:
    """Server-side STFT preview for the browser canvas (no GUI toolkit needed).

    Returns a base64 PNG-free payload: the row/column layout of the shifted log
    power spectrogram plus the min/max used for colour scaling.
    """
    spectrogram = compute_stft_db(iq)
    rows, columns = int(spectrogram.shape[0]), int(spectrogram.shape[1])
    step = max(1, columns // max(1, line_width))
    reduced = spectrogram[:, ::step]
    low = float(np.percentile(reduced, 2.0))
    high = float(np.percentile(reduced, 98.0))
    return {
        "freq_bins": rows,
        "time_bins": int(reduced.shape[1]),
        "frequency_step_hz": SAMPLE_RATE_HZ / rows,
        "time_step_seconds": (HOP_LENGTH * step) / SAMPLE_RATE_HZ,
        "vmin_db": low,
        "vmax_db": high,
        "values": np.round(reduced, 3).tolist(),
    }


def preview_series(iq: np.ndarray, points: int = 2_000) -> dict[str, Any]:
    """Down-sampled I/Q/magnitude series embedded in ``analysis.json``."""
    count = int(iq.shape[0])
    step = max(1, count // max(1, points))
    sampled = iq[::step]
    return {
        "points": int(sampled.shape[0]),
        "decimation": int(step),
        "i": np.round(sampled.real, 6).tolist(),
        "q": np.round(sampled.imag, 6).tolist(),
        "magnitude": np.round(np.abs(sampled), 6).tolist(),
    }


def build_summary_markdown(payload: dict[str, Any]) -> str:
    """Chinese markdown summary of one analysed file."""
    plan = payload["window_plan"]
    summary = payload["file_summary"]
    model = payload["model"]
    environment = payload["environment"]
    lines: list[str] = []
    lines.append(f"# 任意IQ信号分析报告：{payload['input']['display_name']}")
    lines.append("")
    lines.append(f"- 分析时间（UTC）：{payload['created_at']}")
    lines.append(f"- 输入文件：`{payload['input']['path']}`")
    lines.append(
        f"- 检测格式：{payload['inspection']['detected_format']}；"
        f"总点数：{payload['inspection'].get('samples', 'unknown')}；"
        f"文件大小：{payload['inspection']['size_bytes']} 字节"
    )
    lines.append(
        f"- 采样率：{payload['input']['sample_rate_hz'] / 1e6:g} MS/s"
        f"（用户确认：{payload['input']['sample_rate_confirmed']}）；"
        f"协议：{payload['input']['rf_channel']}；未添加任何噪声"
    )
    lines.append(
        f"- 窗口：{WINDOW_SAMPLES} 点 = {WINDOW_SAMPLES / payload['input']['sample_rate_hz'] * 1e3:g} ms；"
        f"起点 {plan['start_sample']}，步长 {plan['stride_samples']}，"
        f"已分析 {plan['windows']} 窗（可选 {plan['requested_windows']} 窗）；"
        f"尾部丢弃 {plan['dropped_samples']} 点（不足一窗不分析、不补零）"
    )
    lines.append(
        f"- 环境：{environment['device']}"
        + (f"（{environment['gpu_name']}）" if environment.get("gpu_name") else "")
        + f"，torch {environment['torch_version']}，AMP float16={environment['amp_float16']}，"
        f"项目 .venv={environment['in_project_venv']}"
    )
    for warning in environment.get("warnings", []):
        lines.append(f"- ⚠️ {warning}")
    lines.append(
        f"- 冻结配置 SHA256：`{model['cascade_config_sha256']}`"
        f"（匹配={model['cascade_config_sha256_match']}）；六权重校验：{model['weights_verified']}"
    )
    lines.append("")
    lines.append("## 结论")
    lines.append("")
    lines.append(f"- 主导结论：**{summary['dominant_text']}**")
    lines.append(
        f"- 是否出现过无人机窗：{'是' if summary['any_drone_window'] else '否'}"
        f"（无人机窗 {summary['drone_windows']} / {summary['analysed_windows']} = "
        f"{summary['drone_ratio'] * 100:.1f}%）"
    )
    lines.append(f"- 机型描述：{summary['machine_summary']}")
    lines.append(
        f"- 模型一致率（三种子全部一致的窗口比例）：{summary['model_agreement_rate'] * 100:.1f}%"
    )
    for name in TIMELINE_ORDER:
        lines.append(
            f"- {class_label_zh(name) if name != UNCERTAIN_LABEL else UNCERTAIN_NAME_ZH}："
            f"{summary['counts'][name]} 窗（{summary['ratios'][name] * 100:.1f}%）"
        )
    lines.append(f"- 总耗时：{payload['timing']['total_seconds']:.2f} 秒")
    lines.append("")
    lines.append("## 边界说明")
    lines.append("")
    lines.append("- 本界面只支持当前三类（背景 / DJI Phantom 4 Pro / DJI Air 2S）与 100 MS/s、RF0 协议。")
    lines.append("- 决策分数（margin）是 logit 差值，**不是概率**，未做任何校准。")
    lines.append("- 未知设备可能被强行分到已知类别；结果仅代表这三种已训练类别。")
    lines.append("- 文件级结论是已分析窗口的多数描述，不是经过校准的事件检测器。")
    lines.append("- 短于 1,000,000 点的信号直接拒绝；采样率非 100 MHz 时只允许预览，不做分类。")
    lines.append("")
    lines.append("## 产物")
    lines.append("")
    for name in payload["artifacts"]:
        lines.append(f"- `{name}`")
    lines.append("")
    return "\n".join(lines)


def export_analysis(
    output_dir: Path,
    payload: dict[str, Any],
    per_seed_rows: dict[int, list[dict[str, Any]]],
    ensemble_rows: Sequence[dict[str, Any]],
    preview: np.ndarray | None,
    stft_window: dict[str, Any] | None = None,
    seeds: Sequence[int] = TRAINING_SEEDS,
) -> dict[str, Any]:
    """Write the full artefact set and return the artefact manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

    flat_rows = flatten_prediction_rows(per_seed_rows, seeds)
    predictions_path = output_dir / "window_predictions.csv"
    write_rows_csv(predictions_path, flat_rows, PREDICTION_COLUMNS)
    artifacts.append(predictions_path.name)

    ensemble_path = output_dir / "ensemble_predictions.csv"
    write_rows_csv(ensemble_path, ensemble_rows, ENSEMBLE_COLUMNS)
    artifacts.append(ensemble_path.name)

    timeline_path = output_dir / "timeline.png"
    build_timeline_figure(
        ensemble_rows,
        timeline_path,
        stride_seconds=payload["window_plan"]["stride_samples"] / payload["input"]["sample_rate_hz"],
        window_seconds=WINDOW_SAMPLES / payload["input"]["sample_rate_hz"],
    )
    artifacts.append(timeline_path.name)

    if preview is not None and preview.shape[0] > 0:
        waveform_path = output_dir / "waveform_preview.png"
        build_waveform_figure(
            preview,
            waveform_path,
            payload["input"]["sample_rate_hz"],
            payload["input"]["display_name"],
        )
        artifacts.append(waveform_path.name)

    stft_meta: dict[str, Any] = {}
    if stft_window is not None and stft_window.get("iq") is not None:
        index = int(stft_window["window_index"])
        stft_path = output_dir / f"stft_preview_window_{index}.png"
        build_stft_figure(
            stft_window["iq"],
            stft_path,
            f"{payload['input']['display_name']} 第 {index} 窗 STFT（n_fft={N_FFT}, hop={HOP_LENGTH}）",
        )
        artifacts.append(stft_path.name)
        stft_meta = {
            "window_index": index,
            "start_sample": int(stft_window["start_sample"]),
            "stop_sample": int(stft_window["stop_sample"]),
            "figure": stft_path.name,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
        }
    payload["stft_preview"] = stft_meta

    hash_manifest = {
        "cascade_config": str(CASCADE_CONFIG_PATH),
        "cascade_config_sha256": payload["model"]["cascade_config_sha256"],
        "cascade_config_sha256_expected": payload["model"]["cascade_config_sha256_expected"],
        "cascade_config_sha256_match": payload["model"]["cascade_config_sha256_match"],
        "ui_config": str(UI_CONFIG_PATH),
        "ui_config_sha256": sha256_file(UI_CONFIG_PATH),
        "weights": payload["model"]["weights"],
        "weights_verified": payload["model"]["weights_verified"],
    }
    hash_path = output_dir / "model_hashes.json"
    write_json(hash_path, hash_manifest)
    artifacts.append(hash_path.name)

    payload["artifacts"] = artifacts + ["analysis.json", "summary.md"]
    analysis_path = output_dir / "analysis.json"
    write_json(analysis_path, payload)
    artifacts.append(analysis_path.name)

    summary_path = output_dir / "summary.md"
    summary_path.write_text(build_summary_markdown(payload), encoding="utf-8")
    artifacts.append(summary_path.name)

    payload["artifacts"] = artifacts
    return {
        "output_dir": str(output_dir),
        "artifacts": artifacts,
        "window_predictions_csv": str(predictions_path),
        "ensemble_csv": str(ensemble_path),
        "analysis_json": str(analysis_path),
        "summary_md": str(summary_path),
    }


def sha256_paths(paths: Iterable[Path]) -> dict[str, str]:
    """Hash a list of files (used by the verification script for frozen artefacts)."""
    return {str(Path(path)): sha256_file(Path(path)) for path in paths}


__all__ = [
    "BATCH_SIZE",
    "CLASS_NAMES",
    "CLASS_NAMES_ZH",
    "DEFAULT_STRIDE_SAMPLES",
    "ENSEMBLE_COLUMNS",
    "ModelBundle",
    "PREDICTION_COLUMNS",
    "SAMPLE_RATE_HZ",
    "SignalAnalyzerError",
    "SignalsCancelled",
    "TRAINING_SEEDS",
    "UNCERTAIN_LABEL",
    "UNCERTAIN_NAME_ZH",
    "WINDOW_SAMPLES",
    "WindowPlan",
    "IQSource",
    "build_stft_figure",
    "build_summary_markdown",
    "build_timeline_figure",
    "build_waveform_figure",
    "class_label",
    "class_label_zh",
    "compute_stft_db",
    "compute_stft_image",
    "preview_series",
    "detect_format",
    "detect_npy_layout",
    "environment_report",
    "export_analysis",
    "features_from_signals",
    "inspect_path",
    "iq_to_tensor",
    "load_model_bundle",
    "majority_label",
    "open_source",
    "plan_windows",
    "raw_layout_from_params",
    "run_stream_analysis",
    "safe_name",
    "sha256_file",
    "sha256_paths",
    "summarize_file",
]
