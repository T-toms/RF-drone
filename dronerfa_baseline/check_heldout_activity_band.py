#!/usr/bin/env python3
"""Activity-band check on fixed windows of the heldout files, compared with the originals.

Sampling scheme is fixed in advance and identical for every file of both sets:
the first acquisition block's *first* 10 ms window, the middle block's *middle*
10 ms window, and the last block's *last* 10 ms window.  Nothing is selected by
model performance, and no window is dropped for being difficult.

For each window and channel (RF0, and RF1 for comparison) the script reports:

* mean power (dB relative to the window's own RMS, so it is a *relative* measure),
* a 1024-point FFT power spectrum, the peak offset from the channel centre,
  the -3 dB occupied width and the 95 % occupied bandwidth,
* an activity flag based on the peak standing above the median spectrum level.

Frequencies are expressed as offsets from the channel centre frequency, converted
with the documented 100 MS/s sample rate.  The MAT containers carry no centre
frequency, so absolute RF bands are **not** asserted here; the report only points
out which of the two centre-frequency configurations documented in the dataset
paper the window's occupied span would be compatible with.  A background file is
never marked invalid merely because it shows no drone activity.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dronerfa_data import read_vector_slice  # noqa: E402


def read_iq(handle: h5py.File, channel: str, start: int, length: int) -> np.ndarray:
    """Read one window's I and Q in a single HDF5 access per sample block.

    The stored datasets are single-chunk gzip arrays, so any partial read
    decompresses the whole chunk; asking for I and Q together at least pays that
    cost once per window instead of once per component.
    """
    i_dataset = handle[f"{channel}_I"]
    q_dataset = handle[f"{channel}_Q"]
    if i_dataset.ndim == 2 and i_dataset.shape[0] == 1:
        # Fancy-index the row list so I and Q come back in one call each.
        i_values = np.asarray(i_dataset[[0], start : start + length], dtype=np.float32)
        q_values = np.asarray(q_dataset[[0], start : start + length], dtype=np.float32)
        return i_values.reshape(-1) + 1j * q_values.reshape(-1)
    return read_vector_slice(i_dataset, start, length) + 1j * read_vector_slice(
        q_dataset, start, length
    )

HELDOUT_ROOT = Path(r"F:\DroneRFa数据集\heldout_20260917")
ORIGINAL_ROOT = Path(r"C:\Users\Administrator\Documents\DroneRFa数据集")
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "heldout_preparation_20260917"
CANDIDATES = (
    SCRIPT_DIR / "outputs" / "noise_ab_replicates_20260917" / "candidate_heldout_files.json"
)

WINDOW_SIZE = 1_000_000
BLOCK_SIZE = 10_000_000
SAMPLE_RATE_HZ = 100_000_000
NFFT = 4096
ACTIVITY_MARGIN_DB = 6.0
_print_lock = threading.Lock()

# Documented in the dataset paper (lines 243-250 of the local extraction):
#   default   -> RF0 centre 2440 MHz, RF1 centre 5800 MHz
#   variant   -> RF0 centre  915 MHz, RF1 centre 2440 MHz
# S0000-S0111 are the "initial communication band" group, which the paper says uses
# 915 MHz or 2.4 GHz.  The containers themselves do not record which one applies.
CENTRE_CONFIGURATIONS = {
    "default": {"RF0": 2440e6, "RF1": 5800e6},
    "ism_variant": {"RF0": 915e6, "RF1": 2440e6},
}


def log(message: str) -> None:
    with _print_lock:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def parse_name(filename: str) -> dict[str, str]:
    stem = filename[:-4] if filename.lower().endswith(".mat") else filename
    target, distance, segment = stem.split("_")
    return {
        "target_code": target[1:],
        "distance_code": distance[1:],
        "segment_code": segment[1:],
    }


def choose_windows(source_samples: int) -> list[dict]:
    """First window of first block, middle window of middle block, last window of last block."""
    blocks = source_samples // BLOCK_SIZE
    if blocks < 1:
        return []
    windows_per_block = BLOCK_SIZE // WINDOW_SIZE
    middle_block = blocks // 2
    picks = [
        ("first_block_first_window", 0, 0),
        ("middle_block_middle_window", middle_block, windows_per_block // 2),
        ("last_block_last_window", blocks - 1, windows_per_block - 1),
    ]
    return [
        {
            "label": label,
            "block_index": block,
            "window_in_block": window,
            "start": block * BLOCK_SIZE + window * WINDOW_SIZE,
            "length": WINDOW_SIZE,
        }
        for label, block, window in picks
    ]


def welch_psd(iq: np.ndarray, nfft: int = 1024, overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Averaged periodogram over many sub-segments of one 10 ms window.

    Averaging matters here: a single 1024-point periodogram of a noise-like window
    fluctuates by several dB, so a single-segment peak would be dominated by
    random fluctuation.  With 1,000,000 samples and 50 % overlap there are about
    1950 segments, giving a stable estimate.
    """
    step = int(nfft * (1.0 - overlap))
    count = (iq.size - nfft) // step + 1
    window = np.hanning(nfft)
    normalisation = float(np.sum(window**2))
    accumulator = np.zeros(nfft, dtype=np.float64)
    for index in range(count):
        segment = iq[index * step : index * step + nfft] * window
        spectrum = np.fft.fft(segment, n=nfft)
        accumulator += (spectrum.real**2 + spectrum.imag**2)
    psd = accumulator / (count * normalisation)
    freqs = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / SAMPLE_RATE_HZ))
    return freqs, np.fft.fftshift(psd)


def analyse_window(iq: np.ndarray) -> dict:
    power = float(np.mean(np.abs(iq) ** 2, dtype=np.float64))
    rms = float(np.sqrt(max(power, 1e-30)))
    # Remove any residual DC before the spectral estimate; the project's frontend
    # de-means each window too, so this matches how the data is actually consumed.
    centered = iq - np.mean(iq, dtype=np.complex128)
    freqs, psd = welch_psd(centered.astype(np.complex64))
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-30))

    # Exclude the outermost bins from peak search: they are the edges of the
    # baseband span, where a slice boundary would otherwise always win.
    guard = max(1, freqs.size // 64)
    interior = slice(guard, freqs.size - guard)
    peak_offset_index = int(np.argmax(psd_db[interior])) + guard
    peak_freq = float(freqs[peak_offset_index])
    peak_db = float(psd_db[peak_offset_index])
    median_db = float(np.median(psd_db[interior]))

    half_power = peak_db - 3.0
    left = peak_offset_index
    while left > 0 and psd_db[left] > half_power:
        left -= 1
    right = peak_offset_index
    while right < psd_db.size - 1 and psd_db[right] > half_power:
        right += 1
    width_3db = float(freqs[right] - freqs[left])

    linear = psd[interior]
    order = np.argsort(linear)[::-1]
    cumulative = np.cumsum(linear[order])
    total = float(cumulative[-1])
    interior_freqs = freqs[interior]
    index_95 = int(np.searchsorted(cumulative, 0.95 * total))
    index_05 = int(np.searchsorted(cumulative, 0.05 * total))
    occupied_95 = float(
        abs(
            interior_freqs[order[min(index_95, order.size - 1)]]
            - interior_freqs[order[min(index_05, order.size - 1)]]
        )
    )

    return {
        "mean_power_dbfs": float(10.0 * np.log10(max(power, 1e-30))),
        "window_rms": rms,
        "peak_freq_offset_hz": peak_freq,
        "peak_db_above_median": float(peak_db - median_db),
        "median_psd_db": median_db,
        "width_3db_hz": width_3db,
        "occupied_95_hz": occupied_95,
        "active": bool(peak_db - median_db >= ACTIVITY_MARGIN_DB),
        "finite": bool(np.isfinite(iq).all()),
        "welch_segments": int((iq.size - 1024) // 512 + 1),
    }


def analyse_file(path: Path) -> dict:
    record: dict = {"filename": path.name, "file_path": str(path)}
    codes = parse_name(path.name)
    record.update(codes)
    with h5py.File(path, "r") as handle:
        source_samples = int(handle["RF0_I"].shape[1] if handle["RF0_I"].ndim == 2 else handle["RF0_I"].shape[0])
        record["source_samples"] = source_samples
        record["complete_blocks"] = source_samples // BLOCK_SIZE
        picks = choose_windows(source_samples)
        record["windows"] = []
        for pick in picks:
            entry = dict(pick)
            for channel in ("RF0", "RF1"):
                iq = read_iq(handle, channel, pick["start"], pick["length"])
                entry[channel] = analyse_window(iq)
            record["windows"].append(entry)
    return record


def summarise(records: list[dict]) -> dict:
    summary: dict = {}
    for target in sorted({record["target_code"] for record in records}):
        subset = [record for record in records if record["target_code"] == target]
        entry: dict = {"files": len(subset), "windows_per_channel": {}}
        for channel in ("RF0", "RF1"):
            offsets: list[float] = []
            widths: list[float] = []
            occupancy: list[float] = []
            active = 0
            total_windows = 0
            for record in subset:
                for window in record["windows"]:
                    stats = window[channel]
                    offsets.append(stats["peak_freq_offset_hz"])
                    widths.append(stats["width_3db_hz"])
                    occupancy.append(stats["occupied_95_hz"])
                    active += int(stats["active"])
                    total_windows += 1
            entry["windows_per_channel"][channel] = {
                "windows": total_windows,
                "active_windows": active,
                "active_fraction": active / total_windows if total_windows else None,
                "peak_offset_mean_hz": float(np.mean(offsets)) if offsets else None,
                "peak_offset_min_hz": float(np.min(offsets)) if offsets else None,
                "peak_offset_max_hz": float(np.max(offsets)) if offsets else None,
                "width_3db_median_hz": float(np.median(widths)) if widths else None,
                "occupied_95_median_hz": float(np.median(occupancy)) if occupancy else None,
            }
        summary[target] = entry
    return summary


def compatibility(records: list[dict]) -> dict:
    """Which documented centre-frequency configuration each target's RF0 span fits."""
    result: dict = {}
    for target in sorted({record["target_code"] for record in records}):
        subset = [record for record in records if record["target_code"] == target]
        spans = []
        for record in subset:
            for window in record["windows"]:
                stats = window["RF0"]
                if stats["active"]:
                    spans.append((stats["peak_freq_offset_hz"], stats["occupied_95_hz"]))
        if not spans:
            result[target] = {
                "active_rf0_windows": 0,
                "note": "RF0 固定窗口未见明显活动；不据此判定文件不合格",
            }
            continue
        max_abs_offset = max(abs(offset) + width / 2 for offset, width in spans)
        result[target] = {
            "active_rf0_windows": len(spans),
            "max_abs_offset_plus_halfwidth_hz": float(max_abs_offset),
            "absolute_band_if_default_centres": [
                CENTRE_CONFIGURATIONS["default"]["RF0"] - 50e6,
                CENTRE_CONFIGURATIONS["default"]["RF0"] + 50e6,
            ],
            "absolute_band_if_ism_variant_centres": [
                CENTRE_CONFIGURATIONS["ism_variant"]["RF0"] - 50e6,
                CENTRE_CONFIGURATIONS["ism_variant"]["RF0"] + 50e6,
            ],
            "note": "采样率100 MS/s给出±50 MHz基带范围；MAT内无中心频率元数据，"
            "故只列出声明的两种配置下的绝对范围，不宣称实际频段",
        }
    return result


def make_figure(summary: dict, output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001 - optional
        log(f"未生成图：{type(exc).__name__}: {exc}")
        return False

    targets = sorted(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    for axis, channel in zip(axes, ("RF0", "RF1")):
        labels = [f"T{t}\n({summary[t]['files']} files)" for t in targets]
        active = [
            summary[t]["windows_per_channel"][channel]["active_fraction"] or 0.0 for t in targets
        ]
        axis.bar(labels, active, color="#3b6ea5")
        axis.set_ylim(0, 1.05)
        axis.set_title(f"{channel}: active fraction of fixed windows")
        axis.set_ylabel("active window fraction")
        axis.grid(axis="y", alpha=0.3)
    fig.suptitle(
        "Fixed windows (first / middle / last block): T0000 background, "
        "T0010 Phantom 4 Pro, T0101 Air 2S"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    log(f"图：{output_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-root", type=Path, default=HELDOUT_ROOT)
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    new_paths = sorted(args.heldout_root.glob("*.mat"))
    old_paths = sorted(args.original_root.glob("*.mat"))
    new_records = [analyse_file(path) for path in new_paths]
    old_records = [analyse_file(path) for path in old_paths]

    new_summary = summarise(new_records)
    old_summary = summarise(old_records)

    # Protocol comparison: same target, old vs new activity pattern and peak offset.
    protocol: dict = {}
    for target in sorted(new_summary):
        if target not in old_summary:
            continue
        protocol[target] = {}
        for channel in ("RF0", "RF1"):
            new_entry = new_summary[target]["windows_per_channel"][channel]
            old_entry = old_summary[target]["windows_per_channel"][channel]
            protocol[target][channel] = {
                "heldout_active_fraction": new_entry["active_fraction"],
                "original_active_fraction": old_entry["active_fraction"],
                "heldout_peak_offset_mean_hz": new_entry["peak_offset_mean_hz"],
                "original_peak_offset_mean_hz": old_entry["peak_offset_mean_hz"],
                "offset_difference_hz": (
                    None
                    if new_entry["peak_offset_mean_hz"] is None
                    or old_entry["peak_offset_mean_hz"] is None
                    else new_entry["peak_offset_mean_hz"] - old_entry["peak_offset_mean_hz"]
                ),
                "heldout_occupied_95_median_hz": new_entry["occupied_95_median_hz"],
                "original_occupied_95_median_hz": old_entry["occupied_95_median_hz"],
            }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampling_scheme": {
            "description": "每个文件取首块首窗、中间块中间窗、末块末窗，共3个固定10ms窗；"
            "新12文件与原12文件使用完全相同的取窗规则",
            "window_size_samples": WINDOW_SIZE,
            "blocks_used": ["first", "middle", "last"],
            "windows_per_file": 3,
            "selection_basis": "预先固定，未参考任何模型表现，未因困难而剔除文件",
        },
        "frequency_convention": {
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "baseband_span_hz": [-50e6, 50e6],
            "absolute_frequency_available": False,
            "reason": "MAT容器内无采集中心频率元数据",
            "documented_centre_configurations": CENTRE_CONFIGURATIONS,
            "documented_source": "dronerfa_baseline/outputs/literature_review/text/"
            "01_DroneRFa_Dataset_JEIT2024.txt 第243-250行（本地提取文本）",
        },
        "activity_threshold_db": ACTIVITY_MARGIN_DB,
        "heldout_summary": new_summary,
        "original_summary": old_summary,
        "protocol_comparison": protocol,
        "absolute_band_compatibility": compatibility(new_records),
        "heldout_windows": new_records,
        "original_windows": old_records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "activity_band_check.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figure(
        new_summary, args.output_dir / "activity_band_heldout.png"
    )
    make_figure(old_summary, args.output_dir / "activity_band_original.png")

    for target in sorted(new_summary):
        rf0 = new_summary[target]["windows_per_channel"]["RF0"]
        rf1 = new_summary[target]["windows_per_channel"]["RF1"]
        log(
            f"T{target}: RF0 活动 {rf0['active_windows']}/{rf0['windows']}，"
            f"峰值偏移均值 {rf0['peak_offset_mean_hz']}；"
            f"RF1 活动 {rf1['active_windows']}/{rf1['windows']}"
        )
    log(f"报告：{output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
