from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "figure" / "signal_aware_dataset_small"
IMAGE_DIR = OUT_DIR / "images"
CONTACT_DIR = OUT_DIR / "sample_contact_sheets"

SEGMENT_SECONDS = 0.1
STEP_SECONDS = 0.05
TARGET_SNRS_DB = [0, -5, -10, -15, -20]
NFFTS = [512, 1024, 2048]
PSD_NFFT = 8192
IMAGE_SIZE = 256
DYNAMIC_RANGE_DB = 80.0

NOISE_BW_HZ = 1e6
NOISE_GUARD_HZ = 2e6
EDGE_GUARD_HZ = 5e6
SIGNAL_EDGE_GUARD_HZ = 5e6
MIN_RAW_SNR_DB = 0.0
MAX_MEASURED_ERROR_DB = 0.15


@dataclass(frozen=True)
class Scenario:
    class_name: str
    scenario_name: str
    data_dir: Path
    iq_file: str
    signal_bw_hz: float


SCENARIOS = [
    Scenario(
        class_name="DAUTEL_EVO_NANO",
        scenario_name="DAUTEL_EVO_NANO_pack1",
        data_dir=ROOT / "DAUTEL EVO NANO",
        iq_file="pack1_0-1s.iq",
        signal_bw_hz=20e6,
    ),
    Scenario(
        class_name="DJI_AVATA2",
        scenario_name="DJI_AVATA2_VTSBW40_pack4",
        data_dir=ROOT / "DJI AVATA2" / "VTSBW=40",
        iq_file="pack4_0-1s.iq",
        signal_bw_hz=40e6,
    ),
]


PARULA_ANCHORS = np.array(
    [
        [0.2422, 0.1504, 0.6603],
        [0.2578, 0.1818, 0.7511],
        [0.2751, 0.2342, 0.8710],
        [0.2803, 0.2782, 0.9221],
        [0.2795, 0.3750, 0.9712],
        [0.2602, 0.5343, 0.9815],
        [0.2206, 0.6459, 0.9506],
        [0.1834, 0.7362, 0.8858],
        [0.2798, 0.8067, 0.7564],
        [0.5418, 0.8384, 0.4964],
        [0.8897, 0.8040, 0.1830],
        [0.9769, 0.9839, 0.0805],
    ],
    dtype=np.float32,
)


def make_cmap() -> np.ndarray:
    idx = np.linspace(0, len(PARULA_ANCHORS) - 1, 256)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    w = idx - lo
    cmap = (1 - w)[:, None] * PARULA_ANCHORS[lo] + w[:, None] * PARULA_ANCHORS[hi]
    return np.clip(cmap * 255, 0, 255).astype(np.uint8)


CMAP = make_cmap()


def read_metadata(data_dir: Path) -> dict[str, str]:
    xml_files = sorted(data_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML metadata found in {data_dir}")
    root = ET.parse(xml_files[0]).getroot()
    return {child.tag: (child.text or "").strip() for child in root}


def open_iq(data_dir: Path, iq_file: str) -> tuple[np.memmap, int]:
    iq_path = data_dir / iq_file
    sample_count = iq_path.stat().st_size // 4 // 2
    raw = np.memmap(iq_path, dtype=np.float32, mode="r", shape=(sample_count * 2,))
    return raw, sample_count


def read_iq_window(raw: np.memmap, start_sample: int, sample_count: int, scale_factor: float) -> np.ndarray:
    block = raw[start_sample * 2 : (start_sample + sample_count) * 2]
    x = block[0::2].astype(np.float32) + 1j * block[1::2].astype(np.float32)
    return np.asarray(x / np.float32(scale_factor), dtype=np.complex64)


def average_psd(x: np.ndarray) -> np.ndarray:
    starts = np.arange(0, len(x) - PSD_NFFT + 1, PSD_NFFT, dtype=np.int64)
    window = np.hamming(PSD_NFFT).astype(np.float32)
    win_power = float(np.sum(window**2))
    acc = np.zeros(PSD_NFFT, dtype=np.float64)
    for start in starts:
        z = np.fft.fftshift(np.fft.fft(x[start : start + PSD_NFFT] * window, n=PSD_NFFT))
        acc += np.abs(z) ** 2 / win_power
    return acc / max(len(starts), 1)


def average_psd_components(x: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    starts = np.arange(0, len(x) - PSD_NFFT + 1, PSD_NFFT, dtype=np.int64)
    window = np.hamming(PSD_NFFT).astype(np.float32)
    win_power = float(np.sum(window**2))
    x_power = np.zeros(PSD_NFFT, dtype=np.float64)
    n_power = np.zeros(PSD_NFFT, dtype=np.float64)
    cross = np.zeros(PSD_NFFT, dtype=np.float64)
    for start in starts:
        xz = np.fft.fftshift(np.fft.fft(x[start : start + PSD_NFFT] * window, n=PSD_NFFT))
        nz = np.fft.fftshift(np.fft.fft(n[start : start + PSD_NFFT] * window, n=PSD_NFFT))
        x_power += np.abs(xz) ** 2 / win_power
        n_power += np.abs(nz) ** 2 / win_power
        cross += np.real(xz * np.conj(nz)) / win_power
    denom = max(len(starts), 1)
    return x_power / denom, n_power / denom, cross / denom


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(x, kernel, mode="same")


def locate_signal_band(psd: np.ndarray, fs: float, signal_bw_hz: float) -> tuple[np.ndarray, float, float]:
    n = len(psd)
    freqs = np.linspace(-fs / 2, fs / 2, n, endpoint=False)
    signal_bins = max(4, int(round(n * signal_bw_hz / fs)))
    edge_guard_bins = max(1, int(round(n * SIGNAL_EDGE_GUARD_HZ / fs)))
    smooth_bins = max(3, int(round(n * 0.25e6 / fs)))
    smooth = moving_average(psd, smooth_bins)
    csum = np.concatenate([[0.0], np.cumsum(smooth)])

    best_score = -np.inf
    best_start = edge_guard_bins
    min_start = edge_guard_bins
    max_start = n - signal_bins - edge_guard_bins
    if max_start < min_start:
        raise RuntimeError("Signal bandwidth is too wide for edge guard constraints")

    for start in range(min_start, max_start + 1):
        stop = start + signal_bins
        score = csum[stop] - csum[start]
        if score > best_score:
            best_score = score
            best_start = start

    signal_mask = np.zeros(n, dtype=bool)
    signal_mask[best_start : best_start + signal_bins] = True
    return signal_mask, float(freqs[best_start]), float(freqs[best_start + signal_bins - 1])


def locate_noise_band(
    psd: np.ndarray,
    fs: float,
    signal_low_hz: float,
    signal_high_hz: float,
) -> tuple[np.ndarray, float, float]:
    n = len(psd)
    freqs = np.linspace(-fs / 2, fs / 2, n, endpoint=False)
    noise_bins = max(4, int(round(n * NOISE_BW_HZ / fs)))
    edge_low = -fs / 2 + EDGE_GUARD_HZ
    edge_high = fs / 2 - EDGE_GUARD_HZ

    candidates: list[tuple[float, int, int]] = []
    for start in range(0, n - noise_bins + 1):
        stop = start + noise_bins
        low = freqs[start]
        high = freqs[stop - 1]
        if low < edge_low or high > edge_high:
            continue
        if high > signal_low_hz - NOISE_GUARD_HZ and low < signal_high_hz + NOISE_GUARD_HZ:
            continue
        candidates.append((float(np.mean(psd[start:stop])), start, stop))

    if not candidates:
        raise RuntimeError("No valid noise band found")

    _, start, stop = min(candidates, key=lambda item: item[0])
    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[start:stop] = True
    return noise_mask, float(freqs[start]), float(freqs[stop - 1])


def estimate_snr_from_psd(psd: np.ndarray, signal_mask: np.ndarray, noise_mask: np.ndarray) -> float:
    signal_plus_noise = float(np.mean(psd[signal_mask]))
    noise = float(np.mean(psd[noise_mask]))
    signal = signal_plus_noise - noise
    if signal <= 0 or noise <= 0:
        return float("-inf")
    return 10.0 * math.log10(signal / noise)


def estimate_snr(x: np.ndarray, signal_mask: np.ndarray, noise_mask: np.ndarray) -> float:
    return estimate_snr_from_psd(average_psd(x), signal_mask, noise_mask)


def add_noise_to_target(
    x: np.ndarray,
    target_snr_db: float,
    signal_mask: np.ndarray,
    noise_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float, str]:
    unit_noise = (
        rng.standard_normal(x.shape).astype(np.float32)
        + 1j * rng.standard_normal(x.shape).astype(np.float32)
    ).astype(np.complex64)

    raw_psd, unit_psd, cross_psd = average_psd_components(x, unit_noise)
    signal_plus_noise = float(np.mean(raw_psd[signal_mask]))
    noise_power = float(np.mean(raw_psd[noise_mask]))
    signal_power = signal_plus_noise - noise_power
    if signal_power <= 0 or noise_power <= 0:
        return x.copy(), float("-inf"), 0.0, "invalid_signal_or_noise_power"

    raw_snr = 10.0 * math.log10(signal_power / noise_power)
    if raw_snr <= target_snr_db:
        return x.copy(), raw_snr, 0.0, "raw_snr_below_target"

    def snr_for_scale(scale: float) -> float:
        psd = raw_psd + (scale * scale) * unit_psd + 2.0 * scale * cross_psd
        return estimate_snr_from_psd(psd, signal_mask, noise_mask)

    target_noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
    additional_noise_power = max(target_noise_power - noise_power, 0.0)

    unit_noise_power = float(np.mean(unit_psd[noise_mask]))
    if unit_noise_power <= 0:
        return x.copy(), raw_snr, 0.0, "invalid_unit_noise_power"

    noise_scale = math.sqrt(additional_noise_power / unit_noise_power)
    measured_snr = snr_for_scale(noise_scale)

    if not math.isfinite(measured_snr) or abs(measured_snr - target_snr_db) > MAX_MEASURED_ERROR_DB:
        fallback_candidates = []
        for factor in np.geomspace(0.25, 4.0, 33):
            candidate_scale = noise_scale * float(factor)
            candidate_snr = snr_for_scale(candidate_scale)
            if math.isfinite(candidate_snr):
                fallback_candidates.append((abs(candidate_snr - target_snr_db), candidate_scale, candidate_snr))
        if fallback_candidates:
            _, noise_scale, measured_snr = min(fallback_candidates, key=lambda item: item[0])

        if math.isfinite(measured_snr) and measured_snr < target_snr_db:
            lo = 0.0
            hi = noise_scale
        else:
            lo = noise_scale
            hi = noise_scale * 2.0
            for _ in range(8):
                hi_snr = snr_for_scale(hi)
                if math.isfinite(hi_snr) and hi_snr <= target_snr_db:
                    break
                lo = hi
                hi *= 2.0

        best = noise_scale
        best_snr = measured_snr
        for _ in range(14):
            mid = (lo + hi) / 2.0
            mid_snr = snr_for_scale(mid)
            if math.isfinite(mid_snr) and (
                not math.isfinite(best_snr)
                or abs(mid_snr - target_snr_db) < abs(best_snr - target_snr_db)
            ):
                best = mid
                best_snr = mid_snr
            if math.isfinite(mid_snr) and mid_snr > target_snr_db:
                lo = mid
            else:
                hi = mid
        noise_scale = best
        measured_snr = best_snr

    y = np.asarray(x + noise_scale * unit_noise, dtype=np.complex64)
    status = "ok" if abs(measured_snr - target_snr_db) <= MAX_MEASURED_ERROR_DB else "snr_not_converged"
    return y, measured_snr, noise_scale, status


def spectrogram_db(x: np.ndarray, nfft: int) -> np.ndarray:
    starts = np.linspace(0, len(x) - nfft, IMAGE_SIZE).astype(np.int64)
    rows = np.linspace(0, nfft - 1, IMAGE_SIZE).astype(np.int64)
    window = np.hamming(nfft).astype(np.float32)
    frame_indices = starts[:, None] + np.arange(nfft, dtype=np.int64)[None, :]
    frames = x[frame_indices] * window[None, :]
    z = np.fft.fftshift(np.fft.fft(frames, n=nfft, axis=1), axes=1)
    spec = 10 * np.log10(np.abs(z[:, rows]).T ** 2 + np.finfo(np.float32).eps)
    return spec.astype(np.float32, copy=False)


def colorize(spec: np.ndarray, ref_db: float) -> Image.Image:
    norm = np.clip((spec - (ref_db - DYNAMIC_RANGE_DB)) / DYNAMIC_RANGE_DB, 0, 1)
    return Image.fromarray(CMAP[np.round(norm * 255).astype(np.uint8)], "RGB")


def make_contact_sheet(image_paths: list[Path], labels: list[str], out_path: Path) -> None:
    label_h = 28
    pad = 18
    cell = IMAGE_SIZE
    sheet = Image.new("RGB", (pad * 2 + len(image_paths) * cell, pad * 2 + label_h + cell), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(zip(image_paths, labels)):
        x = pad + i * cell
        y = pad + label_h
        draw.text((x + 6, pad + 6), label, fill=(0, 0, 0))
        sheet.paste(Image.open(path).convert("RGB"), (x, y))
    sheet.save(out_path)


def run_scenario(scenario: Scenario) -> tuple[list[dict[str, str | int | float]], list[dict[str, str | int | float]]]:
    meta = read_metadata(scenario.data_dir)
    fs = float(meta["SampleRate"])
    scale = float(meta["ScaleFactor"])
    raw, sample_count = open_iq(scenario.data_dir, scenario.iq_file)
    segment_samples = int(round(fs * SEGMENT_SECONDS))
    step_samples = int(round(fs * STEP_SECONDS))
    starts = list(range(0, sample_count - segment_samples + 1, step_samples))
    rng = np.random.default_rng(abs(hash(scenario.scenario_name)) % (2**32))

    label_rows: list[dict[str, str | int | float]] = []
    qc_rows: list[dict[str, str | int | float]] = []

    print(f"\n=== {scenario.scenario_name}: {len(starts)} slices ===")
    for slice_idx, start in enumerate(starts):
        x = read_iq_window(raw, start, segment_samples, scale)
        psd = average_psd(x)
        signal_mask, sig_low, sig_high = locate_signal_band(psd, fs, scenario.signal_bw_hz)
        noise_mask, noise_low, noise_high = locate_noise_band(psd, fs, sig_low, sig_high)
        raw_snr = estimate_snr_from_psd(psd, signal_mask, noise_mask)
        qc_status = "ok" if raw_snr >= MIN_RAW_SNR_DB else "raw_snr_too_low"

        qc_rows.append(
            {
                "scenario": scenario.scenario_name,
                "class_name": scenario.class_name,
                "source_file": scenario.iq_file,
                "slice_index": slice_idx,
                "slice_start_sample": start,
                "slice_start_s": start / fs,
                "slice_end_s": (start + segment_samples) / fs,
                "estimated_raw_snr_db": raw_snr,
                "signal_low_hz": sig_low,
                "signal_high_hz": sig_high,
                "noise_low_hz": noise_low,
                "noise_high_hz": noise_high,
                "signal_bw_hz": scenario.signal_bw_hz,
                "noise_bw_hz": NOISE_BW_HZ,
                "qc_status": qc_status,
            }
        )

        # Fixed display reference per slice/NFFT from the 0 dB signal-aware image.
        noisy_by_snr: dict[int, tuple[np.ndarray, float, float, str]] = {}
        for target_snr in TARGET_SNRS_DB:
            y, measured_snr, noise_scale, noise_status = add_noise_to_target(
                x, target_snr, signal_mask, noise_mask, rng
            )
            noisy_by_snr[target_snr] = (y, measured_snr, noise_scale, noise_status)

        for nfft in NFFTS:
            ref_spec = spectrogram_db(noisy_by_snr[0][0], nfft)
            fixed_ref = float(np.max(ref_spec))
            contact_paths: list[Path] = []
            contact_labels: list[str] = []

            for target_snr in TARGET_SNRS_DB:
                y, measured_snr, noise_scale, noise_status = noisy_by_snr[target_snr]
                spec = ref_spec if target_snr == 0 else spectrogram_db(y, nfft)
                filename = (
                    f"{scenario.scenario_name}_slice{slice_idx:03d}_"
                    f"snr{target_snr}_nfft{nfft}.png"
                )
                path = IMAGE_DIR / filename
                colorize(spec, fixed_ref).save(path)

                status = qc_status if qc_status != "ok" else noise_status
                row = {
                    "filename": filename,
                    "class_name": scenario.class_name,
                    "scenario": scenario.scenario_name,
                    "source_file": scenario.iq_file,
                    "slice_index": slice_idx,
                    "slice_start_sample": start,
                    "slice_start_s": start / fs,
                    "target_snr_db": target_snr,
                    "measured_snr_db": measured_snr,
                    "estimated_raw_snr_db": raw_snr,
                    "snr_method": "signal_aware_slice",
                    "nfft": nfft,
                    "signal_low_hz": sig_low,
                    "signal_high_hz": sig_high,
                    "noise_low_hz": noise_low,
                    "noise_high_hz": noise_high,
                    "signal_bw_hz": scenario.signal_bw_hz,
                    "noise_bw_hz": NOISE_BW_HZ,
                    "noise_scale": noise_scale,
                    "display_ref_db": fixed_ref,
                    "status": status,
                }
                label_rows.append(row)

                if slice_idx in (0, len(starts) // 2, len(starts) - 1) and nfft == 1024:
                    contact_paths.append(path)
                    contact_labels.append(f"{target_snr} dB")

            if contact_paths:
                contact_name = f"{scenario.scenario_name}_slice{slice_idx:03d}_nfft1024.png"
                make_contact_sheet(contact_paths, contact_labels, CONTACT_DIR / contact_name)

        print(
            f"slice {slice_idx:02d} start={start / fs:.2f}s raw_snr={raw_snr:6.2f} dB "
            f"signal={sig_low / 1e6:6.1f}..{sig_high / 1e6:6.1f} MHz status={qc_status}"
        )

    return label_rows, qc_rows


def write_csv(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    labels: list[dict[str, str | int | float]] = []
    qc: list[dict[str, str | int | float]] = []
    for scenario in SCENARIOS:
        label_rows, qc_rows = run_scenario(scenario)
        labels.extend(label_rows)
        qc.extend(qc_rows)

    write_csv(OUT_DIR / "labels.csv", labels)
    write_csv(OUT_DIR / "qc_summary.csv", qc)

    print("\nDone")
    print(f"images: {IMAGE_DIR}")
    print(f"labels: {OUT_DIR / 'labels.csv'}")
    print(f"qc: {OUT_DIR / 'qc_summary.csv'}")
    print(f"contact sheets: {CONTACT_DIR}")
    print(f"image count: {len(labels)}")


if __name__ == "__main__":
    main()
