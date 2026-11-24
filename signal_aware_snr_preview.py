from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "figure" / "signal_aware_snr_preview"

TARGET_SNRS_DB = [0, -5, -10, -15, -20]
DISPLAY_NFFT = 1024
PSD_NFFT = 8192
SEGMENT_SECONDS = 0.1
IMAGE_SIZE = 256
DYNAMIC_RANGE_DB = 80.0


@dataclass(frozen=True)
class Scenario:
    name: str
    data_dir: Path
    iq_file: str
    signal_bw_hz: float
    noise_bw_hz: float = 1e6


SCENARIOS = [
    Scenario(
        name="DAUTEL_EVO_NANO",
        data_dir=ROOT / "DAUTEL EVO NANO",
        iq_file="pack1_0-1s.iq",
        signal_bw_hz=20e6,
    ),
    Scenario(
        name="DJI_AVATA2_VTSBW40",
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


def make_colormap() -> np.ndarray:
    idx = np.linspace(0, len(PARULA_ANCHORS) - 1, 256)
    lo = np.floor(idx).astype(int)
    hi = np.ceil(idx).astype(int)
    w = idx - lo
    cmap = (1 - w)[:, None] * PARULA_ANCHORS[lo] + w[:, None] * PARULA_ANCHORS[hi]
    return np.clip(cmap * 255, 0, 255).astype(np.uint8)


CMAP = make_colormap()


def read_metadata(data_dir: Path) -> dict[str, str]:
    xml_files = sorted(data_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML metadata found in {data_dir}")
    root = ET.parse(xml_files[0]).getroot()
    return {child.tag: (child.text or "").strip() for child in root}


def read_segment(scenario: Scenario, meta: dict[str, str]) -> tuple[np.ndarray, float]:
    fs = float(meta["SampleRate"])
    scale = np.float32(float(meta["ScaleFactor"]))
    count = int(fs * SEGMENT_SECONDS) * 2
    iq_path = scenario.data_dir / scenario.iq_file
    raw = np.memmap(iq_path, dtype=np.float32, mode="r", shape=(count,))
    x = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    return np.asarray(x / scale, dtype=np.complex64), fs


def average_spectrum_power(x: np.ndarray, nfft: int) -> np.ndarray:
    hop = nfft
    usable = len(x) - nfft
    starts = np.arange(0, usable + 1, hop, dtype=np.int64)
    window = np.hamming(nfft).astype(np.float32)
    win_power = float(np.sum(window**2))
    acc = np.zeros(nfft, dtype=np.float64)
    for start in starts:
        frame = x[start : start + nfft] * window
        z = np.fft.fftshift(np.fft.fft(frame, n=nfft))
        acc += (np.abs(z) ** 2) / win_power
    return (acc / max(len(starts), 1)).astype(np.float64)


def moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(x, kernel, mode="same")


def locate_signal_and_noise(
    psd: np.ndarray,
    fs: float,
    signal_bw_hz: float,
    noise_bw_hz: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    n = len(psd)
    freqs = np.linspace(-fs / 2, fs / 2, n, endpoint=False)
    signal_bins = max(4, int(round(n * signal_bw_hz / fs)))
    noise_bins = max(4, int(round(n * noise_bw_hz / fs)))

    smooth_bins = max(3, int(round(n * 0.25e6 / fs)))
    smooth = moving_average(psd, smooth_bins)

    csum = np.concatenate([[0.0], np.cumsum(smooth)])
    window_power = csum[signal_bins:] - csum[:-signal_bins]
    signal_start = int(np.argmax(window_power))
    signal_stop = signal_start + signal_bins

    signal_mask = np.zeros(n, dtype=bool)
    signal_mask[signal_start:signal_stop] = True

    guard_bins = max(2, int(round(n * 0.75e6 / fs)))
    edge_guard_bins = max(1, int(round(n * 5e6 / fs)))
    candidates: list[tuple[float, int, int]] = []
    for start in range(edge_guard_bins, n - noise_bins - edge_guard_bins + 1):
        stop = start + noise_bins
        if stop > signal_start - guard_bins and start < signal_stop + guard_bins:
            continue
        candidates.append((float(np.mean(smooth[start:stop])), start, stop))

    if not candidates:
        raise RuntimeError("Cannot find a noise band outside the signal band")
    _, noise_start, noise_stop = min(candidates, key=lambda item: item[0])

    noise_mask = np.zeros(n, dtype=bool)
    noise_mask[noise_start:noise_stop] = True

    info = {
        "signal_low_hz": float(freqs[signal_start]),
        "signal_high_hz": float(freqs[signal_stop - 1]),
        "signal_center_hz": float((freqs[signal_start] + freqs[signal_stop - 1]) / 2),
        "noise_low_hz": float(freqs[noise_start]),
        "noise_high_hz": float(freqs[noise_stop - 1]),
        "noise_center_hz": float((freqs[noise_start] + freqs[noise_stop - 1]) / 2),
    }
    return signal_mask, noise_mask, info


def estimate_snr_from_masks(x: np.ndarray, signal_mask: np.ndarray, noise_mask: np.ndarray) -> float:
    psd = average_spectrum_power(x, PSD_NFFT)
    signal_plus_noise = float(np.mean(psd[signal_mask]))
    noise = float(np.mean(psd[noise_mask]))
    signal = max(signal_plus_noise - noise, np.finfo(np.float64).tiny)
    return 10.0 * math.log10(signal / max(noise, np.finfo(np.float64).tiny))


def add_noise_to_target_snr(
    x: np.ndarray,
    target_snr_db: float,
    signal_mask: np.ndarray,
    noise_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, float]:
    unit_noise = (
        rng.standard_normal(x.shape).astype(np.float32)
        + 1j * rng.standard_normal(x.shape).astype(np.float32)
    ).astype(np.complex64)

    raw_snr = estimate_snr_from_masks(x, signal_mask, noise_mask)
    if raw_snr <= target_snr_db:
        return x.copy(), raw_snr, 0.0

    lo = 0.0
    hi = float(np.sqrt(np.mean(np.abs(x) ** 2)))
    for _ in range(12):
        snr = estimate_snr_from_masks(x + hi * unit_noise, signal_mask, noise_mask)
        if snr <= target_snr_db:
            break
        hi *= 2.0

    best = hi
    best_snr = estimate_snr_from_masks(x + best * unit_noise, signal_mask, noise_mask)
    for _ in range(18):
        mid = (lo + hi) / 2.0
        snr = estimate_snr_from_masks(x + mid * unit_noise, signal_mask, noise_mask)
        if abs(snr - target_snr_db) < abs(best_snr - target_snr_db):
            best = mid
            best_snr = snr
        if snr > target_snr_db:
            lo = mid
        else:
            hi = mid

    y = np.asarray(x + best * unit_noise, dtype=np.complex64)
    return y, best_snr, best


def spectrogram_db(x: np.ndarray, nfft: int) -> np.ndarray:
    starts = np.linspace(0, len(x) - nfft, IMAGE_SIZE).astype(np.int64)
    rows = np.linspace(0, nfft - 1, IMAGE_SIZE).astype(np.int64)
    window = np.hamming(nfft).astype(np.float32)
    spec = np.empty((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for col, start in enumerate(starts):
        frame = x[start : start + nfft] * window
        z = np.fft.fftshift(np.fft.fft(frame, n=nfft))
        spec[:, col] = 10 * np.log10(np.abs(z[rows]) ** 2 + np.finfo(np.float32).eps)
    return spec


def colorize(spec: np.ndarray, ref_db: float) -> Image.Image:
    floor_db = ref_db - DYNAMIC_RANGE_DB
    norm = np.clip((spec - floor_db) / DYNAMIC_RANGE_DB, 0, 1)
    rgb = CMAP[np.round(norm * 255).astype(np.uint8)]
    return Image.fromarray(rgb, "RGB")


def draw_contact_sheet(image_paths: list[Path], labels: list[str], out_path: Path) -> None:
    label_h = 28
    pad = 18
    cell = IMAGE_SIZE
    sheet = Image.new("RGB", (pad * 2 + len(image_paths) * cell, pad * 2 + label_h + cell), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (path, label) in enumerate(zip(image_paths, labels)):
        x = pad + idx * cell
        y = pad + label_h
        draw.text((x + 6, pad + 6), label, fill=(0, 0, 0))
        sheet.paste(Image.open(path).convert("RGB"), (x, y))
    sheet.save(out_path)


def draw_psd_debug(
    psd: np.ndarray,
    fs: float,
    signal_mask: np.ndarray,
    noise_mask: np.ndarray,
    out_path: Path,
) -> None:
    width, height = 1100, 420
    margin_l, margin_r, margin_t, margin_b = 70, 25, 35, 55
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    psd_db = 10 * np.log10(psd + np.finfo(np.float64).tiny)
    lo = float(np.percentile(psd_db, 1))
    hi = float(np.percentile(psd_db, 99.5))
    if hi <= lo:
        hi = lo + 1.0

    def x_pos(idx: int) -> int:
        return margin_l + int(round(idx * plot_w / (len(psd_db) - 1)))

    def y_pos(value: float) -> int:
        norm = (value - lo) / (hi - lo)
        return margin_t + plot_h - int(round(np.clip(norm, 0, 1) * plot_h))

    img = Image.new("RGB", (width, height), "white")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)

    for mask, color in [(signal_mask, (255, 210, 0, 80)), (noise_mask, (0, 120, 255, 75))]:
        idxs = np.flatnonzero(mask)
        if len(idxs):
            odraw.rectangle(
                [x_pos(int(idxs[0])), margin_t, x_pos(int(idxs[-1])), margin_t + plot_h],
                fill=color,
            )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    draw.rectangle([margin_l, margin_t, margin_l + plot_w, margin_t + plot_h], outline=(0, 0, 0))
    step = max(1, len(psd_db) // 900)
    points = [(x_pos(i), y_pos(float(psd_db[i]))) for i in range(0, len(psd_db), step)]
    if len(points) > 1:
        draw.line(points, fill=(20, 20, 20), width=1)

    for mhz in [-50, -25, 0, 25, 50]:
        idx = int(round((mhz * 1e6 + fs / 2) / fs * len(psd_db)))
        idx = min(max(idx, 0), len(psd_db) - 1)
        x = x_pos(idx)
        draw.line([(x, margin_t + plot_h), (x, margin_t + plot_h + 5)], fill=(0, 0, 0))
        draw.text((x - 16, margin_t + plot_h + 12), f"{mhz}", fill=(0, 0, 0))

    draw.text((margin_l, 8), "PSD debug: yellow=signal band, blue=noise band", fill=(0, 0, 0))
    draw.text((margin_l + plot_w // 2 - 45, height - 25), "Baseband frequency (MHz)", fill=(0, 0, 0))
    draw.text((8, margin_t + 8), "Power dB", fill=(0, 0, 0))
    img.save(out_path)


def run_scenario(scenario: Scenario) -> list[dict[str, str | float]]:
    meta = read_metadata(scenario.data_dir)
    x, fs = read_segment(scenario, meta)
    scenario_out = OUT_ROOT / scenario.name
    image_dir = scenario_out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    raw_psd = average_spectrum_power(x, PSD_NFFT)
    signal_mask, noise_mask, band_info = locate_signal_and_noise(
        raw_psd, fs, scenario.signal_bw_hz, scenario.noise_bw_hz
    )
    draw_psd_debug(raw_psd, fs, signal_mask, noise_mask, scenario_out / "psd_debug_bands.png")
    raw_est_snr = estimate_snr_from_masks(x, signal_mask, noise_mask)

    rng = np.random.default_rng(abs(hash(scenario.name)) % (2**32))
    specs: dict[float, tuple[np.ndarray, float, float]] = {}
    for target in TARGET_SNRS_DB:
        y, measured, noise_scale = add_noise_to_target_snr(x, target, signal_mask, noise_mask, rng)
        specs[target] = (spectrogram_db(y, DISPLAY_NFFT), measured, noise_scale)
        print(
            f"{scenario.name} target {target:>4} dB -> measured {measured:7.3f} dB "
            f"(noise_scale={noise_scale:.4g})"
        )

    fixed_ref = float(np.max(specs[0][0]))
    image_paths: list[Path] = []
    labels: list[str] = []
    rows: list[dict[str, str | float]] = []

    for target in TARGET_SNRS_DB:
        spec, measured, noise_scale = specs[target]
        filename = f"{scenario.name}_signalAware_snr{target}_nfft{DISPLAY_NFFT}.png"
        path = image_dir / filename
        colorize(spec, fixed_ref).save(path)
        image_paths.append(path)
        labels.append(f"{target} dB")

        row = {
            "scenario": scenario.name,
            "drone": meta["Drone"],
            "source_file": scenario.iq_file,
            "xml_reference_snr_db": float(meta["ReferenceSNRLevel"]),
            "scale_factor": float(meta["ScaleFactor"]),
            "signal_bw_hz": scenario.signal_bw_hz,
            "noise_bw_hz": scenario.noise_bw_hz,
            "estimated_raw_snr_db": raw_est_snr,
            "target_snr_db": target,
            "measured_snr_db": measured,
            "noise_scale": noise_scale,
            "nfft": DISPLAY_NFFT,
            **band_info,
            "filename": filename,
        }
        rows.append(row)

    draw_contact_sheet(image_paths, labels, scenario_out / "contact_fixed_clim_nfft1024.png")
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str | float]] = []
    for scenario in SCENARIOS:
        print(f"\n=== {scenario.name} ===")
        rows = run_scenario(scenario)
        all_rows.extend(rows)

    csv_path = OUT_ROOT / "summary.csv"
    fieldnames = list(all_rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nsummary: {csv_path}")


if __name__ == "__main__":
    main()
