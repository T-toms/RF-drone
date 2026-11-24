from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DAUTEL EVO NANO"
IQ_FILE = "pack1_0-1s.iq"
OUT_DIR = ROOT / "figure" / "DAUTEL_band_stability"

WINDOW_SECONDS = 0.1
STEP_SECONDS = 0.05
PSD_NFFT = 8192
NOISE_BW_HZ = 1e6
NOISE_GUARD_HZ = 2e6
EDGE_GUARD_HZ = 5e6

CANDIDATE_BANDS_MHZ = [
    (-50, -30),
    (-40, -20),
    (-30, -10),
    (-20, 0),
    (-10, 10),
    (0, 20),
    (10, 30),
    (20, 40),
    (30, 50),
]


def read_meta() -> dict[str, str]:
    root = ET.parse(DATA_DIR / "pack1.xml").getroot()
    return {child.tag: (child.text or "").strip() for child in root}


def read_iq(meta: dict[str, str]) -> tuple[np.memmap, float, float]:
    fs = float(meta["SampleRate"])
    scale = float(meta["ScaleFactor"])
    iq_path = DATA_DIR / IQ_FILE
    sample_count = iq_path.stat().st_size // 4 // 2
    raw = np.memmap(iq_path, dtype=np.float32, mode="r", shape=(sample_count * 2,))
    return raw, fs, scale


def get_window(raw: np.memmap, start: int, count: int, scale: float) -> np.ndarray:
    block = raw[start * 2 : (start + count) * 2]
    x = block[0::2].astype(np.float32) + 1j * block[1::2].astype(np.float32)
    return np.asarray(x / np.float32(scale), dtype=np.complex64)


def avg_psd(x: np.ndarray, nfft: int) -> np.ndarray:
    starts = np.arange(0, len(x) - nfft + 1, nfft, dtype=np.int64)
    win = np.hamming(nfft).astype(np.float32)
    win_power = float(np.sum(win**2))
    acc = np.zeros(nfft, dtype=np.float64)
    for start in starts:
        z = np.fft.fftshift(np.fft.fft(x[start : start + nfft] * win, n=nfft))
        acc += np.abs(z) ** 2 / win_power
    return acc / max(len(starts), 1)


def band_mask(freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return (freqs >= low_hz) & (freqs < high_hz)


def choose_noise_band(psd: np.ndarray, freqs: np.ndarray, sig_low: float, sig_high: float, fs: float):
    n = len(psd)
    noise_bins = max(4, int(round(n * NOISE_BW_HZ / fs)))
    edge_low = -fs / 2 + EDGE_GUARD_HZ
    edge_high = fs / 2 - EDGE_GUARD_HZ
    candidates = []
    for start in range(0, n - noise_bins + 1):
        low = freqs[start]
        high = freqs[start + noise_bins - 1]
        if low < edge_low or high > edge_high:
            continue
        if high > sig_low - NOISE_GUARD_HZ and low < sig_high + NOISE_GUARD_HZ:
            continue
        candidates.append((float(np.mean(psd[start : start + noise_bins])), start, start + noise_bins))
    if not candidates:
        return None, math.nan, math.nan
    _, start, stop = min(candidates, key=lambda item: item[0])
    mask = np.zeros(n, dtype=bool)
    mask[start:stop] = True
    return mask, float(freqs[start]), float(freqs[stop - 1])


def estimate_snr(psd: np.ndarray, signal: np.ndarray, noise: np.ndarray) -> float:
    ps = float(np.mean(psd[signal]))
    pn = float(np.mean(psd[noise]))
    signal_power = ps - pn
    if signal_power <= 0 or pn <= 0:
        return float("-inf")
    return 10 * math.log10(signal_power / pn)


def color_for_value(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if not math.isfinite(value):
        return (40, 40, 40)
    t = (value - vmin) / max(vmax - vmin, 1e-9)
    t = max(0.0, min(1.0, t))
    # Blue -> cyan -> yellow -> red
    if t < 0.33:
        u = t / 0.33
        return (0, int(180 * u), 255)
    if t < 0.66:
        u = (t - 0.33) / 0.33
        return (int(255 * u), 180 + int(75 * u), int(255 * (1 - u)))
    u = (t - 0.66) / 0.34
    return (255, int(255 * (1 - u)), 0)


def draw_heatmap(rows: list[dict[str, float | str]], window_count: int, out_path: Path) -> None:
    cell_w, cell_h = 56, 34
    left, top = 110, 40
    width = left + window_count * cell_w + 160
    height = top + len(CANDIDATE_BANDS_MHZ) * cell_h + 70

    values = [float(r["estimated_snr_db"]) for r in rows if math.isfinite(float(r["estimated_snr_db"]))]
    vmin = min(values)
    vmax = max(values)

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((left, 12), "DAUTEL 20 MHz candidate-band SNR over time windows", fill=(0, 0, 0))

    row_lookup = {
        (int(r["band_index"]), int(r["window_index"])): float(r["estimated_snr_db"])
        for r in rows
    }
    for wi in range(window_count):
        draw.text((left + wi * cell_w + 4, top - 22), f"{wi}", fill=(0, 0, 0))

    for bi, (low, high) in enumerate(CANDIDATE_BANDS_MHZ):
        y = top + bi * cell_h
        draw.text((8, y + 9), f"{low}..{high} MHz", fill=(0, 0, 0))
        for wi in range(window_count):
            value = row_lookup[(bi, wi)]
            x = left + wi * cell_w
            draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], fill=color_for_value(value, vmin, vmax), outline=(255, 255, 255))
            draw.text((x + 5, y + 10), f"{value:.1f}", fill=(0, 0, 0))

    legend_x = left + window_count * cell_w + 30
    draw.text((legend_x, top), f"min {vmin:.1f} dB", fill=(0, 0, 0))
    draw.text((legend_x, top + 22), f"max {vmax:.1f} dB", fill=(0, 0, 0))
    draw.text((left, height - 32), "Columns are 100 ms windows with 50 ms step across pack1_0-1s.iq.", fill=(0, 0, 0))
    img.save(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = read_meta()
    raw, fs, scale = read_iq(meta)
    window_samples = int(round(fs * WINDOW_SECONDS))
    step_samples = int(round(fs * STEP_SECONDS))
    max_start = (len(raw) // 2) - window_samples
    starts = list(range(0, max_start + 1, step_samples))
    freqs = np.linspace(-fs / 2, fs / 2, PSD_NFFT, endpoint=False)

    rows: list[dict[str, float | str]] = []
    for wi, start in enumerate(starts):
        x = get_window(raw, start, window_samples, scale)
        psd = avg_psd(x, PSD_NFFT)
        for bi, (low_mhz, high_mhz) in enumerate(CANDIDATE_BANDS_MHZ):
            sig_low = low_mhz * 1e6
            sig_high = high_mhz * 1e6
            signal = band_mask(freqs, sig_low, sig_high)
            noise, noise_low, noise_high = choose_noise_band(psd, freqs, sig_low, sig_high, fs)
            snr = estimate_snr(psd, signal, noise) if noise is not None else float("nan")
            rows.append(
                {
                    "window_index": wi,
                    "window_start_s": start / fs,
                    "window_end_s": (start + window_samples) / fs,
                    "band_index": bi,
                    "candidate_low_mhz": low_mhz,
                    "candidate_high_mhz": high_mhz,
                    "estimated_snr_db": snr,
                    "noise_low_mhz": noise_low / 1e6,
                    "noise_high_mhz": noise_high / 1e6,
                }
            )

    csv_path = OUT_DIR / "temporal_band_snr.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    draw_heatmap(rows, len(starts), OUT_DIR / "temporal_band_snr_heatmap.png")

    print(f"windows: {len(starts)}")
    print(f"csv: {csv_path}")
    print(f"heatmap: {OUT_DIR / 'temporal_band_snr_heatmap.png'}")
    for bi, (low, high) in enumerate(CANDIDATE_BANDS_MHZ):
        vals = [float(r["estimated_snr_db"]) for r in rows if int(r["band_index"]) == bi]
        finite = [v for v in vals if math.isfinite(v)]
        print(
            f"{low:>4}..{high:<4} MHz "
            f"mean={np.mean(finite):6.2f} dB std={np.std(finite):5.2f} "
            f"min={np.min(finite):6.2f} max={np.max(finite):6.2f}"
        )


if __name__ == "__main__":
    main()
