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
OUT_DIR = ROOT / "figure" / "DAUTEL_band_scan"

SEGMENT_SECONDS = 0.1
PSD_NFFT = 8192
SPEC_NFFT = 1024
IMAGE_SIZE = 256
SIGNAL_BW_HZ = 20e6
NOISE_BW_HZ = 1e6
NOISE_GUARD_HZ = 2e6
EDGE_GUARD_HZ = 5e6


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
    return np.clip(((1 - w)[:, None] * PARULA_ANCHORS[lo] + w[:, None] * PARULA_ANCHORS[hi]) * 255, 0, 255).astype(np.uint8)


CMAP = make_cmap()


def read_meta() -> dict[str, str]:
    root = ET.parse(DATA_DIR / "pack1.xml").getroot()
    return {child.tag: (child.text or "").strip() for child in root}


def read_segment(meta: dict[str, str]) -> tuple[np.ndarray, float]:
    fs = float(meta["SampleRate"])
    scale = np.float32(float(meta["ScaleFactor"]))
    count = int(fs * SEGMENT_SECONDS) * 2
    raw = np.memmap(DATA_DIR / IQ_FILE, dtype=np.float32, mode="r", shape=(count,))
    x = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
    return np.asarray(x / scale, dtype=np.complex64), fs


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


def choose_noise_band(psd: np.ndarray, freqs: np.ndarray, sig_low: float, sig_high: float, fs: float) -> tuple[np.ndarray, float, float]:
    n = len(psd)
    noise_bins = max(4, int(round(n * NOISE_BW_HZ / fs)))
    edge_low = -fs / 2 + EDGE_GUARD_HZ
    edge_high = fs / 2 - EDGE_GUARD_HZ
    candidates: list[tuple[float, int, int]] = []

    for start in range(0, n - noise_bins + 1):
        low = freqs[start]
        high = freqs[start + noise_bins - 1]
        if low < edge_low or high > edge_high:
            continue
        if high > sig_low - NOISE_GUARD_HZ and low < sig_high + NOISE_GUARD_HZ:
            continue
        candidates.append((float(np.mean(psd[start : start + noise_bins])), start, start + noise_bins))

    if not candidates:
        raise RuntimeError("No valid noise candidate")
    _, start, stop = min(candidates, key=lambda item: item[0])
    mask = np.zeros(n, dtype=bool)
    mask[start:stop] = True
    return mask, float(freqs[start]), float(freqs[stop - 1])


def estimate_snr(psd: np.ndarray, signal: np.ndarray, noise: np.ndarray) -> float:
    ps = float(np.mean(psd[signal]))
    pn = float(np.mean(psd[noise]))
    return 10 * math.log10(max(ps - pn, np.finfo(float).tiny) / max(pn, np.finfo(float).tiny))


def spec_db(x: np.ndarray) -> np.ndarray:
    starts = np.linspace(0, len(x) - SPEC_NFFT, IMAGE_SIZE).astype(np.int64)
    rows = np.linspace(0, SPEC_NFFT - 1, IMAGE_SIZE).astype(np.int64)
    win = np.hamming(SPEC_NFFT).astype(np.float32)
    out = np.empty((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for col, start in enumerate(starts):
        z = np.fft.fftshift(np.fft.fft(x[start : start + SPEC_NFFT] * win, n=SPEC_NFFT))
        out[:, col] = 10 * np.log10(np.abs(z[rows]) ** 2 + np.finfo(np.float32).eps)
    return out


def colorize(spec: np.ndarray, ref_db: float, dyn_db: float = 80.0) -> Image.Image:
    norm = np.clip((spec - (ref_db - dyn_db)) / dyn_db, 0, 1)
    return Image.fromarray(CMAP[np.round(norm * 255).astype(np.uint8)], "RGB")


def make_psd_plot(psd: np.ndarray, freqs: np.ndarray, signal: np.ndarray, noise: np.ndarray, title: str, out: Path) -> None:
    w, h = 560, 260
    ml, mr, mt, mb = 48, 12, 30, 36
    pw, ph = w - ml - mr, h - mt - mb
    psd_db = 10 * np.log10(psd + np.finfo(float).tiny)
    lo, hi = float(np.percentile(psd_db, 1)), float(np.percentile(psd_db, 99.5))
    if hi <= lo:
        hi = lo + 1

    def xp(i: int) -> int:
        return ml + int(round(i * pw / (len(psd_db) - 1)))

    def yp(v: float) -> int:
        return mt + ph - int(round(np.clip((v - lo) / (hi - lo), 0, 1) * ph))

    img = Image.new("RGB", (w, h), "white")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for mask, color in [(signal, (255, 215, 0, 80)), (noise, (0, 120, 255, 75))]:
        idx = np.flatnonzero(mask)
        if len(idx):
            od.rectangle([xp(int(idx[0])), mt, xp(int(idx[-1])), mt + ph], fill=color)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([ml, mt, ml + pw, mt + ph], outline=(0, 0, 0))
    step = max(1, len(psd_db) // 450)
    pts = [(xp(i), yp(float(psd_db[i]))) for i in range(0, len(psd_db), step)]
    d.line(pts, fill=(0, 0, 0), width=1)
    d.text((ml, 8), title, fill=(0, 0, 0))
    d.text((ml + 180, h - 20), "MHz", fill=(0, 0, 0))
    for mhz in [-50, -25, 0, 25, 50]:
        idx = int(round((mhz * 1e6 - freqs[0]) / (freqs[-1] - freqs[0]) * (len(freqs) - 1)))
        idx = min(max(idx, 0), len(freqs) - 1)
        x = xp(idx)
        d.line([x, mt + ph, x, mt + ph + 4], fill=(0, 0, 0))
        d.text((x - 12, mt + ph + 7), str(mhz), fill=(0, 0, 0))
    img.save(out)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = read_meta()
    x, fs = read_segment(meta)
    freqs = np.linspace(-fs / 2, fs / 2, PSD_NFFT, endpoint=False)
    psd = avg_psd(x, PSD_NFFT)
    spec = spec_db(x)
    ref_db = float(np.max(spec))

    candidates_mhz = [(-50, -30), (-40, -20), (-30, -10), (-20, 0), (-10, 10), (0, 20), (10, 30), (20, 40), (30, 50)]
    rows = []
    panels = []

    for low_mhz, high_mhz in candidates_mhz:
        sig_low = low_mhz * 1e6
        sig_high = high_mhz * 1e6
        signal = band_mask(freqs, sig_low, sig_high)
        noise, noise_low, noise_high = choose_noise_band(psd, freqs, sig_low, sig_high, fs)
        snr = estimate_snr(psd, signal, noise)
        title = f"{low_mhz}..{high_mhz} MHz | SNR {snr:.2f} dB"

        psd_path = OUT_DIR / f"psd_{low_mhz}_{high_mhz}.png".replace("-", "m")
        make_psd_plot(psd, freqs, signal, noise, title, psd_path)

        spec_img = colorize(spec, ref_db)
        draw = ImageDraw.Draw(spec_img)
        y1 = int(round((sig_low + fs / 2) / fs * IMAGE_SIZE))
        y2 = int(round((sig_high + fs / 2) / fs * IMAGE_SIZE))
        # Image y-axis is top-to-bottom; frequency bins are low-to-high.
        top = IMAGE_SIZE - max(y1, y2)
        bottom = IMAGE_SIZE - min(y1, y2)
        draw.rectangle([0, top, IMAGE_SIZE - 1, bottom], outline=(255, 255, 255), width=2)
        spec_path = OUT_DIR / f"spec_{low_mhz}_{high_mhz}.png".replace("-", "m")
        spec_img.save(spec_path)

        rows.append(
            {
                "candidate_low_mhz": low_mhz,
                "candidate_high_mhz": high_mhz,
                "estimated_snr_db": snr,
                "noise_low_mhz": noise_low / 1e6,
                "noise_high_mhz": noise_high / 1e6,
                "psd_file": psd_path.name,
                "spec_file": spec_path.name,
            }
        )
        panels.append((title, psd_path, spec_path))

    with (OUT_DIR / "band_scan_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    cell_w, cell_h = 560, 260 + 256 + 40
    sheet = Image.new("RGB", (cell_w * 3, cell_h * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (title, psd_path, spec_path) in enumerate(panels):
        col = i % 3
        row = i // 3
        x0 = col * cell_w
        y0 = row * cell_h
        draw.text((x0 + 8, y0 + 6), title, fill=(0, 0, 0))
        sheet.paste(Image.open(psd_path).convert("RGB"), (x0, y0 + 24))
        sheet.paste(Image.open(spec_path).convert("RGB"), (x0 + 152, y0 + 24 + 260))
    sheet.save(OUT_DIR / "DAUTEL_20MHz_band_scan_contact.png")

    print(f"output: {OUT_DIR}")
    for row in rows:
        print(
            f"{row['candidate_low_mhz']:>4}..{row['candidate_high_mhz']:<4} MHz "
            f"SNR={row['estimated_snr_db']:7.2f} dB "
            f"noise={row['noise_low_mhz']:.2f}..{row['noise_high_mhz']:.2f} MHz"
        )


if __name__ == "__main__":
    main()
