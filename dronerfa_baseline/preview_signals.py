from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("WINDIR", r"C:\Windows")
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_DIR / "outputs" / ".matplotlib")
)

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dronerfa_data import (
    TARGET_NAMES,
    add_complex_awgn,
    load_index,
    read_vector_slice,
    rms_normalize,
)


SAMPLE_RATE = 100_000_000.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PSD, spectrogram, and low-SNR previews."
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "window_index.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "figures",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def read_iq(row: dict[str, object]) -> np.ndarray:
    with h5py.File(str(row["file_path"]), "r") as handle:
        start = int(row["start"])
        length = int(row["length"])
        i_values = read_vector_slice(handle["RF0_I"], start, length)
        q_values = read_vector_slice(handle["RF0_Q"], start, length)
    return rms_normalize(i_values + 1j * q_values)


def power_spectrum(iq: np.ndarray, n_fft: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(iq) // n_fft
    frames = iq[: frame_count * n_fft].reshape(frame_count, n_fft)
    window = np.hanning(n_fft).astype(np.float32)
    spectra = np.fft.fftshift(np.fft.fft(frames * window, axis=1), axes=1)
    power = np.mean(np.abs(spectra) ** 2, axis=0)
    power_db = 10.0 * np.log10(power + 1e-12)
    power_db -= np.max(power_db)
    frequency_mhz = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / SAMPLE_RATE)) / 1e6
    return frequency_mhz, power_db


def spectrogram(
    iq: np.ndarray, n_fft: int = 2048, hop: int = 1024
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_view = np.lib.stride_tricks.sliding_window_view(iq, n_fft)[::hop]
    window = np.hanning(n_fft).astype(np.float32)
    spectra = np.fft.fftshift(np.fft.fft(frame_view * window, axis=1), axes=1)
    power_db = 10.0 * np.log10(np.abs(spectra) ** 2 + 1e-12)
    power_db = power_db.T
    frequency_mhz = np.fft.fftshift(np.fft.fftfreq(n_fft, 1.0 / SAMPLE_RATE)) / 1e6
    time_ms = (np.arange(power_db.shape[1]) * hop + n_fft / 2) / SAMPLE_RATE * 1e3
    return time_ms, frequency_mhz, power_db


def representative_rows(index_path: Path) -> list[dict[str, object]]:
    rows = load_index(index_path, split="train")
    selected: list[dict[str, object]] = []
    for target in TARGET_NAMES:
        candidates = [
            row
            for row in rows
            if row["target_code"] == target
            and int(row["block_index"]) == 0
            and int(row["window_index"]) == 0
        ]
        if not candidates:
            raise RuntimeError(f"No representative row found for target T{target}")
        selected.append(candidates[0])
    return selected


def plot_overview(rows: list[dict[str, object]], output_path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 10), constrained_layout=True)
    for row_index, row in enumerate(rows):
        iq = read_iq(row)
        model_name = str(row["target_name"])

        time_us = np.arange(100_000) / SAMPLE_RATE * 1e6
        axes[row_index, 0].plot(time_us, np.abs(iq[:100_000]), linewidth=0.45)
        axes[row_index, 0].set_title(f"{model_name}: envelope")
        axes[row_index, 0].set_xlabel("Time (us)")
        axes[row_index, 0].set_ylabel("|IQ|")
        axes[row_index, 0].grid(alpha=0.2)

        frequency_mhz, psd_db = power_spectrum(iq)
        axes[row_index, 1].plot(frequency_mhz, psd_db, linewidth=0.7)
        axes[row_index, 1].set_title(f"{model_name}: normalized PSD")
        axes[row_index, 1].set_xlabel("Offset from 2.440 GHz (MHz)")
        axes[row_index, 1].set_ylabel("Relative power (dB)")
        axes[row_index, 1].set_ylim(-70, 3)
        axes[row_index, 1].grid(alpha=0.2)

        time_ms, spec_freq_mhz, spec_db = spectrogram(iq)
        vmax = float(np.percentile(spec_db, 99.7))
        image = axes[row_index, 2].imshow(
            spec_db,
            origin="lower",
            aspect="auto",
            extent=[time_ms[0], time_ms[-1], spec_freq_mhz[0], spec_freq_mhz[-1]],
            cmap="viridis",
            vmin=vmax - 55,
            vmax=vmax,
        )
        axes[row_index, 2].set_title(f"{model_name}: STFT")
        axes[row_index, 2].set_xlabel("Time (ms)")
        axes[row_index, 2].set_ylabel("Offset from 2.440 GHz (MHz)")
        fig.colorbar(image, ax=axes[row_index, 2], label="Power (dB)")

    fig.suptitle(
        "DroneRFa RF0 examples (1,000,000 IQ samples = 10 ms)", fontsize=14
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_low_snr(
    drone_row: dict[str, object], output_path: Path, seed: int
) -> None:
    clean = read_iq(drone_row)
    rng = np.random.default_rng(seed)
    signals = [
        ("Clean", clean),
        ("Added SNR 0 dB", add_complex_awgn(clean, 0.0, rng)),
        ("Added SNR -10 dB", add_complex_awgn(clean, -10.0, rng)),
        ("Added SNR -15 dB", add_complex_awgn(clean, -15.0, rng)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    for axis, (title, iq) in zip(axes.flat, signals):
        time_ms, frequency_mhz, spec_db = spectrogram(iq)
        vmax = float(np.percentile(spec_db, 99.7))
        image = axis.imshow(
            spec_db,
            origin="lower",
            aspect="auto",
            extent=[time_ms[0], time_ms[-1], frequency_mhz[0], frequency_mhz[-1]],
            cmap="viridis",
            vmin=vmax - 55,
            vmax=vmax,
        )
        axis.set_title(title)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Offset from 2.440 GHz (MHz)")
        fig.colorbar(image, ax=axis, label="Power (dB)")

    fig.suptitle(
        f"{drone_row['target_name']}: reproducible complex-AWGN stress test",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = representative_rows(args.index)
    overview_path = args.output_dir / "signal_overview.png"
    low_snr_path = args.output_dir / "low_snr_stft.png"
    plot_overview(rows, overview_path)
    plot_low_snr(rows[1], low_snr_path, args.seed)
    print(f"Overview: {overview_path.resolve()}")
    print(f"Low-SNR preview: {low_snr_path.resolve()}")


if __name__ == "__main__":
    main()
