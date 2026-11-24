#!/usr/bin/env python3
"""Read-only causal probe of the frozen Stage2 model: does it key on WHERE the video block sits?

Edits the normalised 256x256 spectrogram (rows = frequency, 2390-2490 MHz) of
clean windows and re-runs the frozen seed-2027 epoch-4 Stage2 checkpoint:

  none           unmodified (must reproduce the frozen predictions)
  move_down      copy rows 2440-2470 onto 2400-2430 (elementwise max), fill source with floor
                 -> puts an S0101-style video block where S0110 late windows have it
  move_down_ctl  same, but only 20 MHz down (2440-2470 -> 2420-2450), stays in training range
  move_up        copy rows 2398-2423 onto 2445-2470 (max), fill source with floor
                 -> the reverse: lifts the low block of S0110 late windows back up
  mask_low       fill 2400-2425 (lab-WiFi-like region) with floor
  mask_ctl       fill 2455-2480 with floor (control occlusion of equal width)

"floor" = per-image median of the normalised spectrogram. Nothing is trained or
selected; the checkpoint, index and split are only read.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

from dronerfa_data import read_vector_slice, rms_normalize
from model import ResNet18, SpectrogramFrontEnd

HERE = Path(__file__).resolve().parent
CKPT = HERE / "outputs/stage2_12drones_20260927/seed_2027/best_model.pt"
VAL_SCORES = HERE / "outputs/stage2_12drones_20260927/seed_2027/validation_scores_epoch_04.csv"
DEV_PRED = HERE / "outputs/cascade12_development_20260927/window_predictions.csv"
CODES = ("T0001", "T0010", "T0011", "T0100", "T0101", "T0110", "T0111",
         "T1000", "T1001", "T1010", "T1011", "T1100")
WINDOW = 1_000_000
F_LO, F_HI, ROWS = 2390.0, 2490.0, 256


def rows_of(f0: float, f1: float) -> slice:
    r = lambda f: int(round((f - F_LO) / (F_HI - F_LO) * ROWS))
    return slice(r(f0), r(f1))


def move(img: torch.Tensor, src: tuple, dst: tuple) -> torch.Tensor:
    out = img.clone()
    floor = img.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    s, d = rows_of(*src), rows_of(*dst)
    n = min(s.stop - s.start, d.stop - d.start)
    block = img[:, :, s.start:s.start + n].clone()
    out[:, :, s] = floor
    out[:, :, d.start:d.start + n] = torch.maximum(out[:, :, d.start:d.start + n], block)
    return out


def mask(img: torch.Tensor, band: tuple) -> torch.Tensor:
    out = img.clone()
    out[:, :, rows_of(*band)] = img.flatten(1).median(dim=1).values.view(-1, 1, 1, 1)
    return out


EDITS = {
    "none": lambda x: x,
    "move_down": lambda x: move(x, (2440, 2470), (2400, 2430)),
    "move_down_ctl": lambda x: move(x, (2440, 2470), (2420, 2450)),
    "move_up": lambda x: move(x, (2398, 2423), (2445, 2470)),
    "mask_low": lambda x: mask(x, (2400, 2425)),
    "mask_ctl": lambda x: mask(x, (2455, 2480)),
}


def frozen_predictions() -> dict:
    out = {}
    with VAL_SCORES.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] == "clean":
                out[(r["filename"], int(r["row_index"]))] = CODES[int(r["prediction"])]
    with DEV_PRED.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["condition"] == "clean" and r["stage2_prediction"]:
                out[(r["filename"], int(r["window_index"]), "dev")] = CODES[int(r["stage2_prediction"])]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--targets", default="T1010_S0101.mat,T1011_S0101.mat,T1100_S0101.mat,T1011_S0110.mat")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    frontend = SpectrogramFrontEnd(n_fft=ck["n_fft"], hop_length=ck["hop_length"], image_size=ck["image_size"]).to(device)
    frontend.load_state_dict(ck["frontend_state"])
    model = ResNet18(input_channels=1, num_classes=12).to(device)
    model.load_state_dict(ck["model_state"])
    frontend.eval(); model.eval()

    index = HERE / "outputs/stage2_12drones_20260927/contract/stage2_index.csv"
    with index.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["filename"] in args.targets.split(",")]
    frozen = frozen_predictions()
    records = []
    for fn in args.targets.split(","):
        fr = sorted((r for r in rows if r["filename"] == fn), key=lambda r: int(r["window_index"]))
        with h5py.File(args.data_root / fn, "r") as h:
            for r in fr:
                s, n = int(r["start"]), int(r["length"])
                iq = rms_normalize(read_vector_slice(h["RF0_I"], s, n) + 1j * read_vector_slice(h["RF0_Q"], s, n))
                x = torch.from_numpy(np.stack((iq.real, iq.imag)))[None].to(device)
                with torch.inference_mode():
                    img = frontend(x.float())
                    rec = {"filename": fn, "split": r["split"], "window_index": int(r["window_index"]),
                           "target_code": r["target_code"]}
                    for name, edit in EDITS.items():
                        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=torch.float16):
                            logits = model(edit(img))
                        rec[name] = CODES[int(logits.float().argmax(1))]
                key = (fn, int(r["row_index"])) if r["split"] == "val" else (fn, int(r["window_index"]), "dev")
                rec["frozen"] = frozen.get(key, "")
                records.append(rec)
        print(fn, len(fr), "windows", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "probe_windows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    summary = {}
    groups = defaultdict(list)
    for r in records:
        part = "" if r["filename"] != "T1011_S0110.mat" else ("_w0-69" if r["window_index"] < 70 else "_w70-159")
        groups[r["filename"] + part].append(r)
    for g, rs in groups.items():
        summary[g] = {"windows": len(rs),
                      "reproduces_frozen": sum(r["none"] == r["frozen"] for r in rs) / len(rs),
                      **{name: {"correct": round(sum(r[name] == r["target_code"] for r in rs) / len(rs), 3),
                                "top_predictions": Counter(r[name] for r in rs).most_common(3)}
                         for name in EDITS}}
    (args.output / "probe_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
