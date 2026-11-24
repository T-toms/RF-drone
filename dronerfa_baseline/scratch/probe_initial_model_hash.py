"""Probe: is the stage-1 initial model hash a function of the seed alone?

Answers whether the reworked driver changed model initialisation, or whether the
historical smoke simply ran a different torch/model revision.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parent.parent
if str(BASELINE) not in sys.path:
    sys.path.insert(0, str(BASELINE))

import torch

import run_stage1_12drones_training as drv
from model import ResNet18
from train_resnet18 import seed_everything


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hashes: dict[str, str] = {}
    for seed in (2026, 2027):
        seed_everything(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = ResNet18(input_channels=1, num_classes=2).to(device)
        hashes[str(seed)] = drv.state_hash(model.state_dict())
        del model
    payload = {
        "torch": torch.__version__,
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "initial_model_sha256_by_seed": hashes,
        "model_py_sha256": hashlib.sha256((BASELINE / "model.py").read_bytes()).hexdigest(),
        "driver_sha256": hashlib.sha256(
            (BASELINE / "run_stage1_12drones_training.py").read_bytes()
        ).hexdigest(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
