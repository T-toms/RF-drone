"""Run two predeclared training-seed replications with identical validation IQ."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_noise_ab import ROOT, run_experiment, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((args.reference_run / "experiment_manifest.json").read_text(encoding="utf-8"))
    if reference["status"] != "completed":
        raise ValueError("Reference experiment must be complete")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "reference_run": str(args.reference_run.resolve()), "reference_training_seed": 2026,
        "new_training_seeds": [2027, 2028], "validation_noise_seed": 12026,
        "epochs_per_arm": 5, "batch_size": 4,
        "selection": "mean balanced accuracy across clean,-10,-12,-15; earliest tie",
        "purpose": "paired training-seed replication, same reused validation recordings",
        "status": "running", "completed_seeds": [], "started_at": datetime.now().isoformat(),
        "driver_sha256": sha256_file(Path(__file__)),
    }
    if reference["arguments"]["seed"] != 2026 or reference["arguments"]["epochs"] != 5:
        raise ValueError("This replication protocol expects the seed-2026 five-epoch pilot")
    shutil.copy2(__file__, args.output_dir / Path(__file__).name)
    write_json(args.output_dir / "replication_manifest.json", manifest)
    shared_cache = None
    try:
        for seed in manifest["new_training_seeds"]:
            run_dir = args.output_dir / f"seed_{seed}"
            run_args = argparse.Namespace(
                index=Path(reference["arguments"]["index"]), output_dir=run_dir,
                epochs=5, batch_size=4, seed=seed, validation_seed=12026,
                reference_validation_run=args.reference_run,
                max_train_samples=None, max_val_samples=None,
            )
            print(f"Starting paired replication: training seed {seed}, validation seed 12026", flush=True)
            shared_cache = run_experiment(run_args, shared_cache)
            for script in ["summarize_noise_ab.py", "verify_noise_ab_inputs.py"]:
                subprocess.run([sys.executable, "-u", str(ROOT / script), str(run_dir)], check=True)
            manifest["completed_seeds"].append(seed)
            write_json(args.output_dir / "replication_manifest.json", manifest)
        manifest.update(status="completed", completed_at=datetime.now().isoformat())
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output_dir / "replication_manifest.json", manifest)
    print(f"All replications and checks completed: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
