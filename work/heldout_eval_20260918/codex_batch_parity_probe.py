"""Acceptance probe: replay original batch boundaries; diagnose spot-check drift."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dronerfa_baseline'))
from dronerfa_data import DroneRFaWindowDataset
from model import ResNet18, SpectrogramFrontEnd
from train_resnet18 import extract_features

config = json.loads((ROOT / 'work/heldout-eval-config-20260918.json').read_text(encoding='utf-8-sig'))
model_entry = next(m for m in config['models'] if m['training_seed'] == 2028 and m['arm'] == 'A_original')
index = list(csv.DictReader(Path(config['index']).open(encoding='utf-8-sig', newline='')))
groups = {}
for i, row in enumerate(index):
    groups.setdefault(row['filename'], []).append(i)
positions = sorted({p for g in groups.values() for p in (g[0], g[len(g)//2], g[-1])})
csv_path = Path(config['output_root']) / 'run1_evidence/window_predictions_run1.csv'
if not csv_path.exists():
    csv_path = Path(config['output_root']) / 'window_predictions.csv'
stored = {int(r['heldout_index']): r for r in csv.DictReader(csv_path.open(encoding='utf-8-sig', newline=''))
          if int(r['training_seed']) == 2028 and r['arm'] == 'A_original' and r['condition'] == 'clean'}
checkpoint = torch.load(model_entry['checkpoint'], map_location='cpu', weights_only=False)
frontend = SpectrogramFrontEnd().cuda().eval()
frontend.load_state_dict(checkpoint['frontend_state'])
network = ResNet18(input_channels=1, num_classes=2).cuda().eval()
network.load_state_dict(checkpoint['model_state'])
dataset = DroneRFaWindowDataset(Path(config['index']), split='heldout', apply_fixed_snr=False, seed=22026)
original_features = {}
original_logits = {}
with torch.inference_mode():
    for start in sorted({p // 4 * 4 for p in positions}):
        stop = min(start + 4, len(index))
        iq = torch.stack([dataset[i][0] for i in range(start, stop)]).cuda()
        f = extract_features(frontend, iq)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = network(f).float().cpu().numpy()
        for i in range(start, stop):
            if i in positions:
                original_features[i] = f[i-start].cpu().numpy()
                original_logits[i] = logits[i-start]
    selected_iq = torch.stack([dataset[p][0] for p in positions]).cuda()
    selected_features = extract_features(frontend, selected_iq).cpu().numpy()
    regrouped_logits = {}
    for offset in range(0, len(positions), 4):
        selected_batch = torch.from_numpy(np.stack([original_features[p] for p in positions[offset:offset+4]])).cuda()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = network(selected_batch).float().cpu().numpy()
        for j, p in enumerate(positions[offset:offset+4]):
            regrouped_logits[p] = logits[j]
dataset.close()
expected = {p: np.array([stored[p]['logit_background'], stored[p]['logit_drone']], dtype=np.float32) for p in positions}
result = {
    'training_seed': 2028, 'arm': 'A_original', 'condition': 'clean', 'fixed_windows': positions,
    'original_aligned_batch4_max_abs_logit_gap_vs_csv': float(max(np.max(np.abs(original_logits[p]-expected[p])) for p in positions)),
    'original_aligned_batch4_prediction_mismatches_vs_csv': [p for p in positions if int(original_logits[p][1] > original_logits[p][0]) != int(stored[p]['prediction'])],
    'batch36_feature_digest_mismatches_vs_original_batch4': [p for j,p in enumerate(positions) if hashlib.sha256(selected_features[j].tobytes()).digest() != hashlib.sha256(original_features[p].tobytes()).digest()],
    'batch36_max_abs_feature_gap': float(max(np.max(np.abs(selected_features[j]-original_features[p])) for j,p in enumerate(positions))),
    'regrouped_batch4_logits_max_abs_gap_vs_original_alignment': float(max(np.max(np.abs(regrouped_logits[p]-original_logits[p])) for p in positions)),
}
(ROOT / 'work/heldout_eval_20260918/codex-batch-parity-probe.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2), flush=True)
