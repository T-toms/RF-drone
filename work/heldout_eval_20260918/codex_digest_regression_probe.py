"""Check that a multi-window cache emits every digest, not one aggregate digest."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'dronerfa_baseline'))
import evaluate_heldout_models as evaluator

class SyntheticDataset:
    def __init__(self, *args, **kwargs):
        self.epoch = 0
    def __len__(self):
        return 5
    def __getitem__(self, position):
        return torch.full((2,8),float(position)), position % 2
    def close(self):
        pass

class SyntheticFrontend:
    def __call__(self, batch):
        return batch[:,0,0,None,None,None].expand(-1,1,256,256).contiguous()

evaluator.DroneRFaWindowDataset = SyntheticDataset
features, cache = evaluator.compute_condition_features(Path('unused.csv'), 'clean', 22026, 4, 5, torch.device('cpu'), SyntheticFrontend())
raw = cache['window_digests']
checks = {
    'five_windows_emit_160_digest_bytes': len(raw)==5*32,
    'each_window_hash_matches_feature': all(raw[i*32:(i+1)*32] == hashlib.sha256(np.full((1,256,256),i,dtype=np.float32).tobytes()).digest() for i in range(5)),
    'aggregate_digest_matches_concatenated_digests': hashlib.sha256(raw).hexdigest()==cache['metadata']['window_digests_concat_sha256'],
    'full_feature_hash_matches_cache': hashlib.sha256(features.tobytes()).hexdigest()==cache['metadata']['feature_sha256'],
    'last_partial_batch_is_recorded': len(features)==5 and cache['labels'].tolist()==[0,1,0,1,0],
}
result={'passed':all(checks.values()),'checks':checks}
(ROOT/'work/heldout_eval_20260918/codex-digest-regression-probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
raise SystemExit(0 if result['passed'] else 1)
