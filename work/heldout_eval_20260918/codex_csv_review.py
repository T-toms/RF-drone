"""Codex acceptance review using CSV arithmetic, independently of evaluator helpers."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'dronerfa_baseline/outputs/heldout_evaluation_20260918'
CFG = json.loads((ROOT / 'work/heldout-eval-config-20260918.json').read_text(encoding='utf-8-sig'))

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def sha(path):
    d = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            d.update(block)
    return d.hexdigest()

checks = []
def check(name, ok, detail=None):
    checks.append({'name': name, 'passed': bool(ok), 'detail': detail})

index = read_csv(Path(CFG['index']))
prediction_path = OUT / 'window_predictions.csv'
if not prediction_path.exists():
    prediction_path = OUT / 'run1_evidence/window_predictions_run1.csv'
predictions = read_csv(prediction_path)
per_file = read_csv(OUT / 'per_file_metrics.csv')
metrics = read_csv(OUT / 'model_condition_metrics.csv')
pairs = read_csv(OUT / 'paired_comparison.csv')
models = {(str(m['training_seed']), m['arm']): str(m['epoch']) for m in CFG['models']}
expected_combinations = {(s, a, c) for s,a in models for c in CFG['conditions']}
check('heldout index hash and length', sha(Path(CFG['index'])) == CFG['index_sha256'] and len(index) == 1820)
check('product row counts', [len(predictions), len(per_file), len(metrics)] == [43680,288,24], [len(predictions),len(per_file),len(metrics)])
grouped = collections.defaultdict(list)
bad_identity = bad_values = 0
for r in predictions:
    key = (r['training_seed'], r['arm'], r['condition'])
    grouped[key].append(r)
    i = int(r['heldout_index'])
    if i < 0 or i >= len(index) or any(r[k] != index[i][k] for k in index[i]) or r['epoch'] != models.get(key[:2]):
        bad_identity += 1
    values = [float(r[k]) for k in ('logit_background','logit_drone','logit_margin','drone_score')]
    expected_score = 1/(1+math.exp(-values[2]))
    if not all(math.isfinite(v) for v in values) or int(r['prediction']) != int(values[2] > 0) or not math.isclose(values[2],values[1]-values[0],abs_tol=1e-6,rel_tol=1e-7) or not math.isclose(values[3], expected_score, abs_tol=1e-12):
        bad_values += 1
check('index fields labels frozen epoch', bad_identity == 0, bad_identity)
check('finite float32-derived logits margin threshold score', bad_values == 0, bad_values)
check('24 combinations full ordered coverage', set(grouped) == expected_combinations and all([int(r['heldout_index']) for r in rows] == list(range(1820)) for rows in grouped.values()))
files = {r['filename'] for r in index}
expected_file_keys = {(s,a,c,f) for s,a,c in expected_combinations for f in files}
file_map = {(r['training_seed'],r['arm'],r['condition'],r['filename']):r for r in per_file}
metric_map = {(r['training_seed'],r['arm'],r['condition']):r for r in metrics}
check('288 unique file keys and 24 unique metric keys', len(file_map) == len(per_file) == 288 and set(file_map) == expected_file_keys and len(metric_map) == len(metrics) == 24 and set(metric_map) == expected_combinations)
calculated = {}
file_errors = metric_errors = 0
max_metric_gap = 0.0
for key, rows in grouped.items():
    by_file = collections.defaultdict(list)
    for r in rows:
        by_file[r['filename']].append(r)
    rates = collections.defaultdict(list)
    totals = dict(tp=0,tn=0,fp=0,fn=0)
    category_predictions = collections.defaultdict(list)
    for name, fr in by_file.items():
        code = fr[0]['target_code']
        counts = dict(tp=0,tn=0,fp=0,fn=0)
        for r in fr:
            label, p = int(r['label']), int(r['prediction'])
            count_key = ('tn' if p == 0 else 'fp') if label == 0 else ('fn' if p == 0 else 'tp')
            counts[count_key] += 1
            category_predictions[code].append(p)
        for k,v in counts.items():
            totals[k] += v
        rate = sum(int(r['prediction']) for r in fr) / len(fr)
        rates[code].append(rate)
        actual = file_map.get((*key,name),{})
        if any(int(actual.get(k,-1)) != v for k,v in counts.items()) or int(actual.get('windows',-1)) != len(fr):
            file_errors += 1
        rate_column = 'background_false_positive_rate' if code == '0000' else 'drone_detection_rate'
        if not math.isclose(float(actual.get(rate_column,'nan')),rate,abs_tol=1e-12):
            file_errors += 1
    bg, phantom, air2s = (sum(rates[k])/4 for k in ('0000','0010','0101'))
    recall = totals['tp'] / (totals['tp']+totals['fn'])
    specificity = totals['tn'] / (totals['tn']+totals['fp'])
    expected = {
        'windows':1820, 'background_file_equal_fpr':bg, 'background_file_equal_specificity':1-bg,
        'phantom_file_equal_recall':phantom, 'air2s_file_equal_recall':air2s,
        'drone_file_equal_recall':(phantom+air2s)/2, 'file_equal_balanced_accuracy':((1-bg)+(phantom+air2s)/2)/2,
        **{'pooled_'+k:v for k,v in totals.items()}, 'pooled_recall':recall, 'pooled_specificity':specificity,
        'pooled_window_fpr':1-specificity, 'pooled_balanced_accuracy':(recall+specificity)/2,
        'pooled_phantom_window_recall':sum(category_predictions['0010'])/len(category_predictions['0010']),
        'pooled_air2s_window_recall':sum(category_predictions['0101'])/len(category_predictions['0101']),
    }
    calculated[key] = expected
    for k,v in expected.items():
        gap = abs(float(metric_map.get(key,{}).get(k,'nan'))-v)
        max_metric_gap = max(max_metric_gap,gap)
        if not math.isfinite(gap) or gap > 1e-12:
            metric_errors += 1
check('per-file CSV independently recounted', file_errors == 0, file_errors)
check('main file-equal and pooled metrics independently recomputed', metric_errors == 0, {'errors':metric_errors,'max_abs_gap':max_metric_gap})
pair_keys = [(r['training_seed'],r['condition']) for r in pairs]
expected_pair_keys = {(s,c) for s,a in models for c in CFG['conditions']+['four_condition_mean']}
check('15 unique complete paired keys', len(pair_keys)==len(set(pair_keys))==15 and set(pair_keys)==expected_pair_keys, {'rows':len(pair_keys),'unique':len(set(pair_keys))})
pair_errors = 0
for r in pairs:
    s,c = r['training_seed'],r['condition']
    cs = CFG['conditions'] if c == 'four_condition_mean' else [c]
    for prefix, field in (('file_equal','file_equal_balanced_accuracy'),('pooled','pooled_balanced_accuracy')):
        av,bv = (sum(calculated[(s,a,cc)][field] for cc in cs)/len(cs) for a in ('A_original','B_plus_minus15'))
        columns = [('A_'+prefix+'_balacc',av),('B_'+prefix+'_balacc',bv),('B_minus_A_'+prefix+'_balacc',bv-av)]
        if any(not math.isclose(float(r[k]),v,abs_tol=1e-12) for k,v in columns):
            pair_errors += 1
check('paired values independently recomputed', pair_errors==0, pair_errors)
preserved = json.loads((ROOT / 'work/codex-preserved-inputs-20260917.json').read_text(encoding='utf-8-sig'))
preserved_checks = [{'path':f['path'],'matches':sha(Path(f['path']))==f['sha256']} for f in preserved['files']]
check('14 original scripts index and selected weights preserved', len(preserved_checks)==14 and all(f['matches'] for f in preserved_checks),preserved_checks)
result = {'scope':'CSV identity completeness threshold and independently computed metrics; no complete MAT scan or GPU replay in this script',
          'passed':all(c['passed'] for c in checks), 'checks':checks,
          'prediction_sha256':sha(prediction_path), 'prediction_path':str(prediction_path)}
(ROOT / 'work/heldout_eval_20260918/codex-csv-review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'passed':result['passed'],'checks':[dict(c,detail=None) for c in checks]},ensure_ascii=False,indent=2))
