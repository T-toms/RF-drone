import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dronerfa_baseline'))
import stage1_loader_optimization as opt
import torch

BASE = ROOT / 'dronerfa_baseline/outputs/stage1_12drones_20260926'
IO = ROOT / 'dronerfa_baseline/outputs/stage1_io_optimization_20260927'
def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))
def rows(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

idx = rows(BASE/'window_index_stage1_12drones.csv')
run = BASE/'seed_2027'
audit = read(run/'read_audit.json')
expected = Counter()
for r in idx:
    if r['split'] in ('train','val'):
        expected[r['filename']] += 5 if r['split']=='train' else 4
observed = Counter({Path(k).name:v for k,v in audit['windows_by_file'].items()})
metrics = rows(run/'validation_metrics.csv')
errors = []
valkeys = {(r['filename'],int(r['start'])):int(r['label']) for r in idx if r['split']=='val'}
for ep in range(1,6):
    scores = rows(run/f'validation_scores_epoch_{ep:02d}.csv')
    for cond in ('clean','-10dB','-12dB','-15dB'):
        group=[r for r in scores if r['condition']==cond]
        keys=[(r['filename'],int(r['start'])) for r in group]
        if len(keys)!=len(valkeys) or set(keys)!=set(valkeys): errors.append([ep,cond,'coverage'])
        if any(int(r['label'])!=valkeys.get((r['filename'],int(r['start']))) for r in group): errors.append([ep,cond,'labels'])
        counts=Counter((int(r['label']),int(r['prediction'])) for r in group)
        ba=(counts[0,0]/280+counts[1,1]/2080)/2
        recorded=next(r for r in metrics if int(r['epoch'])==ep and r['condition']==cond)
        if abs(ba-float(recorded['balanced_accuracy']))>1e-12: errors.append([ep,cond,'metric'])

audit_repro=opt.LoaderAudit('codex-noncontiguous')
audit_repro.tallies['worker0']=opt.WorkerEventTally(windows={'x.mat':2}, window_positions={'x.mat':{10,20}}, epochs_seen={1})
consumed=opt.ConsumptionLog(split='train', epoch=1,positions=[10,20],batches=1,batches_delivered=1)
audit_repro.record_epoch(1,'train',consumed,sampler_positions=[10,20],expected_positions=2)
prefetch=audit_repro.prefetch_report('train')

smokes=[]
for name in ('smoke_serial_refactor','smoke_parallel_workers4'):
    config=read(IO/name/'run_config.json')
    ckpt=torch.load(IO/name/'best_model.pt',map_location='cpu',weights_only=False)
    smokes.append({'name':name,'initial_model_sha256':config.get('initial_model_sha256'),
       'epoch_order_sha256':config.get('epoch_order_sha256'),'train_loss':read(IO/name/'summary.json')['history'][0]['train_loss'],
       'model_state':ckpt['model_state']})
diffs={k:float((smokes[0]['model_state'][k].to(torch.float64)-smokes[1]['model_state'][k].to(torch.float64)).abs().max()) for k in smokes[0]['model_state']}
for s in smokes: del s['model_state']
result={'seed2027':{'audit_exact_budget':expected==observed,'opened_files':audit['opened_file_count'],
    'windows_read':audit['windows_read'],'metric_or_coverage_errors':errors,
    'best_model_sha256':sha(run/'best_model.pt'),'selected_epoch':read(run/'summary.json')['selected_epoch'],
    'selected_score':read(run/'summary.json')['selected_epoch_score']},
    'optimizer_audit_repro':{'expected_unconsumed':0,'actual':prefetch},
    'smoke_comparison':{'runs':smokes,'different_state_tensors':sum(v>0 for v in diffs.values()),'max_tensor_delta':max(diffs.values())},
    'benchmark_large':read(IO/'benchmark_large.json'),
    'benchmark_csv':rows(IO/'benchmark_summary.csv')}
out=ROOT/'work/stage1-codex-review-20260927.json'
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ('benchmark_large','benchmark_csv')},ensure_ascii=False,indent=2))
