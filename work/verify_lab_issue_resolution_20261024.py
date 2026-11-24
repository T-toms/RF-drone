"""Independent coverage/boundary checks for the prepared analysis indices."""
from pathlib import Path
import csv, json, hashlib, subprocess, sys
from drone_audit_support import replace_markdown_section

W=Path(__file__).parent;O=W/'audit_drone_20261023';checks={}
def read(name):return json.loads((O/name).read_text(encoding='utf8'))
def table(name):return list(csv.DictReader((O/name).open(encoding='utf-8-sig')))
def check(name,ok):
    checks[name]=bool(ok)
    if not ok:raise AssertionError(name)

expected=read('preprocessing_evidence_hashes.json')
check('evidence_unchanged',all(hashlib.sha256((O/p).read_bytes()).hexdigest()==h for p,h in expected.items()))
data=table('data_preprocessing_manifest.csv');raw=table('raw_preprocessing_manifest.csv')
check('source_coverage',len(data)==3933 and len(raw)==890 and len({x['path'] for x in data})==3933 and len({x['path'] for x in raw})==890)
dv={v['path']:v for v in map(json.loads,(O/'data_values.jsonl').read_text(encoding='utf8').splitlines())}
di={x['path']:x for x in data};eligible={p for p,r in di.items() if r['analysis_eligible']=='True'}
check('no_data_duplicates_or_zeros_admitted',all(r['duplicate']=='False' and r['all_zero']=='False' for r in data if r['path'] in eligible))
check('unconfirmed_parameters_unfilled',all(not r['sample_rate_hz'] and not r['exact_receiver_center_hz'] and r['training_split']=='unassigned' for r in data))
check('only_known_prefixes_trimmed',all(int(r['start_sample'])==(dv[r['path']]['first_nonzero'] if dv[r['path']]['zero_blocks_61440'] else 0) for r in data if r['path'] in eligible))
parts=[];coordinates=True
for run in read('data_usable_runs.json')['runs']:
    prev=run['start_sample'];count=0
    for f in run['parts']:
        r=di[f['path']];ordinal=int(r['epoch'])*20+int(r['seg']);n=f['end_sample_exclusive']-f['start_sample']
        coordinates &= f['path'] in eligible and 0<=f['start_sample']<f['end_sample_exclusive']==6144000
        coordinates &= f['start_sample']==int(r['start_sample']) and f['recording_sample_start']==ordinal*6144000+f['start_sample']==prev
        coordinates &= (r['folder'],r['session'])==(run['folder'],run['session'])
        prev+=n;count+=n;parts.append(f['path'])
    coordinates &= count==run['samples'] and prev==run['end_sample_exclusive']
check('all_usable_parts_covered_once',len(parts)==len(set(parts))==len(eligible) and set(parts)==eligible)
check('coordinates_contiguous_and_gaps_preserved',coordinates)
ri={r['path']:r for r in raw};rh={r['path']:r for r in map(json.loads,(O/'raw_full_hash.jsonl').read_text(encoding='utf8').splitlines())}
good=[r for r in raw if r['analysis_eligible']=='True']
check('raw_hash_zeros_and_duplicate_exclusion',all(rh[r['path']]['matches'] and rh[r['path']]['zero_iq_samples']*4<rh[r['path']]['bytes'] and r['duplicate']=='False' for r in good) and len({rh[r['path']]['actual'] for r in good})==len(good))
check('short_captures_not_extended',all((r['supports_10ms']=='True')==(r['analysis_eligible']=='True' and float(r['duration_ms'])>=10) and (r['supports_20ms']=='True')==(r['analysis_eligible']=='True' and float(r['duration_ms'])>=20) for r in raw))
meta={str((Path('G:/drone')/x['path']).with_suffix('.bin')):x['metadata'] for x in read('raw_json_complete.json')}
pairs=read('raw_usable_background_pairs.json')['pairs'];pair_ok=True
for x in pairs:
    m=meta[x['drone']];pair_ok &= ri[x['drone']]['analysis_eligible']=='True' and m['declared']['class']=='drone'
    for p in x['backgrounds']:
        b=meta[p]
        pair_ok &= ri[p]['analysis_eligible']=='True' and b['declared']['class']=='ambient' and b['session']['id']==m['session']['paired_bg']
        pair_ok &= all(b['capture'][k]==m['capture'][k] for k in ['sample_rate_hz','center_freq_hz','usable_bw_hz'])
        pair_ok &= all(b['receiver'][k]==m['receiver'][k] for k in ['gain_db','agc'])
check('paired_backgrounds_remain_valid',pair_ok and len({x['drone'] for x in pairs})==214)

# Directly exercise the report-loss failure with a later peer section.
text='# Report\n\n### 9.4 Old\n\nold\n\n### 9.5 Keep\n\nkeep\n\n## 10. Later\n\nuser notes\n'
updated=replace_markdown_section(text,'### 9.4 Old','### 9.4 Old\n\nnew')
check('later_sections_preserved',updated.endswith('## 10. Later\n\nuser notes\n') and '### 9.5 Keep\n\nkeep' in updated)
check('section_replacement_repeat_stable',replace_markdown_section(updated,'### 9.4 Old','### 9.4 Old\n\nnew')==updated)
report=W.parent/'reports/实验室数据复核与改进方案_20261023.md'
section10=report.read_text(encoding='utf8').split('## 10. 已知问题处理与复查',1)[1]
for i in range(2):
    subprocess.run([sys.executable,str(W/'finalize_continued_drone_audit_20261023.py')],check=True,capture_output=True)
    now=report.read_bytes()
    if i==0:first=now
    else:check('actual_report_rerun_stable',now==first)
check('actual_section10_preserved',report.read_text(encoding='utf8').split('## 10. 已知问题处理与复查',1)[1]==section10)
print(json.dumps({'passed':all(checks.values()),'checks':checks},ensure_ascii=False,indent=2))
(O/'issue_resolution_verification.json').write_text(json.dumps({'passed':all(checks.values()),'checks':checks},ensure_ascii=False,indent=2),encoding='utf8')
