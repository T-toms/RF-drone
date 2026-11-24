"""Exercise the observer on real data and export longer review examples."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from analysis import WORKSPACE,AUDIT,inspect_mat,make_plan,analyze,save_result
from plots import draw_result

out=WORKSPACE/'output/signal_observer';out.mkdir(parents=True,exist_ok=True)
rows=list(csv.DictReader((AUDIT/'data_preprocessing_manifest.csv').open(encoding='utf-8-sig')))
r=next(r for r in rows if r['folder']=='009_2.4GHZ_10M-signal' and r['epoch']=='0' and r['seg']=='0' and r['duplicate']=='False')
path=Path('G:/drone')/r['path'];info=inspect_mat(path)
single=analyze(make_plan(info,'data'),nfft=4096)
assert single['stop']-single['start']==6144000 and single['frames']*4096==6144000
save_result(single,out/'demo_single_mat.npz')
fig=Figure(figsize=(14,10),dpi=110);draw_result(fig,single,122880000,None,'assumed');fig.savefig(out/'demo_single_mat_assumed_12288.png',dpi=130)
print('Single MAT verified: 6,144,000 samples; 50 ms only if Fs=122.88 MS/s.',flush=True)
long=analyze(make_plan(info,'data',recording=True),nfft=4096,progress=lambda a,b,m:print(f'{a}/{b} {m}',flush=True) if a%5==0 else None)
assert long['stop']-long['start']==245760000 and long['frames']*4096==245760000 and len(long['sources'])==40
old=np.load(WORKSPACE/'work/long_signal_pilot_20261024/009_2.4GHZ_10M-signal__2026-09-20-18-04-10_statistics.npz')['mean_psd']
error=float(np.max(np.abs(long['spectrum']-old)/np.maximum(old,1e-20)));assert error<1e-4
save_result(long,out/'demo_full_recording.npz')
for fs,name in [(None,'demo_full_recording_samples.png'),(122880000,'demo_full_recording_assumed_12288.png')]:
    fig=Figure(figsize=(15,11),dpi=110);draw_result(fig,long,fs,None,'assumed' if fs else 'unknown');fig.savefig(out/name,dpi=130)
report=dict(passed=True,single_file_samples=6144000,recording_samples=245760000,recording_files=40,comparison_with_previous_full_recording_psd_relative_error=error,
    displayed_durations_under_explicit_12288_assumption=dict(previous_zoom_ms=262144/122880000*1000,single_mat_ms=50,full_recording_s=2),
    source_fs_metadata=info.fs_hz,source_fc_metadata=info.fc_hz,default_export_clock='unknown')
(out/'real_data_verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(report,ensure_ascii=False,indent=2))
