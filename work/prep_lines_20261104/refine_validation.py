"""Retry persistent unsuccessful sinusoidal cancellation with shorter blocks."""
import sys,json,csv,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_analysis import OUT,ROOT,PROJECT,psd,csvout,dump
from lab_preproc import cancel_tone
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
def main():
    manifest=list(csv.DictReader((PROJECT/'work/audit_drone_20261023/data_preprocessing_manifest.csv').open(encoding='utf-8-sig')))
    starts={m['path'].replace('\\','/'):int(m['start_sample']) for m in manifest}
    sources=list(csv.DictReader((OUT/'source_groups.csv').open(encoding='utf-8-sig')))
    vals=list(csv.DictReader((OUT/'cancel_validation.csv').open(encoding='utf-8-sig'))); additional=[]; choices=[];examples={}
    for cp in sorted(OUT.glob('checkpoint*.json')):
        saved=json.loads(cp.read_text(encoding='utf-8'));r=saved['record'];fs=r['fs'];fc=r['fc'];retry=[]
        for l in saved['lines']:
            old=[v for v in saved['validations'] if v['freq_mhz']==l['freq_mhz'] and v['method']=='cancel']
            if l['presence_rate']>=.9 and not all(v['passed'] for v in old):retry.append((l,old))
        if not retry:continue
        files=sorted(s['path'] for s in sources if s['include']=='True' and s['group_id']==r['dataset']+':'+r['key'])
        if not files:continue
        var='data' if r['dataset']=='data' else 'IQ';x=io.loadmat(ROOT/files[0],variable_names=[var])[var].ravel();x=x[starts.get(files[0],0):]
        for l,old in retry:
            new=[]
            for bi in range(3):
                z=x[bi*round(fs*.01):(bi+1)*round(fs*.01)];y=cancel_tone(z,fs,(l['freq_mhz']-fc)*1e6,block_ms=1)
                f,before=psd(z,fs);_,after=psd(y,fs);f=f/1e6+fc;dist=abs(f-l['freq_mhz']);local=(dist>.15)&(dist<=1);region=(dist>=.15)&(dist<=2);core=dist<=.025
                residual=float(10*np.log10(after[core].max()/np.median(after[local])));delta=float(10*np.log10(after[region].sum()/before[region].sum()))
                row=dict(group=r['group'],recording=r['key'],freq_mhz=l['freq_mhz'],overlaps_signal=l['overlaps_signal'],block=bi,method='cancel_short',block_ms=1,residual_db=residual,off_line_delta_db=delta,error_energy_ratio=float(np.mean(abs(y-z)**2)/np.mean(abs(z)**2)),passed=residual<=3 and abs(delta)<=.1)
                additional.append(row);new.append(row)
                category=(r['dataset'],r['label'],r['band'])
                if bi==0 and category not in examples:examples[category]=(f,before,after,l['freq_mhz'],r['key'])
            choose_new=sum(v['passed'] for v in new)>sum(v['passed'] for v in old)
            choices.append(dict(recording=r['key'],group=r['group'],freq_mhz=l['freq_mhz'],drift_khz=l['drift_khz'],chosen_block_ms=1 if choose_new else 10,passed_10ms=sum(v['passed'] for v in old),passed_1ms=sum(v['passed'] for v in new),n_validation_blocks=3))
        print(r['key'],len(retry),'persistent lines retried',flush=True)
    if additional:csvout('cancel_validation.csv',vals+additional);csvout('cancel_short_blocks.csv',additional);csvout('cancel_final_parameters.csv',choices)
    dump('short_block_summary.json',dict(lines_retried=len(choices),lines_improved=sum(c['chosen_block_ms']==1 for c in choices),extra_cases=len(additional)))
    if examples:
        fig,axes=plt.subplots(len(examples),1,figsize=(13,4*len(examples)));axes=np.atleast_1d(axes)
        for ax,(category,(f,b,a,f0,key)) in zip(axes,examples.items()):
            ax.plot(f,10*np.log10(b+1e-30),label='Before',lw=.8);ax.plot(f,10*np.log10(a+1e-30),label='After 1 ms cancellation',lw=.8);ax.set_xlim(f0-2,f0+2);ax.set_xlabel('Absolute frequency (MHz)');ax.set_ylabel('PSD (dB/Hz)');ax.set_title(key);ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(OUT/'cancel_examples.png',dpi=150);plt.close(fig)
    plans=list(csv.DictReader((OUT/'crop_plan.csv').open(encoding='utf-8-sig')))
    for plan in plans:
        actions=json.loads(plan['lines']);center=float(plan['center_mhz']) if plan['center_mhz'] else None
        for action in actions:
            if center is not None and 28<abs(action['freq_mhz']-center)<30.72:
                source=next((l for cp in OUT.glob('checkpoint*.json') for l in json.loads(cp.read_text(encoding='utf-8'))['lines'] if l['recording']==plan['recording'] and l['freq_mhz']==action['freq_mhz']),None)
                action['method']='cancel' if source and source['overlaps_signal'] else 'notch'
                action['in_transition_band']=True
            if action['method']=='cancel':
                choice=next((c for c in choices if c['recording']==plan['recording'] and c['freq_mhz']==action['freq_mhz']),None)
                action['block_ms']=choice['chosen_block_ms'] if choice else 10
        plan['lines']=json.dumps(actions)
    csvout('crop_plan.csv',plans)
    report=(OUT/'REPORT.md').read_text(encoding='utf-8').replace('未达标的线禁止直接视为已消除；漂移/调制导致的残留需在F之前复核，当前未自动缩块调优。',f'对{len(choices)}条稳定但10ms对消未全达标的线追加1ms块验证；{sum(c["chosen_block_ms"]==1 for c in choices)}条改善，最终参数见cancel_final_parameters.csv；仍失败的线在F前复核。')
    (OUT/'REPORT.md').write_text(report,encoding='utf-8')
if __name__=='__main__':main()
