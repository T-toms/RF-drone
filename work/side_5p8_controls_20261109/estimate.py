from pathlib import Path
import csv,json,time
import numpy as np
P=Path('C:/Users/Administrator/Desktop/数据集处理'); O=Path(__file__).resolve().parent
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
FS={'2.4GHz':61440000,'5.8GHz':100000000}; FC={'2.4GHz':2430.,'5.8GHz':5800.}
bs=rows(P/'work/crop_slice_20261107/blocks.csv'); splits={r['group_id']:r['split'] for r in rows(P/'work/crop_slice_20261107/group_split.csv')}; notches=rows(P/'work/crop_slice_20261107/notch_table.csv')
summ=[]; records=[]; arrays={}
for band in FS:
 n=512 if band=='2.4GHz' else 1024; freq=FC[band]+np.fft.fftshift(np.fft.fftfreq(n,1/FS[band]))/1e6
 for receiver in ['A','B']:
  selected=[b for b in bs if b['band']==band and splits[b['group_id']]=='train' and (b['dataset']=='data' if receiver=='A' else b['dataset']=='v3' and b['label']=='background' and '20260710' in b['group_id'])]
  p10=[]; occ=[]
  for b in selected:
   c=np.load(P/'work/crop_slice_20261107/cells'/Path(b['block_file']).stem/'mean.npy')
   med=[np.median(c[:,(freq>=lo)&(freq<lo+1)]) for lo in np.arange(FC[band]-25,FC[band]+25)]
   floor=float(min(med)); p10.append(np.percentile(c,10,axis=0)); occ.append((c>floor+3).mean(0))
   records.append(dict(band=band,receiver=receiver,group_id=b['group_id'],block_file=b['block_file'],floor_db=floor))
  a=np.array(p10); valid=np.array(occ)<.5; counts=valid.sum(0)
  values=np.ma.median(np.ma.array(a,mask=~valid),axis=0).filled(np.nan)
  notch=np.logical_or.reduce([abs(freq-float(r['freq_mhz']))<=.3 for r in notches if r['band']==band])
  bad=(counts<5)|notch|~np.isfinite(values); good=~bad
  print(band,receiver,'blocks',len(selected),'valid bins',good.sum(),'count range',counts.min(),counts.max(),flush=True)
  if not good.any():raise RuntimeError('No valid frequency bins: '+band+receiver)
  shape=np.interp(freq,freq[good],values[good]); key=band+'_'+receiver
  arrays[key]=shape; arrays[key+'_freq']=freq; arrays[key+'_counts']=counts; arrays[key+'_bad']=bad
  summ.append(dict(band=band,receiver=receiver,blocks=len(selected),groups=len(set(b['group_id'] for b in selected)),counts_percentiles=np.percentile(counts,[0,25,50,75,100]).tolist(),interpolated=int(bad.sum()),interpolated_freq_mhz=freq[bad].tolist(),edge_extrapolated=int(((freq<freq[good].min())|(freq>freq[good].max())).sum())))
np.savez(O/'shapes.npz',**arrays); dump(O/'shape_summary.json',summ)
with (O/'shape_blocks.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
