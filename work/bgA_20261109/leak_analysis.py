from pathlib import Path
import json,csv,numpy as np
O=Path(__file__).resolve().parent;P=O.parents[1];rr=json.loads((O/'full_revised_final.json').read_text(encoding='utf-8'));assert len(rr)==61
passed=[r for r in rr if r['image_pass']];out=[]
for lo in [2405,2406,2407]:
 for kind in ['all_nonempty','n_hot_ge2000']:
  v=[v['delta20'] for r in passed for v in r['subbands'] if v['lo_mhz']==lo and v['delta20'] is not None and (kind=='all_nonempty' or v['n_hot']>=2000)];out.append(dict(lo_mhz=lo,kind=kind,N=len(v),median=float(np.median(v)) if v else None,max=max(v,default=None)))
with (O/'passed_lowband_summary.csv').open('w',encoding='utf-8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
power={Path(r['block_file']).stem:float(r['power_db_median']) for r in csv.DictReader((P/'work/crop_slice_20261107/blocks.csv').open(encoding='utf-8-sig'))};rank=sorted(rr,key=lambda r:power[r['block']]);obs=[]
for i,r in enumerate(rank):obs.append(dict(block=r['block'],power_db_median=power[r['block']],power_rank_ascending=i+1,delta20=r['delta20'],image_pass=r['image_pass']))
(O/'power_delta20.json').write_text(json.dumps(obs,indent=2),encoding='utf-8');print('LEAK_ANALYSIS_DONE',out,flush=True)
