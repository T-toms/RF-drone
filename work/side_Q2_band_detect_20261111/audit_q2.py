"""Read-only acceptance checks; write audit.json inside the output directory."""
import json,hashlib,importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
OUT=Path(__file__).resolve().parent
lock=json.loads((OUT/'threshold_lock.json').read_text(encoding='utf-8'));ks=lock['k_star']
locked_report=(OUT/'calibration_report_locked.md').read_bytes()
assert hashlib.sha256(locked_report).hexdigest()==lock['report_sha256']
assert (OUT/'REPORT.md').read_bytes().startswith(locked_report)
assert len((OUT/'REPORT.md').read_text(encoding='utf-8').splitlines())<=80
cal=pd.read_csv(OUT/'calibration.csv');assert len(cal)==472*4
counts=cal[cal.k==ks].groupby('cal_group').size().to_dict();assert counts=={'a':342,'b':10,'c':120}
expected_ids={c['clip_id'] for c in json.loads((OUT/'clips.json').read_text(encoding='utf-8'))}
checks=[]
for k in (2,3,4,6):
    for g in (.5,2):
        tag=f'k{k}_g{g:g}';s=pd.read_csv(OUT/f'segments_{tag}.csv');b=pd.read_csv(OUT/f'bands_{tag}.csv')
        assert len(s)==1426 and set(s.clip_id)==expected_ids and s.clip_id.is_unique
        assert s.loc[s.band=='5.8GHz','threshold_calibration_note'].str.contains('5.8GHz').all()
        assert s.loc[s.band=='5.8GHz','threshold_calibration_note'].str.contains('假设').all()
        assert set(b.clip_id)<=expected_ids
        assert ((s.coverage.dropna()>=-1e-10)&(s.coverage.dropna()<=1+1e-10)).all()
        assert s.band_fraction.between(0,1.005).all()
        if len(b):
            assert np.allclose(b.width_mhz,b.f_end_mhz-b.f_start_mhz)
            assert b.time_duty.between(0,1).all() and (b.box_count>0).all()
            bc=b.groupby('clip_id').size().reindex(s.clip_id,fill_value=0).to_numpy()
            assert np.array_equal(bc,s.band_count.to_numpy())
            summed=b.groupby('clip_id').width_mhz.sum().reindex(s.clip_id,fill_value=0).to_numpy()
            assert np.allclose(summed,s.width_mhz)
            for cid,v in b.groupby('clip_id',sort=False):
                assert np.array_equal(v.band_id.to_numpy(),np.arange(1,len(v)+1))
                assert (v.f_start_mhz.to_numpy()[1:]-v.f_end_mhz.to_numpy()[:-1]>=g-1e-9).all()
        checks.append(dict(parameter=tag,segments=len(s),bands=len(b),boxes=int(s.box_count.sum())))
boxfiles=list(OUT.glob('boxes_*.csv'));assert [p.name for p in boxfiles]==[f'boxes_k{ks}.csv']
boxes=pd.read_csv(boxfiles[0]);assert (boxes.area_px>=50).all()
seg=pd.read_csv(OUT/f'segments_k{ks}_g0.5.csv')
assert len(boxes)==seg.box_count.sum()
for key,v in boxes.groupby('clip_id'):
    assert len(v)==int(seg.loc[seg.clip_id==key,'box_count'].iloc[0])
plots=pd.read_csv(OUT/'plot_index.csv');assert plots.groupby('figure').size().max()<=6
figures=[]
for p in OUT.glob('图_*.png'):
    assert p.stat().st_size<=5_000_000
    with Image.open(p) as im:
        assert max(im.info.get('dpi',(0,0)))<=90.01
        figures.append(dict(name=p.name,bytes=p.stat().st_size,size=list(im.size),dpi=list(im.info.get('dpi',(0,0)))))
assert set(plots.figure)=={f['name'] for f in figures}
provenance=json.loads((OUT/'input_provenance.json').read_text(encoding='utf-8'))
for name,v in provenance.items():assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==v['sha256']
v=json.loads((OUT/'verification.json').read_text(encoding='utf-8'));assert v['protected_identical'] and v['peak_bytes']<8*1024**3
audit=dict(passed=True,calibration_counts=counts,parameters=checks,figures=figures,boxes_rows=len(boxes),report_lines=len((OUT/'REPORT.md').read_text(encoding='utf-8').splitlines()))
(OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS',len(checks),'parameter sets;',len(figures),'figures;',len(boxes),'boxes')
