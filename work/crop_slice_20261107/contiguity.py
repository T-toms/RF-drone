from pathlib import Path
import sys,json,csv,re
import numpy as np
from scipy import io,signal
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
P=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent; R=Path('G:/drone')
def dump(p,d): p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def match(y,x):
 c=signal.fftconvolve(y,np.conj(x[::-1]),mode='valid'); e=np.r_[0,np.cumsum(abs(y.astype(np.complex128))**2)]; v=abs(c)/(np.sqrt(np.maximum(e[len(x):]-e[:-len(x)],0))*np.linalg.norm(x)+1e-30); k=int(v.argmax());return k,float(v[k])
def main():
 rows=list(csv.DictReader((P/'work/prep_lines_20261104/source_groups.csv').open(encoding='utf-8-sig'))); raw=list((R/'raw').rglob('*.bin')); result=[]
 inv=[(str(p.relative_to(R)),p.stat().st_size,p.stat().st_mtime_ns) for p in R.rglob('*') if p.is_file()];dump(O/'source_before.json',inv)
 for tag in ['07-10_lab','07-11_home','075302']:
  row=next(r for r in rows if tag in r['path'] and '_s0001_' in r['path'] and (R/r['path'].replace('s0001','s0002')).exists()); p=R/row['path']; q=p.with_name(p.name.replace('s0001','s0002')); nn=re.search(r'_(\d{6})_s',p.name)[1]; session=re.search(r'(\d{8}-\d{6})',p.name)[1]
  b=next(b for b in raw if session in str(b) and b.stem.endswith('_'+nn)); z=np.fromfile(b,dtype='<i2').astype(np.float32); y=signal.resample_poly(z[::2]+1j*z[1::2],4 if tag=='075302' else 2,5); del z
  checks=[]
  for f in [p,q]:
   x=io.loadmat(f,variable_names=['IQ'])['IQ'].ravel().astype(np.complex64); k,c=match(y,x[:100000]); t,ct=match(y,x[-100000:]);checks.append(dict(file=str(f.relative_to(R)),start=k,end=k+len(x),head_corr=c,tail_corr=ct,tail_start=t,tail_expected=k+len(x)-100000))
  ok=all(v['head_corr']>=.999 and v['tail_corr']>=.999 and v['tail_start']==v['tail_expected'] for v in checks) and checks[0]['end']==checks[1]['start']; result.append(dict(case=tag,raw=str(b.relative_to(R)),status='pass' if ok else 'fail',contiguous=ok,checks=checks)); print(result[-1],flush=True)
 dump(O/'contiguity.json',result)
if __name__=='__main__':main()
