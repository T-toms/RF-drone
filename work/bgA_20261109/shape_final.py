from pathlib import Path
import sys,json,csv
sys.dont_write_bytecode=True
import numpy as np
O=Path(__file__).resolve().parent;P=O.parents[1];E=Path('E:/drone_lab_bgA_20261109');F=P/'work/crop_slice_20261107'
bs=[b for b in json.loads((F/'plan.json').read_text(encoding='utf-8')) if b['dataset']=='data' and b['band']=='2.4GHz'];f=2430+(np.arange(512)-256)*.12;mask=(f>=2405)&(f<=2455)
for r in csv.DictReader((F/'notch_table.csv').open(encoding='utf-8-sig')):
 if r['band']=='2.4GHz':mask &=abs(f-float(r['freq_mhz']))>.3
sub=[mask&(f>=lo)&(f<lo+1) for lo in range(2405,2455)]
def cells(b):return np.load(F/'cells'/Path(b['block_file']).stem/'mean.npy')
def floor(b):
 c=cells(b);med=[np.median(10*np.log10((10**(c[:,ix].astype(np.float64)/10)).mean(1))) for ix in sub if ix.any()];return float(min(med))
floors={Path(b['block_file']).stem:floor(b) for b in bs};median=float(np.median(list(floors.values())));(O/'floor_noise.json').write_text(json.dumps(dict(median=median,min=min(floors.values()),max=max(floors.values()),floors=floors,skip=[k for k,v in floors.items() if v>median+3]),indent=2),encoding='utf-8')
def shape(b,path):
 stem=Path(b['block_file']).stem;fl=floors[stem];c=cells(b);cache=O/'shape_cache';cache.mkdir(exist_ok=True);cp=cache/(path.stem+'.npy')
 if cp.exists():avg=np.load(cp)
 else:
  z=np.load(path,mmap_mode='r');avg=np.zeros(512);w=np.hanning(512).astype(np.float32);nc=len(z)//61440
  for j in range(nc):
   x=np.asarray(z[j*61440:(j+1)*61440])[:120*512].reshape(120,512);avg+=(abs(np.fft.fftshift(np.fft.fft(x*w,axis=1),axes=1))**2/np.sum(w**2)).mean(0)
  avg/=nc;np.save(cp,avg)
 rs=[]
 for lo,ix in zip(range(2405,2455),sub):
  quiet=np.all(c[:,ix]<fl+3,axis=1);n=int(quiet.sum());v=None
  if n>=50 and ix.any():v=float(10*np.log10(avg[ix].mean()/(10**(c[quiet][:,ix].astype(np.float64)/10)).mean()))
  rs.append(dict(lo_mhz=lo,N=n,diff_db=v,status='judged' if v is not None else 'not_judged'))
 vals=[abs(r['diff_db']) for r in rs if r['diff_db'] is not None];skip=fl>median+3;return dict(block=stem,s=int(path.stem.rsplit('_s',1)[1]),floor_noise=fl,skip=skip,shape_judged=len(vals),shape_max=max(vals,default=None),shape_pass=skip or max(vals,default=0)<=.5,subbands=rs,low_2405_2411=[r for r in rs if r['lo_mhz']<2411])
if __name__=='__main__':
 out=[]
 for b in json.loads((O/'pilot_plan.json').read_text(encoding='utf-8')):
  for s in [0,2]:
   path=E/'2.4GHz'/f'{Path(b["block_file"]).stem}__bgA_s{s}.npy'
   if not path.exists():continue
   r=shape(b,path);out.append(r);(O/'pilot_shape_final.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print({k:v for k,v in r.items() if k not in ['subbands','low_2405_2411']},flush=True)
