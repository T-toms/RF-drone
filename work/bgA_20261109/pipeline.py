import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from chain_copy import *
import concurrent.futures
E=Path('E:/drone_lab_bgA_20261109');E.mkdir(exist_ok=True)
def dump(name,v): (O/name).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def process(b,dest,conjugate=False,shift=0):
 dest=Path(dest)
 if dest.exists():return str(dest)
 assert shutil.disk_usage('E:/').free>20e9
 fs=b['fs_in'];band=b['band'];fo=FS[band];up,down=ratio(fs,band);trim=round(fo*.0002);nout=math.ceil(b['n_in']*up/down);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.partial.npy');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.complex64,shape=(nout-2*trim,));src=Source(b);chunk=round(fs*.05);halo=math.ceil(fs*.002/down)*down
 for s in range(0,b['n_in'],chunk):
  e=min(b['n_in'],s+chunk);a=max(0,s-halo);a=a//down*down;c=min(b['n_in'],e+halo);raw=src.get(a,c);raw=np.conj(raw) if conjugate else raw;y=chain(raw,fs,b['fc_in']-shift,band,a);lo=max(trim,s*up//down);hi=min(nout-trim,math.ceil(e*up/down));out[lo-trim:hi-trim]=y[lo-a*up//down:hi-a*up//down]
 out.flush();del out;tmp.replace(dest);print('processed',dest.name,flush=True);return str(dest)
def feasibility():
 allrows=[];occs={};rank=[]
 for d in sorted(Path('E:/drone_lab_cache_20261031').iterdir()):
  if not (d/'meta.json').exists():continue
  m=json.loads((d/'meta.json').read_text(encoding='utf-8'));x=np.load(d/'cell_mean_db.npy').astype(np.float32);valid=np.load(d/'cell_valid.npy') & ~np.isnan(x).any(1);f=m['fc_hz']/1e6+(np.arange(1024)-512)*.12;b=m['band'];hot=np.nan_to_num(x,nan=999)>m['floor_db']+3;sel=(f>=(2405 if b=='2.4GHz' else 5755))&(f<=(2455 if b=='2.4GHz' else 5845))
  for e in ([2400,2426.88,2440,2450,2457.6] if b=='2.4GHz' else [5800,5795.5,5740]):sel &=abs(f-e)>.25
  busy=hot[:,sel].any(1)|~valid;edges=np.diff(np.r_[False,~busy,False].astype(int));runs=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1);occ=hot[valid].mean(0);occs.setdefault(b,[]).append(occ);allrows.append(dict(rec=d.name,band=b,longest=int(max(runs,default=0)),ge20=int(runs[runs>=20].sum())))
  if b=='2.4GHz':rank.append((float(occ[(f>=2393.5)&(f<=2398.6)].max()),d.name))
 for b,v in occs.items():np.save(O/f'occupancy_{b}.npy',np.array(v))
 dump('feasibility.json',allrows);return sorted(rank,reverse=True)
def main():
 bs=json.loads((P/'work/crop_slice_20261107/plan.json').read_text(encoding='utf-8'));bs=[b for b in bs if b['dataset']=='data' and b['band']=='2.4GHz'];control=next(b for b in bs if '012_2.4GHz_10M-signal__2026-09-20-18-58-55' in b['group_id']);p=process(control,E/'control.npy');x=np.load(p,mmap_mode='r');y=np.load(control['block_file'],mmap_mode='r');err=0.;peak=0.;point=0.
 for j in range(0,len(x),1000000):
  a=x[j:j+1000000];b=y[j:j+1000000];err=max(err,float(abs(a-b).max()));peak=max(peak,float(abs(b).max()));point=max(point,float((abs(a-b)/np.maximum(abs(b),1e-30)).max()))
 checks=dict(n_copy=len(x),n_original=len(y),max_absolute_error=err,relative_to_peak=err/peak,max_point_relative=point);dump('implementation.json',checks);assert len(x)==len(y) and point<1e-5,checks
 mapping=[];fs=122880000;n=1228800;rng=np.random.default_rng(20261109)
 for a,s in [(2370,0),(2370,2),(2395,0),(2345,0)]:
  raw=np.exp(2j*np.pi*(a-2400)*1e6*np.arange(n)/fs)+.01*(rng.standard_normal(n)+1j*rng.standard_normal(n));z=chain(np.conj(raw),fs,2400-s,'2.4GHz');z=z[20000:-20000];freq=2430+np.fft.fftfreq(len(z),1/FS['2.4GHz'])[abs(np.fft.fft(z)).argmax()]/1e6;mapping.append(dict(input_mhz=a,s=s,expected=4800-s-a,measured=float(freq)));assert abs(freq-(4800-s-a))<.001
 dump('mapping.json',mapping);print('CHECKS_PASS',checks,mapping,flush=True)
 rank=feasibility();br=list(csv.DictReader((P/'work/crop_slice_20261107/blocks.csv').open(encoding='utf-8-sig')));power={r['block_file']:float(r['power_db_median']) for r in br};chosen=sorted(bs,key=lambda b:-power[b['block_file']])[:4]
 for _,rec in rank[:3]:
  c=next(b for b in bs if b['group_id']=='data:'+rec)
  if c not in chosen:chosen.append(c)
 if control not in chosen:chosen.append(control)
 for b in sorted(bs,key=lambda b:-power[b['block_file']]):
  if len(chosen)>=8:break
  if b not in chosen:chosen.append(b)
 dump('pilot_plan.json',chosen)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  fsu=[ex.submit(process,b,E/'2.4GHz'/f'{Path(b["block_file"]).stem}__bgA_s{s}.npy',True,s) for b in chosen for s in [0,2]]
  for f in concurrent.futures.as_completed(fsu):f.result()
 print('PILOT_GENERATION_DONE',flush=True)
if __name__=='__main__':main()
