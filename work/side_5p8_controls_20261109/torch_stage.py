from pathlib import Path
import sys,os,csv,json,time
os.environ['PYTHONIOENCODING']='utf-8';sys.dont_write_bytecode=True
import numpy as np,torch
P=Path('C:/Users/Administrator/Desktop/数据集处理'); O=Path(__file__).resolve().parent; E=Path('E:/side_5p8_controls_20261109'); E.mkdir(exist_ok=True)
sys.path.insert(0,str(P/'work/train_stage1_L20_20261108'))
import common
NAMES={'a':'white_raw','b':'white_filtered','A':'proxy_A','B':'proxy_B'}
sys.path.insert(0,str(P/'dronerfa_baseline')); from model import ResNet18
def rows(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(p,rs):
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
if sys.argv[1]=='stream':
 band=sys.argv[2]; kind=sys.argv[3]; n=common.FS[band]//50
 a=np.lib.format.open_memmap(E/f'{band}_{NAMES[kind]}_spec.npy',mode='w+',dtype=np.float16,shape=(200,256,256))
 for i in range(200):
  b=sys.stdin.buffer.read(n*8)
  if len(b)!=n*8:raise RuntimeError(f'short pipe {i}: {len(b)}')
  a[i]=common.spec(np.frombuffer(b,dtype=np.complex64).copy(),band)
 a.flush();print('spec complete',band,kind,flush=True)
 save(O/f'gpu_stream_{band}_{NAMES[kind]}.json',dict(peak_mb=torch.cuda.max_memory_allocated()/2**20))
 sys.exit(0)
def fill(x,band):
 fs=common.FS[band];fc=common.FC[band];half=25 if band=='2.4GHz' else 45
 fr=np.fft.fftshift(np.fft.fftfreq(1024,1/fs))/1e6+fc;fr=fr[abs(fr-fc)<=half]
 rf=np.array([fr[int(np.floor(i*len(fr)/256)):int(np.ceil((i+1)*len(fr)/256))].mean() for i in range(256)])
 bad=np.logical_or.reduce([abs(rf-float(r['freq_mhz']))<=.45 for r in rows(P/'work/crop_slice_20261107/notch_table.csv') if r['band']==band]); good=np.flatnonzero(~bad)
 x=np.asarray(x,dtype=np.float32).copy()
 for k in np.flatnonzero(bad):
  lo=good[good<k][-1];hi=good[good>k][0];x[:,k,:]=x[:,lo,:]+(x[:,hi,:]-x[:,lo,:])*(k-lo)/(hi-lo)
 x=(x-x.mean((1,2),keepdims=True))/np.maximum(x.std((1,2),keepdims=True),1e-8)
 return x.astype(np.float16)
def inputstats(a,b):
 pa=a.astype(np.float32).mean((0,2));pb=b.astype(np.float32).mean((0,2))
 return dict(correlation=float(np.corrcoef(pa,pb)[0,1]),maxdiff_std=float(abs(pa-pb).max()),proxy_pixels_gt3=float((a>3).mean()),real_pixels_gt3=float((b>3).mean()))
results=[]; stats=[]; meta_all={}
for band in common.BANDS:
 tm=common.meta(band,'test');ta=np.load(common.cachepath(band,'test'),mmap_mode='r')
 sets={k:np.load(E/f'{band}_{NAMES[k]}_spec.npy',mmap_mode='r') for k in ['a','b','A','B']}
 bgids=[i for i,r in enumerate(tm) if r['dataset']=='v3' and r['label']=='background'];drids=[i for i,r in enumerate(tm) if r['dataset']=='data' and r['label']=='drone']
 sets['e_bg']=np.ascontiguousarray(ta[bgids]);sets['e_drone']=np.ascontiguousarray(ta[drids])
 # All test sessions retained, not only July 10, and report each session separately.
 for gid in sorted(set(tm[i]['group_id'] for i in bgids)):
  sets['session:'+gid]=np.ascontiguousarray(ta[[i for i in bgids if tm[i]['group_id']==gid]])
 if band=='2.4GHz':
  bm=rows(Path('E:/drone_lab_spec_bgA_L20_20261109/bgA_meta.csv'));ba=np.load(Path('E:/drone_lab_spec_bgA_L20_20261109/bgA_spec.npy'),mmap_mode='r')
  bids=[i for i,r in enumerate(bm) if r['keep']=='True' and r['split']=='test'];assert len(bids)==1287,len(bids)
  sets['bgA_test']=np.ascontiguousarray(ba[bids]);sets['bgB_0710']=np.ascontiguousarray(ta[[i for i in bgids if '20260710' in tm[i]['group_id']]])
  for proxy,real in [('A','bgA_test'),('B','bgB_0710')]:stats.append(dict(proxy=proxy,n_proxy=len(sets[proxy]),n_real=len(sets[real]),**inputstats(sets[proxy],sets[real])))
 for k in ['b','A','e_bg','e_drone']:sets['f_'+k]=fill(sets[k],band)
 for k in list(sets):
  if k.startswith('session:'):sets['f_'+k]=fill(sets[k],band)
 for model in (['H','J'] if band=='2.4GHz' else ['H']):
  for seed in common.SEEDS:
   base=Path('E:/drone_lab_spec_L20_20261108') if model=='H' else Path('E:/drone_lab_stage1_L20_bgA_20261109')
   ck=torch.load(base/f'{band}_seed{seed}_best.pt',map_location='cpu',weights_only=False)
   m=ResNet18(1,2).cuda().to(memory_format=torch.channels_last);m.load_state_dict(ck['model']);m.eval()
   for k,a in sets.items():
    ps=common.predict(m,torch.from_numpy(np.array(a)).unsqueeze(1));prob=ps[:,1]
    for decision,mask in [('argmax',ps.argmax(1)==1),('thr_fa1',prob>=float(ck['thr_fa1']))]:
     results.append(dict(band=band,model=model,seed=seed,input=k,decision=decision,k=int(mask.sum()),n=len(mask),rate=float(mask.mean()),threshold=float(ck['thr_fa1']) if decision=='thr_fa1' else .5))
   print('evaluated',band,model,seed,flush=True);del m
   csvout(O/'inference.csv',results)
save(O/'input_statistics.json',stats);save(O/'gpu_eval.json',dict(peak_mb=torch.cuda.max_memory_allocated()/2**20));print('all inference complete',flush=True)
