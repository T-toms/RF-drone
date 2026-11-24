from pathlib import Path
import sys,json,csv,concurrent.futures
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
import pipeline as p
O=p.O;P=p.P;E=p.E
def rows(path):return list(csv.DictReader(path.open(encoding='utf-8-sig')))
def write(path,rs,keys=None):
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys or list(rs[0]));w.writeheader();w.writerows(rs)
def generate(s):
 bs=[b for b in json.loads((P/'work/crop_slice_20261107/plan.json').read_text(encoding='utf-8')) if b['dataset']=='data' and b['band']=='2.4GHz'];assert len(bs)==61;(O/'full_plan.json').write_text(json.dumps(bs,ensure_ascii=False,indent=2),encoding='utf-8')
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  fs=[ex.submit(p.process,b,E/'2.4GHz'/f'{Path(b["block_file"]).stem}__bgA_s{s}.npy',True,s) for b in bs]
  for i,f in enumerate(concurrent.futures.as_completed(fs)):f.result();print('FULL_PROGRESS',i+1,61,flush=True)
 print('FULL_GENERATION_DONE',flush=True)
def index(s):
 metrics=json.loads((O/'full_revised_final.json').read_text(encoding='utf-8'));bad={r['block'] for r in metrics if not r['image_pass']};assert len(bad)<=15,('COORDINATOR_STOP_IF_OVER15',sorted(bad));counts=[]
 for L in [10,20,30,40,50]:
  source=rows(P/f'work/crop_slice_20261107/index_L{L}.csv');bg=[]
  for r in source:
   if r['dataset']!='data' or r['band']!='2.4GHz':continue
   v=dict(r);stem=Path(r['block_file']).stem;v.update(block_file=str(E/'2.4GHz'/f'{stem}__bgA_s{s}.npy'),dataset='dataA_mirror',label='background',model='none',state='unknown',bw_label='unknown',sample_id=f'bgA_L{L}_{len(bg):08d}',paired_sample_id=r['sample_id'],mirror_shift_mhz=s,keep=stem not in bad)
   for k in ['lack_feature','primary_weak','power_shift','floor_raised','interference']:v[k]=False
   assert Path(v['block_file']).exists();assert int(v['start'])+int(v['n_samples'])<=len(p.np.load(v['block_file'],mmap_mode='r'));bg.append(v)
  write(O/f'index_bgA_L{L}.csv',bg,list(source[0])+['paired_sample_id','mirror_shift_mhz'])
  for sp in ['train','val','test']:
   a=[r for r in bg if r['split']==sp and r['keep']];v3=[r for r in source if r['dataset']=='v3' and r['band']=='2.4GHz' and r['label']=='background' and r['split']==sp and r['keep']=='True'];counts.append(dict(L=L,split=sp,bgA_N=len(a),bgA_seconds=len(a)*L/1000,v3_N=len(v3),v3_seconds=len(v3)*L/1000))
 write(O/'background_counts.csv',counts);(O/'excluded_blocks.json').write_text(json.dumps(sorted(bad)),encoding='utf-8');print('INDEX_DONE',flush=True)
if __name__=='__main__':
 mode=sys.argv[1];s=int(sys.argv[2]);generate(s) if mode=='generate' else index(s)
