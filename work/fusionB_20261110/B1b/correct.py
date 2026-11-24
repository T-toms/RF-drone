import sys,time
from pathlib import Path
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O))
from screen import *
import screen_original as orig
if __name__=='__main__':
 t=time.time();rng=np.random.default_rng(2026);white=[]
 for i in range(20):
  z=(rng.standard_normal(L)+1j*rng.standard_normal(L))/np.sqrt(2);z=notch(z,FS,(2432.9412-2430)*1e6,50000);q,ex=quiet(z);white.append(dict(index=i,quiet=q,max_excess_db=ex))
 dump('white_check.json',white)
 # Additional appendix: no new notch, only exclude full 2432/2433 MHz cells.
 for side in ['bgA','bgB']:
  rr=pd.read_csv(O/f'original_{side}_index.csv',dtype={'nn':str}).to_dict('records');out=[]
  def app3work(r):
   c=orig.cells(orig.getz(r));c=np.delete(c,[27,28],axis=1);ex=float(10*np.log10(c.max()/np.median(c)));return dict(r,quiet=ex<=3,max_excess_db=ex)
  with ThreadPoolExecutor(max_workers=4) as pool:
   for i,r in enumerate(pool.map(app3work,rr)):
    out.append(r)
    if i%1000==0:print('APP3_SCREEN',side,i,flush=True)
  pd.DataFrame(out).to_csv(O/f'appendix3_{side}_index.csv',index=False,encoding='utf-8')
 for side in ['bgA','bgB']:
  rr=pd.read_csv(O/f'original_{side}_index.csv',dtype={'nn':str});rr=rr.drop(columns=['quiet','max_excess_db']).to_dict('records');out,xx=evaluate(rr,side=='bgB');df=pd.DataFrame(out);df.to_csv(O/f'{side}_index.csv',index=False,encoding='utf-8')
  if side=='bgA':
   # Authorized 1500 random quiet A segments, deterministic seed 2026.
   sel=df[df.quiet];sel=sel.iloc[np.random.default_rng(2026).choice(len(sel),min(1500,len(sel)),replace=False)].sort_index();sel.to_csv(O/'bgA_analysis_index.csv',index=False,encoding='utf-8')
   vals=[]
   with ThreadPoolExecutor(max_workers=4) as pool:
    for j,v in enumerate(pool.map(lambda r:feature(getz(r),'2.4GHz'),sel.to_dict('records'))):
     vals.append(v)
     if j%250==0:print('A_FEATURE',j,len(sel),flush=True)
   np.save(O/'bgA_features.npy',np.array(vals))
  else:np.save(O/'bgB_features.npy',xx)
  print('CORRECTED_QUIET',side,int(df.quiet.sum()),len(df),flush=True)
 # In-band references retain original F-chain IQ, quiet rule and original feature mask.
 plan=json.loads((P/'work/crop_slice_20261107/plan.json').read_text(encoding='utf-8'));bs=[b for b in plan if b['group_id']=='v3:20260715-075302_ambient@2440.0'];cr,cf=orig.evaluate(orig.sliced(bs,'v3'),True);pd.DataFrame(cr).to_csv(O/'B2440_index.csv',index=False,encoding='utf-8');np.save(O/'B2440_features.npy',cf)
 runs=json.loads((P/'work/bgA_20261109/quiet013_runs.json').read_text(encoding='utf-8'));rr=[]
 for r in runs:
  for j in range(r['start_ms'],r['end_ms']-19,20):rr.append(dict(block_file=r['block'],start=j*61440,n_samples=L,sample_id=f'A013_{len(rr):08d}',nn='013',group_id='A013'))
 assert len(rr)==19;dr,df=orig.evaluate(rr,True);pd.DataFrame(dr).to_csv(O/'A013_index.csv',index=False,encoding='utf-8');np.save(O/'A013_features.npy',df)
 dump('screen_runtime.json',dict(seconds=time.time()-t));print('SCREEN_DONE',flush=True)