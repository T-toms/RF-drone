from pathlib import Path
import sys,time,json,importlib.util
sys.dont_write_bytecode=True
import numpy as np
O=Path(__file__).resolve().parent;P=O.parents[1];E=Path('E:/drone_lab_spec_bgA_L20_20261109');E.mkdir(exist_ok=True)
s=importlib.util.spec_from_file_location('common',P/'work/train_stage1_L20_20261108/common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
rr=[r for r in c.rows(P/'work/crop_slice_20261107/index_L20.csv') if r['dataset']=='data' and r['band']=='2.4GHz'];dest=E/'bgA_spec.npy';tmp=dest.with_suffix('.partial.npy');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.float16,shape=(len(rr),256,256));last=None
for i,r in enumerate(rr):
 path=Path('E:/drone_lab_bgA_20261109/2.4GHz')/(Path(r['block_file']).stem+'__bgA_s0.npy')
 while not path.exists():time.sleep(3)
 if str(path)!=last:z=np.load(path,mmap_mode='r');last=str(path)
 j=int(r['start']);out[i]=c.spec(np.array(z[j:j+int(r['n_samples'])]),'2.4GHz')
 if i%200==0:print('SPEC',i,len(rr),flush=True)
out.flush();del out;tmp.replace(dest);(E/'paired_ids.json').write_text(json.dumps([r['sample_id'] for r in rr]),encoding='utf-8');print('SPEC_PRECOMPUTE_DONE',len(rr),flush=True)
