from pathlib import Path
import sys,json,time
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
import shape_p10 as sh
import numpy as np
O=sh.O;E=sh.E
bs=json.loads((O/'full_plan.json').read_text(encoding='utf-8'));dest=O/'full_shape_profile.json';rs=json.loads(dest.read_text(encoding='utf-8')) if dest.exists() else []
for b in bs:
 stem=Path(b['block_file']).stem
 if any(r['block']==stem for r in rs):continue
 path=E/'2.4GHz'/f'{stem}__bgA_s0.npy'
 while not path.exists():time.sleep(3)
 r=sh.shape(b,0);vv=[v['p10_diff'] for v in r['subbands'] if v['quiet_fraction']>=.9];level=float(np.median(vv)) if vv else None;res=max([abs(v-level) for v in vv],default=None);r.update(level_offset=level,shape_residual=res,n_quiet90_subbands=len(vv),shape_residual_over05=res is not None and res>.5);rs.append(r);dest.write_text(json.dumps(rs,indent=2),encoding='utf-8');print(stem,'level',level,'residual',res,'N',len(vv),flush=True)
print('FULL_SHAPE_DONE',flush=True)
