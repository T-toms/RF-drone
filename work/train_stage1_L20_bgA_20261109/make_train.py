from pathlib import Path
import difflib
O=Path(__file__).resolve().parent
src=(O.parent/'train_stage1_L20_20261108/train.py').read_text(encoding='utf-8-sig')
s=src.replace('from common import *','from support import *\nimport ctypes\nCACHE=DEST')
s=s.replace("b=a.band;s=a.seed;", "assert a.band=='2.4GHz'\nb=a.band;s=a.seed;")
s=s.replace("(OUT/f'{tag}_done.json')","(CACHE/f'{tag}_done.json')")
s=s.replace("tx,ty,tr=dataset(b,'train');vx,vy,vr=dataset(b,'val')", "tx,ty,tr,tk=joined('train');vx,vy,vr,vk=joined('val')")
s=s.replace('counts=torch.bincount(ty,minlength=2).float();w=1/counts;w=w/w.mean();w=w.cuda()', 'counts=torch.bincount(tk,minlength=3).float();w=torch.tensor([1.,.5,.5])/counts;w=w*(len(tx)/2);w=w.cuda()\ntw=w[tk.cuda()];vw=w[vk.cuda()]')
s=s.replace("best=(float('inf'),float('inf'));logs=[];torch.cuda.reset_peak_memory_stats()", """best=(float('inf'),float('inf'));logs=[];torch.cuda.reset_peak_memory_stats()
first_epoch=1
lastpath=CACHE/f'{tag}_{"smoke_" if smoke else ""}last.pt'
if lastpath.exists():
    resume=torch.load(lastpath,map_location='cpu',weights_only=False)
    m.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);sched.load_state_dict(resume['scheduler'])
    scaler.load_state_dict(resume['scaler']);best=tuple(resume['best']);first_epoch=resume['epoch']+1
    logs=rows(OUT/f'train_log_{tag}{"_smoke" if smoke else ""}.csv')
    random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
""")
s=s.replace('range(1,a.epochs+1)', 'range(first_epoch,a.epochs+1)')
s=s.replace('with torch.autocast(\'cuda\',dtype=dtype):loss=F.cross_entropy(m(x),y,weight=w)', "with torch.autocast('cuda',dtype=dtype):\n            ce=F.cross_entropy(m(x),y,reduction='none');bw=tw[ids.cuda()];loss=(ce*bw).sum()/bw.sum()")
s=s.replace("vl+=float(F.cross_entropy(logits.float(),y,weight=w,reduction='sum'));vd+=float(w[y].sum())", "bw=vw[i:i+64];vl+=float((F.cross_entropy(logits.float(),y,reduction='none')*bw).sum());vd+=float(bw.sum())")
s=s.replace("fa=float((pred[yy==0]==1).mean());", "fa=float((pred[vk.numpy()==1]==1).mean());fa_bgA=float((pred[vk.numpy()==2]==1).mean());")
s=s.replace('val_fa=fa,','val_fa_v3=fa,val_fa_bgA=fa_bgA,')
s=s.replace('class_weights=w.cpu().tolist()', 'sample_weights=w.cpu().tolist(),train_counts=counts.tolist()')
s=s.replace('(fa+miss,vl)', '(fa+fa_bgA+miss,vl)')
s=s.replace("bg=np.sort(probs[yy==0,1]);allowed=int(np.floor(.01*len(bg)));thr=float(np.nextafter(bg[-1-allowed],np.float32(np.inf)))", "bg=np.sort(probs[vk.numpy()==1,1]);bga=np.sort(probs[vk.numpy()==2,1]);allowed=int(np.floor(.01*len(bg)));thr_v3=float(np.nextafter(bg[-1-allowed],np.float32(np.inf)));thr_bgA=float(np.nextafter(bga[-1-int(np.floor(.01*len(bga)))],np.float32(np.inf)));thr=max(thr_v3,thr_bgA)")
s=s.replace('thr_fa1=thr,val_background_n=len(bg)', 'thr_fa1=thr,thr_v3=thr_v3,thr_bgA=thr_bgA,val_v3_background_n=len(bg),val_bgA_background_n=len(bga),val_background_n=len(bg)+len(bga)')
s=s.replace("scheduler=sched.state_dict())", "scheduler=sched.state_dict(),scaler=scaler.state_dict(),best=best,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())")
s=s.replace("dump(OUT/f'{tag}_", "dump(CACHE/f'{tag}_")
s=s.replace("done=dict(band=", "class PMC(ctypes.Structure):\n    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(x,ctypes.c_size_t) for x in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]\npm=PMC();pm.cb=ctypes.sizeof(pm);ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1),ctypes.byref(pm),pm.cb)\ndone=dict(peak_host_bytes=pm.PeakWorkingSetSize,band=")
(O/'train.py').write_text(s,encoding='utf-8')
(O/'train_vs_H.diff').write_text(''.join(difflib.unified_diff(src.splitlines(True),s.splitlines(True),fromfile='TaskH/train.py',tofile='TaskJ/train.py')),encoding='utf-8')
