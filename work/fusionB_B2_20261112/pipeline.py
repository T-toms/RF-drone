import sys,subprocess,time,json,math
from pathlib import Path
sys.dont_write_bytecode=True
from base import *
PY=sys.executable
assert (OUT/'cache_done.json').exists() and (OUT/'protected_B2b_before.json').exists()
start=time.time()
with (OUT/'run.log').open('a',encoding='utf-8') as log:
 def run(args):
  if args[0]=='train.py':
   b=args[args.index('--band')+1];g=args[args.index('--group')+1];seed=args[args.index('--seed')+1];suffix='_smoke' if '--epochs' in args else ''
   if (OUT/f'{g}_{b}_seed{seed}{suffix}_done.json').exists():return
  print('RUN',args,flush=True);log.write('\nRUN '+repr(args)+'\n');log.flush();subprocess.run([PY,'-B',str(OUT/args[0])]+args[1:],stdout=log,stderr=subprocess.STDOUT,check=True)
 smoke=[]
 for b in ['2.4GHz','5.8GHz']:
  for g in ['F','F5']:
   run(['train.py','--band',b,'--group',g,'--seed','2026','--epochs','3'])
   rr=rows(OUT/f'{g}_{b}_seed2026_smoke_epochs.csv');assert len(rr)==3 and all(math.isfinite(float(r[k])) for r in rr for k in ['train_loss','val_loss'])
   smoke.append(dict(band=b,group=g,epochs=3,finite_losses=True))
 dump(OUT/'smoke_checks.json',smoke)
 for s in [2026,2027,2028]:
  for b in ['2.4GHz','5.8GHz']:
   for g in ['R','Z','F','F5']:
    assert sum(p.stat().st_size for d in [CACHE,OUT] for p in d.rglob('*') if p.is_file())<20*1024**3
    run(['train.py','--band',b,'--group',g,'--seed',str(s)])
  if s==2026:
   if not (OUT/'interim_seed2026.csv').exists():run(['interim.py'])
   if not (OUT/'interim_sent.json').exists():
    body=(OUT/'interim_seed2026.md').read_text(encoding='utf-8')
    r=subprocess.run(['C:/Users/Administrator/AppData/Local/Programs/orca/resources/bin/orca.exe','orchestration','ask','--from','term_b0dbf216-935e-44f8-acbd-ee8eef552634','--question',body,'--timeout-ms','1'],capture_output=True,text=True,encoding='utf-8')
    dump(OUT/'interim_sent.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
 if not (OUT/'evaluation_done.json').exists():run(['evaluate.py'])
 for g in ['R','Z','F','F5']:
  if not (OUT/f'gradcam_done_{g}.json').exists():run(['gradcam.py',g])
 run(['finish_b2b.py'])
dump(OUT/'pipeline_done.json',dict(seconds=time.time()-start))
