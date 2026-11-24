import sys,subprocess,time,json
from pathlib import Path
O=Path(__file__).resolve().parent
with (O/'run.log').open('a',encoding='utf-8',buffering=1) as log:
    for s,n in [(2026,3),(2026,40),(2027,40),(2028,40)]:
        p=subprocess.run([sys.executable,'-I','-B',str(O/'train.py'),'--band','2.4GHz','--seed',str(s),'--epochs',str(n)],stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'training failed {s}/{n}: {p.returncode}')
        if n==3:
            d=json.loads(Path('E:/drone_lab_stage1_L20_bgA_20261109/2.4GHz_seed2026_smoke_done.json').read_text())
            assert d['peak_host_bytes']<12*1024**3 and d['seconds']<600,d
    (O/'training_complete.json').write_text(json.dumps(dict(completed=time.time())),encoding='utf-8')
