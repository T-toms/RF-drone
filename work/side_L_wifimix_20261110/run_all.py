import sys,subprocess,time,json
from pathlib import Path
O=Path(__file__).resolve().parent
D=Path('E:/side_L_wifimix_20261110')
def call(name,*args):
    subprocess.run([sys.executable,'-I','-B',str(O/name),*args],check=True)
try:
    while not (D/'cache/generation_done.json').exists():time.sleep(5)
    call('train.py','--band','2.4GHz','--seed','2026','--variant','La','--epochs','3')
    smoke=json.loads((D/'2.4GHz_La_seed2026_smoke_done.json').read_text(encoding='utf-8'))
    print('SMOKE VALIDATED',smoke,flush=True)
    for v in ['La','Lb']:
        for s in [2026,2027,2028]:call('train.py','--band','2.4GHz','--seed',str(s),'--variant',v)
    (O/'training_complete.json').write_text(json.dumps({'completed':True}),encoding='utf-8')
    call('controls.py');call('evaluate.py');call('finish.py')
except Exception as e:
    (O/'pipeline_failed.json').write_text(json.dumps({'error':str(e)}),encoding='utf-8');raise
