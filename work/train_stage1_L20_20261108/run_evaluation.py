import sys,subprocess,os,time,json
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['PYTHONIOENCODING']='utf-8'
# Dependency wait observes only the six authorized training completion records.
while not all((P/f'{b}_seed{s}_done.json').exists() for b in ['2.4GHz','5.8GHz'] for s in [2026,2027,2028]):time.sleep(10)
with (P/'run.log').open('a',encoding='utf-8',buffering=1) as log:
    for script in ['evaluate.py','mix_test.py','gradcam.py','verify.py','write_report.py']:
        log.write('\nEVALUATION '+script+'\n');log.flush();subprocess.run([sys.executable,'-I',str(P/script)],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ)
