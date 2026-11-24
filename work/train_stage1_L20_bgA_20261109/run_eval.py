import sys,time,subprocess
from pathlib import Path
O=Path(__file__).resolve().parent
while not (O/'training_complete.json').exists():time.sleep(5)
with (O/'evaluation.log').open('a',encoding='utf-8',buffering=1) as log:
    for name in ['evaluate.py','finish.py','verify_outputs.py']:
        subprocess.run([sys.executable,'-I','-B',str(O/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
