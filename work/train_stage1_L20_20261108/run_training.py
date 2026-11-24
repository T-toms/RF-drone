import sys,os,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
os.environ['PYTHONIOENCODING']='utf-8'
with (P/'run.log').open('a',encoding='utf-8',buffering=1) as log:
    for b in ['2.4GHz','5.8GHz']:
        for s in [2026,2027,2028]:
            cmd=[sys.executable,'-I',str(P/'train.py'),'--band',b,'--seed',str(s)]
            if '--smoke' in sys.argv:cmd+=['--epochs','3']
            log.write('\nSTART '+repr(cmd)+'\n');log.flush()
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ)
            if '--smoke' in sys.argv:sys.exit(0)
