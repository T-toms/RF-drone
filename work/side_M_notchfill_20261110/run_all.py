from pathlib import Path
import subprocess,os,sys
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1')
with (O/'run.log').open('a',encoding='utf-8') as log:
    for s,ep in [(2026,3),(2026,40),(2027,40),(2028,40)]:
        subprocess.run([sys.executable,'-I',str(O/'train.py'),'--band','2.4GHz','--seed',str(s),'--epochs',str(ep),'--variant','Lb'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for fn,args in [('evaluate.py',[]),('cross.py',[]),('finish.py',[]),('verify.py',['--final'])]:
        subprocess.run([sys.executable,'-I',str(O/fn)]+args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
