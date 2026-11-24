"""Sequential resumable background stages; no scheduled wakeups and no training."""
from pathlib import Path
import json,subprocess,sys,datetime
sys.stdout.reconfigure(encoding='utf-8');sys.stderr.reconfigure(encoding='utf-8')
O=Path(__file__).resolve().parent
def main():
 smoke=json.loads((O/'smoke_verification.json').read_text(encoding='utf-8'))
 assert len(smoke['notch']['samples'])>=39,'Complete smoke three-snippet checks before full run'
 assert all(smoke[k]['status']=='pass' for k in ['v3_contiguity','filter_chain','recompute','notch'])
 stages=[('crop.py','full'),('finalize.py',),('verify.py','full'),('report.py',)]
 with (O/'run.log').open('a',encoding='utf-8',buffering=1) as log:
  for stage in stages:
   state=dict(stage=list(stage),time=datetime.datetime.now().isoformat(),status='running');(O/'pipeline_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8');log.write('\nSTAGE '+repr(stage)+' '+state['time']+'\n');log.flush()
   p=subprocess.run([sys.executable,'-I','-u',str(O/stage[0]),*stage[1:]],stdout=log,stderr=log)
   if p.returncode:
    state.update(status='failed',returncode=p.returncode);(O/'pipeline_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8');return p.returncode
  state.update(status='complete',time=datetime.datetime.now().isoformat());(O/'pipeline_state.json').write_text(json.dumps(state,indent=2),encoding='utf-8');log.write('PIPELINE_COMPLETE\n')
 return 0
if __name__=='__main__':sys.exit(main())
