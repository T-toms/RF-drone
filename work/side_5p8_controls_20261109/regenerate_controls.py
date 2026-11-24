from pathlib import Path
import sys,os,time,shutil,subprocess,concurrent.futures,json
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parent))
from synthesis import *
from run_synthesis import task,TORCH
if __name__=='__main__':
 start=time.time();E=Path('E:/side_5p8_controls_20261109')
 # Windows is case-insensitive. Keep prior files and copy proxies to unambiguous names.
 for band in FS:
  for kind in ['A','B']:shutil.copyfile(E/f'{band}_{kind}_spec.npy',E/f'{band}_proxy_{kind}_spec.npy')
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
  for band in FS:
   for kind in ['a','b']:
    proc=subprocess.Popen([str(TORCH),'-I','-B',str(O/'torch_stage.py'),'stream',band,kind],stdin=subprocess.PIPE)
    for first in range(0,200,8):
     for x in pool.map(task,[(band,kind,i,None) for i in range(first,first+8)]):proc.stdin.write(x.tobytes())
    proc.stdin.close();assert proc.wait()==0
 (O/'control_regeneration.json').write_text(json.dumps(dict(reason='Windows filenames a/A and b/B alias; proxies copied, white controls regenerated under distinct semantic names',seconds=time.time()-start)),encoding='utf-8')
