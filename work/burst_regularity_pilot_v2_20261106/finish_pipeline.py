from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pilot import OUT,dump
from recluster import redo
from render import render
from verify_boundary import boundary
from final_qa import finish
results=[redo(r) for r in json.loads((OUT/'results.json').read_text(encoding='utf8'))]
dump(OUT/'results.json',results)
print('FINAL RECLUSTER COMPLETE',flush=True)
render()
boundary()
finish()
print('PIPELINE COMPLETE',flush=True)
