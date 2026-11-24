from pathlib import Path
import json
B=Path(__file__).resolve().parent
for x in json.loads((B/'r3_audit.json').read_text(encoding='utf-8')):
 print(x['page'],x['chars'],x['note_chars'])
 for v in x['overflow']:print(v)
