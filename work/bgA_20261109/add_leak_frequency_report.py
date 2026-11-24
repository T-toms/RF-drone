from pathlib import Path
p=Path('work/bgA_20261109/report.py');s=p.read_text(encoding='utf-8-sig');s=s.replace("'image_pass']} for r in images]","'image_pass']} | {'failed_1mhz':';'.join(f'{v[\"lo_mhz\"]}:{v[\"delta20\"]:.6f}:n={v[\"n_hot\"]}' for v in r['subbands'] if v['status']=='judged' and v['delta20'] is not None and v['delta20']>1)} for r in images]");p.write_text(s,encoding='utf-8')
