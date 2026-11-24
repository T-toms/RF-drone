from pathlib import Path
p=Path('work/bgA_20261109/report.py');s=p.read_text(encoding='utf-8-sig');s=s.replace("f'{r[\"positive\"]}/{r[\"N\"]} ({pct(r[\"cp95_upper\"])})'","f'{r[\"positive\"]}/{r[\"N\"]}={pct(r[\"rate\"])} ({pct(r[\"cp95_upper\"])})'")
s=s.replace("{r[\"positive\"]}/{r[\"N\"]}，上界","{r[\"positive\"]}/{r[\"N\"]}={pct(r[\"rate\"])}，上界");s=s.replace("{r[\"k\"]}/{r[\"N\"]},上界","{r[\"k\"]}/{r[\"N\"]}={pct(r[\"rate\"])},上界")
p.write_text(s,encoding='utf-8')
