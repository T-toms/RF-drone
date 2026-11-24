from pathlib import Path
for name in ['evaluate_bg.py','quiet013.py','raw_inference.py']:
 p=Path('work/bgA_20261109')/name;s=p.read_text(encoding='utf-8-sig');s=s.replace('pred_argmax=int(pd>=.5)','pred_argmax=int(pd>.5)').replace('pred_argmax=int(v>=.5)','pred_argmax=int(v>.5)').replace('(pr[::2]>=.5)==(pr[1::2]>=.5)','(pr[::2]>.5)==(pr[1::2]>.5)');p.write_text(s,encoding='utf-8')
