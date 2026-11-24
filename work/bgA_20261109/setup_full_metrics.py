from pathlib import Path
p=Path('work/bgA_20261109/revised_final.py');s=p.read_text(encoding='utf-8-sig');s=s.replace("ss=shape(b,path)","ss={}").replace("r['pass']=img and ss['shape_pass']","r['pass']=img")
Path('work/bgA_20261109/full_image_metrics.py').write_text(s,encoding='utf-8')
