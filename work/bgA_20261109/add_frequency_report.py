from pathlib import Path
p=Path('work/bgA_20261109/report.py');s=p.read_text(encoding='utf-8-sig');s=s.replace("'局限：源2345", "'频率空隙复现：2.4G 2342.6–2393.5MHz共424bin，其中413bin的跨录制最大占用率<1%，最大2.462%；2393.5–2398.6MHz最大23.385%；5.8G 1024bin中共同<1%为0，跨录制最大占用率的最小值29.7%。','局限：源2345");p.write_text(s,encoding='utf-8')
