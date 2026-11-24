from pathlib import Path
p=Path('work/bgA_20261109/report.py');s=p.read_text(encoding='utf-8-sig');s=s.replace('短块偏移范围由协调者确认；未改F/H文件','短块偏移范围由协调者确认；argmax平票按第一轮归背景；日志已统一UTF8；未改F/H文件');p.write_text(s,encoding='utf-8')
