# -*- coding: utf-8 -*-
from pathlib import Path
import json,zipfile,hashlib,xml.etree.ElementTree as E
B=Path(__file__).resolve().parent;N=B/'r4_assets';D=B.parent
before=json.loads((N/'desktop_before.json').read_text(encoding='utf-8-sig'));after=json.loads((N/'desktop_after.json').read_text(encoding='utf-8-sig'))
doc='C:/Users/Administrator/Desktop/中期检查表.doc';target='C:/Users/Administrator/Desktop/中期答辩_模板稿.pptx'
assert before[doc]==after[doc]
assert not Path('C:/Users/Administrator/Desktop/~$中期答辩_模板稿.pptx').exists()
with zipfile.ZipFile(target) as z:
 root=E.fromstring(z.read('ppt/presentation.xml'));pages=len(root.find('{http://schemas.openxmlformats.org/presentationml/2006/main}sldIdLst'))
assert pages==21
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(target)==sha(B/'r4_work.pptx')
checks=json.loads((N/'checks.json').read_text(encoding='utf-8'));checks.update({'desktop_delivery':'complete','desktop_pages':pages,'desktop_size':after[target]['size'],'desktop_sha256':sha(target),'desktop_form_metadata_unchanged':True,'preview_rounds':[1,2,3,4,5,6,7]});(N/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
p=B/'REPORT_R4.md';t=p.read_text(encoding='utf-8').replace('最终复开工作PPT只读导出全部21页','最终第7轮复开工作PPT只读导出全部21页').replace('DroneRFa：时频图ResNet18与97.73%基线，对齐探索方法。','DroneRFa：时频图ResNet18基线，对齐探索方法（基线性能已在第4页说明）。')
t+='\n## 桌面交付核对（已完成）\n\n覆盖前锁文件不存在，WPS/wpp进程无该文档可见窗口；未关闭用户窗口。全部自查通过后复制工作文件写回桌面原PPT，复制后直接读包核对21页，并与工作副本SHA256相同。没有将桌面原文件用COM打开或另存。\n\n|文件|覆盖前大小|交付后大小|原修改时间|交付后修改时间|结论|\n|---|---:|---:|---|---|---|\n'
for path in [target,doc]:t+=f"|{path}|{before[path]['size']}|{after[path]['size']}|{before[path]['modified']}|{after[path]['modified']}|{'按3B有意更新，21页' if path==target else '大小和修改时间完全未变'}|\n"
t+='\n最终PPT SHA256：`'+sha(target)+'`。报告目录不保留另一个R4交付文件；第2轮草稿归档到r4_assets/draft_round2.pptx，供r4_repair复现。第7轮仅修复目录段落保留与网络图标签字号，已分别打开最终第2、9页复查；其他页与第6轮一致。\n'
p.write_text(t,encoding='utf-8');print('DELIVERED',pages,after[target]['size'],sha(target),'FORM UNCHANGED')
