# -*- coding: utf-8 -*-
"""Final OOXML compatibility repair after WPS save, including native label overlays.
WPS changes some fixed text-box heights during SaveAs. Never resave the repaired
package in WPS: reopen read-only for preview and audit only.
"""
from pathlib import Path
import zipfile,xml.etree.ElementTree as E,json,re,posixpath,hashlib,copy
B=Path(__file__).resolve().parent;D=B.parent;A=B/'r3_assets';N=B/'r4_assets'
SRC=D/'中期答辩_R4.pptx';OUT=B/'r4_work.pptx'
if not SRC.exists():SRC=N/'draft_round2.pptx'
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
for k,v in ns.items():E.register_namespace(k,v)
def q(k):p,n=k.split(':');return '{'+ns[p]+'}'+n
def text(sp):return ''.join(t.text or '' for t in sp.findall('.//a:t',ns))
def settext(sp,t):
 ts=sp.findall('.//a:t',ns)
 if ts:
  if '　' in t:
   rs=[r for r in sp.findall('.//a:r',ns) if r.find('a:t',ns) is not None and r.find('a:t',ns).text]
   headpr=copy.deepcopy(rs[0].find('a:rPr',ns));bodypr=copy.deepcopy(rs[-1].find('a:rPr',ns));p=sp.find('p:txBody/a:p',ns)
   for r in sp.findall('.//a:r',ns):
    rr=r.find('a:t',ns)
    if rr is not None:rr.text=''
   head,body=t.split('　',1)
   for label,pr in [(head+'　',headpr),(body,bodypr)]:
    r=E.SubElement(p,q('a:r'))
    if pr is not None:r.append(pr)
    if label==body:
     pr=r.find('a:rPr',ns)
     if pr is not None:
      pr.set('b','0');fill=pr.find('a:solidFill/a:srgbClr',ns)
      if fill is not None:fill.set('val','404040')
    E.SubElement(r,q('a:t')).text=label
   return
  ts[0].text=t
  for tt in ts[1:]:tt.text=''
def shape(root,x,y,w,h,t='',size=10,fill=None,color='404040'):
 tree=root.find('p:cSld/p:spTree',ns);ids=[int(v.get('id','0')) for v in root.findall('.//p:cNvPr',ns)];sp=E.SubElement(tree,q('p:sp'));nv=E.SubElement(sp,q('p:nvSpPr'));E.SubElement(nv,q('p:cNvPr'),{'id':str(max(ids+[0])+1),'name':'r4_native_label'});E.SubElement(nv,q('p:cNvSpPr'),{'txBox':'1'});E.SubElement(nv,q('p:nvPr'))
 pr=E.SubElement(sp,q('p:spPr'));xf=E.SubElement(pr,q('a:xfrm'));E.SubElement(xf,q('a:off'),{'x':str(round(x*12700)),'y':str(round(y*12700))});E.SubElement(xf,q('a:ext'),{'cx':str(round(w*12700)),'cy':str(round(h*12700))});geo=E.SubElement(pr,q('a:prstGeom'),{'prst':'rect'});E.SubElement(geo,q('a:avLst'))
 if fill:E.SubElement(E.SubElement(pr,q('a:solidFill')),q('a:srgbClr'),{'val':fill})
 else:E.SubElement(pr,q('a:noFill'))
 E.SubElement(E.SubElement(pr,q('a:ln')),q('a:noFill'))
 tx=E.SubElement(sp,q('p:txBody'));E.SubElement(tx,q('a:bodyPr'),{'wrap':'square','lIns':'0','rIns':'0','tIns':'0','bIns':'0'});E.SubElement(tx,q('a:lstStyle'));p=E.SubElement(tx,q('a:p'));r=E.SubElement(p,q('a:r'));rp=E.SubElement(r,q('a:rPr'),{'lang':'zh-CN','sz':str(round(size*100))});E.SubElement(E.SubElement(rp,q('a:solidFill')),q('a:srgbClr'),{'val':color});E.SubElement(rp,q('a:latin'),{'typeface':'Arial'});E.SubElement(rp,q('a:ea'),{'typeface':'微软雅黑'});E.SubElement(r,q('a:t')).text=t
 return sp

aud=json.loads((N/'audit_round2.json').read_text(encoding='utf-8'))
norm=lambda s:re.sub(r'[\r\n]','',s)
fixes={d['page']:{norm(o['text']):o for o in d['overflow']} for d in aud}
replots={hashlib.sha256((A/name).read_bytes()).hexdigest():N/name for name in ['source_probe.png','wifi_focus.png','fusion.png','window_miss.png','window_recall.png']}
bg_hash=hashlib.sha256((A/'background_clean.png').read_bytes()).hexdigest();fa_hash=hashlib.sha256((A/'background_fa_clean.png').read_bytes()).hexdigest()
summary='探索性实验确定了处理流程和评测原则，正式实验在课题组采集的实测数据上开展。'
new7='两批实测数据分别在实验室和真实环境采集，均包含无人机与遥控器之间的通信信号。实验室数据覆盖待机与飞行，包含不同图传带宽；真实环境数据同时录有无人机和不含无人机的环境背景，日常无线通信干扰普遍存在。两批数据由不同设备录制，因此实验分别统计来源、核验设备杂散，再通过保持信号不变的背景替换检查环境影响。中间表给出采样、型号与规模，右侧显示来源与类别的分布。'
with zipfile.ZipFile(SRC) as z:
 entries={n:z.read(n) for n in z.namelist()}
 note_pages={}
 for name,data in list(entries.items()):
  if name.startswith('ppt/media/'):
   match=replots.get(hashlib.sha256(data).hexdigest())
   if match:entries[name]=match.read_bytes()
 for name,data in list(entries.items()):
  m=re.fullmatch(r'ppt/slides/slide(\d+)\.xml',name)
  if not m:continue
  i=int(m[1]);root=E.fromstring(data)
  relname=posixpath.dirname(name)+'/_rels/'+posixpath.basename(name)+'.rels';rels=E.fromstring(entries[relname]);relmap={r.get('Id'):posixpath.normpath(posixpath.join('ppt/slides',r.get('Target'))) for r in rels}
  for r in rels:
   if r.get('Type','').endswith('/notesSlide'):note_pages[posixpath.normpath(posixpath.join('ppt/slides',r.get('Target')))]=i
  for sp in root.findall('.//p:sp',ns):
   old=text(sp);xf=sp.find('p:spPr/a:xfrm',ns)
   if i==2:
    for tt in sp.findall('.//a:t',ns):
     if tt.text:tt.text=tt.text.replace('公开数据集阶段','探索性实验（公开数据集）').replace('实测数据阶段','实测数据实验')
    continue
   if norm(old) in fixes.get(i,{}) and xf is not None:
    o=fixes[i][norm(old)];ext=xf.find('a:ext',ns);ext.set('cx',str(round(max(int(ext.get('cx'))/12700,o['bound'][0]+2)*12700)));ext.set('cy',str(round(max(int(ext.get('cy'))/12700,o['bound'][1]+2)*12700)))
   t=old.replace('公开数据集阶段','探索性实验').replace('实测数据阶段','实测数据实验')
   if i==2:t=t.replace('公开数据集','探索性实验（公开数据集）')
   if i==4 and old=='公开数据集与基线方法':t='探索性实验：公开数据集上的基线'
   if i in [5,6] and old in ['高分背后的依据：信号周期与频率位置','由此确立的评测原则']:t='探索性实验 · '+old
   if i in [4,5,6] and old.startswith('层次'):t='探索性实验 · '+t
   if i==3 and old.startswith('一 能否识别'):t='一 能否识别'
   if i==3 and old.startswith('技术路线'):t=t.replace('随后转向实测数据','正式实验在实测数据开展')
   if i==3 and old=='公开数据集阶段':t='研究目标与脉络'
   if i==7 and old.startswith('研究如何转向实测数据'):t='实测数据概况　'+new7
   if i==7 and old.startswith('DroneRFa，'):t='两批实测数据按设备来源分别统计'
   if i in [6,7] and ('带着这些原则转向' in old or '按录音、分来源、单因素对照原则 → 按录音划分与单因素' in old):
    t=summary if not old.startswith('形成') else '形成按录音、分来源、单因素对照原则 → '+summary
   if i==6 and old.startswith('归一化：'):t='WiSig归一化以避免功率含位置信息；Glüge混合背景后，−12 dB以上至少85%；本项目沿用功率归一化。'
   if i==14 and old.startswith('改进方法'):
    t=t.replace('沿用WiSig防止功率形成位置信息与Glüge混合前归一化的做法','WiSig跨设备发射机识别由99%降至33%以下，Glüge混合背景在−12 dB以上至少85%；本项目参照两者先归一化')
   if i==14 and old.startswith('效果与启示'):t='效果与启示　筛选得到两种输入方案，尚不认定融合有效。右表列出含义、公式及保留情况；下一页用置零与两种打乱对照，区分结构变化和特征信息的作用。'
   if i==20:t=t.replace('落实引入中心频率的建议；','').replace('沿用按信号频带截取的时频图','采用信号频带时频图')
   if i==9 and xf is not None and 170<int(xf.find('a:off',ns).get('y'))/12700<420:
    for pr in sp.findall('.//a:rPr',ns):pr.set('sz','1000')
   if i==3 and old in ['公开基线85.4%','平移后0–4%']:
    # Add explicit exploratory positioning on the roadmap, keeping the measurements unchanged.
    t='探索性实验\n'+('公开基线85.4%' if old.startswith('公开') else '平移后0–4%')
   if i==5 and old.startswith('分析与验证') and 'DroneRFa' in old:
    t+='因此先检查窗口中是否包含发射，再评估网络是否识别。观测不足和判断依据变化分别解释不同现象，不能用总体准确率替代这两类核验。'
   if i==5 and xf is not None and int(xf.find('a:off',ns).get('x'))/12700<480 and old.startswith(('改进方法','效果与启示')):
    off=xf.find('a:off',ns);off.set('y',str(int(off.get('y'))+22*12700))
   if t!=old:settext(sp,t)
   if xf is not None and i==3 and old in ['公开基线85.4%','平移后0–4%']:
    off=xf.find('a:off',ns);off.set('y',str(int(off.get('y'))-10*12700));xf.find('a:ext',ns).set('cy',str(58*12700))
   if i==5 and old.startswith('分析与验证') and 'DroneRFa' in old and xf is not None:xf.find('a:ext',ns).set('cy',str(80*12700))
   if i==7 and old.startswith('研究如何转向') and xf is not None:xf.find('a:ext',ns).set('cy',str(220*12700))
   if i==14 and old.startswith('改进方法') and xf is not None:
    xf.find('a:ext',ns).set('cy',str(74*12700));off=xf.find('a:off',ns);off.set('y',str(int(off.get('y'))-6*12700))
   if i==14 and old.startswith('效果与启示') and xf is not None:
    off=xf.find('a:off',ns);off.set('y',str(int(off.get('y'))+1*12700));xf.find('a:ext',ns).set('cy',str(56*12700))
   if i==6 and old.startswith('归一化：') and xf is not None:
    off=xf.find('a:off',ns);off.set('y',str(int(off.get('y'))-8*12700));xf.find('a:ext',ns).set('cy',str(32*12700))
   # Remove excessive spacing in new plan pages; distribute each text block over the available vertical space.
   bp=sp.find('p:txBody/a:bodyPr',ns)
   if bp is not None:
    bp.set('wrap','square')
  # Native overlays replace obsolete R3 labels without altering any plotted values.
  for pic in root.findall('.//p:pic',ns):
   blip=pic.find('.//a:blip',ns);path=relmap.get(blip.get(q('r:embed')))
   if not path or path not in entries:continue
   digest=hashlib.sha256(entries[path]).hexdigest();xf=pic.find('p:spPr/a:xfrm',ns)
   if xf is None:continue
   off=xf.find('a:off',ns);ext=xf.find('a:ext',ns);x=int(off.get('x'))/12700;y=int(off.get('y'))/12700;w=int(ext.get('cx'))/12700;h=int(ext.get('cy'))/12700
   if digest==bg_hash:
    shape(root,x,y,w,h*60/577,fill='FFFFFF');shape(root,x,y,w/2,12,'实验室无人机',9);shape(root,x+w/2,y,w/2,12,'真实环境无人机',9)
    shape(root,x+w*630/1800,y+h*414/577,w*196/1800,h*80/577,fill='FFFFFF')
    shape(root,x,y+h+2,w,12,'蓝：白噪声；橙：真实环境背景',9)
   if digest==fa_hash:
    shape(root,x+w*110/1285,y,w*280/1285,h*113/895,fill='FFFFFF');shape(root,x+w*.1,y+2,w*.88,22,'蓝：首轮；橙：加入同设备背景',9)
    shape(root,x,y+h*780/895,w,h*115/895,fill='FFFFFF')
    for k,(xx,lab) in enumerate(zip([170,465,685,905,1125],['真实背景','同设备背景','安静录音','白噪声','陷波白噪'])):shape(root,x+w*xx/1285-25,y+h*798/895,51,20,lab,9)
  if i==19:
   flows=[(35,263,430,'公开录音 → 功率与频率匹配 → 两类叠加 → 痕迹检验'),(495,263,430,'已知混合频带 → 开关相关性切分 → 边缘与错并核验'),(35,438,430,'0 dB → −5 dB → −10 dB → −15 dB：分档报告'),(495,438,430,'两种融合输入 → 结构一致对照 → 误报置信上界')]
   for x,y,w,t in flows:shape(root,x,y,w,27,t,11,'E8F2FA','0079BA')
  if i==20:
   flows=[(35,267,'信号频带＋独立中心频率 → 型号'),(340,267,'频率平移训练 → 平移测试'),(645,267,'周期分析 → 长窗／连续片段投票'),(35,448,'跨设备／带宽／状态 → 分条件评估'),(340,448,'留出未知型号 → 阈值拒识'),(645,448,'检测 → 型号 → 整体混淆矩阵')]
   for x,y,t in flows:shape(root,x,y,280,25,t,10,'E8F2FA','0079BA')
  entries[name]=E.tostring(root,encoding='utf-8',xml_declaration=True)
 # Recompose all speaker notes with the latest positioning and no data-switch rationale.
 for name,data in list(entries.items()):
  m=re.fullmatch(r'ppt/notesSlides/notesSlide(\d+)\.xml',name)
  if not m:continue
  root=E.fromstring(data)
  for t in root.findall('.//a:t',ns):
   if t.text:t.text=t.text.replace('公开数据集阶段','探索性实验').replace('实测数据阶段','实测数据实验').replace('再将周期分析、按录音划分与单因素对照延续到实测数据','正式实验沿用周期分析、按录音划分与单因素对照')
  if note_pages.get(name)==7:
   for sp in root.findall('.//p:sp',ns):
    if len(text(sp))>100:settext(sp,'这一页先让各位老师了解实测数据的组成。'+summary+'实验室数据与真实环境数据由不同设备采集，包含的型号、环境与背景比例不同。实验室覆盖待机、飞行和不同图传带宽，真实环境同时包含无人机及不含无人机的环境背景，日常干扰普遍存在。需要关注的是来源与类别的相关性：背景全部来自真实环境，而2.4 GHz无人机大多来自实验室。因此后续统一处理后仍需按来源统计，并用对照实验检查环境与设备信息是否影响判断。接下来先说明统一输入流程，再介绍固定的网络与训练设置。')
  entries[name]=E.tostring(root,encoding='utf-8',xml_declaration=True)
 with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as out:
  for name,data in entries.items():out.writestr(name,data)
print('repaired package',OUT)
