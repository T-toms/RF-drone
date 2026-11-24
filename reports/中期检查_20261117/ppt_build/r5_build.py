from copy import deepcopy
from pathlib import Path, PurePosixPath
import json
import posixpath
import re
import zipfile
from lxml import etree as E
from PIL import Image
from r5_content import draw, NOTES, TITLES

B=Path(__file__).resolve().parent
A=B/'r5_assets'
P='http://schemas.openxmlformats.org/presentationml/2006/main'
D='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PKG='http://schemas.openxmlformats.org/package/2006/relationships'
NS={'p':P,'a':D,'r':R}
TXT, BLUE, DARK, PALE, SKY, ORANGE, WARM, GRAY = '243744','0079BA','004B73','E8F2FA','CFE3F3','F0A30A','FFF4DB','5C6C76'
WHITE='FFFFFF'

def make(parent, ns, tag, **attrs):
    return E.SubElement(parent,f'{{{ns}}}{tag}',{k:str(v) for k,v in attrs.items()})

def emu(value):
    return str(round(value*12700))

def transform(parent,x,y,w,h,ns=D,**attrs):
    xf=make(parent,ns,'xfrm',**attrs)
    make(xf,D,'off',x=emu(x),y=emu(y))
    make(xf,D,'ext',cx=emu(w),cy=emu(h))
    return xf

def fill(parent,color):
    if color:
        make(make(parent,D,'solidFill'),D,'srgbClr',val=color)
    else:
        make(parent,D,'noFill')

def style(parent,size,color,bold=False):
    rp=make(parent,D,'rPr',lang='zh-CN',altLang='en-US',sz=round(size*100),b=int(bold))
    fill(rp,color)
    make(rp,D,'latin',typeface='Arial')
    make(rp,D,'ea',typeface='Microsoft YaHei')

def paragraphs(tb,text,size,color,bold,align,linespace=112):
    for line in text.split('\n'):
        p=make(tb,D,'p')
        pp=make(p,D,'pPr',algn=align)
        make(make(pp,D,'lnSpc'),D,'spcPct',val=linespace*1000)
        make(make(pp,D,'spcAft'),D,'spcPts',val=0)
        rr=make(p,D,'r')
        style(rr,size,color,bold)
        make(rr,D,'t').text=line

def textbody(parent,ns,text,size,color,bold,align,anchor='ctr',margin=0):
    tb=make(parent,ns,'txBody')
    make(tb,D,'bodyPr',lIns=emu(margin),rIns=emu(margin),tIns='0',bIns='0',anchor=anchor,wrap='square')
    make(tb,D,'lstStyle')
    paragraphs(tb,text,size,color,bold,align)
    return tb

def line_style(parent,color,weight=0.8,dash=False,arrow=False):
    ln=make(parent,D,'ln',w=emu(weight))
    fill(ln,color)
    if dash:
        make(ln,D,'prstDash',val='dash')
    if arrow:
        make(ln,D,'tailEnd',type='triangle',w='sm',len='sm')

class Canvas:
    def __init__(self,tree,rels,media,page):
        self.tree,self.rels,self.media,self.page=tree,rels,media,page
        self.ident=10
        self.metrics={'body':[],'sizes':[],'visual':[],'images':[],'boxes':[]}

    def node(self,tag,nvtag,role):
        self.ident+=1
        sh=make(self.tree,P,tag)
        nv=make(sh,P,'nv'+nvtag+'Pr')
        make(nv,P,'cNvPr',id=self.ident,name=f'R5 {role} {self.ident}')
        cnv=make(nv,P,'cNv'+nvtag+'Pr')
        make(nv,P,'nvPr')
        return sh,cnv

    def box(self,x,y,w,h,text='',size=14,color=TXT,bold=False,align='ctr',bg=None,
            edge=None,dash=False,role='图内',shape='rect',weight=.8,anchor='ctr',margin=4):
        sh,cnv=self.node('sp','Sp',role)
        if not bg and not edge and text:
            cnv.set('txBox','1')
        prop=make(sh,P,'spPr')
        transform(prop,x,y,w,h)
        make(make(prop,D,'prstGeom',prst=shape),D,'avLst')
        fill(prop,bg)
        line_style(prop,edge,weight,dash)
        if text:
            textbody(sh,P,text,size,color,bold,align,anchor,margin)
            self.metrics['sizes'].append(size)
            self.metrics['boxes'].append([x,y,w,h,text,role])
            if role=='正文':
                self.metrics['body'].append(text)
        return sh

    def text(self,x,y,w,h,text,size=14,color=TXT,bold=False,align='l',role='正文'):
        return self.box(x,y,w,h,text,size,color,bold,align,role=role,margin=0)

    def line(self,x,y,xx,yy,arrow=False,color='404040',weight=.85,dash=False):
        sh,_=self.node('cxnSp','CxnSp','连线')
        p=make(sh,P,'spPr')
        transform(p,min(x,xx),min(y,yy),abs(xx-x),abs(yy-y),
                  flipH=int(xx<x),flipV=int(yy<y))
        make(make(p,D,'prstGeom',prst='line'),D,'avLst')
        line_style(p,color,weight,dash,arrow)

    def arc(self,x,y,w,h):
        sh,_=self.node('cxnSp','CxnSp','镜像箭头')
        p=make(sh,P,'spPr')
        transform(p,x,y,w,h)
        g=make(p,D,'custGeom')
        for child in ('avLst','gdLst','ahLst','cxnLst'):
            make(g,D,child)
        make(g,D,'rect',l='l',t='t',r='r',b='b')
        path=make(make(g,D,'pathLst'),D,'path',w=1000,h=1000,fill='none')
        make(make(path,D,'moveTo'),D,'pt',x=0,y=1000)
        curve=make(path,D,'cubicBezTo')
        for xx,yy in [(90,0),(910,0),(1000,1000)]:
            make(curve,D,'pt',x=xx,y=yy)
        line_style(p,'404040',.9,False,True)

    def group(self,x,y,w,h,name,accent=BLUE,bg=None):
        self.box(x,y,w,h,bg=bg,edge='748895',dash=True,shape='roundRect')
        self.text(x+12,y+7,w-24,24,name,14,accent,True,role='图内')

    def visual(self,x,y,w,h):
        self.metrics['visual'].append([x,y,w,h])

    def pic(self,file,x,y,w,h):
        file=Path(file)
        if not file.is_absolute():
            file=A/file
        with Image.open(file) as im:
            ratio=min(w/im.width,h/im.height)
            ww,hh=im.width*ratio,im.height*ratio
        x+=(w-ww)/2
        y+=(h-hh)/2
        key=f'r5_{file.stem}{file.suffix}'
        self.media['ppt/media/'+key]=file
        rel=f'rIdR5_{self.ident}'
        make(self.rels,PKG,'Relationship',Id=rel,Type=R+'/image',Target='../media/'+key)
        sh,_=self.node('pic','Pic','图像')
        bf=make(sh,P,'blipFill')
        b=make(bf,D,'blip')
        b.set(f'{{{R}}}embed',rel)
        make(make(bf,D,'stretch'),D,'fillRect')
        prop=make(sh,P,'spPr')
        transform(prop,x,y,ww,hh)
        make(make(prop,D,'prstGeom',prst='rect'),D,'avLst')
        self.metrics['images'].append([file.name,x,y,ww,hh])
        return (x,y,ww,hh)

    def table(self,x,y,widths,heights,rows,size=12):
        sh,_=self.node('graphicFrame','GraphicFrame','表格')
        transform(sh,x,y,sum(widths),sum(heights),P)
        g=make(make(sh,D,'graphic'),D,'graphicData',uri=D.replace('/main','/table'))
        tbl=make(g,D,'tbl')
        make(tbl,D,'tblPr',firstRow=0,bandRow=0)
        grid=make(tbl,D,'tblGrid')
        for width in widths:
            make(grid,D,'gridCol',w=emu(width))
        for i,(row,h) in enumerate(zip(rows,heights)):
            tr=make(tbl,D,'tr',h=emu(h))
            for j,word in enumerate(row):
                tc=make(tr,D,'tc')
                textbody(tc,D,word,size,WHITE if i==0 else TXT,i==0 or j==0,'l' if j==0 else 'ctr')
                prop=make(tc,D,'tcPr',marL=emu(7),marR=emu(7),marT=emu(2),marB=emu(2),anchor='ctr')
                for side in ('lnL','lnR','lnT','lnB'):
                    fill(make(prop,D,side,w=emu(.6)),'CFDEE8')
                fill(prop,BLUE if i==0 else (PALE if i%2 else WHITE))
        self.metrics['sizes'].append(size)

def relpath(part):
    f=PurePosixPath(part)
    return str(f.parent/'_rels'/(f.name+'.rels'))

def target(part,rel):
    return posixpath.normpath(posixpath.join(posixpath.dirname(part),rel.get('Target')))

def xml(node):
    return E.tostring(node,encoding='UTF-8',xml_declaration=True,standalone=True)

def build():
    backup=B/'r4b_final_backup.pptx'
    output=B/'r5_work.pptx'
    mods,media,audit={}, {}, {}
    with zipfile.ZipFile(backup) as src:
        prespart='ppt/presentation.xml'
        pres=E.fromstring(src.read(prespart))
        links=E.fromstring(src.read(relpath(prespart)))
        idlst=pres.find(f'{{{P}}}sldIdLst')
        plinks={r.get('Id'):r for r in links}
        entries=list(idlst)
        assert len(entries)>=19
        dropped=set()
        for ent in entries[18:-1]:
            old=plinks[ent.get(f'{{{R}}}id')]
            slidepart=target(prespart,old)
            oldrels=E.fromstring(src.read(relpath(slidepart)))
            note=target(slidepart,next(v for v in oldrels if v.get('Type').endswith('/notesSlide')))
            dropped.update((slidepart,relpath(slidepart),note,relpath(note)))
            idlst.remove(ent)
            links.remove(old)
        mods[prespart]=xml(pres)
        mods[relpath(prespart)]=xml(links)
        types=E.fromstring(src.read('[Content_Types].xml'))
        for el in list(types):
            if (el.get('PartName') or '').lstrip('/') in dropped:
                types.remove(el)
        mods['[Content_Types].xml']=xml(types)
        for page,entry in enumerate(idlst,1):
            part=target(prespart,plinks[entry.get(f'{{{R}}}id')])
            slide=E.fromstring(src.read(part))
            tree=slide.find('.//p:spTree',NS)
            rp=relpath(part)
            rs=E.fromstring(src.read(rp))
            note=target(part,next(v for v in rs if v.get('Type').endswith('/notesSlide')))
            nr=E.fromstring(src.read(note))
            body=nr.xpath('.//p:sp[p:nvSpPr/p:nvPr/p:ph[@type="body"]]/p:txBody',namespaces=NS)[0]
            for para in body.findall('a:p',NS):
                body.remove(para)
            paragraphs(body,NOTES[page-1],12,TXT,False,'l')
            for field in nr.xpath('.//a:fld[@type="slidenum"]/a:t',namespaces=NS):
                field.text=str(page)
            mods[note]=xml(nr)
            if page==2:
                allowed=('研究背景与现状','研究方案','关键技术与阶段成果','创新点与下一步')
                pp=tree.xpath('.//p:txBody/a:p[a:r/a:t]',namespaces=NS)
                assert len(pp)==4
                for para,label in zip(pp,allowed):
                    rr=para.findall('a:r',NS)
                    for run in rr[1:]:
                        para.remove(run)
                    rr[0].find('a:t',NS).text=label
                mods[part]=xml(slide)
            elif 3<=page<=18:
                for sh in list(tree)[2:]:
                    tree.remove(sh)
                for rel in list(rs):
                    if not rel.get('Type').endswith(('/slideLayout','/notesSlide')):
                        rs.remove(rel)
                c=Canvas(tree,rs,media,page)
                c.text(35,12,880,39,TITLES[page-3],24,TXT,True,role='标题')
                c.text(895,520,28,13,str(page),11,GRAY,False,'r','页码')
                draw(c,page)
                audit[page]=c.metrics
                mods[part]=xml(slide)
                mods[rp]=xml(rs)
        app=E.fromstring(src.read('docProps/app.xml'))
        for e in app:
            if E.QName(e).localname in ('Slides','Notes'):
                e.text='19'
        mods['docProps/app.xml']=xml(app)
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as out:
            for path in src.namelist():
                if path not in dropped:
                    out.writestr(path,mods[path] if path in mods else src.read(path))
            for path,file in media.items():
                out.write(file,path)
    (B/'r5_layout.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output)

if __name__=='__main__':
    build()




