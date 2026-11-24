from pathlib import Path
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent.parent/'ppt_preview'
for rnd in [1,2]:
 files=list(P.glob(f'r3_round{rnd}_??.png'))
 if len(files)!=19:continue
 out=Image.new('RGB',(1920,1450),'#dfe5ea');dr=ImageDraw.Draw(out)
 for i,p in enumerate(sorted(files)):
  im=Image.open(p);x=i%4*480;y=i//4*290;out.paste(im.resize((480,270)),(x,y));dr.text((x+7,y+271),str(i+1),fill='black')
 out.save(P/f'contact_r3_round{rnd}.png')
 if rnd==2:out.save(P/'contact_r3.png')
