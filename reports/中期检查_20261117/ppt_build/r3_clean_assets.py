from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
B=Path(__file__).resolve().parent;A=B/'r3_assets';R=B.parents[2]
f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',30)
im=Image.open(B/'background.png').convert('RGB');d=ImageDraw.Draw(im);d.rectangle((0,0,1800,60),fill='white');d.text((120,12),'2.4 GHz：实验室无人机（训练见过的型号）',font=f,fill='#404040');d.text((1040,12),'2.4 GHz：城区无人机',font=f,fill='#404040');d.rectangle((630,414,825,493),fill='white');d.line((640,438,680,438),fill='#277fcb',width=5);d.text((690,417),'白噪声',font=f,fill='#404040');d.line((640,479,680,479),fill='#ed7229',width=5);d.text((690,455),'城区背景',font=f,fill='#404040');im.save(A/'background_clean.png')
im=Image.open(A/'bgA_fa.png').convert('RGB');d=ImageDraw.Draw(im);d.rectangle((0,0,54,895),fill='white');d.text((4,330),'误\n报\n率\n（%）',font=f,fill='#404040');d.rectangle((100,780,292,894),fill='white');d.text((111,806),'城区背景',font=f,fill='#404040');im.save(A/'background_fa_clean.png')
im=Image.open(A/'mirror_feature.png');out=Image.new('RGB',(760,360),'white');out.paste(im.crop((58,50,651,286)),(90,60));d=ImageDraw.Draw(out);d.text((160,12),'谱平坦度：同内容背景的分布',font=f,fill='#404040');d.text((150,305),'实验室接收机',font=f,fill='#404040');d.text((470,305),'城区接收机',font=f,fill='#404040');out.save(A/'mirror_clean.png')
im=Image.open(R/'work/fusionB_B2_20261112/fig0_input_examples_2.4GHz.png');print('input size',im.size)
im.resize((1200,int(im.height*1200/im.width))).save(A/'input_source_review.png')
# Retain plot pixels; only labels above were translated, not data.
