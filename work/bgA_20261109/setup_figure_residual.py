from pathlib import Path
p=Path('work/bgA_20261109/figure_final.py');s=p.read_text(encoding='utf-8-sig');a=s.index('r=next(v for v');z=s.index('ax.set(xlabel=\'MHz\',ylabel=\'功率',a)
new='''profiles=c.load(O/'full_shape_profile.json');assert len(profiles)==61;xx=np.arange(2405,2455)+.5;curves=[]
for r in profiles:
 level=r['level_offset'];yy=[v['p10_diff']-level if level is not None and v['quiet_fraction']>=.9 else np.nan for v in r['subbands']];curves.append(yy);ax.plot(xx,yy,color='gray',alpha=.3,lw=.6)
ax.plot(xx,np.nanmedian(np.array(curves),axis=0),color='black',lw=2.2,label='61块中位数');ax.axhline(.5,color='red',ls=':',lw=.8);ax.axhline(-.5,color='red',ls=':',lw=.8)
'''
s=s[:a]+new+s[z:];s=s.replace("ylabel='功率 (dB，未归一化)'","ylabel='去电平偏移后的 p10 差 (dB)'").replace("title=f'1 MHz p10；已判子带 N={r[\"n_judged\"]}'","title='61块形状残差；仅安静占比≥90%子带'")
Path('work/bgA_20261109/figure_residual.py').write_text(s,encoding='utf-8')
