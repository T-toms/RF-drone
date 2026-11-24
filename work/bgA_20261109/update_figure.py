from pathlib import Path
p=Path('work/bgA_20261109/figure.py');s=p.read_text(encoding='utf-8-sig');a=s.index('cells=np.load');z=s.index('ax.set(xlabel=\'MHz\',ylabel=\'功率',a)
new='''r=next(v for v in c.load(O/'pilot_shape_p10.json') if v['block']==Path(b['block_file']).stem and v['s']==0);xx=[v['lo_mhz']+.5 for v in r['subbands']];ax.plot(xx,[v['drone_p10'] if v['status']=='judged' else np.nan for v in r['subbands']],label='无人机侧 p10（已判子带）');ax.plot(xx,[v['bg_p10'] for v in r['subbands']],label='镜像背景 p10 s=0');bs=c.load(P/'work/crop_slice_20261107/plan.json')
for key in ['20260710','20260709']:
 ref=next(v for v in bs if v['dataset']=='v3' and v['label']=='background' and v['band']=='2.4GHz' and key in v['group_id']);cc=np.load(P/'work/crop_slice_20261107/cells'/Path(ref['block_file']).stem/'mean.npy');yy=[]
 for lo in range(2405,2455):
  ix=sel&(freq>=lo)&(freq<lo+1);yy.append(np.percentile(cc[:,ix],10))
 ax.plot(xx,yy,label='v3 '+key+' p10')
'''
s=s[:a]+new+s[z:];s=s.replace("title=f'1 MHz 平均谱；安静cell N={quiet.sum()}'","title=f'1 MHz p10；已判子带 N={r[\"n_judged\"]}'");Path('work/bgA_20261109/figure_final.py').write_text(s,encoding='utf-8')
