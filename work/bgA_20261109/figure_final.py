from pathlib import Path
import sys,json,csv,importlib.util
sys.dont_write_bytecode=True
import numpy as np,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent;P=O.parents[1];E=Path('E:/drone_lab_bgA_20261109')
s=importlib.util.spec_from_file_location('common',P/'work/train_stage1_L20_20261108/common.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
b=c.load(O/'pilot_plan.json')[0];path=E/'2.4GHz'/f'{Path(b["block_file"]).stem}__bgA_s0.npy';z=np.load(path,mmap_mode='r');d=np.load(b['block_file'],mmap_mode='r');a=c.spec(np.array(d[:1228800]),'2.4GHz');bb=c.spec(np.array(z[:1228800]),'2.4GHz');fig=plt.figure(figsize=(19,5.2));gs=fig.add_gridspec(1,4,width_ratios=[1.2,1,1,1.3]);ax=fig.add_subplot(gs[0]);occ=np.load(O/'occupancy_2.4GHz.npy');f=2400+(np.arange(1024)-512)*.12;ax.plot(f,occ.max(0)*100,label='57次最大');ax.plot(f,occ.mean(0)*100,label='57次均值');ax.axvspan(2405,2455,color='orange',alpha=.15,label='截取频段');ax.axvspan(2345,2395,color='green',alpha=.15,label='s=0镜像源');ax.set(xlabel='MHz',ylabel='占用率 (%)',title='接收机A原频带，阈值+3 dB');ax.legend(fontsize=8)
c.image(fig.add_subplot(gs[1]),a,'2.4GHz','无人机：模型输入');c.image(fig.add_subplot(gs[2]),bb,'2.4GHz','配对镜像背景 s=0');ax=fig.add_subplot(gs[3]);freq=2430+(np.arange(512)-256)*.12;sel=(freq>=2405)&(freq<=2455)
for r in c.rows(P/'work/crop_slice_20261107/notch_table.csv'):
 if r['band']=='2.4GHz':sel &=abs(freq-float(r['freq_mhz']))>.3
r=next(v for v in c.load(O/'pilot_shape_p10.json') if v['block']==Path(b['block_file']).stem and v['s']==0);xx=[v['lo_mhz']+.5 for v in r['subbands']];ax.plot(xx,[v['drone_p10'] if v['status']=='judged' else np.nan for v in r['subbands']],label='无人机侧 p10（已判子带）');ax.plot(xx,[v['bg_p10'] for v in r['subbands']],label='镜像背景 p10 s=0');bs=c.load(P/'work/crop_slice_20261107/plan.json')
for key in ['20260710','20260709']:
 ref=next(v for v in bs if v['dataset']=='v3' and v['label']=='background' and v['band']=='2.4GHz' and key in v['group_id']);cc=np.load(P/'work/crop_slice_20261107/cells'/Path(ref['block_file']).stem/'mean.npy');yy=[]
 for lo in range(2405,2455):
  ix=sel&(freq>=lo)&(freq<lo+1);yy.append(np.percentile(cc[:,ix],10))
 ax.plot(xx,yy,label='v3 '+key+' p10')
ax.set(xlabel='MHz',ylabel='功率 (dB，未归一化)',title=f'1 MHz p10；已判子带 N={r["n_judged"]}');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(O/'图11_接收机A背景_2.4GHz.png',dpi=140);plt.close(fig);print('FIG11_DONE',flush=True)

