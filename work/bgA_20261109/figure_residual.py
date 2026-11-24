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
profiles=c.load(O/'full_shape_profile.json');assert len(profiles)==61;xx=np.arange(2405,2455)+.5;curves=[]
for r in profiles:
 level=r['level_offset'];yy=[v['p10_diff']-level if level is not None and v['quiet_fraction']>=.9 else np.nan for v in r['subbands']];curves.append(yy);ax.plot(xx,yy,color='gray',alpha=.3,lw=.6)
ax.plot(xx,np.nanmedian(np.array(curves),axis=0),color='black',lw=2.2,label='61块中位数');ax.axhline(.5,color='red',ls=':',lw=.8);ax.axhline(-.5,color='red',ls=':',lw=.8)
ax.set(xlabel='MHz',ylabel='去电平偏移后的 p10 差 (dB)',title='61块形状残差；仅安静占比≥90%子带');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(O/'图11_接收机A背景_2.4GHz.png',dpi=140);plt.close(fig);print('FIG11_DONE',flush=True)

