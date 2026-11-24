"""Explain one observed interval with exploratory, unlabeled spectral regions."""
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from scipy.fft import fft,fftshift
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle

W=Path(__file__).parent;O=W/'long_signal_pilot_20261024';tag='009_2.4GHZ_10M-signal__2026-09-20-18-04-10'
r=json.loads((O/(tag+'_summary.json')).read_text(encoding='utf8'))
lo,hi=r['zoom_source_interval'];x=loadmat(Path('G:/drone')/r['zoom_source'],variable_names=['data'])['data'].ravel()[lo:hi]
nfft=2048;hop=512;window=np.hanning(nfft).astype(np.float32);z=np.lib.stride_tricks.sliding_window_view(x,nfft)[::hop]
power=np.abs(fftshift(fft(z*window,axis=1),axes=1))**2/np.sum(window**2)
db=10*np.log10(np.maximum(power,1e-25));freq=np.arange(nfft)/nfft-.5
times=np.arange(len(z))*hop+nfft/2;regions=[('A',.28,.34,'#ff7f0e'),('B',.38,.46,'#e15759'),('C',.005,.03,'#7b4ab5'),('D',.045,.08,'#169f9f')]
results=[];font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(2,1,figsize=(13,8),layout='constrained',gridspec_kw={'height_ratios':[3,1.4]})
ref=np.percentile(db,99.5);im=axs[0].imshow(db.T-ref,origin='lower',aspect='auto',extent=[0,len(x)/1e4,-.5,.5],vmin=-45,vmax=10,interpolation='nearest',cmap='viridis')
for label,flo,fhi,color in regions:
    p=power[:,(freq>=flo)&(freq<fhi)].mean(axis=1);pdb=10*np.log10(p+1e-25);baseline=float(np.percentile(pdb,20));relative=pdb-baseline
    mask=relative>=10;edge=np.diff(np.r_[False,mask,False].astype(np.int8));events=[]
    for a,b in zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)):
        if b-a<3:continue
        start=int(max(0,a*hop));end=int(min(len(x),(b-1)*hop+nfft));events.append({'local_start_sample':start,'local_end_sample_exclusive':end,'duration_samples':end-start})
        axs[0].add_patch(Rectangle((start/1e4,flo),(end-start)/1e4,fhi-flo,fill=False,ec=color,lw=1.7))
        axs[0].text((start+end)/2/1e4,fhi+.008,label,color='white',fontsize=11,ha='center',weight='bold',bbox=dict(facecolor=color,edgecolor='none',alpha=.9,pad=1.5))
    axs[1].plot(times/1e4,relative,label=label+'区',color=color,lw=1.2)
    results.append({'region':label,'frequency_interval_cycles_per_sample':[flo,fhi],'baseline_definition':'20th percentile of local regional power, not independently measured noise','threshold_above_baseline_db':10,'events':events})
axs[0].set(ylim=(-.08,.5),xlim=(0,len(x)/1e4),ylabel='归一化频率（采样率的倍数）',title='同一段 IQ 中，不同频率位置和宽度的突发成分')
fig.colorbar(im,ax=axs[0],label='相对谱功率（dB）')
axs[1].axhline(10,color='gray',lw=.8,ls='--');axs[1].set(xlim=(0,len(x)/1e4),xlabel='从局部起点开始的采样点数（万点）',ylabel='相对各区域低分位基线（dB）');axs[1].legend(ncol=4);axs[1].grid(alpha=.25)
fig.suptitle('区域 A–D 仅用于说明观察结果，未赋予图传、遥控或干扰标签',fontsize=11)
fig.savefig(O/'annotated_signal_regions.png',dpi=140);plt.close(fig)
out={'source':r['zoom_source'],'source_interval':[lo,hi],'recording_interval':r['zoom_recording_interval'],'stft_size':nfft,'hop_samples':hop,'edge_resolution_note':'Windows span 2048 samples; threshold-dependent region proposals, not exact RF frame boundaries or true occupied bandwidth.','regions':results}
(O/'annotated_signal_regions.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(results,ensure_ascii=False,indent=2))
