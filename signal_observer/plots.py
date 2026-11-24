"""Plotting in sample coordinates or explicitly supplied clock coordinates."""
from pathlib import Path
import numpy as np
from matplotlib.font_manager import FontProperties

FONT = Path('C:/Windows/Fonts/msyh.ttc')
def configure_font():
    import matplotlib
    if FONT.exists():matplotlib.rcParams['font.family'] = FontProperties(fname=str(FONT)).get_name()
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 9

def time_scale(result, fs):
    if fs:
        return (1/fs, '原始录制时间（秒，按输入Fs换算）') if (result['stop']-result['start'])/fs >= 1 else (1000/fs, '原始录制时间（毫秒，按输入Fs换算）')
    return (1e-4,'原始采样点坐标（万点）') if result['stop']-result['start'] < 1000000 else (1e-6,'原始采样点坐标（百万点）')

def figure_title(result,fs,fs_source):
    amount=result['stop']-result['start']
    clock='Fs未知：仅使用样点与归一化频率' if fs is None else f"Fs={fs/1e6:g} MS/s（{'来自文件' if fs_source=='file' else '用户输入/推定'}），本范围约{amount/fs:.6g}秒"
    valid=result.get('analyzed_samples',result['frames']*result['nfft'])
    return result['title']+'\n'+f'视窗{amount:,}点，覆盖IQ {valid:,}点；'+clock+'；灰色=无统计帧/缺段'

def update_live(fig,result,fs,fc,fs_source,overview,detail_kind):
    live=getattr(fig,'_observer_live',None)
    key=(id(overview),fs,fc,fs_source,result['nfft'],result['frequency_bins'],time_scale(result,fs))
    if live is None or live['key']!=key:return None
    axes=live['axes'];scale=live['scale'];n=len(result['weights']);left=result['start']*scale
    right=(result['start']+n*result['time_bin_samples'])*scale;stop=result['stop']*scale
    live['image'].set_data(np.ma.masked_invalid(result[detail_kind].T-live['ref']))
    old=live['image'].get_extent();live['image'].set_extent((left,right,old[2],old[3]))
    axes[1].set_xlim(left,stop);axes[1].set_title('局部'+('平均谱' if detail_kind=='mean_db' else '峰值谱')+'（用上方按钮或时间滑块移动）',fontsize=9)
    db=10*np.log10(np.maximum(result['spectrum'],1e-30));live['psd'].set_ydata(db-db.max() if result['frames'] else np.full_like(db,np.nan))
    axes[0].relim();axes[0].autoscale_view(scalex=False)
    te=np.minimum(result['start']+np.arange(n+1)*result['time_bin_samples'],result['stop'])*scale;tc=(te[:-1]+te[1:])/2
    for line,data in zip(live['envelopes'],[result['envelope_db'],result['envelope_max_db']]):line.set_data(tc,data-live['eref'])
    if np.isfinite(result['envelope_db']).any():axes[3].relim();axes[3].autoscale_view(scalex=False)
    nav=axes[4];nav._viewport_patch.set_x(result['start']*nav._sample_scale);nav._viewport_patch.set_width((result['stop']-result['start'])*nav._sample_scale)
    fig._suptitle.set_text(figure_title(result,fs,fs_source))
    return axes,scale

def draw_result(fig, result, fs=None, fc=None, fs_source='unknown', overview=None, detail_kind='mean_db'):
    if overview is not None:
        updated=update_live(fig,result,fs,fc,fs_source,overview,detail_kind)
        if updated is not None:return updated
    fig._observer_live=None
    configure_font();fig.clear()
    offset=1 if overview is not None else 0
    grid=fig.add_gridspec(4,1,height_ratios=[.5,.65,3,.65] if offset else [1,1.6,1.6,1])
    a0=fig.add_subplot(grid[offset]);a1=fig.add_subplot(grid[offset+1])
    a2=a1 if offset else fig.add_subplot(grid[2],sharex=a1)
    a3=fig.add_subplot(grid[3],sharex=a1)
    axes=[a0,a1,a2,a3];n=result['nfft'];nf=result['frequency_bins']
    f=np.arange(n)/n-.5;edges=np.linspace(-.5,.5,nf+1)
    if fs:
        f=f*fs/1e6;edges=edges*fs/1e6
        if fc is not None:f+=fc/1e6;edges+=fc/1e6
        flabel='射频频率（MHz，按输入Fs/Fc换算）' if fc is not None else '相对接收中心的频率偏移（MHz）'
    else:flabel='归一化频率（采样率的倍数）'
    db=10*np.log10(np.maximum(result['spectrum'],1e-30));psd,=a0.plot(f,db-db.max() if result['frames'] else np.full_like(db,np.nan),lw=.8);a0.set(xlabel=flabel,ylabel='相对谱功率/dB');a0.grid(alpha=.2)
    scale,tlabel=time_scale(result,fs)
    te=np.minimum(result['start']+np.arange(len(result['weights'])+1)*result['time_bin_samples'],result['stop'])*scale
    tc=(te[:-1]+te[1:])/2
    ref=float(np.nanpercentile((overview if overview is not None else result)['mean_db'],99.5))
    from matplotlib import colormaps
    cmap=colormaps['viridis'].copy();cmap.set_bad('#bdbdbd')
    maps=[(a1,detail_kind,'局部'+('平均谱' if detail_kind=='mean_db' else '峰值谱')+'（用上方按钮或时间滑块移动）')] if offset else [(a1,'mean_db','当前视窗平均谱'),(a2,'peak_db','当前视窗峰值谱（用于观察稀疏能量）')]
    for ax,key,title in maps:
        if offset:
            # Equal-width bins, with the last bin clipped at the source boundary.
            # A raster image avoids rebuilding half a million mesh cells per move.
            end=(result['start']+len(result['weights'])*result['time_bin_samples'])*scale
            im=ax.imshow(np.ma.masked_invalid(result[key].T-ref),extent=(te[0],end,edges[0],edges[-1]),origin='lower',aspect='auto',interpolation='nearest',cmap=cmap,vmin=-45,vmax=10)
        else:im=ax.pcolormesh(te,edges,np.ma.masked_invalid(result[key].T-ref),shading='flat',cmap=cmap,vmin=-45,vmax=10,rasterized=True)
        ax.set_ylabel(flabel);ax.set_title(title,fontsize=9);fig.colorbar(im,ax=ax,pad=.01,label='相对PSD/dB')
        ax.set_xlim(te[0],te[-1])
    e=result['envelope_db'];eref=float(np.nanmedian((overview if overview is not None else result)['envelope_db']))
    line1,=a3.plot(tc,e-eref,lw=.7,label='平均短帧功率');line2,=a3.plot(tc,result['envelope_max_db']-eref,lw=.5,alpha=.65,label='最大短帧功率')
    a3.set(xlabel=tlabel,ylabel='相对功率/dB');a3.grid(alpha=.2);a3.legend(loc='upper right',fontsize=8)
    if overview is not None:
        nav=fig.add_subplot(grid[0]);oscale,olabel=time_scale(overview,fs)
        oe=np.minimum(overview['start']+np.arange(len(overview['weights'])+1)*overview['time_bin_samples'],overview['stop'])*oscale
        nav.plot((oe[:-1]+oe[1:])/2,overview['envelope_db'],lw=.6,color='#397a9e')
        nav._viewport_patch=nav.axvspan(result['start']*oscale,result['stop']*oscale,color='orange',alpha=.4)
        nav.set(xlim=(oe[0],oe[-1]),ylabel='总览/dB',xlabel=olabel)
        nav.set_title('整段位置参考：橙色为当前视窗（移动请用上方按钮/滑块）',fontsize=9)
        nav._sample_scale=oscale;axes.append(nav)
    fig.suptitle(figure_title(result,fs,fs_source),fontsize=10)
    if offset:a0.set_xlabel('');a0.tick_params(axis='x',labelsize=8)
    fig.subplots_adjust(left=.09,right=.96,bottom=.10,top=.85,hspace=.6 if offset else .55)
    if offset:fig._observer_live=dict(key=(id(overview),fs,fc,fs_source,result['nfft'],nf,time_scale(result,fs)),axes=axes,scale=scale,image=im,psd=psd,envelopes=[line1,line2],ref=ref,eref=eref)
    return axes,scale
