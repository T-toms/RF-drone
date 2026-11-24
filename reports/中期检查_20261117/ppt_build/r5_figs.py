from pathlib import Path
import csv
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

B = Path(__file__).resolve().parent
R = B.parents[2]
A = B / 'r5_assets'
A.mkdir(exist_ok=True)
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':['Arial','Microsoft YaHei'], 'font.size':13,
    'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#404040',
    'text.color':'#243744','axes.labelcolor':'#243744','axes.unicode_minus':True})
BLUE, PALE, GRAY, ORANGE = '#0079BA', '#96BBD1', '#A1A9AF', '#F0A30A'
manifest = {}

def rows(path):
    with (R / 'work' / path).open(encoding='utf-8-sig', newline='') as f:
        return dict(enumerate(csv.DictReader(f),2))

def save(fig, name):
    fig.savefig(A / (name+'.png'), dpi=300, facecolor='white')
    plt.close(fig)

def chart(width, height):
    fig, ax = plt.subplots(figsize=(width, height), layout='constrained')
    ax.grid(axis='y', color='#dce5ec', lw=.65)
    ax.set_axisbelow(True)
    ax.tick_params(length=2)
    return fig, ax

def bars(name, labels, values, colors, ylabel, limit, fmt, err=None, size=(6,3.2)):
    fig, ax = chart(*size)
    xx = np.arange(len(labels))
    ax.bar(xx,values,color=colors,width=.58,yerr=err,capsize=3,
           error_kw={'lw':1, 'ecolor':'#404040'})
    ax.set(xticks=xx,xticklabels=labels,ylabel=ylabel,ylim=(0,limit))
    for i,v in enumerate(values):
        e = err[i] if err is not None else 0
        ax.text(i,v+e+limit*.025,fmt(v),ha='center',va='bottom',fontsize=14,color='#243744')
    save(fig,name)

def plot_features():
    source='fusionB_20261110/B1b/main_table.csv'
    data=rows(source)
    labels=['谱平坦度','形状因子','归一化四阶累积量 C42','瞬时幅度谱峰值','幅度峰度',
            '瞬时频率标准差','峰均功率比','时频占用率','突发占空比','突发次数']
    keys=['spec_flatness','shape_factor','c42n','log_gamma_max','log_kurt_a','sigma_f',
          'papr999_db','occ_frac','duty','log1p_n_bursts']
    fig,ax=plt.subplots(figsize=(6.1,4.2),layout='constrained')
    record=[next((i,float(r['auc'])) for i,r in data.items() if r['feature']==k) for k in keys]
    values=[v for _,v in record]
    pos=np.arange(10)
    ax.barh(pos,np.array(values)-.475,left=.475,color=[GRAY]*5+[BLUE]*5,height=.62)
    ax.set(yticks=pos,yticklabels=labels,xlim=(.475,1.01),xticks=[.5,.6,.7,.8,.9,1],xlabel='区分能力')
    ax.invert_yaxis()
    ax.axvline(.5,c='#404040',ls='--',lw=.8)
    for y,v in zip(pos,values):
        ax.text(v+.013,y,f'{v:.3f}',va='center',fontsize=12)
    save(fig,'features')
    manifest['features']=[source,[i for i,v in record]]

def plot_iq():
    path=next(Path('E:/drone_lab_crop_20261107/2.4GHz/data').glob('data_001_2.4GHz_20M-signal*blk1.npy'))
    iq=np.load(path,mmap_mode='r')
    fs=122_880_000
    frame=int(round(fs*.0001))
    count=int(fs*.02)
    # Pick the opening window with visible bursts using the feature extractor's frame rule.
    candidates=[]
    for index in range(12):
        z=np.asarray(iq[index*count:(index+1)*count],dtype=np.complex128).ravel()
        if len(z)!=count or not np.iscomplexobj(iq):
            raise ValueError((iq.shape, iq.dtype))
        z-=z.mean()
        p=np.abs(z)**2
        fp=(p/p.mean()).reshape(-1,frame).mean(1)
        db=10*np.log10(fp)
        threshold=np.percentile(db,10)+6
        active=db>threshold
        transitions=np.where(np.diff(np.r_[False,active,False]))[0].reshape(-1,2)
        candidates.append((abs(len(transitions)-3)+abs(active.mean()-.5),index,db,threshold,transitions,active.mean()))
    _,i,db,threshold,transitions,duty=min(candidates,key=lambda row:row[0])
    fig,ax=chart(6.0,2.25)
    time=np.arange(len(db))*.1+.05
    ax.plot(time,db,color=BLUE,lw=1.35)
    for lo,hi in transitions:
        ax.axvspan(lo*.1,hi*.1,color=ORANGE,alpha=.19,lw=0)
    ax.axhline(threshold,color='#404040',ls='--',lw=.8)
    j=int(np.argmax(db))
    ax.scatter(time[j],db[j],color=ORANGE,s=22,zorder=3)
    ax.annotate('峰值',(time[j],db[j]),xytext=(11,6),textcoords='offset points',fontsize=14)
    ax.set(xlim=(0,20.45),xticks=np.arange(0,21,5),xlabel='时间（ms）',ylabel='帧功率（dB）')
    ax.set_ylim(db.min()-1,db.max()+3.5)
    ax.text(.58,threshold+.55,'底噪 + 6 dB 门限',fontsize=14,bbox={'facecolor':'white','edgecolor':'none','alpha':.87,'pad':2})
    save(fig,'iq')
    manifest['iq']={'path':str(path),'offset_samples':i*count,'samples':count,
        'frame_samples':frame,'visible_regions':transitions.tolist(),'occupied_fraction':duty}

def plot_background():
    path='review_taskH_20261108/ctrl_whitenoise_results.csv'
    data=rows(path)
    fig,ax=chart(6,3.1)
    levels=[-10,-5,0,5,10,15,20]
    used=[]
    for kind,color,label in [('wn',BLUE,'白噪声'),('v3bg',ORANGE,'真实环境背景')]:
        yy=[]
        for level in levels:
            entries=[(i,float(r['det'])*100) for i,r in data.items() if r['band']=='2.4GHz'
                     and r['group']=='data_drone' and r['seen']=='True' and r['bgtype']==kind
                     and r['x_db']==str(level)]
            used.extend(i for i,v in entries)
            yy.append(np.mean([v for i,v in entries]))
        ax.plot(levels,yy,'o-',color=color,lw=1.7,label=label)
        ax.annotate(f'{yy[2]:.0f}%',(0,yy[2]),xytext=(8,6),textcoords='offset points',
                    fontsize=15,fontweight='bold',color=color)
    ax.axvline(0,c='#8A969F',ls='--',lw=1)
    ax.set(ylim=(0,110),xticks=levels,xlabel='无人机样本功率／背景功率（dB）',ylabel='检出率（%）')
    ax.legend(loc='lower right',frameon=False,fontsize=12)
    save(fig,'background')
    manifest['background']=[path,used]

def plot_results():
    a=json.loads((B/'r4_assets/fusion_sources.json').read_text(encoding='utf-8-sig'))
    labels=['只用时频图','特征全置零','10 个特征','5 个特征']
    for name, block, ylabel, ymax in [('fusion_fa',a[:4],'误报率（%）',8.5),
                                      ('fusion_det',a[4:],'检出率（%）',115)]:
        bars(name,labels,[r['value'] for r in block],[PALE,GRAY,BLUE,ORANGE],ylabel,ymax,
             lambda v:f'{v:.1f}%', [r['std'] for r in block],(7.8,2.30))
        manifest[name]=['fusionB_B2_20261112/metrics_summary.csv',[r['row'] for r in block]]
    bars('mirror',['首轮','加入同设备背景'],[100,0],[GRAY,BLUE],'误报率（%）',120,
         lambda v:f'{v:.0f}%',size=(4,3.2))
    source='side_L_wifimix_20261110/metrics_paired.csv'
    data=rows(source)
    tags=['J','La','Lb']
    cond=['不叠加','只给无人机\n叠加','对称叠加']
    bars('symmetric_det',cond,[float(data[43][tag+'_mean'])*100 for tag in tags],
         [PALE,GRAY,BLUE],'真实环境检出率（%）',115,lambda v:f'{v:.0f}%',
         [float(data[43][tag+'_std'])*100 for tag in tags],(4.6,2.43))
    rr=[r for r in data.values() if r['category']=='C1' and r['mode']=='argmax']
    fa=[[100*sum(int(row[f'{tag}_{k}_k']) for row in rr)/sum(int(row[f'{tag}_{k}_n']) for row in rr)
         for k in (2026,2027,2028)] for tag in tags]
    bars('symmetric_fa',cond,[np.mean(v) for v in fa],[PALE,GRAY,BLUE],
        '模拟测试集误报率（%）',48,lambda v:f'{v:.0f}%' if v>15 else f'{v:.1f}%',
        [np.std(v,ddof=1) for v in fa],(4.6,2.43))
    manifest['symmetric']=[source,43,[i for i,r in data.items() if r in rr]]
    fig,ax=chart(6.3,4.25)
    for xx,yy,color,label,offset in [([0,1,2],[58,62,81],BLUE,'真实环境无人机检出率',7),
                                    ([0,1],[100,0],ORANGE,'同设备无信号背景误报率',8)]:
        ax.plot(xx,yy,'o-',lw=2,c=color,label=label)
        for x,y in zip(xx,yy):
            ax.annotate(f'{y}%',(x,y),xytext=(0,offset),textcoords='offset points',
                        ha='center',color=color,fontsize=15,fontweight='bold')
    ax.set(ylim=(-5,113),xlim=(-.25,2.25),xticks=[0,1,2],
           xticklabels=['首轮','加入同设备背景','对称叠加'],ylabel='比例（%）')
    ax.legend(loc='upper right',frameon=False,fontsize=12)
    save(fig,'progress')

def crop_examples():
    source=R/'work/fusionB_B2_20261112/fig0_input_examples_2.4GHz.png'
    with Image.open(source) as im:
        for name,box in {'lab':(100,73,625,453),'mixed':(742,73,1267,453),
                         'ambient':(1385,73,1910,453)}.items():
            im.crop(box).save(A/f'{name}.png',dpi=(300,300))
    for stem,old,bounds in [
        ('m100_crop','m100_window_r4b',(98,15,596,162)),
        ('split_crop','normal_r4b',(69,29,733,186)),
    ]:
        with Image.open(B/'r4_assets'/f'{old}.png') as im:
            im.crop(bounds).save(A/f'{stem}.png',dpi=(300,300))
    manifest['input_examples']=str(source)

if __name__=='__main__':
    crop_examples()
    plot_features()
    plot_iq()
    plot_background()
    plot_results()
    (A/'sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')


