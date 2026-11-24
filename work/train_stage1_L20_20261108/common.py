from pathlib import Path
import sys, os, csv, json, time, random, collections
os.environ['PYTHONIOENCODING']='utf-8'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CACHE=Path('E:/drone_lab_spec_L20_20261108'); CACHE.mkdir(exist_ok=True)
INPUT=ROOT/'work/crop_slice_20261107'
sys.path.insert(0,str(ROOT/'dronerfa_baseline'))
from model import ResNet18
BANDS=['2.4GHz','5.8GHz']; SEEDS=[2026,2027,2028]
FS={'2.4GHz':61440000,'5.8GHz':100000000}; FC={'2.4GHz':2430.,'5.8GHz':5800.}
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=True
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rs):
    rs=list(rs)
    if not rs:return
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def seed(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def spec(iq,b):
    x=torch.as_tensor(iq,device='cuda',dtype=torch.complex64)
    s=torch.stft(x,1024,512,window=torch.hann_window(1024,device='cuda'),center=False,onesided=False,return_complex=True)
    p=torch.fft.fftshift(s.abs().square(),dim=-2)
    freq=torch.fft.fftshift(torch.fft.fftfreq(1024,1/FS[b],device='cuda'))/1e6+FC[b]
    half=25 if b=='2.4GHz' else 45
    p=p[(freq>=FC[b]-half)&(freq<=FC[b]+half)]
    p=F.adaptive_avg_pool2d(p[None,None],(256,256))[0,0]
    db=10*torch.log10(p+p.median().clamp_min(1e-30)*1e-3)
    return ((db-db.mean())/db.std().clamp_min(1e-8)).cpu().numpy().astype(np.float16)
def cachepath(b,s):return CACHE/f'{b}_{s}_spec.npy'
def meta(b,s):return rows(CACHE/f'{b}_{s}_spec_meta.csv')
def dataset(b,s):
    a=np.load(cachepath(b,s));r=meta(b,s)
    return torch.from_numpy(a).unsqueeze(1),torch.tensor([x['label']=='drone' for x in r],dtype=torch.long),r
def model(b,s):
    c=torch.load(CACHE/f'{b}_seed{s}_best.pt',map_location='cpu',weights_only=False)
    m=ResNet18(1,2).cuda().to(memory_format=torch.channels_last);m.load_state_dict(c['model']);m.eval();return m,c
@torch.no_grad()
def predict(m,x):
    m.eval();ps=[]
    for z in x.split(64):
        with torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            ps.append(m(z.cuda().float().contiguous(memory_format=torch.channels_last)).float().softmax(1).cpu().numpy())
    return np.concatenate(ps)
def extent(b):return [0,20,FC[b]-(25 if b=='2.4GHz' else 45),FC[b]+(25 if b=='2.4GHz' else 45)]
def image(ax,z,b,title='',boundaries=None):
    ax.imshow(z,origin='lower',aspect='auto',extent=extent(b),cmap='viridis',vmin=-2,vmax=3)
    ax.set(xlabel='Time (ms)',ylabel='Frequency (MHz)',title=title)
    for r in rows(INPUT/'notch_table.csv'):
        if r['band']==b:ax.axhline(float(r['freq_mhz']),color='white',ls=':',lw=.65)
    for f in boundaries or []:ax.axhline(f,color='red',ls='--',lw=.8)
def examples(b,split='val'):
    r=meta(b,split);a=np.load(cachepath(b,split),mmap_mode='r'); rng=np.random.default_rng(2026)
    fig,axs=plt.subplots(3,2,figsize=(12,11))
    for ax,(ds,lab) in zip(axs, [('data','drone'),('v3','drone'),('v3','background')]):
        ids=[i for i,x in enumerate(r) if x['dataset']==ds and x['label']==lab]
        for j,k in enumerate(rng.choice(ids,2,replace=False)):image(ax[j],a[k],b,f'{ds} {lab} {r[k]["model"]} {r[k]["sample_id"]}')
    fig.tight_layout();fig.savefig(OUT/f'fig0_input_examples_{b}.png',dpi=140);plt.close(fig)



