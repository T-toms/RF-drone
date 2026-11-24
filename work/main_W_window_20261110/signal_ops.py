import random
from core import FC, FS, np, torch
from torch.nn import functional as F

SAMPLE_FS, CENTER = 61440000, 2430.
freq = np.fft.fftshift(np.fft.fftfreq(1024, 1 / SAMPLE_FS)) / 1e6 + CENTER
keep = (freq >= CENTER - 25) & (freq <= CENTER + 25)
fk = freq[keep]
# 受影响频点（量得的结果，见 notch_hole_profile.py 输出）
HOLES = {2426.88: (2426.76, 2427.00), 2440.0: (2439.84, 2440.20), 2450.0: (2449.80, 2450.16)}
hole_idx = [np.where((fk >= lo - 1e-6) & (fk <= hi + 1e-6))[0] for lo, hi in HOLES.values()]


def fill_bins(p, method):
    p = p.clone()
    for idx in hole_idx:
        a, b = int(idx[0]), int(idx[-1])          # 坑内首末频点
        n = b - a + 1
        if method == 'M':
            half = n // 2
            for k in range(half):                 # 左半：a+k <- a-1-k
                p[a + k] = p[a - 1 - k]
            for k in range(n - half):             # 右半：b-k <- b+1+k
                p[b - k] = p[b + 1 + k]
        else:
            L, R = p[a - 1], p[b + 1]
            for k in range(n):
                w = (k + 1) / (n + 1)
                p[a + k] = (1 - w) * L + w * R
    return p


def spec_new(iq, method='M', dtype=np.float16):
    x = torch.as_tensor(iq, device='cuda', dtype=torch.complex64)
    s = torch.stft(x, 1024, 512, window=torch.hann_window(1024, device='cuda'), center=False, onesided=False, return_complex=True)
    p = torch.fft.fftshift(s.abs().square(), dim=-2)
    p = p[torch.as_tensor(keep, device='cuda')]
    p = fill_bins(p, method) if hole_idx else p
    p = F.adaptive_avg_pool2d(p[None, None], (256, 256))[0, 0]
    db = 10 * torch.log10(p + p.median().clamp_min(1e-30) * 1e-3)
    return ((db - db.mean()) / db.std().clamp_min(1e-8)).cpu().numpy().astype(dtype)


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


@torch.no_grad()
def predict(m,x):
    m.eval();ps=[]
    for z in x.split(64):
        with torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            ps.append(m(z.cuda().float().contiguous(memory_format=torch.channels_last)).float().softmax(1).cpu().numpy())
    return np.concatenate(ps)


def power(z):return float(np.mean(np.abs(z)**2,dtype=np.float64))


def mixing(d,b,x):
    pd=power(d);pb=power(b);g=float(np.sqrt(pb/max(pd,1e-30)*10**(x/10)));return b+g*d,g,pd,pb

