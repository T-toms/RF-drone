# Measure cost of IQ-matrix input on this machine: ResNet18(2,2) (dronerfa_baseline/model.py, unmodified),
# input 2 x H x 1024 (I,Q channels; 1 row = 1024 consecutive samples), bf16 autocast, AdamW, channels_last.
# Reports peak GPU memory and seconds per training sample (fwd+bwd+step) per band/L, plus spectrogram baseline.
import sys, time, torch
from torch.nn import functional as F
sys.path.insert(0, 'C:/Users/Administrator/Desktop/数据集处理/dronerfa_baseline')
from model import ResNet18
torch.backends.cudnn.benchmark = True
FS = {'2.4GHz': 61.44e6, '5.8GHz': 100e6}
def bench(cin, H, W, bs, steps=6):
    m = ResNet18(cin, 2).cuda().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randn(bs, cin, H, W, device='cuda').contiguous(memory_format=torch.channels_last); y = torch.randint(0, 2, (bs,), device='cuda')
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    try:
        for i in range(steps + 2):
            if i == 2: torch.cuda.synchronize(); t = time.time()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(m(x), y)
            loss.backward(); opt.step()
        torch.cuda.synchronize(); dt = (time.time() - t) / steps
        return torch.cuda.max_memory_allocated() / 2**30, dt / bs
    except torch.OutOfMemoryError:
        return None, None
    finally:
        del m, opt, x; torch.cuda.empty_cache()
print('GPU', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2), 'GiB')
g, s = bench(1, 256, 256, 64); print(f'spectrogram 1x256x256 bs64: peak {g:.2f} GiB, {s*1e3:.2f} ms/sample')
for band in FS:
    for L in [10, 20, 50]:
        n = int(FS[band] * L / 1000); H = n // 1024
        for bs in [32, 16, 8, 4, 2]:
            g, s = bench(2, H, 1024, bs)
            if g is not None and g < 7.2:
                print(f'{band} L={L}ms IQ 2x{H}x1024 bs{bs}: peak {g:.2f} GiB, {s*1e3:.1f} ms/sample, fp16 bytes/sample {2*H*1024*2/2**20:.1f} MiB')
                break
            print(f'{band} L={L}ms bs{bs}: OOM')

