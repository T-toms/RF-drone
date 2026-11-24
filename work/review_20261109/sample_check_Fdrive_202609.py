# 抽样检查 F 盘 202609：.bin 能否完整读出、sha256 是否与 JSON 一致、读速；.mat 能否打开（2026-11-11，支线）
import json, hashlib, time, sys
from pathlib import Path
import numpy as np
R = Path('F:/drone/202609')
js = sorted(R.rglob('*.json'))
pick = [js[i] for i in np.linspace(0, len(js) - 1, 8).astype(int)]
tot = 0; t0 = time.time()
for j in pick:
    m = json.loads(j.read_text(encoding='utf-8'))
    b = j.with_suffix('.bin')
    h = hashlib.sha256(); n = 0; err = None; t = time.time()
    try:
        with open(b, 'rb') as f:
            while True:
                blk = f.read(8 << 20)
                if not blk: break
                h.update(blk); n += len(blk)
    except OSError as e:
        err = repr(e)
    tot += n
    ok = (h.hexdigest() == m['capture']['sha256'])
    print(f"{j.parent.name:30s} {b.name[-12:]} read={n} sha_ok={ok} {time.time() - t:.1f}s {err or ''}", flush=True)
print(f'读速约 {tot / 1e6 / (time.time() - t0):.0f} MB/s；全部 735 个 bin 共 74.2 GB，预计 {74.2e3 / (tot / 1e6 / (time.time() - t0)) / 60:.0f} 分钟')
# .bin 格式：ci16_le 交错 I/Q，检查一下数值
x = np.fromfile(pick[0].with_suffix('.bin'), dtype='<i2', count=2_000_000).astype(np.float32)
print('ci16 前 100 万点：I 均值 %.1f 标准差 %.1f；Q 均值 %.1f 标准差 %.1f；最大绝对值 %d' % (x[0::2].mean(), x[0::2].std(), x[1::2].mean(), x[1::2].std(), np.abs(x).max()))
# .mat：看变量名与形状（不读全量）
try:
    from scipy import io
    print('mat 变量', io.whosmat(pick[0].with_suffix('.mat')))
except Exception as e:
    try:
        import h5py
        with h5py.File(pick[0].with_suffix('.mat'), 'r') as f:
            print('mat(h5) 变量', {k: (f[k].shape, str(f[k].dtype)) for k in f.keys()})
    except Exception as e2:
        print('mat 无法用 scipy/h5py 打开:', repr(e), repr(e2))
