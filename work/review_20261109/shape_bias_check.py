# Is the 1-2 dB "floor shape" gap real, or does the drone-side reference contain weak residual signal?
# Falsification test: if the gap is a true floor difference it should NOT depend on the quiet threshold;
# if it is residual signal it should shrink toward 0 as threshold tightens (floor+3 -> +1 -> +0.5).
# Same quantity computed on the background side with the same selection (cells of bgA's own cell grid)
# serves as the control for selection bias.
import sys, numpy as np, json
stem = sys.argv[1]; s = sys.argv[2] if len(sys.argv) > 2 else '0'
c = np.load(f'work/crop_slice_20261107/cells/{stem}/mean.npy').astype(np.float64)   # (cells, 512) dB, 120 kHz bins
z = np.load(f'E:/drone_lab_bgA_20261109/2.4GHz/{stem}__bgA_s{s}.npy', mmap_mode='r')
f = 2430 + (np.arange(512) - 256) * 0.12
mask = (f >= 2405) & (f <= 2455)
for n in [2426.88, 2440, 2450]:
    mask &= np.abs(f - n) > 0.3
# background cells with the same 1 ms x 512-bin grid
cs = 61440; nf = cs // 512; w = np.hanning(512); ncell = len(z) // cs
bg = np.empty((ncell, 512))
for j in range(ncell):
    x = np.asarray(z[j * cs:(j + 1) * cs])[:nf * 512].reshape(nf, 512)
    bg[j] = 10 * np.log10((np.abs(np.fft.fftshift(np.fft.fft(x * w, axis=1), axes=1)) ** 2 / np.sum(w ** 2)).mean(0))
n = min(len(c), len(bg)); c, bg = c[:n], bg[:n]
sub_med = [np.median(c[:, mask & (f >= lo) & (f < lo + 1)]) for lo in range(2405, 2455)]
floor = float(np.median(c[:, mask & (f >= 2405 + int(np.argmin(sub_med))) & (f < 2406 + int(np.argmin(sub_med)))]))
print(stem, 's', s, 'floor_noise', round(floor, 2), ' bg global median', round(float(np.median(bg[:, mask])), 2))
print('  1 MHz band | drone-ref (thr +3/+1/+0.5): diff dB [n cells] | bg-side same selection on bg own cells (+3/+1)')
for lo in [2405, 2408, 2410, 2415, 2420, 2425, 2430, 2440, 2445, 2452]:
    ix = mask & (f >= lo) & (f < lo + 1)
    out = []
    for t in [3, 1, 0.5]:
        q = np.all(c[:, ix] < floor + t, axis=1)
        if q.sum() >= 20:
            d = 10 * np.log10((10 ** (bg[:, ix] / 10)).mean() / (10 ** (c[q][:, ix] / 10)).mean())
            out.append(f'{d:+.2f}[{q.sum()}]')
        else:
            out.append(f'  n/a[{q.sum()}]')
    # control: per-cell distribution of drone quiet cells vs bg cells (median, not mean)
    q3 = np.all(c[:, ix] < floor + 3, axis=1)
    med = (np.median(c[q3][:, ix]) - np.median(bg[:, ix])) if q3.sum() else np.nan
    print(f'  {lo}-{lo+1} |', ' '.join(out), f'| drone-quiet median - bg median = {med:+.2f} dB; bg p99-p50 = {np.percentile(bg[:, ix], 99) - np.median(bg[:, ix]):.2f}')
