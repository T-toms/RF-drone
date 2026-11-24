# Quick look: are the raw/ sessions missing from datasets_v3 valid captures with signal?
# Compares mean PSD + time-occupancy of 033706 (missing) vs 033809 (in v3), and mini3pro 024545 vs a v3 mini3pro 5.8G session.
import sys, glob, os, numpy as np
RAW = 'G:/drone/raw/2026-07-09_urban'
FS = 250e6
NFFT = 1024

def stats(path, n=25_000_000):
    x = np.fromfile(path, dtype='<i2', count=2 * n).astype(np.float32).view(np.complex64)
    if x.size == 0:
        return None
    frames = x[: (x.size // NFFT) * NFFT].reshape(-1, NFFT)
    w = np.hanning(NFFT).astype(np.float32)
    P = np.abs(np.fft.fftshift(np.fft.fft(frames * w, axis=1), axes=1)) ** 2
    f = (np.arange(NFFT) - NFFT // 2) * FS / NFFT / 1e6
    psd = 10 * np.log10(P.mean(0) + 1e-12)
    floor = np.median(psd)
    # band power over time for 5755-5845 (offset -45..+45 MHz), frames of 4.1 us; aggregate to 1 ms
    sel = (f >= -45) & (f <= 45)
    bp = P[:, sel].mean(1)
    k = int(1e-3 * FS / NFFT)
    bp_ms = 10 * np.log10(bp[: (bp.size // k) * k].reshape(-1, k).mean(1) + 1e-12)
    return dict(n=x.size, rms=float(np.sqrt(np.mean(np.abs(x) ** 2))), zeros=float(np.mean(x == 0)),
                floor=float(floor), p99_minus_floor=float(np.percentile(psd, 99) - floor),
                ms_p10=float(np.percentile(bp_ms, 10)), ms_p90=float(np.percentile(bp_ms, 90)),
                top_bins=[round(float(v), 1) for v in f[np.argsort(psd)[-5:]]])

for sess in sys.argv[1:]:
    for b in sorted(glob.glob(os.path.join(RAW, sess, '*.bin'))):
        s = stats(b)
        print(sess, os.path.basename(b)[-11:], s)
