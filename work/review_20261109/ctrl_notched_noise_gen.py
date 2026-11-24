# Control: is "100% drone on receiver-A background" caused by receiver A itself, or by
# "empty band + notch dips" (an input the training background never contained)?
# Generate complex white noise at 61.44 MS/s and pass it through the SAME terminal FIR + notch set as Task F.
# (scipy env). Output: N x 1228800 complex64.
import sys, csv, importlib.util, numpy as np
from pathlib import Path
from scipy import signal
P = Path('C:/Users/Administrator/Desktop/数据集处理')
spec = importlib.util.spec_from_file_location('lab_preproc', P / 'work/prep_lines_20261104/lab_preproc.py')
lab = importlib.util.module_from_spec(spec); spec.loader.exec_module(lab)
FS, FC, N, K = 61440000, 2430.0, 1228800, 80
h = np.load(P / 'work/crop_slice_20261107/filters.npz')['2.4GHz']
notches = [r for r in csv.DictReader(open(P / 'work/crop_slice_20261107/notch_table.csv', encoding='utf-8-sig')) if r['band'] == '2.4GHz']
rng = np.random.default_rng(20261109)
pad = 20000
out = np.lib.format.open_memmap(P / 'work/review_20261109/notched_noise_2.4GHz.npy', mode='w+', dtype=np.complex64, shape=(K, N))
for k in range(K):
    x = ((rng.standard_normal(N + 2 * pad) + 1j * rng.standard_normal(N + 2 * pad)) / np.sqrt(2)).astype(np.complex64)
    z = signal.fftconvolve(x, h, mode='same')
    for r in notches:   # identical call to crop.py chain()
        z = lab.notch(z, FS, (float(r['freq_mhz']) - FC) * 1e6, float(r['half_width_khz']) * 500.)
    out[k] = z[pad:pad + N].astype(np.complex64)
out.flush(); print('done', K)
