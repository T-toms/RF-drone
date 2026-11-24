"""Numerical core of the local MAT signal observer. Source files are read-only."""
from dataclasses import dataclass, asdict
from collections import OrderedDict
from pathlib import Path
import csv, json, math, threading
import h5py
import numpy as np
from scipy.io import loadmat, whosmat
from scipy.fft import fft, fftshift

WORKSPACE = Path(__file__).resolve().parent.parent
AUDIT = WORKSPACE / 'work/audit_drone_20261023'

class Cancelled(Exception):
    pass

@dataclass
class MatInfo:
    path: str
    variables: list
    lengths: dict
    fs_hz: float | None = None
    fc_hz: float | None = None

@dataclass
class Part:
    path: str
    variable: str
    length: int
    offset: int
    valid_start: int
    valid_stop: int

@dataclass
class Plan:
    title: str
    parts: list
    total_samples: int
    scope: str
    source_path: str

def scalar(value):
    a = np.asarray(value)
    if a.size == 1 and np.isrealobj(a):
        v = float(a.ravel()[0])
        return v if np.isfinite(v) else None

def inspect_mat(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('找不到所选MAT文件。')
    names, lengths, metadata = [], {}, {}
    keys = ['fs', 'Fs', 'sample_rate_hz', 'centfreq', 'center_freq_hz']
    if h5py.is_hdf5(path):
        with h5py.File(path, 'r') as f:
            for name, ds in f.items():
                if not isinstance(ds, h5py.Dataset):
                    continue
                if name in keys:
                    metadata[name] = scalar(ds[()])
                if ds.ndim in (1, 2) and sum(n > 1 for n in ds.shape) <= 1 and ds.size >= 32:
                    if ds.dtype.kind == 'c' or ds.dtype.fields and {'real', 'imag'} <= ds.dtype.fields.keys():
                        names.append(name); lengths[name] = int(ds.size)
    else:
        for name, shape, kind in whosmat(path):
            if name not in keys and len(shape) <= 2 and sum(n > 1 for n in shape) <= 1 and math.prod(shape) >= 32 and kind in ('single', 'double'):
                names.append(name); lengths[name] = math.prod(shape)
        small = loadmat(path, variable_names=keys)
        metadata = {k: scalar(small[k]) for k in keys if k in small}
    names.sort(key=lambda n: (n not in ('data', 'IQ'), n != 'data', n))
    if not names:
        raise ValueError('未找到可读取的一维IQ变量；当前支持复数向量data/IQ等，不自动解释实数矩阵。')
    fs = next((metadata[k] for k in ('sample_rate_hz', 'fs', 'Fs') if metadata.get(k) is not None), None)
    fc = next((metadata[k] for k in ('center_freq_hz', 'centfreq') if metadata.get(k) is not None), None)
    if fs is not None and fs <= 0:
        fs = None
    return MatInfo(str(path), names, lengths, fs, fc)

def make_plan(info, variable, recording=False, exclude_invalid=True, audit=AUDIT):
    if variable not in info.variables:
        raise ValueError('请选择文件中存在的IQ变量。')
    manifest = audit / 'data_preprocessing_manifest.csv'
    rows = list(csv.DictReader(manifest.open(encoding='utf-8-sig'))) if manifest.exists() else []
    root = Path('G:/drone')
    key = str(Path(info.path).resolve()).casefold()
    found = next((r for r in rows if str((root/r['path']).resolve()).casefold() == key), None)
    if not recording:
        n = info.lengths[variable]
        lo, hi = 0, n
        if found and exclude_invalid:
            if found['all_zero'] == 'True':
                lo = hi = 0
            else:
                lo, hi = int(found['start_sample']), int(found['end_sample_exclusive'])
        return Plan(Path(info.path).name, [Part(info.path, variable, n, 0, lo, hi)], n, 'single_file', info.path)
    if not found or variable != 'data':
        raise ValueError('整次录制模式仅适用于已审计索引中的G:/drone/data文件；其他MAT请选择“仅当前文件”。')
    if info.lengths[variable] != 6144000:
        raise ValueError('所选文件长度与已审计索引不符，请先核对源文件。')
    group = [r for r in rows if (r['folder'],r['session']) == (found['folder'],found['session']) and r['duplicate'] == 'False']
    group.sort(key=lambda r:(int(r['epoch']),int(r['seg'])))
    parts = []
    for r in group:
        lo, hi = 0, 6144000
        if exclude_invalid:
            if r['all_zero'] == 'True':
                lo = hi = 0
            else:
                lo, hi = int(r['start_sample']), int(r['end_sample_exclusive'])
        parts.append(Part(str(root/r['path']), 'data', 6144000, (int(r['epoch'])*20+int(r['seg']))*6144000, lo, hi))
    total = max(p.offset+p.length for p in parts)
    return Plan(found['folder']+' / '+found['session'], parts, total, 'recording', info.path)

class IQCache:
    """Bounded read-only cache for compressed v5 MAT vectors; never stores HDF handles."""
    def __init__(self,max_bytes=128*1024*1024):
        self.max_bytes=int(max_bytes);self.bytes=0;self.loads=0;self.hits=0;self.entries=OrderedDict()
    def clear(self):self.entries.clear();self.bytes=0
    def get(self,part):
        stat=Path(part.path).stat();key=(part.path,part.variable,part.length,stat.st_size,stat.st_mtime_ns)
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key);return self.entries[key]
        data=loadmat(part.path,variable_names=[part.variable])[part.variable].reshape(-1);self.loads+=1
        if data.size!=part.length or not np.iscomplexobj(data):raise ValueError('所选变量必须是一维复数IQ向量；不把实数信号自动当作IQ。')
        data.flags.writeable=False
        if data.nbytes<=self.max_bytes:
            while self.entries and self.bytes+data.nbytes>self.max_bytes:
                _,old=self.entries.popitem(last=False);self.bytes-=old.nbytes
            self.entries[key]=data;self.bytes+=data.nbytes
        return data

class Reader:
    def __init__(self, part,cache=None):
        self.part = part; self.h5 = None; self.data = None;self.cache=cache
    def __enter__(self):
        p = self.part
        if h5py.is_hdf5(p.path):
            self.h5 = h5py.File(p.path, 'r'); self.data = self.h5[p.variable]
            if self.data.size != p.length:
                self.h5.close();self.h5=None
                raise ValueError('源变量长度已变化。')
        else:
            self.data = self.cache.get(p) if self.cache is not None else loadmat(p.path, variable_names=[p.variable])[p.variable].reshape(-1)
            if self.data.size != p.length or not np.iscomplexobj(self.data):
                raise ValueError('所选变量必须是一维复数IQ向量；不把实数信号自动当作IQ。')
        return self
    def read(self, a, b):
        if self.h5:
            d = self.data
            z = d[a:b] if d.ndim == 1 else d[0,a:b] if d.shape[0] == 1 else d[a:b,0]
            if z.dtype.fields:
                z = z['real'] + 1j*z['imag']
        else:
            z = self.data[a:b]
        z = np.asarray(z, dtype=np.complex64)
        if not np.isfinite(z).all():
            raise ValueError('所选区间存在NaN或Inf，已停止绘图。')
        return z
    def __exit__(self, *args):
        if self.h5:
            self.h5.close()
        self.data = None

def analyze(plan, start=0, count=None, nfft=4096, max_bins=2000, progress=None, cancel=None, allow_empty=False, hop=None, max_frequency_bins=512, cache=None):
    start = int(start)
    if start < 0 or start >= plan.total_samples:
        raise ValueError('起点超出源数据范围。')
    if count is not None and int(count) <= 0:
        raise ValueError('样点数应为正整数，留空表示到末尾。')
    stop = min(plan.total_samples, start+int(count)) if count is not None else plan.total_samples
    nfft = int(nfft); max_bins = int(max_bins)
    if nfft < 32 or nfft & (nfft-1) or max_bins < 1:
        raise ValueError('FFT点数必须为不小于32的2次幂，显示格数应为正数。')
    hop=nfft if hop is None else int(hop)
    if hop<1 or hop>nfft or nfft%hop:raise ValueError('帧移必须为FFT点数的正整数约数。')
    if stop-start < nfft:
        raise ValueError(f'所选范围少于{nfft}点，请扩大范围或减小FFT点数。')
    width = max(1, math.ceil((stop-start)/(max_bins*hop)))*hop
    nbins = math.ceil((stop-start)/width); nf = min(int(max_frequency_bins),nfft); factor = nfft//nf
    if nf<1 or nfft%nf:raise ValueError('频率格数必须为FFT点数的正整数约数。')
    sums = np.zeros((nbins,nf),np.float64); maxima = np.zeros((nbins,nf),np.float32)
    weights = np.zeros(nbins,np.int64); esums = np.zeros(nbins,np.float64); emax = np.zeros(nbins,np.float32)
    spectrum = np.zeros(nfft,np.float64); processed = 0; used = [];analyzed_samples=0
    win = np.hanning(nfft).astype(np.float32); norm = float(np.sum(win*win))
    parts = [p for p in plan.parts if p.offset+p.valid_stop>start and p.offset+p.valid_start<stop]
    for index,p in enumerate(parts):
        if cancel and cancel.is_set():raise Cancelled()
        lo = max(p.valid_start,start-p.offset); hi = min(p.valid_stop,stop-p.offset)
        lo = math.ceil(lo/hop)*hop
        frame_count=(hi-lo-nfft)//hop+1
        if frame_count<=0:continue
        hi=lo+(frame_count-1)*hop+nfft
        if progress:progress(index,len(parts),Path(p.path).name)
        with Reader(p,cache) as reader:
            for first_frame in range(0,frame_count,256):
                if cancel and cancel.is_set():raise Cancelled()
                size=min(256,frame_count-first_frame);a=lo+first_frame*hop;b=a+(size-1)*hop+nfft
                raw=reader.read(a,b)
                z=raw.reshape(-1,nfft) if hop==nfft else np.lib.stride_tricks.sliding_window_view(raw,nfft)[::hop]
                power = np.abs(fftshift(fft(z*win,axis=1),axes=1))**2/norm
                spectrum += power.sum(axis=0,dtype=np.float64)
                e = np.mean(np.abs(z)**2,axis=1)
                meanf = power.reshape(-1,nf,factor).mean(axis=2); maxf = power.reshape(-1,nf,factor).max(axis=2)
                centers = p.offset+a+np.arange(len(z))*hop+nfft/2
                ids = ((centers-start)//width).astype(np.int64)
                unique,first = np.unique(ids,return_index=True); num = np.diff(np.r_[first,len(z)])
                sums[unique] += np.add.reduceat(meanf,first,axis=0)
                maxima[unique] = np.maximum(maxima[unique],np.maximum.reduceat(maxf,first,axis=0))
                esums[unique] += np.add.reduceat(e,first); emax[unique] = np.maximum(emax[unique],np.maximum.reduceat(e,first))
                weights[unique] += num; processed += len(z)
        used.append(dict(path=p.path,variable=p.variable,source_interval=[lo,hi],recording_interval=[p.offset+lo,p.offset+hi]))
        analyzed_samples+=hi-lo
    if progress:progress(len(parts),len(parts),'完成')
    if not processed and not allow_empty:
        raise ValueError('所选范围没有可用的完整IQ帧，可能落在缺段或已排除的全零区间。')
    covered = weights>0
    means = np.full_like(sums,np.nan,dtype=np.float32); peaks = np.full_like(means,np.nan)
    means[covered] = sums[covered]/weights[covered,None];peaks[covered] = maxima[covered]
    env = np.full(nbins,np.nan,np.float32); envmax = np.full_like(env,np.nan)
    env[covered] = esums[covered]/weights[covered]; envmax[covered] = emax[covered]
    def db(z):return 10*np.log10(np.maximum(z,1e-30))
    return dict(title=plan.title,scope=plan.scope,source_path=plan.source_path,start=start,stop=stop,total_samples=plan.total_samples,
        requested_count=count,nfft=nfft,hop=hop,time_bin_samples=width,frequency_bins=nf,frames=processed,analyzed_samples=analyzed_samples,sources=used,
        mean_db=db(means),peak_db=db(peaks),envelope_db=db(env),envelope_max_db=db(envmax),weights=weights,
        spectrum=(spectrum/max(1,processed)).astype(np.float32))

def save_result(result,path,fs_hz=None,fc_hz=None,fs_source='unknown'):
    path = Path(path)
    arrays = {k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    metadata = {k:v for k,v in result.items() if k not in arrays}
    metadata.update(fs_hz=fs_hz,fc_hz=fc_hz,fs_source=fs_source,clock_status='file metadata' if fs_source=='file' else 'user assumption' if fs_hz else 'unknown')
    np.savez_compressed(path,**arrays,metadata_json=json.dumps(metadata,ensure_ascii=False))
    path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf8')

def load_result(path):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json']));arrays={k:z[k] for k in z.files if k!='metadata_json'}
    return {**meta,**arrays}
