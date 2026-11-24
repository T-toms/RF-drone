# Functions copied verbatim from Q pilot; never import or execute pilot.py.
from pathlib import Path
import numpy as np
from scipy import io
import hashlib
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
PROTECTED=['crop_slice_20261107','side_M_notchfill_20261110','main_W_window_20261110']
def snapshot():
    d = {}
    for name in PROTECTED:
        assert (ROOT/'work'/name).is_dir()
        for p in sorted((ROOT/'work'/name).rglob('*')):
            if p.is_file():
                h = hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
                d[str(p.relative_to(ROOT))] = {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    return d

def read_iq(c):
    arrays=[]
    for p in c['parts']:
        path=Path(p['path']); assert path.drive.upper()=='G:'
        if c['source']=='data':
            # One bounded 50-ms MAT container at a time (49 MB), not the recording.
            a=io.loadmat(path,variable_names=['data'])['data'].ravel()
            arrays.append(a[p['start']:p['start']+p['count']].astype(np.complex64,copy=True))
            del a
        else:
            with path.open('rb') as f:
                f.seek(p['start']*4); a=np.fromfile(f,dtype='<i2',count=p['count']*2)
            assert len(a)==p['count']*2
            arrays.append(a[::2].astype(np.float32)+1j*a[1::2].astype(np.float32))
    x=np.concatenate(arrays) if len(arrays)>1 else arrays[0]
    assert len(x)==c['length']; return x

def merge(boxes,g,nt):
    ordered=sorted(boxes,key=lambda b:b['f_start_mhz']); groups=[]
    for b in ordered:
        if not groups or b['f_start_mhz']-groups[-1]['hi']>=g:
            groups.append(dict(lo=b['f_start_mhz'],hi=b['f_end_mhz'],boxes=[b]))
        else:
            groups[-1]['hi']=max(groups[-1]['hi'],b['f_end_mhz']); groups[-1]['boxes'].append(b)
    bands=[]
    for i,h in enumerate(groups,1):
        active=np.zeros(nt,bool)
        for b in h['boxes']: active[b['t0']:b['t1']]=True
        bands.append(dict(band_id=i,f_start_mhz=h['lo'],f_end_mhz=h['hi'],width_mhz=h['hi']-h['lo'],
                          time_duty=float(active.mean()),box_count=len(h['boxes']),max_excess_db=max(b['max_excess_db'] for b in h['boxes'])))
    return bands

def scene_for(c):
    p=c['parts'][0]['path']
    return '07-11家里' if '2026-07-11_home' in p else ('07-10实验室' if '2026-07-10_lab' in p else ('07-09城区' if c['source']=='raw' else '实验室'))
