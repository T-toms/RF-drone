from chain_copy import *
def process(b,dest,conjugate=False,shift=0):
 dest=Path(dest)
 if dest.exists():return str(dest)
 assert shutil.disk_usage('E:/').free>20e9
 fs=b['fs_in'];band=b['band'];fo=FS[band];up,down=ratio(fs,band);trim=round(fo*.0002);nout=math.ceil(b['n_in']*up/down);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.partial.npy');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.complex64,shape=(nout-2*trim,));src=Source(b);chunk=round(fs*.05);halo=math.ceil(fs*.002/down)*down
 for s in range(0,b['n_in'],chunk):
  e=min(b['n_in'],s+chunk);a=max(0,s-halo);a=a//down*down;c=min(b['n_in'],e+halo);raw=src.get(a,c);raw=np.conj(raw) if conjugate else raw;y=chain(raw,fs,b['fc_in']-shift,band,a);lo=max(trim,s*up//down);hi=min(nout-trim,math.ceil(e*up/down));out[lo-trim:hi-trim]=y[lo-a*up//down:hi-a*up//down]
 out.flush();del out;tmp.replace(dest);print('processed',dest.name,flush=True);return str(dest)
