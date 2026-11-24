"""Read-only: RF0(2440MHz)/RF1(5800MHz) PSD + per-window power over time for indoor classes."""
import sys, json, h5py, numpy as np
from pathlib import Path
root = Path(sys.argv[1]); out = Path(sys.argv[2])
files = [f"{t}_{s}.mat" for t in ("T1010","T1011","T1100") for s in ("S0100","S0101","S0110","S0111","S1000","S1001")]
NF=1024; W=1_000_000
res={}
for fn in files:
    p=root/fn
    if not p.exists(): print("missing",fn); continue
    with h5py.File(p,'r') as h:
        n=min(h[k].shape[-1] for k in ("RF0_I","RF1_I"))
        nwin=n//W
        r={"nwin":int(nwin)}
        for ch in ("RF0","RF1"):
            pw=[];psd=np.zeros(NF)
            for w in range(nwin):
                s=w*W
                x=h[ch+"_I"][0,s:s+65536]+1j*h[ch+"_Q"][0,s:s+65536]
                pw.append(float(np.sqrt(np.mean(np.abs(x)**2))))
                X=np.fft.fftshift(np.abs(np.fft.fft(x[:NF*64].reshape(64,NF)*np.hamming(NF),axis=1))**2,axes=1).mean(0)
                psd+=X
            psd/=max(nwin,1); db=10*np.log10(psd+1e-20)
            floor=np.median(db)
            r[ch]={"rms_per_win":pw,"psd_db":db.tolist(),"floor":float(floor),
                   "occ_frac_6dB":float(np.mean(db>floor+6)),"peak_over_floor":float(db.max()-floor)}
        res[fn]=r
        print(fn,nwin,"RF0 rms med %.4f occ %.2f pk %.1f | RF1 rms med %.4f occ %.2f pk %.1f"%(
            np.median(r["RF0"]["rms_per_win"]),r["RF0"]["occ_frac_6dB"],r["RF0"]["peak_over_floor"],
            np.median(r["RF1"]["rms_per_win"]),r["RF1"]["occ_frac_6dB"],r["RF1"]["peak_over_floor"]),flush=True)
json.dump(res,open(out,'w'))
