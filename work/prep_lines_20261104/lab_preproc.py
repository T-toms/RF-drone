"""Read-only IQ analysis helpers; frequency arguments Hz except crop center MHz."""
import numpy as np
from scipy import signal, optimize
FS_OUT = 61_440_000
PASS_HZ = 28_000_000
STOP_HZ = 30_720_000

def crop_band(iq, fs_in, fc_in, f_center_mhz):
    x=np.asarray(iq).ravel(); fs_in=float(fs_in)
    x=x*np.exp(-2j*np.pi*(f_center_mhz*1e6-fc_in)*np.arange(len(x))/fs_in)
    # 90 dB Kaiser, explicit pass/stop edges; polyphase kernel at upsampled rate.
    up,down=(1,2) if fs_in==122880000 else (384,625)
    if fs_in not in (122880000,100000000): raise ValueError('unsupported input rate')
    rate=fs_in*up
    n,beta=signal.kaiserord(90,(STOP_HZ-PASS_HZ)/(rate/2))
    h=signal.firwin(n|1,(PASS_HZ+STOP_HZ)/2,fs=rate,window=('kaiser',beta))
    return signal.resample_poly(x,up,down,window=h)

def notch(iq,fs,offset_hz,width_hz):
    x=np.asarray(iq).ravel(); t=np.arange(len(x))/fs
    z=x*np.exp(-2j*np.pi*offset_hz*t)
    w=max(2*width_hz,60000)
    sos=signal.butter(6,w,fs=fs,btype='highpass',output='sos')
    return signal.sosfiltfilt(sos,z)*np.exp(2j*np.pi*offset_hz*t)

def cancel_tone(iq,fs,offset_hz,block_ms=10,search_hz=20000,mean_only=False):
    x=np.asarray(iq).ravel(); y=x.copy(); size=round(fs*block_ms/1000)
    for start in range(0,len(x),size):
        z=x[start:start+size]; t=np.arange(len(z))/fs
        if mean_only:
            y[start:start+size]=z-z.mean(); continue
        mixed=z*np.exp(-2j*np.pi*offset_hz*t)
        dec=max(1,int(fs/200000)); a=signal.resample_poly(mixed,1,dec)
        rate=fs/dec; nfft=2**int(np.ceil(np.log2(max(16,len(a)*8))))
        spec=np.abs(np.fft.fftshift(np.fft.fft(a*np.hanning(len(a)),nfft)))**2
        f=np.fft.fftshift(np.fft.fftfreq(nfft,1/rate)); valid=np.where(abs(f)<=search_hz)[0]
        k=valid[np.argmax(spec[valid])]; log=np.log(spec+1e-30)
        d=0.5*(log[k-1]-log[k+1])/(log[k-1]-2*log[k]+log[k+1]) if 0<k<nfft-1 else 0
        guess=f[k]+d*rate/nfft; td=np.arange(len(a))/rate
        fit=optimize.minimize_scalar(lambda u:-abs(np.vdot(np.exp(2j*np.pi*u*td),a))**2,
          bounds=(guess-1000/ block_ms,guess+1000/block_ms),method='bounded',options={'xatol':0.01})
        tone=np.exp(2j*np.pi*(offset_hz+fit.x)*t)
        amp=np.vdot(tone,z)/len(z)
        y[start:start+size]=z-amp*tone
    return y
