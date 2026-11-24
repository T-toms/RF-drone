from pathlib import Path
import numpy as np
from scipy import io,signal,fft
import math,csv,json,shutil,time,sys
P=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent;R=Path('G:/drone')
FS={'2.4GHz':61440000,'5.8GHz':100000000};FC={'2.4GHz':2430.,'5.8GHz':5800.};WIDTH={'2.4GHz':25.,'5.8GHz':45.}
FILTERS=dict(np.load(P/'work/crop_slice_20261107/filters.npz'))
NOTCHES=list(csv.DictReader((P/'work/crop_slice_20261107/notch_table.csv').open(encoding='utf-8-sig')))
for r in NOTCHES:
 for k in ['freq_mhz','half_width_khz']:r[k]=float(r[k])
KERNELS={}
def notch(iq,fs,offset_hz,width_hz):
    x=np.asarray(iq).ravel(); t=np.arange(len(x))/fs
    z=x*np.exp(-2j*np.pi*offset_hz*t)
    w=max(2*width_hz,60000)
    sos=signal.butter(6,w,fs=fs,btype='highpass',output='sos')
    return signal.sosfiltfilt(sos,z)*np.exp(2j*np.pi*offset_hz*t)
class Source:
 def __init__(self,b):self.b=b;self.cache={};self.ends=np.cumsum([p['end']-p['start'] for p in b['parts']])
 def get(self,s,e):
  pieces=[];offset=0;used=set()
  for i,p in enumerate(self.b['parts']):
   end=int(self.ends[i]);lo=max(s,offset);hi=min(e,end)
   if hi>lo:
    if i not in self.cache:
     key='data' if self.b['dataset']=='data' else 'IQ';self.cache[i]=io.loadmat(R/p['path'],variable_names=[key])[key].ravel().astype(np.complex64)
    pieces.append(self.cache[i][p['start']+lo-offset:p['start']+hi-offset]);used.add(i)
   offset=end
  self.cache={i:v for i,v in self.cache.items() if i in used};return np.concatenate(pieces)

def ratio(fs,b):
 g=math.gcd(fs,FS[b]);return FS[b]//g,fs//g

def kernel(fs,b):
 key=(fs,b)
 if key not in KERNELS:
  up,down=ratio(fs,b);rate=fs*up; stop=min(fs,FS[b])/2; pas=WIDTH[b]*1e6;n,beta=signal.kaiserord(100,(stop-pas)/(rate/2));KERNELS[key]=signal.firwin(n|1,(stop+pas)/2,fs=rate,window=('kaiser',beta))
 return KERNELS[key]

def chain(x,fs,fc,b,start=0,notches=True):
 n=np.arange(len(x),dtype=np.float64)+start;z=x*np.exp(-2j*np.pi*((FC[b]-fc)*1e6/fs)*n);up,down=ratio(fs,b)
 if up!=down:z=signal.resample_poly(z,up,down,window=kernel(fs,b))
 h=FILTERS[b];z=signal.fftconvolve(z,h,mode='same')
 if notches:
  for r in NOTCHES:
   if r['band']==b:z=notch(z,FS[b],(r['freq_mhz']-FC[b])*1e6,r['half_width_khz']*500.)
 return z.astype(np.complex64)
