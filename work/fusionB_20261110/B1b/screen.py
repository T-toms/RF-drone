import os,sys,time,json,csv
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';sys.dont_write_bytecode=True
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
O=Path(__file__).resolve().parent;P=O.parents[2];sys.path.insert(0,str(P/'work/fusionB_20261110'));sys.path.insert(0,str(O))
from extract_copy import feature,NAMES,config
from generate import dump
FS=61440000;L=1228800;F=2430+(np.arange(1024)-512)*.06;MASK=config('2.4GHz')[2];WIN=np.hanning(1024)
FB=np.floor(F-2405).astype(int);BINS=[np.flatnonzero(MASK&(FB==i)) for i in range(50)]
CACHE=Path('E:/fusionB_features_20261110')
def rawz(r):return np.array(np.load(r['block_file'],mmap_mode='r')[int(r['start']):int(r['start'])+int(r['n_samples'])])
from chain_copy import notch
def getz(r):
 z=rawz(r)
 return notch(z,FS,(2432.9412-2430)*1e6,50000)
def spectrum(z):
 z=np.asarray(z,dtype=np.complex128);z=z-z.mean();z=z/np.sqrt(np.mean(abs(z)**2));fr=z[:len(z)//1024*1024].reshape(-1,1024);return np.fft.fftshift(abs(np.fft.fft(fr*WIN,axis=1))**2,axes=1)
def cells(z):
 ps=spectrum(z);ts=np.floor((np.arange(len(ps))+.5)*1024/(FS*.001)).astype(int)
 q=np.stack([ps[:,ix].mean(1) for ix in BINS],axis=1)
 return np.stack([q[ts==j].mean(0) for j in range(20)])
def quiet(z):
 c=cells(z);exc=float(10*np.log10(c.max()/np.median(c)));return exc<=3,exc

def sliced(bs,dataset):
 out=[]
 for b in bs:
  n=len(np.load(b['block_file'],mmap_mode='r'))
  for st in range(0,n-L+1,L):out.append(dict(block_file=b['block_file'],dataset=dataset,band='2.4GHz',label='background',group_id=b['group_id'],nn=b.get('nn',str(b['k'])),sample_id=f'bgB_L20_{len(out):08d}' if dataset=='dataB_mirror' else f'B2440_{len(out):08d}',start=st,n_samples=L,keep=True))
 return out

def evaluate(rr,features=False):
 t=time.time();out=[];vals=[]
 def work(r):
  z=getz(r);q,ex=quiet(z);return dict(r,quiet=q,max_excess_db=ex),feature(z,'2.4GHz') if features else None
 with ThreadPoolExecutor(max_workers=4) as pool:
  for i,(r,v) in enumerate(pool.map(work,rr)):
   out.append(r)
   if features:vals.append(v)
   if i%250==0:print('EVAL',i,len(rr),round(time.time()-t,1),flush=True)
 return out,np.array(vals)
