"""Fixed absolute-band IQ pipeline. Source root is read-only; bounded-memory overlap processing."""
from pathlib import Path
import sys, json, csv, re, math, shutil, time, importlib.util, collections,threading,concurrent.futures
import numpy as np
from scipy import io,signal,fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
P=Path(__file__).resolve().parents[2]; O=Path(__file__).resolve().parent; R=Path('G:/drone'); E=Path('E:/drone_lab_crop_20261107')
spec=importlib.util.spec_from_file_location('lab_preproc',P/'work/prep_lines_20261104/lab_preproc.py'); lab=importlib.util.module_from_spec(spec);spec.loader.exec_module(lab)
FS={'2.4GHz':61440000,'5.8GHz':100000000}; FC={'2.4GHz':2430.,'5.8GHz':5800.}; WIDTH={'2.4GHz':25.,'5.8GHz':45.}
DONE_LOCK=threading.Lock()
MODELS=['mini4 pro','Flip','mini 4k','neo','neo2','air 3s','mini5 pro','mavic 4','avata2','evo lite 6k','evo ii pro v3','avata','atom se','atom']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(p,rs):
 if not rs:return
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def disk():
 free=shutil.disk_usage(E.anchor).free;print('DISK_FREE_GB',free/1e9,flush=True)
 if free<20e9:raise RuntimeError('DISK_BELOW_20GB: stop and ask coordinator')
def design():
 table=[]; ls=rows(P/'work/prep_lines_20261104/lines_classified.csv')
 for band in FS:
  lo,hi=FC[band]-WIDTH[band],FC[band]+WIDTH[band]; a=sorted([r for r in ls if r['classification']=='persistent_tone_candidate' and lo<=float(r['freq_mhz'])<=hi],key=lambda r:float(r['freq_mhz'])); clusters=[]
  for r in a:
   if clusters and float(r['freq_mhz'])-np.median([float(x['freq_mhz']) for x in clusters[-1]])<=.020:clusters[-1].append(r)
   else:clusters.append([r])
  centers=[2440.,2450.] if band=='2.4GHz' else [5800.]
  for c in centers:
   rr=[r for r in a if abs(float(r['freq_mhz'])-c)<=.025];table.append(dict(band=band,freq_mhz=c,half_width_khz=max(2*np.median([float(r['width_khz']) for r in rr]),100.) if rr else 100.,origin='fc_spike',sources=';'.join(sorted({r['group'] for r in rr}))))
  for cl in clusters:
   f=float(np.median([float(r['freq_mhz']) for r in cl])); sources={r['recording'] for r in cl}
   if len(sources)>=2 and all(abs(f-c)>.025 for c in centers):table.append(dict(band=band,freq_mhz=f,half_width_khz=max(2*np.median([float(r['width_khz']) for r in cl]),60.),origin='stable_line',sources=';'.join(sorted({r['group'] for r in cl}))))
 filters={};params={}
 for b in FS:
  stop=27.5 if b=='2.4GHz' else 48.; n,beta=signal.kaiserord(90,(stop-WIDTH[b])*1e6/(FS[b]/2)); h=signal.firwin(n|1,(stop+WIDTH[b])*5e5,fs=FS[b],window=('kaiser',beta));filters[b]=h
  f,H=signal.freqz(h,worN=262144,fs=FS[b]);db=20*np.log10(np.maximum(abs(H),1e-20));params[b]=dict(taps=len(h),pass_ripple_db=float(np.ptp(db[f<=WIDTH[b]*1e6])),stop_atten_db=float(-db[f>=stop*1e6].max()),trim_samples=round(FS[b]*.0002))
 np.savez(O/'filters.npz',**filters);write(O/'notch_table.csv',table)
 resampling=[]
 for fs,b in [(122880000,'2.4GHz'),(122880000,'5.8GHz'),(100000000,'2.4GHz'),(100000000,'5.8GHz')]:
  g=math.gcd(fs,FS[b]);up,down=FS[b]//g,fs//g;stop=min(fs,FS[b])/2;pas=WIDTH[b]*1e6
  n,beta=signal.kaiserord(100,(stop-pas)/(fs*up/2));taps=(n|1) if up!=down else 1
  resampling.append(dict(fs_in=fs,band=b,up=up,down=down,kaiser_attenuation_db=100,pass_hz=pas,stop_hz=stop,taps=taps,equivalent_output_samples=math.ceil(taps/down)))
 dump(O/'config.json',dict(fs_out=FS,centers_mhz=FC,band_halfwidth_mhz=WIDTH,filters=params,resampling=resampling,notches=table,notch_implementation='lab_preproc.notch: complex mixing + Butterworth order 6 highpass + sosfiltfilt; width argument = half_width/2',chunk_ms=50,halo_ms=2,cell_ms=1,nfft={'2.4GHz':512,'5.8GHz':1024},bin_hz={'2.4GHz':120000,'5.8GHz':97656.25},folding=False,normalization=False,screen_thresholds=dict(appear=.01,persist=.9,weak=.25,power_db=6,floor_db=3,out_min=.05,out_mult=3,threshold_db=10),seed=2026))
 return filters,table
FILTERS,NOTCHES=design()
def plan():
 sg=rows(P/'work/prep_lines_20261104/source_groups.csv');by={r['path'].replace('\\','/'):r for r in sg if r['include']=='True'}; bs=[];excluded=[];counts=collections.Counter()
 for run in read(P/'work/audit_drone_20261023/data_usable_runs.json')['runs']:
  parts=run['parts']; assert all(x['path'].replace('\\','/') in by for x in parts); r=by[parts[0]['path'].replace('\\','/')];b=r['band'];gid=r['group_id']; counts[gid]+=1;code=int(re.search(r'\d{3}',run['folder'])[0]);n=run['samples']
  bs.append(dict(dataset='data',band=b,label='drone',model=MODELS[code-1],state='flight' if run['folder'].startswith('F') else 'standby',bw_label=(re.search(r'_(\d+M)',run['folder']) or ['','unknown'])[1],group_id=gid,parts=[dict(path=x['path'].replace('\\','/'),start=x['start_sample'],end=x['end_sample_exclusive']) for x in parts],fs_in=122880000,fc_in=2400. if b=='2.4GHz' else 5800.,n_in=n,source_start=run['start_sample'],k=counts[gid]))
 cont=read(O/'contiguity.json');sweep=next(x['contiguous'] for x in cont if x['case']=='075302'); buckets=collections.defaultdict(list)
 for r in sg:
  if r['dataset']!='v3':continue
  f=float(re.search(r'_iq_[\d.]+M_([\d.]+)M_',r['path'])[1]);b=r['band']
  if r['include']!='True' or (f not in [2440.,2450.] if b=='2.4GHz' else f!=5800.):excluded.append(dict(**r,fc_filename_mhz=f,task_f_reason='not_included' if r['include']!='True' else 'crop_band_not_covered'));continue
  key=re.sub(r'_s\d{4}_new.mat$','',r['path']); join='_250M_' in key and 'noise_adrv' not in key or ('075302' in key and sweep)
  buckets[(key if join else r['path'])].append((r,f))
 for key,rs in sorted(buckets.items()):
  rs.sort(key=lambda v:v[0]['path']);r,f=rs[0];parts=[]
  for v,_ in rs:
   m=io.whosmat(R/v['path']); n=next(math.prod(shape) for name,shape,t in m if name=='IQ');parts.append(dict(path=v['path'],start=0,end=n))
  gid=r['group_id'];counts[gid]+=1;bs.append(dict(dataset='v3',band=r['band'],label=r['label'],model='none' if r['label']=='background' else ('fpv' if 'digital_fpv' in key else Path(r['path']).parts[2]),state='unknown',bw_label='unknown',group_id=gid,parts=parts,fs_in=100000000,fc_in=f,n_in=sum(x['end'] for x in parts),source_start=0,k=counts[gid]))
 for b in bs:
  gid=re.sub(r'[^A-Za-z0-9_.@-]','_',b['group_id']);b['block_file']=str(E/b['band']/b['dataset']/f'{gid}__blk{b["k"]}.npy')
 dump(O/'plan.json',bs);write(O/'excluded_files.csv',excluded);return bs
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
KERNELS={}
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
   if r['band']==b:z=lab.notch(z,FS[b],(r['freq_mhz']-FC[b])*1e6,r['half_width_khz']*500.)
 return z.astype(np.complex64)
def process(b):
 dest=Path(b['block_file']);meta=O/'block_meta'/f'{dest.stem}.json';meta.parent.mkdir(exist_ok=True)
 if meta.exists() and dest.exists():return read(meta)
 fs=b['fs_in'];band=b['band'];fo=FS[band];up,down=ratio(fs,band);trim=round(fo*.0002);nout=math.ceil(b['n_in']*up/down);dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.partial.npy');out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.complex64,shape=(nout-2*trim,));src=Source(b);chunk=round(fs*.05);halo=math.ceil(fs*.002/down)*down
 for s in range(0,b['n_in'],chunk):
  e=min(b['n_in'],s+chunk);a=max(0,s-halo);a=a//down*down;c=min(b['n_in'],e+halo);y=chain(src.get(a,c),fs,b['fc_in'],band,a);lo=max(trim,s*up//down);hi=min(nout-trim,math.ceil(e*up/down));out[lo-trim:hi-trim]=y[lo-a*up//down:hi-a*up//down]
 out.flush();del out;tmp.replace(dest);z=np.load(dest,mmap_mode='r');cs=round(fo/1000);powers=[float(10*np.log10(np.mean(abs(z[s:s+cs])**2)+1e-30)) for s in range(0,len(z)-cs+1,cs)];m={k:b[k] for k in ['block_file','dataset','band','label','model','state','bw_label','group_id']};m.update(source_paths=';'.join(x['path'] for x in b['parts']),fs_out=fo,n_samples=len(z),t0_in_source_ms=b['source_start']/fs*1000+trim/fo*1000,power_db_median=float(np.median(powers)));dump(meta,m)
 with DONE_LOCK:
  with (O/'blocks_done.txt').open('a',encoding='utf-8') as f:f.write(b['block_file']+'\n')
 return m
def smoke_select(bs):
 return [next(b for b in bs if b['dataset']=='data' and x in b['group_id']) for x in ['012_2.4GHz_10M-signal__2026-09-20-18-58-55','007_5.8GHz_20M-signal__2026-09-20-20-34-08']]+[next(b for b in bs if b['dataset']=='v3' and c(b)) for c in [lambda b:'consumer_drone/mini5pro' in b['parts'][0]['path'] and b['fc_in']==2440,lambda b:'07-11_home' in b['parts'][0]['path'] and b['fc_in']==2450,lambda b:'07-09_urban' in b['parts'][0]['path'] and b['label']=='background' and b['fc_in']==5800]]
def smoke_plot(m):
 z=np.load(m['block_file'],mmap_mode='r')[:round(m['fs_out']*.05)]; nfft=2048;_,t,S=signal.spectrogram(z,fs=m['fs_out'],window='hann',nperseg=nfft,noverlap=0,mode='psd',return_onesided=False);S=10*np.log10(np.maximum(np.fft.fftshift(S,axes=0),1e-30));f=np.fft.fftshift(np.fft.fftfreq(nfft,1/m['fs_out']))/1e6+FC[m['band']];fig,ax=plt.subplots(figsize=(12,5));v=np.percentile(S,99.5);im=ax.pcolormesh(t*1000,f,S,vmin=v-55,vmax=v,shading='auto');fig.colorbar(im,ax=ax,label='dB');ax.set(xlabel='Time (ms)',ylabel='Absolute frequency (MHz)',title=Path(m['block_file']).stem)
 for r in NOTCHES:
  if r['band']==m['band']:ax.axhline(r['freq_mhz'],color='red',ls='--',lw=.7)
 for edge in [-1,1]:ax.axhline(FC[m['band']]+edge*WIDTH[m['band']],color='white',ls=':')
 fig.tight_layout();fig.savefig(O/('smoke_'+Path(m['block_file']).stem+'.png'),dpi=120);plt.close(fig)
def main():
 disk();bs=read(O/'plan.json') if (O/'plan.json').exists() else plan();mode=sys.argv[1]
 if mode=='plan':print(len(bs));return
 if mode=='smoke':bs=smoke_select(bs)
 if mode=='full':assert (O/'smoke_review.json').exists(),'Visual smoke review required'
 if mode=='full':
  # Independent blocks, three bounded workers; all phase gates remain sequential.
  done={};it=iter(bs);started={};completed_groups=set()
  def work(b):
   disk();t=time.time();return process(b),time.time()-t
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for _ in range(3):
    b=next(it,None)
    if b is not None:started[pool.submit(work,b)]=b
   while started:
    finished,_=concurrent.futures.wait(started,return_when=concurrent.futures.FIRST_COMPLETED)
    for future in finished:
     b=started.pop(future)
     try:m,elapsed=future.result()
     except BaseException:
      for pending in started:pending.cancel()
      raise
     done[b['block_file']]=m;print(len(done),len(bs),Path(b['block_file']).stem,'seconds',round(elapsed,2),flush=True);completed_groups.add(b['group_id'])
     if len(completed_groups)%10==0:disk()
     b=next(it,None)
     if b is not None:started[pool.submit(work,b)]=b
  write(O/'blocks.csv',[done[b['block_file']] for b in bs]);print('CROP_COMPLETE full',flush=True);return
 done=[];last=None;ng=0
 for i,b in enumerate(bs):
  if b['group_id']!=last:
   ng+=1;last=b['group_id']
   if ng%10==0:disk()
  t=time.time();m=process(b);done.append(m);print(i+1,len(bs),Path(b['block_file']).stem,'seconds',round(time.time()-t,2),flush=True)
  if mode=='smoke':smoke_plot(m)
 write(O/('smoke_blocks.csv' if mode=='smoke' else 'blocks.csv'),done);print('CROP_COMPLETE',mode,flush=True)
if __name__=='__main__':main()
