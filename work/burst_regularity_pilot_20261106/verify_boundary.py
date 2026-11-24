from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pilot import *
def boundary():
    rec=RECS[0]; r=json.loads((OUT/(rec+'.json')).read_text(encoding='utf8'));df=pd.read_csv(OUT/f'bursts_{rec}.csv')
    chosen=None
    for b in r['boundaries']:
        es=df[(df.t_start_ms<b['boundary_ms'])&(df.t_end_ms>b['boundary_ms'])&(df.parts>=2)]
        if len(es):chosen=(b,es.iloc[0]);break
    b,e=chosen;cut=b['boundary_ms'];ord=int(round(cut/50));folder,session=rec.split('__');tab=pd.read_csv(ROOT/'work/audit_drone_20261023/data_preprocessing_manifest.csv',dtype=str);tab=tab[(tab.folder==folder)&(tab.session==session)&(tab.duplicate=='False')];tab['ord']=tab.epoch.astype(int)*20+tab.seg.astype(int)
    x=np.concatenate([loadmat(RAW/tab[tab.ord==j].iloc[0].path,variable_names=['data'])['data'].ravel().astype('complex64') for j in [ord-1,ord]])
    fs=r['fs'];lo=round((50-1)*fs/1000);hi=round(51*fs/1000);p=stft(x[lo:hi],fs);freq=5800+fft.fftshift(fft.fftfreq(1024,1/fs))/1e6
    fig,ax=plt.subplots(figsize=(12,6));ax.imshow(p.T-r['floor_db'],origin='lower',aspect='auto',extent=[cut-1,cut+1,freq[0],freq[-1]],vmin=-5,vmax=35,cmap='magma');ax.add_patch(Rectangle((e.t_start_ms,e.f_lo_MHz),e.duration_ms,e.bandwidth_MHz,fill=False,edgecolor='cyan',linewidth=2));ax.axvline(cut,color='white',ls='--');ax.set(xlim=(cut-1,cut+1),ylim=(e.f_lo_MHz-2,e.f_hi_MHz+2),xlabel='Time (ms)',ylabel='Frequency (MHz)',title=f'Cross-boundary union: one output event, {int(e.parts)} parts');fig.tight_layout();fig.savefig(OUT/'boundary_merge.png',dpi=160);plt.close(fig)
    dump(OUT/'boundary_evidence.json',dict(recording=rec,boundary_ms=cut,output_event=e.to_dict(),matches_in_output=len(es),numeric_pass=bool(e.t_start_ms<cut<e.t_end_ms and e.parts>=2)))
if __name__=='__main__':boundary()
