import sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from extract import *
chosen=noise_selection(inputs())
saved=rows(CACHE/'noise_meta.csv')
assert [r['sample_id'] for r in saved]==[r['sample_id'] for r in chosen[:len(saved)]]
array=np.load(CACHE/'noise_features.npy',mmap_mode='r')
assert array.shape==(len(chosen),5,19)
for i,r in enumerate(chosen):
    r['noise_seed']=str(int.from_bytes(hashlib.sha256(('2026|'+r['sample_id']).encode()).digest()[:8],'little'))
    clean=np.load(CACHE/(r['meta_name']+'_features.npy'),mmap_mode='r')[int(r['row_index'])]
    assert np.array_equal(array[i,0],clean,equal_nan=True),(i,r['sample_id'])
writecsv(CACHE/'noise_meta.csv',chosen)
seconds=(CACHE/'noise_features.npy').stat().st_mtime-(OUT/'extraction.json').stat().st_mtime
dump(OUT/'noise_runtime.json',dict(seconds=seconds,read_bytes=sum(int(r['n_samples'])*8 for r in chosen),n=len(chosen),snr=['clean',0,-5,-10,-15],timing_note='Estimate from noise array completion minus extraction completion, includes stage launch overhead',recovery='Mixed meta schemas CSV field union; deterministic selection restored and all clean rows matched exactly',matched_clean_rows=len(chosen)))
print('RECOVERED',len(chosen),'rows; clean alignment exact; seconds_estimate',seconds)
