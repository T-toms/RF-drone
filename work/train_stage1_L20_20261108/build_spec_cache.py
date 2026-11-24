import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--band');p.add_argument('--split');args=p.parse_args()
allrows=rows(INPUT/'index_L20.csv')
for b in ([args.band] if args.band else BANDS):
    for split in ([args.split] if args.split else ['train','val','test']):
        dest=cachepath(b,split)
        if dest.exists() and dest.with_suffix('.done.json').exists():continue
        rs=[dict(r) for r in allrows if r['band']==b and r['split']==split and r['keep']=='True']
        rs.sort(key=lambda r:(r['block_file'],int(r['start'])))
        start=time.time(); a=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float16,shape=(len(rs),256,256))
        old=None
        for i,r in enumerate(rs):
            if r['block_file']!=old:
                block=np.load(r['block_file'],mmap_mode='r');old=r['block_file']
            lo=int(r['start']); n=int(r['n_samples']);iq=np.array(block[lo:lo+n]);assert len(iq)==n
            a[i]=spec(iq,b);r['cache_index']=i
            if i%100==0:print(b,split,i,len(rs),round(time.time()-start,1),flush=True)
        a.flush();writecsv(CACHE/f'{b}_{split}_spec_meta.csv',rs)
        dump(dest.with_suffix('.done.json'),dict(n=len(rs),seconds=time.time()-start,bytes=dest.stat().st_size))
        if split=='val':examples(b,split)


