import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'work/train_stage1_L20_20261108'))
from common import *
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
OLD=Path('E:/drone_lab_spec_L20_20261108')
BG=Path('E:/drone_lab_spec_bgA_L20_20261109')
DEST=Path('E:/side_L_wifimix_20261110')
DEST.mkdir(exist_ok=True)
def bgrows(split):
    return [(i,r) for i,r in enumerate(rows(BG/'bgA_meta.csv')) if r['keep']=='True' and r['split']==split]
def original_joined(split):
    a=np.load(OLD/f'2.4GHz_{split}_spec.npy');rs=rows(OLD/f'2.4GHz_{split}_spec_meta.csv')
    br=bgrows(split);ba=np.load(BG/'bgA_spec.npy',mmap_mode='r')
    x=np.concatenate([a,ba[[i for i,r in br]]]);rs=rs+[r for i,r in br]
    y=torch.tensor([r['label']=='drone' for r in rs],dtype=torch.long)
    kinds=torch.tensor([2 if r['dataset']=='dataA_mirror' else (0 if r['label']=='drone' else 1) for r in rs])
    return torch.from_numpy(x).unsqueeze(1),y,rs,kinds
def getmodel(folder,s,variant=None):
    ck=torch.load(folder/(f'2.4GHz_{variant}_seed{s}_best.pt' if variant else f'2.4GHz_seed{s}_best.pt'),map_location='cpu',weights_only=False)
    m=ResNet18(1,2).cuda().to(memory_format=torch.channels_last);m.load_state_dict(ck['model']);m.eval()
    return m,ck

MIXCACHE=DEST/'cache'
MIXCACHE.mkdir(exist_ok=True)
def joined(split,variant='La'):
    tx,ty,tr,tk=original_joined(split)
    if split!='train':return tx,ty,tr,tk
    extras=[('drone',3)] + ([('bg',4)] if variant=='Lb' else [])
    xs=[tx.numpy()[:,0]];ys=[ty];ks=[tk]
    for name,kind in extras:
        a=np.load(MIXCACHE/f'mix_train_{name}.npy',mmap_mode='r');rs=rows(OUT/f'mix_train_{name}.pairs.csv')
        xs.append(a);ys.append(torch.full((len(a),),int(name=='drone'),dtype=torch.long));ks.append(torch.full((len(a),),kind,dtype=torch.long));tr+=rs
    return torch.from_numpy(np.concatenate(xs)).unsqueeze(1),torch.cat(ys),tr,torch.cat(ks)
