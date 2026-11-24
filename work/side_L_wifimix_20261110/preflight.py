import sys
sys.dont_write_bytecode=True
from pathlib import Path
O=Path(__file__).resolve().parent
ROOT=O.parents[1]
sys.path.insert(0,str(ROOT/'work/train_stage1_L20_20261108'))
from common import *
import hashlib
OLD=Path('E:/drone_lab_spec_L20_20261108'); BG=Path('E:/drone_lab_spec_bgA_L20_20261109')
PROTECTED=['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109','train_stage1_L20_bgA_20261109']
def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for d in PROTECTED for p in sorted((ROOT/'work'/d).rglob('*')) if p.is_file()}
def iq(r):
    a=np.load(r['block_file'],mmap_mode='r');lo=int(r['start']);return np.array(a[lo:lo+int(r['n_samples'])])
def power(z):return float(np.mean(np.abs(z)**2,dtype=np.float64))
def mixing(d,b,x):
    pd=power(d);pb=power(b);g=float(np.sqrt(pb/max(pd,1e-30)*10**(x/10)));return b+g*d,g,pd,pb
def main():
    dump(O/'protected_before.json',hashes())
    tr=rows(OLD/'2.4GHz_train_spec_meta.csv');br=rows(BG/'bgA_meta.csv')
    did=[i for i,r in enumerate(tr) if r['dataset']=='data' and r['label']=='drone'];vid=[i for i,r in enumerate(tr) if r['dataset']=='v3' and r['label']=='drone'];bid=[i for i,r in enumerate(tr) if r['dataset']=='v3' and r['label']=='background']
    ai=[i for i,r in enumerate(br) if r['keep']=='True' and r['split']=='train']
    counts=[len(did),len(vid),len(bid)]+[sum(r['keep']=='True' and r['split']==s for r in br) for s in ['train','val','test']]
    checks={'counts':counts};assert counts==[2832,152,234,2861,990,1287],checks
    gs={r['group_id']:r['split'] for r in rows(INPUT/'group_split.csv')}
    checks['split_mismatches']=sum(gs.get(r['group_id'])!='train' for r in [tr[i] for i in did+bid]+[br[i] for i in ai]);assert checks['split_mismatches']==0
    arr=np.load(OLD/'2.4GHz_train_spec.npy',mmap_mode='r');ba=np.load(BG/'bgA_spec.npy',mmap_mode='r');rng=np.random.default_rng(20261110)
    checks['recompute']=[]
    for name,ids,rs,a in [('data',did,tr,arr),('v3_background',bid,tr,arr),('bgA',ai,br,ba)]:
        for i in rng.choice(ids,3,replace=False):
            diff=float(np.max(np.abs(spec(iq(rs[i]),'2.4GHz').astype(float)-a[i].astype(float))))
            checks['recompute'].append(dict(source=name,sample_id=rs[i]['sample_id'],max_abs=diff));assert diff<=.01,checks
    te=rows(OLD/'2.4GHz_test_spec_meta.csv');lookup={r['sample_id']:r for r in te};mix=np.load(OLD/'2.4GHz_mix_data_drone.npy',mmap_mode='r');sources=[r for r in te if r['dataset']=='data' and r['label']=='drone'];si={r['sample_id']:i for i,r in enumerate(sources)}
    checks['mix_recompute']=[]
    for r in rows(OLD/'2.4GHz_mix_data_drone.pairs.csv')[:3]:
        d=iq(lookup[r['sample_id']]);b=iq(lookup[r['background_sample_id']]);z=spec(b+float(r['g'])*d,'2.4GHz');k=[-10,-5,0,5,10,15,20].index(int(r['x_db']))
        diff=float(np.max(np.abs(z.astype(float)-mix[si[r['sample_id']],k].astype(float))));checks['mix_recompute'].append(diff);assert diff<=.01,checks
    ck=torch.load(Path('E:/drone_lab_stage1_L20_bgA_20261109/2.4GHz_seed2026_best.pt'),map_location='cpu',weights_only=False)
    m=ResNet18(1,2).cuda().to(memory_format=torch.channels_last);m.load_state_dict(ck['model'])
    probs=predict(m,torch.from_numpy(np.load(OLD/'2.4GHz_test_spec.npy')).unsqueeze(1))[:,1];old={r['sample_id']:float(r['p_drone']) for r in rows(ROOT/'work/train_stage1_L20_bgA_20261109/test_predictions_2.4GHz_seed2026.csv')}
    checks['checkpoint_max_abs']=max(abs(float(p)-old[r['sample_id']]) for p,r in zip(probs,te));assert checks['checkpoint_max_abs']<1e-3,checks
    dump(O/'preflight.json',checks);print(json.dumps(checks,ensure_ascii=False),flush=True)
    del m;torch.cuda.empty_cache()
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    rng=np.random.default_rng(20261110);chosen=[]
    for model in sorted({tr[i]['model'] for i in did}):chosen.append(('data '+model,tr[int(rng.choice([i for i in did if tr[i]['model']==model]))]))
    assert len(chosen)==11
    chosen += [('bgA '+str(j+1),br[int(i)]) for j,i in enumerate(rng.choice(ai,2,replace=False))]
    background=next(r for r in tr if r['group_id']=='v3:20260709-020956_ambient' and r['label']=='background');b=iq(background);bs=spec(b,'2.4GHz')
    fig,axs=plt.subplots(len(chosen),6,figsize=(24,3*len(chosen)));duty=[]
    for axes,(name,r) in zip(axs,chosen):
        d=iq(r);z=spec(d,'2.4GHz');fraction=float((z.astype(float)>np.median(z)+3).mean());duty.append(dict(model=name,sample_id=r['sample_id'],burst_duty=fraction))
        image(axes[0],z,'2.4GHz',name+' 原片');image(axes[1],bs,'2.4GHz','统一 v3 训练背景')
        for ax,x in zip(axes[2:],[-5,0,5,15]):image(ax,spec(mixing(d,b,x)[0],'2.4GHz'),'2.4GHz',f'叠加 {x} dB')
    fig.suptitle('任务 L 试算：原片、环境 WiFi 背景与不同功率比叠加',fontsize=20);fig.tight_layout(rect=(0,0,1,.99));fig.savefig(O/'fig_L_pilot_2.4GHz.png',dpi=110);plt.close(fig)
    writecsv(O/'pilot_duty.csv',duty);dump(O/'pilot_done.json',dict(background_sample_id=background['sample_id'],models=len(chosen),checks=checks));print('PILOT DONE',flush=True)
if __name__=='__main__':main()
