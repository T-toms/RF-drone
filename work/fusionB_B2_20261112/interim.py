from evaluate import *
def run():
 result=[];perms=[];deps=[]
 for b in ['2.4GHz','5.8GHz']:
  data=sets(b)
  for g in ['R','Z','F','F5']:
   ck=torch.load(CACHE/f'{g}_{b}_seed2026_best.pt',map_location='cpu',weights_only=False)
   m=(ResNet18(1,2) if g=='R' else Fusion(5 if g=='F5' else 10)).cuda().to(memory_format=torch.channels_last);m.load_state_dict(ck['model']);m.eval()
   for name in (['test','bgA_test','C1','notched_noise'] if b=='2.4GHz' else ['test']):
    x,raw,rs=data[name];f=transform(raw,ck['normalization']);p=predict(m,x,f,g)
    result += [r for r in metrics(b,g,2026,name,rs,p,ck['thr_fa1']) if r['category'] in ['main','receiver'] or (r['category']=='v3_model_recall' and r['subset'] in ['mavic2','mini3pro'])]
    if g in ['F','F5'] and name=='test':
     baseline={(r['mode'],r['subset']):r['rate'] for r in metrics(b,g,2026,name,rs,p,ck['thr_fa1']) if r['category']=='main'}
     for j in (F5COLS if g=='F5' else range(10)):
      rng=np.random.default_rng(20261116)
      for rep in range(5):
       ff=f.copy();ff[:,j]=f[rng.permutation(len(f)),j];pp=predict(m,x,ff,g)
       for r in metrics(b,g,2026,name,rs,pp,ck['thr_fa1']):
        if r['category']=='main' and r['subset']=='v3_drone_recall':perms.append(dict(r,feature=NAMES[j],repeat=rep,delta=r['rate']-baseline[r['mode'],r['subset']]))
     for method in ['shuffle_all','zero']:
      rng=np.random.default_rng(20261112)
      for rep in range(5 if method=='shuffle_all' else 1):
       ff=f[rng.permutation(len(f))] if method=='shuffle_all' else np.zeros_like(f);pp=predict(m,x,ff,g)
       for r in metrics(b,g,2026,name,rs,pp,ck['thr_fa1']):
        if r['category']=='main' and r['subset'] in ['v3_drone_recall','nonG_miss']:deps.append(dict(r,dependence=method,repeat=rep,delta=r['rate']-baseline[r['mode'],r['subset']]))
   del m;torch.cuda.empty_cache()
 writecsv(OUT/'interim_seed2026.csv',result)
 writecsv(OUT/'interim_seed2026_permutation.csv',perms);writecsv(OUT/'interim_seed2026_dependence.csv',deps)
 lines=['# 中间结果，不需要等回复','仅seed2026；数值为比例。协调者已纠正32次笔误：正式24次，此时完成8次，剩余2027/2028两轮。','|频段/指标|argmax R/Z/F/F5|thr_fa1 R/Z/F/F5|','|---|---|---|']
 keys=list(dict.fromkeys((r['band'],r['category'],r['subset']) for r in result if r['subset'] not in ['all_drone_miss_with_G','data_nonG_recall','A_nonG_recall','A_FA'] and (r['category']!='receiver' or r['subset'].startswith('B_'))))
 for b,cat,sub in keys:
  cells=[]
  for mode in ['argmax','thr_fa1']:
   cells.append(' / '.join(f"{next(r['rate'] for r in result if r['band']==b and r['category']==cat and r['subset']==sub and r['mode']==mode and r['group']==g):.4f}" for g in ['R','Z','F','F5']))
  lines.append('|'+b+' '+sub+'|'+'|'.join(cells)+'|')
 for b in ['2.4GHz','5.8GHz']:
  for g in ['F','F5']:
   single=aggregate([r for r in perms if r['band']==b and r['group']==g and r['mode']=='argmax'],['feature'],['delta']);top=sorted(single,key=lambda r:abs(r['delta_mean']),reverse=True)[:3]
   lines.append(b+' '+g+' 单列打乱argmax召回变化前三：'+'；'.join(r['feature']+f"={r['delta_mean']:.4f}" for r in top))
   dd=aggregate([r for r in deps if r['band']==b and r['group']==g],['mode','dependence','subset'],['delta']);lines.append(b+' '+g+' 依赖度变化：'+'；'.join('/'.join(r[k] for k in ['mode','dependence','subset'])+f"={r['delta_mean']:.4f}" for r in dd))
   vals=[]
   for mode in ['argmax','thr_fa1']:
    for sub in ['v3_test_FA','nonG_miss','v3_drone_recall']+(['C1_FA'] if b=='2.4GHz' else []):
     v=[next(r['rate'] for r in result if r['band']==b and r['group']==gg and r['mode']==mode and r['subset']==sub and r['category']=='main') for gg in [g,'Z']];vals.append(mode+'/'+sub+f'={v[0]-v[1]:.4f}')
   lines.append(b+' '+g+'−Z：'+'；'.join(vals))
 lines.append('单种子结果；逐片k/N与CP95在CSV中；三种子标准差及方向一致性待后两轮，不下结论。')
 assert len(lines)<=40
 (OUT/'interim_seed2026.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':run()
