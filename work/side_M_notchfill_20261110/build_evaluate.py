from pathlib import Path
O=Path(__file__).resolve().parent;L=O.parent/'side_L_wifimix_20261110'
s=(L/'evaluate.py').read_text(encoding='utf-8')
a=s.index('    plan=');b=s.index('    MIX=[]',a)
s=s[:a]+'''    qr=rows(OUT/'quiet013_metadata.csv');qx=tensor(np.load(MIXCACHE/'quiet013.npy'))
    wx=tensor(np.load(MIXCACHE/'white_noise.npy'));nx=tensor(np.load(MIXCACHE/'notched_noise.npy'))
    import shutil
    for p in L.glob('Lb_*predictions*.csv'):shutil.copyfile(p,OUT/p.name)
    for p in L.glob('C1_predictions_Lb_*.csv'):shutil.copyfile(p,OUT/p.name)
    MET.extend(r for r in rows(L/'metrics_all.csv') if r['round']=='Lb')
'''+s[b:]
s=s.replace("    MIX=[]","    MIX=[r for r in rows(L/'mix_results.csv') if r['round']=='Lb']")
s=s.replace("for round,folder in [('J',Path('E:/drone_lab_stage1_L20_bgA_20261109')),('La',DEST),('Lb',DEST)]:","for round,folder in [('M',DEST)]:")
s=s.replace("getmodel(folder,s,None if round=='J' else round)","getmodel(folder,s,'Lb')")
(O/'evaluate.py').write_text(s,encoding='utf-8')
