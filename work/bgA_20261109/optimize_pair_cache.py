from pathlib import Path
p=Path('work/bgA_20261109/evaluate_bg.py');s=p.read_text(encoding='utf-8-sig');a=s.index("  missing=[source[r['paired_sample_id']]");b=s.index('  if missing:',a)
new='''  missing=[source[r['paired_sample_id']] for r in rr if r['paired_sample_id'] not in known];specs=[];last=None;mp=E/'paired_missing_spec.npy'
  if mp.exists():
   assert json.loads((E/'paired_missing_ids.json').read_text(encoding='utf-8'))==[r['sample_id'] for r in missing];specs=np.load(mp)
  else:
   for r in missing:
    if r['block_file']!=last:z=np.load(r['block_file'],mmap_mode='r');last=r['block_file']
    j=int(r['start']);specs.append(c.spec(np.array(z[j:j+int(r['n_samples'])]),'2.4GHz'))
   np.save(mp,np.array(specs));(E/'paired_missing_ids.json').write_text(json.dumps([r['sample_id'] for r in missing]),encoding='utf-8')
'''
s=s[:a]+new+s[b:];p.write_text(s,encoding='utf-8')
