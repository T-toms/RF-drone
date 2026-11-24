from pathlib import Path
p=Path('work/bgA_20261109/full_driver.py');s=p.read_text(encoding='utf-8-sig').replace("assert len(bad)<=5,('ASK_REQUIRED',sorted(bad))","assert len(bad)<=15,('COORDINATOR_STOP_IF_OVER15',sorted(bad))");p.write_text(s,encoding='utf-8')
p=Path('work/bgA_20261109/summarize_fa.py');s=p.read_text(encoding='utf-8-sig');s=s.replace(" bg=[r for r in rows('bgA_predictions.csv') if r['keep']=='True'];pair=", " allbg=rows('bgA_predictions.csv');bg=[r for r in allbg if r['keep']=='True'];pair=")
s=s.replace(" write('bgA_FA.csv',out);", " for seed in [2026,2027,2028]:\n  failed=[r for r in allbg if r['keep']=='False' and int(r['seed'])==seed]\n  if failed:\n   for rule in ['argmax','thr']:out.append(dict(kind='failed_blocks',value='all_failed',seed=seed,rule=rule,source_model='',source_state='',source_bw_label='',**stat(failed,rule)))\n write('bgA_FA.csv',out);")
p.write_text(s,encoding='utf-8')
