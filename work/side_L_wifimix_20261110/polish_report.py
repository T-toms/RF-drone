from pathlib import Path
p=Path(__file__).resolve().parent/'finish.py';s=p.read_text(encoding='utf-8-sig')
old="            report.append('|'+mode+'/'+g+'|'+'|'.join(f'{a*100:.3f}±{b*100:.3f}' for a,b in [summary('main',g,mode,t) for t in TAGS])+'|')"
new='''            cells=[]
            for t in TAGS:
                a,b=summary('main',g,mode,t);cell=f'{a*100:.3f}±{b*100:.3f}'
                rates=[r for r in rr if r['category']=='main' and r['group']==g and r['mode']==mode and r['round']==t]
                if rates[0]['fa']=='True':cell+='；'+', '.join(f"{r['seed']}:{r['k']}/{r['n']},U95={float(r['cp95_upper'])*100:.3f}%" for r in rates)
                cells.append(cell)
            report.append('|'+mode+'/'+g+'|'+'|'.join(cells)+'|')'''
assert old in s;s=s.replace(old,new)
anchor="    disk=sum(p.stat().st_size for d in [OUT,DEST] for p in d.rglob('*') if p.is_file())"
new='''    report += ['data按型号mix重点项（argmax，三种子均值%，每格J/L-a/L-b）：','|X dB|atom se|mavic 4|','|---|---|---|']
    for x in XS:
        report.append('|'+str(x)+'|'+'|'.join('/'.join(f"{summary('mix_data_model_'+str(x),model,'argmax',t)[0]*100:.3f}" for t in TAGS) for model in ['atom se','mavic 4'])+'|')
'''+anchor
assert anchor in s;s=s.replace(anchor,new)
anchor="    assert len(report)<=100,len(report)"
new='''    formatted=[]
    for i,line in enumerate(report):
        if i and (line.startswith('#') or line.startswith('|')!=report[i-1].startswith('|') or report[i-1].startswith('#')):formatted.append('')
        formatted.append(line)
    report=formatted
'''+anchor
s=s.replace(anchor,new)
p.write_text(s,encoding='utf-8')
