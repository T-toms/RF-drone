"""Run the existing refinement with empty invalid-segment starts treated as zero.
Only included files are subsequently loaded; original refinement script is unchanged.
"""
from pathlib import Path
p=Path(__file__).resolve().with_name('refine_validation.py')
s=p.read_text(encoding='utf-8').replace("int(m['start_sample'])", "int(m['start_sample'] or 0)")
exec(compile(s,str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
