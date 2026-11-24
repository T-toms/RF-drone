from pathlib import Path
import ast,json,hashlib
P=Path.cwd();O=P/'work/bgA_20261109'
def hashes():
 return {str(p.relative_to(P)):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for d in ['crop_slice_20261107','train_stage1_L20_20261108'] for p in sorted((P/'work'/d).rglob('*')) if p.is_file()}
(O/'hash_before.json').write_text(json.dumps(hashes(),indent=2),encoding='utf-8')
s=(P/'work/crop_slice_20261107/crop.py').read_text(encoding='utf-8');t=ast.parse(s)
names={'Source','ratio','kernel','chain'}
head='''from pathlib import Path
import numpy as np
from scipy import io,signal,fft
import math,csv,json,shutil,time,sys
P=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;R=Path('G:/drone')
FS={'2.4GHz':61440000,'5.8GHz':100000000};FC={'2.4GHz':2430.,'5.8GHz':5800.};WIDTH={'2.4GHz':25.,'5.8GHz':45.}
FILTERS=dict(np.load(P/'work/crop_slice_20261107/filters.npz'))
NOTCHES=list(csv.DictReader((P/'work/crop_slice_20261107/notch_table.csv').open(encoding='utf-8-sig')))
for r in NOTCHES:
 for k in ['freq_mhz','half_width_khz']:r[k]=float(r[k])
KERNELS={}
'''
lab=ast.parse((P/'work/prep_lines_20261104/lab_preproc.py').read_text(encoding='utf-8'))
head+=next(ast.get_source_segment((P/'work/prep_lines_20261104/lab_preproc.py').read_text(encoding='utf-8'),n) for n in lab.body if isinstance(n,ast.FunctionDef) and n.name=='notch')+'\n'
head+='\n\n'.join(ast.get_source_segment(s,n).replace('lab.notch','notch') for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
(O/'chain_copy.py').write_text(head+'\n',encoding='utf-8')
print('baseline recorded',len(hashes()))
