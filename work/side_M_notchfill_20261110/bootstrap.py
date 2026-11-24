from pathlib import Path
import sys
sys.dont_write_bytecode=True
O=Path(__file__).resolve().parent
L=O.parent/'side_L_wifimix_20261110'
# Copy the reviewed pilot definitions as text; never execute or import its module.
p=(O.parent/'review_20261109/notch_fill_pilot.py').read_text(encoding='utf-8')
part=p[p.index('FS, FC ='):p.index('\n\nif __name__')]
part=part.replace('FS, FC =','SAMPLE_FS, CENTER =').replace('1 / FS','1 / SAMPLE_FS').replace('+ FC','+ CENTER').replace('FC - 25','CENTER - 25').replace('FC + 25','CENTER + 25')
part=part.replace('p = fill_bins(p, method)','p = fill_bins(p, method) if hole_idx else p')
part=part.replace("def spec_new(iq, method='M'):","def spec_new(iq, method='M', dtype=np.float16):").replace('.astype(np.float16)','.astype(dtype)')
(O/'fill.py').write_text("import sys\nsys.dont_write_bytecode=True\nimport csv\nfrom pathlib import Path\nimport numpy as np, torch\nfrom torch.nn import functional as F\n"+part+"\n\nspec_fill=spec_new\n",encoding='utf-8')
s=(L/'support.py').read_text(encoding='utf-8').replace("DEST=Path('E:/side_L_wifimix_20261110')","LDEST=Path('E:/side_L_wifimix_20261110')\nL=ROOT/'work/side_L_wifimix_20261110'\nDEST=Path('E:/side_M_notchfill_20261110')\nHOLD=OLD\nBOLD=BG\nOLD=DEST/'H'\nBG=DEST/'BG'")
s += "\nfrom fill import spec_fill\nspec_original=spec\ndef spec(iq,b='2.4GHz'): return spec_fill(iq)\n"
(O/'support.py').write_text(s,encoding='utf-8')
(O/'train.py').write_text((L/'train.py').read_text(encoding='utf-8'),encoding='utf-8')
import shutil
for f in ['mix_train_drone.pairs.csv','mix_train_bg.pairs.csv','C1.pairs.csv']:
    shutil.copyfile(L/f,O/f)
