# 抽样核对：G:/raw 与 G:/drone/raw 的重合 .bin 内容是否真的一样（按实际文件算 sha256，不只看 JSON）（2026-11-12，支线）
import csv, hashlib, json
from pathlib import Path
import numpy as np
NEW = Path('G:/raw'); OLD = Path('G:/drone/raw')
rows = list(csv.DictReader(open(Path(__file__).with_name('compare_new_raw.csv'), encoding='utf-8')))
pick = [rows[i] for i in np.linspace(0, len(rows) - 1, 6).astype(int)]
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for r in pick:
    nb = (NEW / r['new_rel']).with_suffix('.bin'); ob = (OLD / r['old_rel'].split(';')[0]).with_suffix('.bin')
    a, b = h(nb), h(ob)
    print(r['new_rel'][-60:], 'new==old', a == b, 'json_sha_ok', a == r['sha'])
