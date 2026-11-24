# 对比 G:/raw（用户说的新数据）与 G:/drone/raw（已有）的重合情况（2026-11-12，支线）
# 只读。重合判据：同一次采集 = JSON 里 capture.sha256 相同（.bin 内容的哈希）；再核对 .bin 大小。
# 不删除任何文件，只输出清单。
import json, collections, csv
from pathlib import Path
NEW = Path('G:/raw'); OLD = Path('G:/drone/raw')
OUT = Path(__file__).resolve().parent


def load(root):
    rows = {}
    for j in root.rglob('*.json'):
        try:
            m = json.loads(j.read_text(encoding='utf-8'))
        except Exception as e:
            rows[str(j)] = dict(path=str(j), err=repr(e)); continue
        c = m.get('capture', {}); d = m.get('declared', {})
        b = j.with_suffix('.bin')
        rows[str(j)] = dict(path=str(j), rel=str(j.relative_to(root)), sha=c.get('sha256'), t=c.get('t_start_utc'), fc=(c.get('center_freq_hz') or 0) / 1e6,
                            fs=c.get('sample_rate_hz'), n=c.get('sample_count'), cls=d.get('class'), model=';'.join(d.get('model') or []),
                            bin_bytes=b.stat().st_size if b.exists() else -1, session=(m.get('session') or {}).get('id'))
    return rows


new, old = load(NEW), load(OLD)
old_by_sha = collections.defaultdict(list)
for r in old.values():
    if r.get('sha'):
        old_by_sha[r['sha']].append(r)
dup, uniq = [], []
for r in new.values():
    hit = old_by_sha.get(r.get('sha'))
    (dup if hit else uniq).append((r, hit))
print(f'新目录 JSON {len(new)}；已有目录 JSON {len(old)}')
print(f'新目录中与已有目录 sha256 相同的采集：{len(dup)}；新目录独有：{len(uniq)}')
print('独有采集按 (类别, 中心频率, 采样率, 日期) 统计：', collections.Counter((r['cls'], round(r['fc'], 2), r['fs'], (r['t'] or '')[:10]) for r, _ in uniq).most_common(30))
print('重合采集按 (类别, 中心频率) 统计：', collections.Counter((r['cls'], round(r['fc'], 2)) for r, _ in dup).most_common(10))
same_rel = sum(1 for r, h in dup if any(x['rel'] == r['rel'] for x in h))
size_ok = sum(1 for r, h in dup if all(x['bin_bytes'] == r['bin_bytes'] for x in h))
print(f'重合采集中相对路径也相同：{same_rel}；.bin 大小全部一致：{size_ok}')
old_only = [r for r in old.values() if r.get('sha') and r['sha'] not in {x.get('sha') for x in new.values()}]
print(f'已有目录独有（新目录没有）的采集：{len(old_only)}', collections.Counter((r['cls'], round(r['fc'], 2), (r['t'] or '')[:10]) for r in old_only).most_common(10))
with open(OUT / 'compare_new_raw.csv', 'w', encoding='utf-8', newline='') as f:
    w = csv.writer(f); w.writerow(['new_rel', 'status', 'old_rel', 'sha', 'cls', 'fc', 'fs', 't', 'new_bin_bytes', 'old_bin_bytes'])
    for r, h in sorted(dup + uniq, key=lambda x: x[0].get('rel', '')):
        w.writerow([r.get('rel'), 'dup' if h else 'new_only', ';'.join(x['rel'] for x in h) if h else '', r.get('sha'), r.get('cls'), r.get('fc'), r.get('fs'), r.get('t'), r.get('bin_bytes'), ';'.join(str(x['bin_bytes']) for x in h) if h else ''])
