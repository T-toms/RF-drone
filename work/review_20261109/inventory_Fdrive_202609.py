# 清点 F 盘 F:/drone/202609 的 JSON 元数据（只读 JSON，不读 IQ）（2026-11-11，支线）
import json, collections, csv
from pathlib import Path
R = Path('F:/drone/202609')
rows = []
for j in sorted(R.rglob('*.json')):
    try:
        m = json.loads(j.read_text(encoding='utf-8'))
    except Exception as e:
        rows.append(dict(session=j.parent.name, file=j.name, error=repr(e))); continue
    c, d, s = m.get('capture', {}), m.get('declared', {}), m.get('session', {})
    b = j.with_suffix('.bin'); mt = j.with_suffix('.mat')
    rows.append(dict(model_dir=j.parent.parent.name, session=j.parent.name, file=j.stem,
                     fs=c.get('sample_rate_hz'), fc_mhz=(c.get('center_freq_hz') or 0) / 1e6, n=c.get('sample_count'),
                     t=c.get('t_start_utc'), gain=m.get('receiver', {}).get('gain_db'), rx=m.get('receiver', {}).get('id'),
                     cls=d.get('class'), model=';'.join(d.get('model') or []), state=';'.join(d.get('state') or []),
                     bw=';'.join(str(int(x / 1e6)) for x in (d.get('bw') or [])), dfc=';'.join(str(x / 1e6) for x in (d.get('fc') or [])),
                     paired_bg=s.get('paired_bg'), notes=m.get('notes'),
                     bin_bytes=b.stat().st_size if b.exists() else -1, mat_bytes=mt.stat().st_size if mt.exists() else -1))
out = Path(__file__).resolve().parent / 'inventory_Fdrive_202609.csv'
with open(out, 'w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
ok = [r for r in rows if 'error' not in r]
print('JSON 总数', len(rows), '解析失败', len(rows) - len(ok))
print('接收机', collections.Counter(r['rx'] for r in ok), ' 采样率', collections.Counter(r['fs'] for r in ok))
print('bin 大小异常', sum(r['bin_bytes'] != 101000000 for r in ok), ' 缺 mat', sum(r['mat_bytes'] < 0 for r in ok))
by = collections.defaultdict(list)
for r in ok:
    by[r['session']].append(r)
print('\n会话汇总：模型 | 状态 | 中心频率 | 带宽 | 增益 | 备注 | 文件数 | 总秒数')
agg = collections.Counter()
for sname, rs in sorted(by.items()):
    r0 = rs[0]
    secs = sum(r['n'] / r['fs'] for r in rs)
    band = '2.4G' if r0['fc_mhz'] < 3000 else '5.8G'
    agg[(r0['model'], r0['state'], band, r0['bw'], r0['cls'])] += secs
    print(f"  {sname:32s} {r0['model']:13s} {r0['state']:8s} {r0['fc_mhz']:8.2f} bw={r0['bw']:4s} g={r0['gain']} {str(r0['notes'])[:24]:24s} n={len(rs):2d} {secs:.2f}s")
print('\n按 (模型, 状态, 频段, 带宽, 类别) 汇总总秒数：')
for k, v in sorted(agg.items()):
    print('  ', k, f'{v:.2f}s')
