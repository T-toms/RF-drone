# Find where the "spectrogram input for round 1, IQ later" decision came from (user answers / tool results, 2026-11-07..09)
import json, glob, os, sys, re
sys.stdout.reconfigure(encoding='utf-8')
D = [p for p in glob.glob(os.path.join(os.path.expanduser('~'), '.claude', 'projects', '*')) if glob.glob(os.path.join(p, '44ca6a8f*.jsonl'))][0]
for f in sorted(glob.glob(os.path.join(D, '*.jsonl')), key=os.path.getmtime)[-6:]:
    for line in open(f, encoding='utf-8', errors='ignore'):
        if 'IQ' not in line or '时频' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get('timestamp', '')[:16]
        if not ts.startswith('2026-11-0') or o.get('type') != 'user':
            continue
        s = json.dumps(o['message']['content'], ensure_ascii=False)
        m = re.search(r'answered[^\]]{0,700}', s)
        if m:
            print(ts, os.path.basename(f)[:8], '|', m.group(0)[:700])
            print('---')
