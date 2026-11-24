# Search all past transcripts for discussion of "引入中心频率" (report step 2), both user and assistant text.
import json, glob, os, sys, re
sys.stdout.reconfigure(encoding='utf-8')
D = [p for p in glob.glob(os.path.join(os.path.expanduser('~'), '.claude', 'projects', '*')) if glob.glob(os.path.join(p, '44ca6a8f*.jsonl'))][0]
pat = re.compile(r'(引入[^。\n]{0,8}中心频率|中心频率[^。\n]{0,30}(输入|引入|通道|编码|条件))')
seen = set()
for f in sorted(glob.glob(os.path.join(D, '*.jsonl')), key=os.path.getmtime):
    for line in open(f, encoding='utf-8', errors='ignore'):
        if '中心频率' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        role = o.get('type')
        c = o.get('message', {}).get('content')
        texts = []
        if isinstance(c, str):
            texts = [c]
        elif isinstance(c, list):
            for x in c:
                if isinstance(x, dict) and x.get('type') == 'text':
                    texts.append(x.get('text', ''))
        for t in texts:
            for m in pat.finditer(t):
                a = max(0, m.start() - 250); s = t[a:m.end() + 250].replace('\n', ' ')
                key = s[:120]
                if key in seen:
                    continue
                seen.add(key)
                print(o.get('timestamp', '')[:16], os.path.basename(f)[:8], role, '|', s)
                print('---')
