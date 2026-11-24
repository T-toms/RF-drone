# Pull the user's own messages (not assistant) from past session transcripts that mention input/IQ/feature/crop,
# to recover the original agreements verbatim.
import json, glob, re, os, sys
sys.stdout.reconfigure(encoding='utf-8')
D = os.path.join(os.path.expanduser('~'), '.claude', 'projects', 'C--Users-Administrator-Desktop-----') + os.sep
if not os.path.isdir(D):
    D = [p for p in glob.glob(os.path.join(os.path.expanduser('~'), '.claude', 'projects', '*')) if glob.glob(os.path.join(p, '44ca6a8f*.jsonl'))][0] + os.sep
pat = re.compile(r'IQ|特征|输入|截取|截频|频带|频段|矩阵|时频图|中心频率|主线|顺序|步骤')
for f in sorted(glob.glob(D + '*.jsonl'), key=os.path.getmtime):
    for line in open(f, encoding='utf-8', errors='ignore'):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get('type') != 'user':
            continue
        c = o.get('message', {}).get('content')
        if isinstance(c, list):
            c = ' '.join(x.get('text', '') for x in c if isinstance(x, dict) and x.get('type') == 'text')
        if not isinstance(c, str) or not c.strip() or c.startswith('<') or 'tool_use_id' in c:
            continue
        if len(c) > 1500 or not pat.search(c):
            continue
        print(o.get('timestamp', '')[:16], os.path.basename(f)[:8], '|', c.replace('\n', ' ')[:600])
