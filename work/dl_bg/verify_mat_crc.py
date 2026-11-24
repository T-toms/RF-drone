"""Verify .mat files against the CRC32 stored in their RAR5 file headers (read remotely, header only)."""
import json, subprocess, sys, zlib
from pathlib import Path
ROOT = Path('C:/Users/Administrator/Desktop/数据集处理')
URL = 'https://china.scidb.cn/download?fileId=c403fc76444e4b9989e4f3ff570f3b3d'
M = {e['name'].split('/')[-1]: e for e in json.loads((ROOT / 'dronerfa_manifest.json').read_text(encoding='utf-8'))['entries']}

def vint(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1; v |= (c & 0x7F) << s; s += 7
        if not c & 0x80:
            return v, i

def header_crc(name):
    e = M[name]
    for _ in range(5):
        h = subprocess.run(['curl', '-sS', '-L', '-r', f"{e['block_offset']}-{e['block_offset'] + 511}", '-o', '-', URL],
                           capture_output=True).stdout
        if len(h) == 512:
            break
    i = 4
    _, i = vint(h, i)                    # header size
    htype, i = vint(h, i); assert htype == 2, htype
    hflags, i = vint(h, i)
    if hflags & 1: _, i = vint(h, i)     # extra size
    if hflags & 2: _, i = vint(h, i)     # data size
    fflags, i = vint(h, i)
    _, i = vint(h, i)                    # unpacked size
    _, i = vint(h, i)                    # attributes
    if fflags & 2: i += 4                # mtime
    assert fflags & 4, 'no CRC'
    return int.from_bytes(h[i:i + 4], 'little')

def file_crc(path):
    c = 0
    with open(path, 'rb') as f:
        while b := f.read(64 * 2**20):
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF

args = sys.argv[1:]
alias = None
if '--name' in args:  # verify one file stored under a temporary name
    i = args.index('--name'); alias = args[i + 1]; del args[i:i + 2]
ok = bad = 0
for p in args:
    n = alias or Path(p).name
    good = header_crc(n) == file_crc(p)
    ok += good; bad += not good
    print(('OK  ' if good else 'BAD ') + n, flush=True)
print(f'ok {ok} bad {bad}')
sys.exit(1 if bad else 0)
