"""Remote probe of NIST mds2-2731 HDF5 files: list capture keys and storage layout
without downloading whole files (HTTP range reads, stdlib only)."""
import io, json, sys, time, urllib.request
import h5py

BASE = 'https://nist-oar-cache.s3.amazonaws.com/prd/gen0/mds2-2731/'


class RangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, with a small block cache."""

    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache, self.fetched = url, block, 0, {}, 0
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers['Content-Length'])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _get(self, b):
        if b not in self.cache:
            lo = b * self.block
            hi = min(lo + self.block, self.size) - 1
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={lo}-{hi}'})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        assert r.status == 206, r.status
                        data = r.read()
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            assert len(data) == hi - lo + 1
            self.fetched += len(data)
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[b] = data
        return self.cache[b]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        out, done = memoryview(buf), 0
        while done < n:
            b, o = divmod(self.pos + done, self.block)
            chunk = self._get(b)[o:o + n - done]
            out[done:done + len(chunk)] = chunk
            done += len(chunk)
        self.pos += n
        return n


if __name__ == '__main__':
    OUT = sys.argv[1]
    files = sys.argv[2:]
    res = {}
    for name in files:
        t = time.time()
        rf = RangeFile(BASE + name, block=256 << 10)
        with h5py.File(io.BufferedReader(rf, 256 << 10), 'r') as h:
            keys = sorted(h.keys())
            d = h[keys[0]]
            info = dict(size=rf.size, n_keys=len(keys), first_keys=keys[:3], last_keys=keys[-3:],
                        shape=d.shape, dtype=str(d.dtype), chunks=d.chunks, compression=d.compression,
                        attrs={k: str(v) for k, v in d.attrs.items()},
                        file_attrs={k: str(v) for k, v in h.attrs.items()},
                        layout=[dict(key=k, offset=h[k].id.get_offset(), n=int(h[k].shape[0])) for k in keys])
        info['seconds'] = time.time() - t
        info['bytes_fetched'] = rf.fetched
        res[name] = info
        print(name, info['n_keys'], info['shape'], info['dtype'], info['chunks'], info['compression'],
              round(info['seconds'], 1), 's', rf.fetched, 'B', flush=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=str)
