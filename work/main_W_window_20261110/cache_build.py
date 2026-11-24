import json
import time

from core import (
    BANDS, FS, SPLITS, IQReader, bg_index, disk_guard, dump, index, np,
    order, paired, select, train_groups, window_dirs, writecsv,
)
from signal_ops import mixing, spec, spec_new


def to_spec(iq, band):
    return spec_new(iq) if band == "2.4GHz" else spec(iq, band)


def read_into(cache, offset, records, band):
    read = IQReader()
    for count, i in enumerate(sorted(range(len(records)), key=lambda i: order(records[i])), 1):
        cache[offset+i] = to_spec(read(records[i]), band)
        if count % 500 == 0:
            cache.flush()
            print("cache",band,count,len(records),flush=True)


def build(window, band):
    out, cache = window_dirs(window)
    marker = cache / f"{band}_cache_done.json"
    if marker.exists():
        print("SKIP cache",window,band,flush=True)
        return
    start = time.monotonic()
    signal = index(window)
    mirrored = bg_index(window) if band == "2.4GHz" else []
    counts = {}
    for split in SPLITS:
        base = select(signal,band,split)
        mirror = select(mirrored,band,split)
        source = base + mirror
        kinds = [0 if r["label"] == "drone" else 1 for r in base] + [2] * len(mirror)
        backgrounds, groups = [], {}
        if mirror and split == "train":
            backgrounds, groups = train_groups(base,mirror)
            for kind, (records, assignments) in enumerate(groups.values(), 3):
                source.extend(records)
                kinds.extend([kind] * len(records))
        fields = set().union(*(r.keys() for r in source))
        metadata = [
            dict({k:r.get(k,"") for k in sorted(fields)}, cache_index=i, kind=kinds[i])
            for i,r in enumerate(source)
        ]
        name = f"{band}_{split}_spec"
        path = cache / f"{name}.npy"
        disk_guard(len(source) * 256 * 256 * 2 + 256 - (path.stat().st_size if path.exists() else 0))
        images = np.lib.format.open_memmap(path,mode="w+",dtype=np.float16,shape=(len(source),256,256))
        if split == "train":
            if len(source) == len(base) + len(mirror):
                read_into(images,0,source,band)
            else:
                read_into(images,0,base+mirror,band)
                offset = len(base) + len(mirror)
                for group, (records, assignments) in groups.items():
                    pair_rows = paired(records,assignments,backgrounds,mixing,
                                       lambda i,z: images.__setitem__(offset+i,to_spec(z,band)))
                    writecsv(out / f"mix_train_{group}_L{window}.pairs.csv",pair_rows)
                    offset += len(records)
                    images.flush()
                    print("train mixing",window,group,len(records),flush=True)
        else:
            read_into(images,0,source,band)
        images.flush()
        del images
        writecsv(cache / f"{name}_meta.csv",metadata)
        counts[split] = {kind:kinds.count(kind) for kind in sorted(set(kinds))}
        print("cache ready",window,band,split,len(source),flush=True)
    if band == "2.4GHz":
        n = FS[band] // 1000 * int(window.rstrip("t"))
        rng = np.random.default_rng(20261109)
        path = cache / "white_noise.npy"
        disk_guard(80 * 256 * 256 * 2 + 256 - (path.stat().st_size if path.exists() else 0))
        noise = np.lib.format.open_memmap(path,mode="w+",dtype=np.float16,shape=(80,256,256))
        for i in range(len(noise)):
            iq = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
            noise[i] = to_spec(iq,band)
        noise.flush()
        del noise
    dump(marker,dict(window=window,band=band,counts=counts,seconds=time.monotonic()-start))
