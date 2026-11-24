from common import *


DEST = CACHE


def joined(split, variant="La"):
    data, labels, meta = dataset("2.4GHz",split)
    kinds = torch.tensor([int(r["kind"]) for r in meta])
    assert split != "train" or variant == "Lb"
    assert set(kinds.tolist()) == set(range(5 if split == "train" else 3))
    return data, labels, meta, kinds
