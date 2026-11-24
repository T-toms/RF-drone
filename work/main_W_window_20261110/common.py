import os
import random
import time

from torch.nn import functional as F

from core import ResNet18, dump, np, rows, torch, train_window, window_dirs, writecsv
from signal_ops import seed


OUT, CACHE = window_dirs(train_window())
if "WINDOW_W_PROCESS_TOKEN" in os.environ:
    dump(OUT / "training_process.json",
         dict(pid=os.getpid(), token=os.environ["WINDOW_W_PROCESS_TOKEN"]))


class BatchMap:
    """Read a training batch from the float16 cache if its working set exceeds the limit."""
    def __init__(self, array):
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.numpy()
        return torch.from_numpy(np.array(self.array[ids])).unsqueeze(1)


def dataset(band, split):
    from os import environ

    name = f"{band}_{split}_spec"
    use_map = environ.get("WINDOW_W_MMAP") == "1"
    a = np.load(CACHE / f"{name}.npy",mmap_mode="r" if use_map else None)
    meta = rows(CACHE / f"{name}_meta.csv")
    assert len(a) == len(meta)
    data = BatchMap(a) if use_map else torch.from_numpy(a).unsqueeze(1)
    labels = torch.tensor([r["label"] == "drone" for r in meta],dtype=torch.long)
    return data, labels, meta
