from core import ResNet18, np, torch, writecsv
from signal_ops import predict


MODES = ("argmax", "thr_fa1")


def get_model(cache, band, seed):
    tag = f"{band}_Lb" if band == "2.4GHz" else band
    ck = torch.load(cache / f"{tag}_seed{seed}_best.pt", map_location="cpu", weights_only=False)
    model = ResNet18(1, 2).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["thr_fa1"]


def decisions(probs, threshold):
    return dict(argmax=probs.argmax(1), thr_fa1=(probs[:, 1] >= threshold).astype(int))


def infer(model, images, selection=None):
    ids = range(len(images)) if selection is None else selection
    return np.concatenate([
        predict(model, torch.from_numpy(np.array(images[ids[i:i+64]])).unsqueeze(1))
        for i in range(0, len(ids), 64)
    ])


def generalization(row):
    if row["label"] != "drone":
        return False
    if row["band"] == "2.4GHz":
        return row["dataset"] == "data" and row["model"] in {"evo lite 6k", "evo ii pro v3"}
    return row["dataset"] == "v3" and row["model"] in {"evo2pro_v3", "fpv"}


def state_untrained(row):
    return row["dataset"] == "data" and row["state"] == "flight" and (row["band"], row["model"]) in {
        ("2.4GHz", "atom se"), ("2.4GHz", "avata"), ("5.8GHz", "mini 4k"),
    }


def record_predictions(path, meta, probs, pred):
    writecsv(path, (dict(row, p_drone=float(p), pred_argmax=int(a), pred_thr=int(t))
                    for row, p, a, t in zip(meta, probs[:, 1], pred["argmax"], pred["thr_fa1"])))


def evaluate_clean(window, band, seed, mode, test, pred, bg_a=(), pred_a=(), white=()):
    from collections import defaultdict
    from stats import cp

    stats = []

    def emit(category, group, values, fa=False):
        n = len(values)
        if n:
            k = int(sum(values))
            stats.append(dict(L=window, band=band, seed=seed, mode=mode, category=category,
                              group=group, n=n, k=k, rate=k/n,
                              cp95_upper=cp(k,n) if fa else ""))

    def groupby(category, samples, values, keys, fa=False):
        groups = defaultdict(list)
        for row, value in zip(samples, values):
            groups[" | ".join(row[key] for key in keys)].append(value)
        for group, outcomes in sorted(groups.items()):
            emit(category, group, outcomes, fa)

    drone = [(r, int(p)) for r, p in zip(test, pred) if r["label"] == "drone"]
    bg = [(r, int(p)) for r, p in zip(test, pred) if r["label"] == "background"]
    non_g = [(r,p) for r,p in drone if not generalization(r)]
    emit("main", "nonG_miss", [1-p for r,p in non_g])
    emit("main", "all_drone_miss_with_G", [1-p for r,p in drone])
    emit("main", "v3_drone_recall", [p for r,p in drone if r["dataset"] == "v3"])
    emit("main", "v3_test_FA", [p for r,p in bg], True)
    emit("main", "bgA_test_FA", [int(p) for r,p in zip(bg_a,pred_a)
                                if r.get("on_original_grid", "True") == "True"], True)
    if window == "50t":
        emit("main", "bgA_test_FA_all_correlated", list(pred_a))
    emit("main", "white_noise_FA", list(white), True)
    for category, keys in (("miss_by_model", ("model",)),
                           ("miss_by_model_state_bw", ("dataset","model","state","bw_label")),
                           ("miss_by_file", ("block_file",))):
        groupby(category, [r for r,p in drone], [1-p for r,p in drone], keys)
    groupby("fa_by_session", [r for r,p in bg], [p for r,p in bg], ("group_id",), True)
    for category, predicate in (("G_recall", generalization), ("state_untrained_recall", state_untrained)):
        pairs = [(r,p) for r,p in drone if predicate(r)]
        groupby(category, [r for r,p in pairs], [p for r,p in pairs], ("dataset","model","state","bw_label","group_id"))
    models = {"evo ii pro v3", "evo2pro_v3"} if band == "2.4GHz" else {"mini5 pro", "mini5pro"}
    pairs = [(r,p) for r,p in drone if r["model"] in models]
    groupby("same_model_recall", [r for r,p in pairs], [p for r,p in pairs],
            ("dataset","model","bw_label"))
    matrix = dict(TN=sum(1-p for r,p in bg), FP=sum(p for r,p in bg),
                  FN=sum(1-p for r,p in drone), TP=sum(p for r,p in drone))
    matrix.update(L=window, band=band, seed=seed, mode=mode)
    return stats, matrix
