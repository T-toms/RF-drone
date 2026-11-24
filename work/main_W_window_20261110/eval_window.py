import time

from cache_build import to_spec
from core import (
    SEEDS, IQReader, background_iq, dump, np, order, rows, torch, window_dirs, writecsv,
)
from evaluate import decisions, evaluate_clean, get_model, infer, record_predictions
from signal_ops import power
from stats import cp


XS = (-10,-5,0,5,10,15,20)


def mix_test(window, band, test, models, out):
    rs = sorted(test,key=order)
    backgrounds = [r for r in rs if r["dataset"] == "v3" and r["label"] == "background"]
    iq = background_iq(backgrounds)
    powers = [power(b) for b in iq]
    reader = IQReader()
    pairs = []
    result = []
    for group in ("data_drone","v3_drone","v3_background"):
        ds, label = group.split("_")
        records = [r for r in rs if r["dataset"] == ds and r["label"] == label]
        assert records, (window,band,group)
        tally = {(seed,mode,x):0 for seed in SEEDS for mode in ("argmax","thr_fa1") for x in XS}
        for start in range(0,len(records),64):
            batch = records[start:start+64]
            images = np.empty((len(XS),len(batch),256,256),dtype=np.float16)
            for i,row in enumerate(batch,start):
                d = reader(row)
                pd = power(d)
                j = (i + int(label == "background")) % len(backgrounds)
                b,pb = iq[j],powers[j]
                for k,x in enumerate(XS):
                    g = float(np.sqrt(pb/max(pd,1e-30)*10**(x/10)))
                    images[k,i-start] = to_spec(b+g*d,band)
                    pairs.append(dict(band=band,group=group,sample_id=row["sample_id"],
                                      background_sample_id=backgrounds[j]["sample_id"],x_db=x,g=g,
                                      source_power=pd,background_power=pb,
                                      achieved_ratio_db=float(10*np.log10(g*g*pd/pb))))
            for seed, (model,threshold) in zip(SEEDS,models):
                for x,array in zip(XS,images):
                    for mode,values in decisions(infer(model,array),threshold).items():
                        tally[seed,mode,x] += int(values.sum())
            if start % 512 == 0:
                print("test mixing",window,band,group,start,len(records),flush=True)
        n = len(records)
        for (seed,mode,x),k in tally.items():
            result.append(dict(L=window,band=band,seed=seed,mode=mode,category="mix_test",
                               group=group,x_db=x,n=n,k=k,rate=k/n,
                               cp95_upper=cp(k,n) if group == "v3_background" else ""))
    writecsv(out / f"{band}_mix_test.pairs.csv",pairs)
    return result


def evaluate(window,band):
    out, cache = window_dirs(window)
    marker = out / f"{band}_evaluation_done.json"
    if marker.exists():
        print("SKIP eval",window,band,flush=True)
        return
    start = time.monotonic()
    torch.backends.cudnn.benchmark = True
    name = f"{band}_test_spec"
    all_rows = rows(cache / f"{name}_meta.csv")
    images = np.load(cache / f"{name}.npy",mmap_mode="r")
    test_ids = [i for i,r in enumerate(all_rows) if r["dataset"] != "dataA_mirror"]
    bg_ids = [i for i,r in enumerate(all_rows) if r["dataset"] == "dataA_mirror"]
    test = [all_rows[i] for i in test_ids]
    bg = [all_rows[i] for i in bg_ids]
    noise = np.load(cache / "white_noise.npy",mmap_mode="r") if bg else None
    metrics, matrices = [], []
    models = [get_model(cache,band,s) for s in SEEDS]
    for seed,(model,threshold) in zip(SEEDS,models):
        probs = infer(model,images,test_ids)
        preds = decisions(probs,threshold)
        record_predictions(out / f"{band}_test_predictions_seed{seed}.csv",test,probs,preds)
        bg_preds, white = {}, {}
        if bg:
            bg_probs = infer(model,images,bg_ids)
            bg_preds = decisions(bg_probs,threshold)
            record_predictions(out / f"{band}_bgA_predictions_seed{seed}.csv",bg,bg_probs,bg_preds)
            white_probs = infer(model,noise)
            white = decisions(white_probs,threshold)
            record_predictions(out / f"{band}_white_noise_predictions_seed{seed}.csv",
                               [dict(sample_id=f"white_{i:02}") for i in range(len(noise))],white_probs,white)
        for mode,pred in preds.items():
            stats, matrix = evaluate_clean(window,band,seed,mode,test,pred,bg,
                                          bg_preds.get(mode,()),white.get(mode,()))
            metrics.extend(stats)
            matrices.append(matrix)
    writecsv(out / f"{band}_metrics.csv",metrics)
    writecsv(out / f"{band}_confusion.csv",matrices)
    mix = mix_test(window,band,test,models,out)
    writecsv(out / f"{band}_mix_metrics.csv",mix)
    dump(marker,dict(seconds=time.monotonic()-start))
