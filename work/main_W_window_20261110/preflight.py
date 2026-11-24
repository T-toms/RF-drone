import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import (
    BANDS, BGA, FS, H_CACHE, INPUT, M_CACHE, N2, OUT, SEEDS, TASK_H, TASK_L,
    TASK_M, WINDOWS, bg_index, disk_guard, dump, index, inventory, kept, np,
    order, paired, protected_hashes, rows, select, torch, train_groups, writecsv,
    IQReader,
)
from evaluate import decisions, evaluate_clean, get_model, infer
from signal_ops import mixing, spec, spec_new


def check_index():
    group_split = {r["group_id"]: r["split"] for r in rows(INPUT / "group_split.csv")}
    f_counts = rows(INPUT / "counts.csv")
    n_counts = rows(N2 / "counts.csv")
    b_counts = rows(BGA / "background_counts.csv")
    results = []
    for window in WINDOWS:
        raw, mirror = index(window), bg_index(window)
        assert all(r["split"] == group_split[r["group_id"]] for r in raw + mirror), window
        assert all(int(r["n_samples"]) == FS[r["band"]] // 1000 * int(window.rstrip("t"))
                   for r in raw + mirror), window
        assert len({r["sample_id"] for r in raw + mirror}) == len(raw + mirror), window
        counts = inventory(window, raw, mirror)
        key = lambda r: tuple(r[k] for k in ("band", "dataset", "label", "split"))
        expected = {key(r): (int(r["n"]), int(r["keep"]), float(r["keep_seconds"]))
                    for r in f_counts if r["L"] == window}
        if window == "50t":
            expected = {key(r): (int(r["L50t_count"]), int(r["L50t_keep"]),
                                 float(r["L50t_keep_unique_seconds"])) for r in n_counts}
        actual = {key(r): (r["n"], r["keep"], round(r["keep_unique_s"], 3))
                  for r in counts if r["dataset"] != "dataA_mirror"}
        assert actual == expected, (window, actual, expected)

        low_data = [r for r in raw if r["band"] == "2.4GHz" and r["dataset"] == "data"]
        assert [r["sample_id"] for r in low_data] == [r["paired_sample_id"] for r in mirror]
        assert all(tuple(a[k] for k in ("start","n_samples","group_id","split"))
                   == tuple(b[k] for k in ("start","n_samples","group_id","split"))
                   for a,b in zip(low_data,mirror))
        base = mirror
        if window == "50t":
            base = [r for r in mirror if r["on_original_grid"] == "True"]
            old = rows(BGA / "index_bgA_L50.csv")
            fields = ("block_file", "start", "n_samples", "split", "keep")
            assert len(base) == len(old) and all(tuple(a[k] for k in fields) == tuple(b[k] for k in fields)
                                                  for a,b in zip(base,old))
            assert len({r["block_file"] for r in old if not kept(r)}) == 6
            assert len({r["block_file"] for r in mirror if not kept(r)}) == 6
            writecsv(OUT / "index_bgA_L50t.csv", mirror)
        expected_bg = {r["split"]: int(r["bgA_N"]) for r in b_counts if r["L"] == window.rstrip("t")}
        actual_bg = {s: sum(r["split"] == s and kept(r) for r in base) for s in expected_bg}
        assert expected_bg == actual_bg, (window, expected_bg, actual_bg)
        results.extend(counts)
    writecsv(OUT / "inventory.csv", results)
    return {"windows": len(WINDOWS), "groups_checked": len(group_split),
            "L50t_original_bgA_rows": len(base), "L50t_total_bgA_rows": len(mirror)}


def check_spec():
    result = {}
    for band, cache in zip(BANDS, (M_CACHE / "H", H_CACHE)):
        name = f"{band}_test_spec"
        meta = rows(cache / f"{name}_meta.csv")
        images = np.load(cache / f"{name}.npy", mmap_mode="r")
        rng = np.random.default_rng(20261110)
        selected = []
        for dataset in ("data", "v3"):
            selected.extend(rng.choice([i for i,r in enumerate(meta) if r["dataset"] == dataset],
                                       20, replace=False))
        reader = IQReader()
        differences = []
        for i in sorted(selected, key=lambda i: order(meta[i])):
            iq = reader(meta[i])
            computed = spec_new(iq) if band == BANDS[0] else spec(iq, band)
            differences.append(float(np.abs(computed.astype(np.float32) - images[i].astype(np.float32)).max()))
        result[band] = {"n": len(selected), "data": 20, "v3": 20, "max_abs": max(differences)}
        assert max(differences) <= .02, (band, max(differences))
    return result


def check_pairs():
    background, groups = train_groups(select(index("20"), BANDS[0], "train"),
                                      select(bg_index("20"), BANDS[0], "train"))
    result = {}
    for name, (source, assignments) in groups.items():
        reference = rows(TASK_L / f"mix_train_{name}.pairs.csv")
        assert len(source) == len(reference), name
        for i, (r,(x,j),old) in enumerate(zip(source,assignments,reference)):
            actual = r["sample_id"], background[j]["sample_id"]
            expected = old["sample_id"], old["background_sample_id"]
            assert actual == expected, (name, i, actual, expected)
        recomputed = paired(source, assignments, background, mixing)
        diffs = {key: max(abs(float(a[key])-float(b[key])) / max(abs(float(b[key])), 1e-30)
                          for a,b in zip(recomputed,reference)) for key in ("X","g")}
        assert max(diffs.values()) <= 1e-9, (name,diffs)
        result[name] = {"n": len(recomputed), "max_relative": diffs}
        writecsv(OUT / f"mix_train_{name}_L20.pairs.csv", recomputed)
        print("L20 pairing", name, result[name], flush=True)
    return result


def compare_predictions(meta, probs, pred, path):
    old = rows(path)
    assert len(old) == len(meta) and [r["sample_id"] for r in old] == [r["sample_id"] for r in meta], path
    assert np.array_equal(pred["argmax"], [int(r["pred_argmax"]) for r in old]), path
    assert np.array_equal(pred["thr_fa1"], [int(r["pred_thr"]) for r in old]), path
    delta = float(np.abs(probs[:,1] - [float(r["p_drone"]) for r in old]).max())
    assert delta <= 5e-3, (path, delta)
    return delta


def check_eval():
    result = []
    metrics = []
    matrices = []
    m_ref = {(r["mode"],r["group"]):r for r in rows(TASK_M / "metrics_paired.csv") if r["category"] == "main"}
    h_ref = {(r["seed"],r["mode"]):r for r in rows(TASK_H / "main_metrics.csv") if r["band"] == BANDS[1]}
    h_expected = rows(TASK_H / "confusion_matrices.csv")
    low_bg = [(i,r) for i,r in enumerate(rows(M_CACHE / "BG/bgA_meta.csv")) if kept(r) and r["split"] == "test"]
    for band, cache, image_dir in zip(BANDS, (M_CACHE,H_CACHE), (M_CACHE / "H",H_CACHE)):
        images = np.load(image_dir / f"{band}_test_spec.npy", mmap_mode="r")
        meta = rows(image_dir / f"{band}_test_spec_meta.csv")
        for seed in SEEDS:
            model, thr = get_model(cache, band, seed)
            probs = infer(model, images)
            pred = decisions(probs, thr)
            ba_meta, ba, white = [], {}, {}
            if band == BANDS[0]:
                delta = compare_predictions(meta,probs,pred,TASK_M / f"M_test_predictions_{band}_seed{seed}.csv")
                ba_ids, ba_meta = map(list, zip(*low_bg))
                ba_probs = infer(model,np.load(M_CACHE / "BG/bgA_spec.npy",mmap_mode="r"),ba_ids)
                ba = decisions(ba_probs,thr)
                ba_delta = compare_predictions(ba_meta,ba_probs,ba,TASK_M / f"M_bgA_test_predictions_seed{seed}.csv")
                white = decisions(infer(model,np.load(M_CACHE / "cache/white_noise.npy",mmap_mode="r")),thr)
            else:
                delta = compare_predictions(meta,probs,pred,TASK_H / f"test_predictions_{band}_seed{seed}.csv")
                ba_delta = None
            result.append(dict(band=band, seed=seed, n=len(meta), max_abs=delta,
                               bgA_n=len(ba_meta), bgA_max_abs=ba_delta))
            for mode in pred:
                entries, confusion = evaluate_clean("20",band,seed,mode,meta,pred[mode],
                                                   ba_meta,ba.get(mode,()),white.get(mode,()))
                main = {r["group"]:r for r in entries if r["category"] == "main"}
                if band == BANDS[0]:
                    for group in ("nonG_miss","v3_test_FA","bgA_test_FA"):
                        old = m_ref[mode,group]
                        assert (main[group]["n"],main[group]["k"]) == (int(old[f"M_{seed}_n"]),int(old[f"M_{seed}_k"])), (group,main[group],old)
                else:
                    old = h_ref[str(seed),mode]
                    for group,rate,n in (("all_drone_miss_with_G","miss","n_drone"),("v3_test_FA","fa","n_background")):
                        assert main[group]["n"] == int(old[n])
                        assert main[group]["k"] == round(float(old[rate])*int(old[n]))
                    old_matrix = next(r for r in h_expected if r["band"] == band and r["seed"] == str(seed) and r["mode"] == mode)
                    assert all(confusion[k] == int(old_matrix[k]) for k in ("TN","FP","FN","TP"))
                    # H reported all-drone misses. Derive non-G from H's frozen per-clip predictions.
                    old_rows = rows(TASK_H / f"test_predictions_{band}_seed{seed}.csv")
                    from evaluate import generalization
                    non_g = [r for r in old_rows if r["label"] == "drone" and not generalization(r)]
                    col = "pred_argmax" if mode == "argmax" else "pred_thr"
                    assert (main["nonG_miss"]["n"],main["nonG_miss"]["k"]) == (len(non_g),sum(r[col]=="0" for r in non_g))
                metrics.extend(entries)
                matrices.append(confusion)
            print("L20 eval", result[-1], flush=True)
    writecsv(OUT / "L20_reference_metrics.csv", metrics)
    writecsv(OUT / "L20_reference_confusion.csv", matrices)
    return result


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "hash":
        current = protected_hashes()
        dump(OUT / "sha256_after.json", current)
        import json
        before = json.loads((OUT / "sha256_before.json").read_text(encoding="utf-8"))
        assert current == before, [k for k in current.keys() | before.keys() if current.get(k) != before.get(k)]
        print("Protected files identical",len(current), flush=True)
        return
    report = {}
    torch.backends.cudnn.benchmark = True
    for name, check in (("sha256_before",protected_hashes),("index",check_index),("spec",check_spec),
                        ("eval",check_eval),("pairs",check_pairs)):
        value = check()
        if name == "sha256_before":
            dump(OUT / "sha256_before.json", value)
            value = {"files":len(value)}
        report[name] = value
        dump(OUT / "preflight_progress.json", report)
        print("PASS",name,value,flush=True)
    disk_guard()
    dump(OUT / "preflight_pass.json",report)


if __name__ == "__main__":
    main()
