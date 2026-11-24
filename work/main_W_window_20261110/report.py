import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))

from core import (
    BANDS, CACHE_ROOT, FC, FS, OUT, SEEDS, TASK_H, TASK_M, WINDOWS,
    IQReader, disk_guard, index, kept, np, rows, writecsv,
)
from stats import cp


def data():
    metrics = rows(OUT / "L20_reference_metrics.csv")
    metrics = [r for r in metrics if not (r["category"] == "state_untrained_recall"
               and r["band"] == "2.4GHz" and "mini 4k" in r["group"])]
    matrices = rows(OUT / "L20_reference_confusion.csv")
    for path, band in ((TASK_M,BANDS[0]),(TASK_H,BANDS[1])):
        for r in rows(path / "mix_results.csv"):
            if (band == BANDS[0] and r["round"] != "M") or (band == BANDS[1] and r["band"] != band):
                continue
            n = int(r["n"])
            rate = float(r.get("rate") or r["detection_rate"])
            k = round(rate*n)
            metrics.append(dict(L="20",band=band,seed=r["seed"],mode=r["mode"],category="mix_test",
                                group=r["group"],n=n,k=k,rate=rate,
                                cp95_upper=cp(k,n) if r["group"] == "v3_background" else "",x_db=r["x_db"]))
    completed = {("20",b) for b in BANDS}
    for window in WINDOWS:
        if window == "20":
            continue
        out = OUT / f"L{window}"
        for band in BANDS:
            if (out / f"{band}_evaluation_done.json").exists():
                metrics.extend(rows(out / f"{band}_metrics.csv"))
                metrics.extend(rows(out / f"{band}_mix_metrics.csv"))
                matrices.extend(rows(out / f"{band}_confusion.csv"))
                completed.add((window,band))
    return [dict(r,x_db=r.get("x_db","")) for r in metrics], matrices, completed


def low_is_worse(category, group):
    return (category.endswith("_recall") or group == "v3_drone_recall" or
            category == "mix_test" and group != "v3_background")


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        groups[tuple(str(r[k]) for k in ("L","band","mode","category","group","x_db"))].append(r)
    out = {}
    for key, group in groups.items():
        assert sorted(int(r["seed"]) for r in group) == list(SEEDS), key
        group.sort(key=lambda r: int(r["seed"]))
        rates = [float(r["rate"]) for r in group]
        sample_counts = {int(r["n"]) for r in group}
        assert len(sample_counts) == 1, key
        counts = [int(r["k"]) for r in group]
        bounds = [r["cp95_upper"] for r in group]
        out[key] = dict(zip(("L","band","mode","category","group","x_db"),key),
                        n=sample_counts.pop(), k_by_seed="/".join(map(str,counts)),
                        cp95_by_seed="/".join(f"{float(v):.8f}" for v in bounds) if bounds[0] != "" else "",
                        mean=float(np.mean(rates)), std=float(np.std(rates,ddof=1)),
                        worst=min(rates) if low_is_worse(key[3],key[4]) else max(rates))
    return out


def compare_windows(summary):
    groups = defaultdict(dict)
    for (window,band,mode,category,group,x),r in summary.items():
        if category == "miss_by_model" or group in ("nonG_miss","v3_drone_recall"):
            groups[band,mode,category,group][window] = r
    result = []
    for (band,mode,category,group),points in groups.items():
        for a,b in itertools.combinations([w for w in WINDOWS if w in points],2):
            first,second = points[a],points[b]
            difference = second["mean"] - first["mean"]
            spread = max(first["std"],second["std"])
            result.append(dict(band=band,mode=mode,category=category,group=group,L_a=a,L_b=b,
                               mean_difference=difference,max_seed_std=spread,
                               comparison="不能区分" if abs(difference) <= spread else "差值大于最大种子标准差"))
    return result


def cell(r, fa=False):
    if r is None:
        return "—"
    base = f'{100*r["mean"]:.2f}±{100*r["std"]:.2f} [{100*r["worst"]:.2f}]'
    if fa:
        detail = f'[{r["k_by_seed"].replace("/",",")}]/{r["n"]}'
        if r["cp95_by_seed"]:
            upper = "/".join(f"{100*float(p):.2f}" for p in r["cp95_by_seed"].split("/"))
            detail += f" U95 {upper}"
        return f"{detail}<br>{base}"
    return f'{base}<br>N={r["n"]}'


def inventory_lines():
    counts = {(r["L"],r["band"],r["dataset"],r["label"],r["split"]):r for r in rows(OUT / "inventory.csv")}
    cols = (("data","drone"),("v3","drone"),("v3","background"),("dataA_mirror","background"))
    result = [
        "## 片数与覆盖", "",
        "各格按 train / val / test 列出 keep/总片数 · keep 不重复秒数。末列为 v3 test 0 次虚警的单侧 95% 上界。",
        "", "| L ms | 频段 | data 无人机 | v3 无人机 | v3 背景 | bgA | v3 test 0/N · U95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for w,band in itertools.product(WINDOWS,BANDS):
        cells = []
        for ds,lab in cols:
            subsets = [counts[w,band,ds,lab,s] for s in ("train","val","test") if (w,band,ds,lab,s) in counts]
            cells.append("<br>".join(f'{r["keep"]}/{r["n"]} · {float(r["keep_unique_s"]):.3f}' for r in subsets) or "—")
        n = int(counts[w,band,"v3","background","test"]["keep"])
        result.append(f"| {w} | {band} | " + " | ".join(cells) + f" | 0/{n} · {100*cp(0,n):.2f}% |")
    return result


def resolution_lines():
    result = ["## 时间分辨率", "",
              "每格为 STFT 帧数 / ms 每列 / 0.4 ms 对应列数。按帧数 × 512 / fs / 256 计算。",
              "", "| L ms | 2.4 GHz | 5.8 GHz |", "|---|---|---|"]
    for w in WINDOWS:
        cells = []
        for b in BANDS:
            n = int(w.rstrip("t")) * FS[b] // 1000
            frames = 1 + (n-1024)//512
            ms = frames * 512 / FS[b] / 256 * 1000
            cells.append(f"{frames} / {ms:.5f} / {0.4/ms:.2f}")
        result.append(f"| {w} | " + " | ".join(cells) + " |")
    return result


def metrics_lines(summary, differences):
    result = ["## 主表", "",
              "百分比为种子 2026/2027/2028 的均值 ± 样本标准差 [最差]。虚警先列三种子的 k/共同 N 和各自 U95（%）。† 表示长窗的虚警因样本数、覆盖或重叠变化，与短窗不可直接比较。* 表示与 L20 不能区分。",
              "", "| 频段 | 判决 | L ms | nonG 漏检 | v3 无人机召回 | v3 虚警 | bgA 虚警 | 白噪声虚警 |",
              "|---|---|---:|---|---|---|---|---|"]
    tied = {(r["band"],r["mode"],r["group"],r["L_a"] if r["L_b"]=="20" else r["L_b"])
            for r in differences if "20" in (r["L_a"],r["L_b"]) and r["comparison"] == "不能区分"}
    groups = ("nonG_miss","v3_drone_recall","v3_test_FA","bgA_test_FA","white_noise_FA")
    for b,mode,w in itertools.product(BANDS,("argmax","thr_fa1"),WINDOWS):
        r = [summary.get((w,b,mode,"main",group,"")) for group in groups]
        if r[0] is None:
            continue
        mark = "†" if w in ("30","40","50t") else ""
        entries = [cell(item,i>=2) + (" *" if (b,mode,groups[i],w) in tied else "")
                   for i,item in enumerate(r)]
        result.append(f"| {b} | {mode} | {w}{mark} | " + " | ".join(entries) + " |")
    return result


def model_lines(summary, differences):
    result = ["## 按型号漏检", "",
              "每格上行 argmax、下行 thr_fa1，均为均值 ± 标准差 [最差]（百分比）。图中同一判决下三个点左至右为种子 2026/2027/2028，线连均值。* 表示与 L20 的均值差不超过两者较大的种子标准差，标记为「不能区分」。" +
              ("[逐窗比较](model_window_differences.csv) 包含所有两两比较。" if differences else ""),
              "", "| 频段 | 型号 | L10 | L20 | L30 | L40 | L50t |",
              "|---|---|---|---|---|---|---|"]
    groups = sorted({(b,group) for w,b,mode,cat,group,x in summary if cat == "miss_by_model"})
    tied = {(r["band"],r["mode"],r["group"],r["L_a"] if r["L_b"]=="20" else r["L_b"])
            for r in differences if "20" in (r["L_a"],r["L_b"]) and r["comparison"] == "不能区分"}
    for band,group in groups:
        entries = []
        for w in WINDOWS:
            vals = [cell(summary.get((w,band,mode,"miss_by_model",group,""))).split("<br>")[0] +
                    (" *" if (band,mode,group,w) in tied else "") for mode in ("argmax","thr_fa1")]
            entries.append("<br>".join(vals))
        is_g = group in ({"evo lite 6k","evo ii pro v3"} if band == BANDS[0] else {"evo2pro_v3","fpv"})
        label = group + ("（G）" if is_g else "")
        result.append(f"| {band} | {label} | " + " | ".join(entries) + " |")
    return result


def plot_models(plt, records):
    from matplotlib.lines import Line2D

    styles = (("argmax","#18759c",-.12),("thr_fa1","#bb4f1b",.12))
    for band in BANDS:
        names = sorted({r["group"] for r in records if r["band"] == band and r["category"] == "miss_by_model"})
        nrows = (len(names)+3)//4
        fig,axs = plt.subplots(nrows,4,figsize=(14,2.3*nrows),sharex=True,sharey=True,squeeze=False)
        for ax,name in zip(axs.flat,names):
            for mode,color,offset in styles:
                means = []
                for wi,w in enumerate(WINDOWS):
                    rates = [100*float(r["rate"]) for r in records if
                             (str(r["L"]),r["band"],r["mode"],r["category"],r["group"]) ==
                             (w,band,mode,"miss_by_model",name)]
                    if rates:
                        ax.scatter(wi+offset+np.array([-.055,0,.055]),rates,color=color,s=12)
                        means.append((wi+offset,np.mean(rates)))
                if means:
                    ax.plot(*zip(*means),color=color,linewidth=1)
            is_g = name in ({"evo lite 6k","evo ii pro v3"} if band == BANDS[0] else {"evo2pro_v3","fpv"})
            ax.set(title=name + (" (G)" if is_g else ""),ylim=(-4,104),
                   xticks=range(len(WINDOWS)),xticklabels=WINDOWS)
            ax.grid(alpha=.2)
        for ax in list(axs.flat)[len(names):]:
            ax.set_visible(False)
        fig.legend(handles=[Line2D([],[],color=color,marker="o",label=mode) for mode,color,_ in styles],
                   loc="upper center",ncols=2,frameon=False)
        fig.supxlabel("窗长 (ms)")
        fig.supylabel("漏检率 (%)")
        fig.tight_layout(rect=(0,0,.99,.96))
        fig.savefig(OUT / f"model_miss_{band}.png",dpi=145)
        plt.close(fig)


def plot_mix(plt, summary):
    xs = (-10,-5,0,5,10,15,20)
    for b in BANDS:
        fig,axs = plt.subplots(2,3,figsize=(14,7),sharex=True,sharey=True)
        for axes,mode in zip(axs,("argmax","thr_fa1")):
            for ax,group in zip(axes,("data_drone","v3_drone","v3_background")):
                for w in WINDOWS:
                    points = [summary.get((w,b,mode,"mix_test",group,str(x))) for x in xs]
                    if all(points):
                        ax.plot(xs,[100*p["mean"] for p in points],marker="o",label=f"{w} ms")
                ax.set(title=f"{mode} · {group}",xlabel="叠加功率比 X (dB)",
                       ylabel="虚警率 (%)" if group=="v3_background" else "检出率 (%)",ylim=(-2,102))
                ax.grid(alpha=.25)
        axs[0,0].legend(ncols=2)
        fig.tight_layout()
        fig.savefig(OUT / f"test_mix_{b}.png",dpi=145)
        plt.close(fig)


def plot_signal(plt):
    from cache_build import to_spec

    indexes = {w:index(w) for w in WINDOWS}
    for band,half in zip(BANDS,(25,45)):
        by_l = {w:{(r["block_file"],r["start"]):r for r in indexes[w] if
                   r["band"]==band and r["dataset"]=="data" and r["split"]=="test" and kept(r)}
                for w in WINDOWS}
        shared = sorted(set.intersection(*(set(v) for v in by_l.values())))
        key = next((k for k in shared if by_l["20"][k]["model"] == "mini5 pro"),shared[0])
        read = IQReader()
        iq = read(by_l["50t"][key])
        fig,axes = plt.subplots(5,1,figsize=(10,11),sharex=True,sharey=True,layout="constrained")
        for ax,w in zip(axes,WINDOWS):
            n = int(by_l[w][key]["n_samples"])
            image = to_spec(iq[:n],band)
            pic = ax.imshow(image,origin="lower",aspect="auto",cmap="viridis",vmin=-2,vmax=3,
                            extent=(0,int(w.rstrip("t")),FC[band]-half,FC[band]+half))
            ax.set(xlim=(0,50),ylabel="MHz",title=f"{w} ms · {by_l[w][key]['sample_id']}")
        axes[-1].set_xlabel("距同一 IQ 起点的时间 (ms)")
        fig.suptitle(f"{band} · {Path(key[0]).stem} · start={key[1]}",fontsize=10)
        fig.colorbar(pic,ax=axes,label="逐片标准化后的 dB")
        fig.savefig(OUT / f"same_iq_{band}.png",dpi=145)
        plt.close(fig)


def verify_runs():
    assert (OUT / "train_M.py").read_bytes() == (TASK_M / "train.py").read_bytes()
    assert (OUT / "train_H.py").read_bytes() == (TASK_H / "train.py").read_bytes()
    for window in ("10","30","40","50t"):
        local = OUT / f"L{window}"
        cache = CACHE_ROOT / f"L{window}"
        for band in BANDS:
            low = band == BANDS[0]
            marker_root = cache if low else local
            tag = f"{band}_Lb" if low else band
            log_tag = "Lb" if low else band
            runs = [(2026,3,"_smoke")] + [(s,40,"") for s in SEEDS]
            for seed,epochs,suffix in runs:
                run = f"{tag}_seed{seed}{suffix}"
                info = json.loads((marker_root / f"{run}_done.json").read_bytes())
                assert (info["band"],info["seed"],info["epochs"],info["amp"]) == (
                    band,seed,epochs,"torch.bfloat16")
                log = rows(local / f"train_log_{log_tag}_seed{seed}{suffix}.csv")
                assert [int(r["epoch"]) for r in log] == list(range(1,epochs+1)), (window,run)
                assert all((cache / f"{run}_{name}.pt").stat().st_size > 0 for name in ("best","last"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial",action="store_true")
    args = parser.parse_args()
    os.environ["MPLCONFIGDIR"] = str(OUT / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                         "axes.unicode_minus":False})
    records, matrices, completed = data()
    if not args.partial:
        assert completed == set(itertools.product(WINDOWS,BANDS)), sorted(completed)
        verify_runs()
    summary = summarize(records)
    comparisons = compare_windows(summary)
    writecsv(OUT / "metrics_by_seed.csv",records)
    writecsv(OUT / "metrics_summary.csv",summary.values())
    non_g = {(str(r["L"]),r["band"],str(r["seed"]),r["mode"]):r for r in records
             if r["category"]=="main" and r["group"]=="nonG_miss"}
    both_matrices = []
    for matrix in matrices:
        both_matrices.append(dict(matrix,population="all_keep"))
        ref = non_g[str(matrix["L"]),matrix["band"],str(matrix["seed"]),matrix["mode"]]
        restricted = dict(matrix,population="nonG",FN=int(ref["k"]),TP=int(ref["n"])-int(ref["k"]))
        both_matrices.append(restricted)
    writecsv(OUT / "confusion_matrices.csv",both_matrices)
    writecsv(OUT / "model_window_differences.csv",comparisons)
    for category in ("miss_by_model_state_bw","miss_by_file","fa_by_session"):
        writecsv(OUT / f"{category}.csv",[r for r in records if r["category"]==category])
    writecsv(OUT / "same_model_recall.csv",[
        dict(r,G="是" if r["band"]==BANDS[0] and r["group"].startswith("data | ") else "否")
        for r in summary.values() if r["category"]=="same_model_recall"
    ])
    writecsv(OUT / "G_and_untrained_recall.csv",[r for r in summary.values() if
             r["category"] in ("G_recall","state_untrained_recall")])
    plot_models(plt,records)
    plot_mix(plt,summary)
    if not args.partial:
        plot_signal(plt)
    pre = json.loads((OUT / "preflight_pass.json").read_text(encoding="utf-8"))
    before = OUT / "sha256_before.json"
    after = OUT / "sha256_after.json"
    hashes_ok = after.exists() and json.loads(before.read_bytes()) == json.loads(after.read_bytes())
    if not args.partial:
        assert hashes_ok, "Protected input hashes differ or are missing"
    peaks = [f'{r["band"]} {r["seed"]} {r["max_abs"]:.7f}' for r in pre["eval"]]
    bga_all = []
    for mode in ("argmax","thr_fa1"):
        item = summary.get(("50t",BANDS[0],mode,"main","bgA_test_FA_all_correlated",""))
        if item:
            bga_all.append(f'{mode} [{item["k_by_seed"].replace("/",",")}]/{item["n"]}')
    lines = [
        "# 任务 W · 第一阶段窗长比较", "",
        "ResNet18，256×256 时频图，3 个种子，每个新窗长与频段先做 3 epoch 冒烟，再各从头训练 40 epoch。L20 引用 2.4G 任务 M、5.8G 任务 H。缓存与 checkpoint 位于 E:/main_W_window_20261110/。",
        *inventory_lines(), "", *resolution_lines(), "", *metrics_lines(summary,comparisons), "",
        *model_lines(summary,comparisons), "",
        "[逐种子指标](metrics_by_seed.csv) · [三种子统计](metrics_summary.csv) · [型号/状态/带宽](miss_by_model_state_bw.csv) · [逐文件漏检](miss_by_file.csv) · [会话虚警](fa_by_session.csv) · [混淆矩阵](confusion_matrices.csv) · [同型号召回](same_model_recall.csv) · [G 与该状态未训练](G_and_untrained_recall.csv)",
        "主表与 nonG 混淆矩阵排除 G，另列 all_keep 全量矩阵。2.4G 的 G 为 data 的 EVO Lite 6K 和 EVO II Pro V3。任务书 §4.4 写反了归属，同型号表按主线 §2 标记 data 的 EVO II Pro V3 为 G、训练未见，v3 033510 计入 nonG。5.8G 的 G 为 v3 EVO II Pro V3 和 FPV，033706 不在任务 F 索引。",
        "同型号表含 5.8G Mini 5 Pro 的 data/v3 和 2.4G EVO II Pro V3 的 data/v3，按带宽列出。各频段内每个 (dataset, model, state, bw_label) 的 test 只有一次录制，014 atom 无数据，001、009–013 只有 2.4G。",
        "[按型号漏检 2.4G](model_miss_2.4GHz.png) · [按型号漏检 5.8G](model_miss_5.8GHz.png) · [叠加 2.4G](test_mix_2.4GHz.png) · [叠加 5.8G](test_mix_5.8GHz.png)" +
        ("" if args.partial else " · [同起点 IQ 2.4G](same_iq_2.4GHz.png) · [同起点 IQ 5.8G](same_iq_5.8GHz.png)"), "",
        "## L50t", "",
        "原 4092 片，尾窗 471 片，合计 4563 片，keep 4154 片。v3 两窗重叠 1 ms，test 背景全计 CP。data 尾窗与前窗重叠 1–47 ms，中位 1 ms。",
        "bgA 全集片段不独立，仅原网格子集计 CP。" +
        ("全集 k/N 为 " + "、".join(bga_all) + "。" if bga_all else ""),
        "v3 的 49.6 ms 块无窗。2.4G train 的 noise_adrv_2450 由 L20 的 20 片降至 0，原为无 WiFi、有零星突发的接收机 B 背景。075302_ambient@2440 为 14→4。5.8G 的 041728、115406、115702 ambient 各为 18→8。",
        "v3 原窗筛掉 10/371（2.7%），尾窗筛掉 33/371（8.9%），尾窗筛掉的 33 片都是无人机，主要为 lack_feature，v3 无人机 keep 比例偏低。",
        "L50t / L20 keep 不重复秒数为 2.4G train v3 背景 5.148/4.680、v3 无人机 3.813/3.040、data 无人机 59.268/56.640，5.8G train v3 背景 4.653/3.880。", "",
        "## §5 复现", "",
        "L20 每频段复算 40 片（data/v3 各 20），两频段时频图最大差均为 0。训练叠加 2832 无人机片、2861 bgA 片的配对 ID 相同，X/g 最大相对差均为 0。",
        "六组 checkpoint 的样本 ID、两种逐片判决和主表 k/N 一致。2.4G 每种子 1305 测试片 + 1287 bgA，5.8G 每种子 878 测试片。5.8G nonG 由 H 冻结逐片预测重算。",
        "首次 2.4G seed 2026 最大概率差 0.0014518，暂停后协调端将上限从 0.001 修订为 0.005，保留计数和判决一致要求。复核各组最大绝对差依次为 " + "、".join(peaks) + f'，bgA 最大 {max(r["bgA_max_abs"] or 0 for r in pre["eval"]):.8f}。',
        "任务书 §6 缺计数表，按协调端确认采用 F、N2、bgA 的计数表。173 个组与 group_split 一致，bgA L50t 的 2215 个原网格行与原索引逐行一致，尾窗镜像 61 行。",
        f'保护目录 SHA256 共 {pre["sha256_before"]["files"]} 个文件，' + ("交付前复算逐项一致。" if hashes_ok else "待交付前复算。"),
        "两份训练脚本逐字复制，diff 均为空。L 路径由本地加载模块指定，叠加训练背景抽样池按 H 缓存顺序，原始样本按索引行写回，测试叠加按 H 的块顺序配对并逐批计算。" +
        f"本任务文件总长约 {disk_guard()/1e9:.3f} GB。",
    ]
    if args.partial:
        lines.insert(2,f"阶段快照，已完成 {len(completed)}/10 个窗长×频段组合（含引用的 L20）。")
    assert len(lines)<=120,len(lines)
    (OUT / "REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Report lines",len(lines),"output bytes",disk_guard(),flush=True)


if __name__ == "__main__":
    main()
