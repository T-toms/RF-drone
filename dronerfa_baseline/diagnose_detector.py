"""Read-only checkpoint inference and validation-only threshold diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs' / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dronerfa_data import DroneRFaWindowDataset, load_index, parse_snr_list, snr_name
from model import ResNet18, SpectrogramFrontEnd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stats(y, margins, threshold):
    positive = y == 1
    pred = margins > threshold
    tp = int(np.sum(pred & positive))
    fp = int(np.sum(pred & ~positive))
    fn = int(np.sum(~pred & positive))
    tn = int(np.sum(~pred & ~positive))
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    return dict(samples=len(y), tp=tp, fn=fn, fp=fp, tn=tn,
                recall=recall, fpr=fpr, specificity=1-fpr,
                accuracy=(tp+tn)/len(y), balanced_accuracy=(recall+1-fpr)/2)


def auc(y, margins):
    a, b = margins[y == 1], margins[y == 0]
    return float(((a[:, None] > b).sum() + .5*(a[:, None] == b).sum()) / (len(a)*len(b)))


def probability(margin):
    if margin >= 0:
        return 1 / (1 + math.exp(-margin))
    e = math.exp(margin)
    return e / (1 + e)


def summarize(out, records, cfg):
    groups = {}
    for name in cfg['conditions']:
        rr = [r for r in records if r['condition'] == name]
        groups[name] = (np.array([r['label'] for r in rr]),
                        np.array([r['logit_margin'] for r in rr]))
    # One common threshold; never use a true-SNR label to switch thresholds.
    candidates = np.unique(np.r_[0., [r['logit_margin'] for r in records]])
    feasible = []
    sensitivity_choices = {b: [] for b in sorted({cfg['validation_fpr_limit'], .01, .02, .05})}
    for threshold in candidates:
        ss = [stats(y, m, threshold) for y, m in groups.values()]
        candidate = (np.mean([s['recall'] for s in ss]),
                     np.mean([s['specificity'] for s in ss]), float(threshold))
        for budget in sensitivity_choices:
            if max(s['fpr'] for s in ss) <= budget:
                sensitivity_choices[budget].append(candidate)
        if max(s['fpr'] for s in ss) <= cfg['validation_fpr_limit']:
            feasible.append(candidate)
    _, _, threshold = max(feasible)
    chosen = {'logit_margin_threshold': threshold,
              'equivalent_drone_score_threshold': probability(threshold),
              'rule': 'logit_drone - logit_background > threshold',
              'selection_split': 'val',
              'constraint': 'empirical FPR <= limit in EVERY validation noise condition',
              'fpr_limit': cfg['validation_fpr_limit'],
              'objective': 'maximize mean recall; ties maximize specificity, then threshold',
              'status': 'exploratory; same validation used for selection and description; not test performance'}
    (out/'selected_threshold.json').write_text(json.dumps(chosen, indent=2), encoding='utf-8')
    comparisons, curves, breakdown = [], [], []
    for name, (y, m) in groups.items():
        for policy, th in [('default_0.5', 0.), ('validation_selected_common', threshold)]:
            comparisons.append(dict(condition=name, policy=policy, threshold_margin=th,
                                    **stats(y, m, th), roc_auc=auc(y, m)))
        for th in np.r_[np.nextafter(m.min(), -np.inf), np.unique(m), 0., threshold]:
            curves.append(dict(condition=name, threshold_margin=float(th),
                               threshold_score=probability(float(th)), **stats(y, m, th)))
        rr = [r for r in records if r['condition'] == name]
        for target in sorted({r['target_name'] for r in rr}):
            subset = [r for r in rr if r['target_name'] == target]
            yy = np.array([r['label'] for r in subset])
            mm = np.array([r['logit_margin'] for r in subset])
            for policy, th in [('default_0.5', 0.), ('validation_selected_common', threshold)]:
                breakdown.append(dict(condition=name, target=target, policy=policy,
                                      samples=len(subset), predicted_drone=int((mm > th).sum()),
                                      correct=int(((mm > th) == yy).sum())))
    write_csv(out/'validation_comparison.csv', comparisons)
    write_csv(out/'validation_threshold_sweep.csv', curves)
    write_csv(out/'validation_by_model.csv', breakdown)
    sensitivity=[]
    for budget, choices in sensitivity_choices.items():
        _, _, th=max(choices)
        for name,(y,m) in groups.items():
            sensitivity.append(dict(fpr_budget=budget, condition=name,
                                    common_threshold_margin=th,
                                    common_threshold_score=probability(th), **stats(y,m,th)))
    write_csv(out/'validation_budget_sensitivity.csv', sensitivity)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, (name, (y, m)) in zip(axes.flat, groups.items()):
        bins = np.linspace(min(m.min(), 0, threshold)-.1, max(m.max(), 0, threshold)+.1, 36)
        ax.hist(m[y == 0], bins=bins, alpha=.6, label='Background')
        ax.hist(m[y == 1], bins=bins, alpha=.6, label='Drone')
        ax.axvline(0, color='black', ls='--', label='Default')
        ax.axvline(threshold, color='red', ls=':', label='Validation-selected')
        ax.set(title=name, xlabel='Drone logit minus background logit', ylabel='Windows')
    axes.flat[0].legend(fontsize=8)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle('Validation score distributions (same source windows across noise conditions)')
    fig.savefig(out/'score_distributions.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for name in groups:
        cc = sorted([r for r in curves if r['condition'] == name], key=lambda r:(r['fpr'],r['recall']))
        for ax in axes:
            ax.plot([r['fpr'] for r in cc], [r['recall'] for r in cc], label=name)
        for r, marker in zip([r for r in comparisons if r['condition']==name], ['o','x']):
            for ax in axes:
                ax.scatter(r['fpr'], r['recall'], marker=marker, color='black', s=25)
    for ax in axes:
        ax.set(xlabel='False-positive fraction', ylabel='Drone recall', ylim=(0,1.02))
        ax.grid(alpha=.25)
    axes[0].set(xlim=(0,1), title='ROC: circles=default, crosses=selected')
    axes[1].set(xlim=(0,.05), title='Low false-positive region')
    axes[0].legend(fontsize=8)
    fig.savefig(out/'validation_roc.png', dpi=160)
    plt.close(fig)

    negative_count = sum(r['label']==0 for r in records if r['condition']==cfg['conditions'][0])
    lines = ['# 第一阶段检测器：验证集分数与门槛诊断', '',
             '本次加载已有模型重新推理，没有训练新网络，也没有使用测试集选门槛。', '',
             f"- 原模型轮次：{cfg['checkpoint_epoch']}；设备：{cfg['gpu']}。",
             f"- 每档验证窗口：{cfg['validation_windows']}，其中背景 {negative_count} 个。",
             '- 各噪声档使用同一批原始窗口，是相关的压力测试，不能当成多批独立采集。',
             '- 噪声数值表示额外 AWGN 退化，不是现场绝对信噪比。', '',
             '## 本次如何选门槛', '',
             f"预先采用探索性规则：每档验证背景误报比例不超过 {cfg['validation_fpr_limit']:.1%}，在这个限制下尽量提高平均检出比例。",
             f"得到同一个公共分数门槛 **{probability(threshold):.8f}**（原默认值为 0.5），对应 logit 差门槛 {threshold:.8f}。",
             '这个门槛对所有噪声档一致，不需要预先知道输入的真实 SNR。1% 是本轮诊断工作点，不是用户确定的业务要求。', '',
             '## 实际结果', '',
             '| 附加噪声 | 默认检出率 | 公共门槛检出率 | 默认误报数 | 公共门槛误报数 | ROC-AUC |',
             '|---|---:|---:|---:|---:|---:|']
    for name in groups:
        a,b=[r for r in comparisons if r['condition']==name]
        lines.append(f"| {name} | {a['recall']:.2%} | {b['recall']:.2%} | {a['fp']}/{negative_count} | {b['fp']}/{negative_count} | {a['roc_auc']:.4f} |")
    lines += ['', '## 误报要求放宽时会怎样', '',
              '下面是预先列出的 1%、2%、5% 三种探索性误报限制。每一行仍用同一个门槛覆盖全部噪声档；不是按真实 SNR 切换门槛，也不代表建议接受更高误报。', '',
              '| 每档误报限制 | 公共分数门槛 | −15 dB 检出比例 | −20 dB 检出比例 | 各档中最多误报数 |',
              '|---|---:|---:|---:|---:|']
    for budget in sensitivity_choices:
        rr=[r for r in sensitivity if r['fpr_budget']==budget]
        by_name={r['condition']:r for r in rr}
        a=f"{by_name['-15dB']['recall']:.2%}" if '-15dB' in by_name else '未运行'
        b=f"{by_name['-20dB']['recall']:.2%}" if '-20dB' in by_name else '未运行'
        lines.append(f"| {budget:.0%} | {rr[0]['common_threshold_score']:.8f} | {a} | {b} | {max(r['fp'] for r in rr)}/{negative_count} |")
    lines += ['', '## 如何理解', '',
              '分数排序能力较好、默认门槛漏检多时，门槛可能是原因之一；只有改变门槛同时保持可接受误报才有意义。高 AUC 不保证在很低误报下也能检出。',
              '本表是在同一验证集上选择并描述门槛，属于乐观的探索结果，不能称为测试集提升或已经解决低 SNR。原测试集此前已被反复查看，未来正式结论还需要新的独立留出记录。',
              f'每档只有 {negative_count} 个背景窗，一个误报对应 {1/negative_count:.3%}。相邻窗口相关，不能凭少量/零误报宣称现场误报率有统计保证。',
              '本次保留原始权重和原实验；没有修改实际运行默认门槛。后续可基于验证结果决定是否需要调整训练噪声范围和低 SNR 验证策略。', '',
              '## 输出', '',
              '- validation_scores.csv：每窗来源、标签、噪声条件、两类 logits 和分数。',
              '- validation_comparison.csv：默认与公共门槛对比。',
              '- validation_by_model.csv：分别检查两个机型。',
              '- validation_threshold_sweep.csv：完整门槛扫描。',
              '- validation_budget_sensitivity.csv：不同误报要求下的公共门槛对比。',
              '- selected_threshold.json：探索性候选门槛及选择规则。',
              '- run_config.json：输入文件哈希、参数、运行时间和审计。',
              '- score_distributions.png、validation_roc.png：分数分布与误报/检出权衡。', '',
              '![分数分布](score_distributions.png)', '', '![ROC](validation_roc.png)', '']
    if '-15dB' in groups:
        rr=[r for r in breakdown if r['condition']=='-15dB' and r['policy']=='default_0.5' and r['target']!='Background']
        intro=['## 先看结论', '',
               '第一阶段的低 SNR 漏检已经重现，但两个机型表现差异很大。默认门槛下，−15 dB 的结果如下：', '',
               '| 机型 | 正确检出 | 漏检 |', '|---|---:|---:|']
        for r in rr:
            intro.append(f"| {r['target']} | {r['predicted_drone']}/{r['samples']} | {r['samples']-r['predicted_drone']} |")
        intro += ['', '本次每个类别只有一个验证源文件，差异也可能涉及录制条件，不能直接断言某机型天生更难检测。', '',
                  '门槛调整有取舍：更严格限制误报会增加漏检；放宽误报限制能挽回一部分无人机。不能把“调低门槛”当成已经解决问题。',
                  '下一步应优先检查漏检集中的录制窗口，并让训练选模型过程包含低 SNR 验证，再决定是否修改网络；两阶段框架不变。', '']
        lines[2:2]=intro
    (out/'诊断报告.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'selected':chosen, 'comparison':comparisons},ensure_ascii=False),flush=True)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'outputs/pilot_5epochs/best_model.pt')
    parser.add_argument('--index',type=Path,default=ROOT/'outputs/window_index.csv')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--snrs',default='clean,0,-5,-10,-12,-15,-20')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--fpr-limit',type=float,default=.01)
    parser.add_argument('--analyze-only',type=Path,help='Regenerate analysis from an existing completed score directory; no inference')
    args=parser.parse_args()
    if args.analyze_only:
        out=args.analyze_only
        cfg=json.loads((out/'run_config.json').read_text(encoding='utf-8'))
        if cfg['status']!='complete':
            raise ValueError('Score collection must be complete before reanalysis')
        with (out/'validation_scores.csv').open(encoding='utf-8-sig',newline='') as f:
            records=list(csv.DictReader(f))
        for r in records:
            r['label']=int(r['label'])
            r['logit_margin']=float(r['logit_margin'])
        summarize(out,records,cfg)
        return
    if not 0 <= args.fpr_limit <= 1:
        raise ValueError('fpr-limit must lie in [0,1]')
    out=args.output_dir or ROOT/'outputs'/('validation_diagnosis_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    rows=load_index(args.index)
    split_files={s:{r['file_path'] for r in rows if r['split']==s} for s in ('train','val','test')}
    if any(split_files[a]&split_files[b] for a,b in [('train','val'),('val','test'),('train','test')]):
        raise ValueError('Source file split leakage')
    val=[r for r in rows if r['split']=='val']
    if {r['label'] for r in val}!={0,1}:
        raise ValueError('Validation must have both classes')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; diagnose environment before a large CPU run')
    device=torch.device('cuda')
    checkpoint_hash=digest(args.checkpoint)
    # This locally produced checkpoint includes torch.__version__ metadata.
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    old=checkpoint['run_config']['arguments']
    frontend=SpectrogramFrontEnd(n_fft=old['n_fft'],hop_length=old['hop_length'],image_size=old['image_size']).to(device).eval()
    model=ResNet18(input_channels=1,num_classes=2).to(device).eval()
    frontend.load_state_dict(checkpoint['frontend_state'])
    model.load_state_dict(checkpoint['model_state'])
    del checkpoint['model_state']
    snrs=parse_snr_list(args.snrs)
    amp=bool(old['amp'])
    seed=int(old.get('validation_seed', int(old['seed'])+10000))
    cfg=dict(checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checkpoint_hash,
             index=str(args.index.resolve()),index_sha256=digest(args.index),
             checkpoint_epoch=checkpoint['epoch'],split='val',validation_windows=len(val),
             validation_files=sorted(split_files['val']),test_used=False,
             conditions=[snr_name(s) for s in snrs],noise_seed=seed,
             validation_fpr_limit=args.fpr_limit,amp=amp,batch_size=args.batch_size,
             frontend=dict(n_fft=old['n_fft'],hop_length=old['hop_length'],image_size=old['image_size']),
             gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),status='running')
    (out/'run_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(f'OUTPUT {out}\nGPU {cfg["gpu"]}; validation={len(val)}; conditions={cfg["conditions"]}',flush=True)
    started=time.perf_counter()
    records=[]
    for snr in snrs:
        name=snr_name(snr)
        dataset=DroneRFaWindowDataset(args.index,split='val',fixed_snr=snr,apply_fixed_snr=snr is not None,seed=seed)
        loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
        offset=0
        for signals,labels in loader:
            signals=signals.to(device)
            features=frontend(signals.float())
            with torch.amp.autocast('cuda',enabled=amp,dtype=torch.float16):
                logits=model(features)
            ll=logits.float().cpu().numpy()
            if not np.isfinite(ll).all():
                raise ValueError('Non-finite logits')
            for j,pair in enumerate(ll):
                row=dataset.rows[offset+j]
                assert int(labels[j])==int(row['label'])
                margin=float(pair[1]-pair[0])
                records.append(dict(condition=name,filename=row['filename'],file_path=row['file_path'],
                                    start=row['start'],length=row['length'],block_index=row['block_index'],
                                    target_code=row['target_code'],target_name=row['target_name'],
                                    split='val',label=int(labels[j]),logit_background=float(pair[0]),
                                    logit_drone=float(pair[1]),logit_margin=margin,drone_score=probability(margin),
                                    default_prediction=int(margin>0)))
            offset+=len(ll)
            if offset%100==0 or offset==len(dataset):
                print(f'{name}: {offset}/{len(dataset)}; total elapsed {time.perf_counter()-started:.1f}s',flush=True)
        dataset.close()
        assert offset==len(val)
        write_csv(out/'validation_scores.csv',records)
    assert len(records)==len(val)*len(snrs)
    summarize(out,records,cfg)
    cfg.update(status='complete',elapsed_seconds=time.perf_counter()-started,
               output_rows=len(records),checkpoint_unchanged=digest(args.checkpoint)==checkpoint_hash)
    assert cfg['checkpoint_unchanged']
    (out/'run_config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    print(f'COMPLETE {out}',flush=True)


if __name__=='__main__':
    main()
