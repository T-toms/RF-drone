# -*- coding: utf-8 -*-
"""Replot original R3 CSV evidence with public-facing R4 Chinese labels at 300 dpi.
Only r4_assets is writable; original R2/R3 data, scripts and images stay unchanged.
"""
from pathlib import Path
B=Path(__file__).resolve().parent
src=(B/'r3_figs.py').read_text(encoding='utf-8-sig')
src=src.replace("r3_assets","r4_assets").replace('城区','真实环境').replace('dpi=200','dpi=300')
src=src.replace('exec(src,globals())', """
src = (src.replace('城区无人机召回率','检出率')
          .replace('城区无人机召回','真实环境数据中的无人机检出率')
          .replace('城区无人机检出率率（%）','检出率（%）')
          .replace('城区无人机检出率（%）','检出率（%）')
          .replace('城区无人机检出率','真实环境数据中的无人机检出率')
          .replace('城区','真实环境')
          .replace('dpi=200','dpi=300')
          .replace("'N='+rr[1]['n']","rr[1]['n']+'片'"))
exec(src,globals())
""")
exec(compile(src,str(B/'r3_figs.py'),'exec'),globals())


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(enumerate(csv.DictReader(stream), 2))


def background_bars():
    source = R / 'work/train_stage1_L20_bgA_20261109/metrics_paired.csv'
    rows = csv_rows(source)
    groups = ['v3_test_FA', 'bgA_test_FA', 'quiet013_FA', 'white_noise_FA', 'notched_noise_FA']
    labels = ['真实环境\n背景', '同设备\n背景', '安静\n录音', '白噪声', '陷波\n白噪声']
    selected = [next((i, r) for i, r in rows if r['group'] == group
                     and r['category'] == 'main' and r['mode'] == 'argmax')
                for group in groups]
    fig, ax = plt.subplots(figsize=(4.5, 2.22))
    positions = np.arange(len(labels))
    for tag, offset, color, label in [
        ('H', -.19, '#0079BA', '首轮'),
        ('J', .19, '#ED8B2C', '加入同设备背景'),
    ]:
        values = [100 * float(r[tag + '_mean']) for _, r in selected]
        errors = [100 * float(r[tag + '_std']) for _, r in selected]
        ax.bar(positions + offset, values, width=.36, yerr=errors, capsize=2.6,
               color=color, label=label, error_kw={'lw': .85})
    ax.set_xticks(positions, labels, fontsize=9.5)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='y', labelsize=9.5)
    ax.tick_params(axis='x', length=0, pad=4)
    ax.set_ylabel('误报率（%）', fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#E0E6EB', linewidth=.6)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=2,
              frameon=False, fontsize=10, borderaxespad=0)
    fig.subplots_adjust(left=.155, right=.985, bottom=.225, top=.82)
    fig.savefig(A / 'background_fa_r4b.png', dpi=300)
    plt.close(fig)
    return [{'source': str(source), 'output': 'background_fa_r4b.png', 'row': i,
             'mean_std': {tag: [r[tag + '_mean'], r[tag + '_std']] for tag in ('H', 'J')}}
            for i, r in selected]


def background_curves():
    source = R / 'work/review_taskH_20261108/ctrl_whitenoise_results.csv'
    rows = csv_rows(source)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.45), sharey=True)
    levels = [-10, -5, 0, 5, 10, 15, 20]
    records = []
    for ax, group, title in zip(axes, ['data_drone', 'v3_drone'],
                                ['实验室无人机（训练见过的型号）', '真实环境数据中的无人机']):
        subset = [(i, r) for i, r in rows
                  if r['band'] == '2.4GHz' and r['group'] == group and r['seen'] == 'True']
        baseline = np.mean([100 * float(r['det']) for _, r in subset if r['bgtype'] == 'clean'])
        ax.axhline(baseline, color='#404040', linestyle='--', linewidth=1)
        ax.text(20, baseline - 2, f'原样本 {baseline:.0f}%', va='top', ha='right', fontsize=11)
        for kind, color, label in [('wn', '#0079BA', '白噪声'), ('v3bg', '#ED8B2C', '真实环境背景')]:
            values = [[100 * float(r['det']) for _, r in subset
                       if r['bgtype'] == kind and r['x_db'] == str(level)] for level in levels]
            assert all(len(v) == 3 for v in values)
            means = np.mean(values, axis=1)
            deviations = np.std(values, axis=1, ddof=1)
            ax.fill_between(levels, means - deviations, means + deviations, color=color, alpha=.15)
            ax.plot(levels, means, color=color, marker='o', label=label)
        ax.set(title=title, xticks=levels, ylim=(0, 109), xlabel='无人机样本功率／背景功率（dB）')
        ax.tick_params(labelsize=11)
        ax.title.set_fontsize(14)
        ax.xaxis.label.set_fontsize(12)
        ax.grid(axis='y', alpha=.2)
        records.extend(i for i, _ in subset)
    axes[0].set_ylabel('检出率（%）', fontsize=12)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2,
               frameon=False, fontsize=12)
    fig.subplots_adjust(left=.07, right=.99, bottom=.27, top=.87, wspace=.15)
    fig.savefig(A / 'background_r4b.png', dpi=300)
    plt.close(fig)
    return [{'source': str(source), 'output': 'background_r4b.png', 'rows': records}]


manifest = A / 'manifest.json'
records = json.loads(manifest.read_text(encoding='utf-8'))
wifi_source = R / 'work/side_L_wifimix_20261110/metrics_paired.csv'
wifi_rows = csv_rows(wifi_source)
records.append({
    'source': str(wifi_source), 'output': 'wifi_focus.png',
    'detection_row': next(i for i, r in wifi_rows
                          if r['category'] == 'main' and r['group'] == 'v3_drone_recall' and r['mode'] == 'argmax'),
    'background_rows': [i for i, r in wifi_rows if r['category'] == 'C1' and r['mode'] == 'argmax'],
    'aggregation': 'Combine k/n across levels within each repeat, then use mean and sample standard deviation.',
})
records.extend(background_bars())
records.extend(background_curves())
manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
