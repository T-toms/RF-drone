# 核查 B1 (a) 组：bgA vs v3 07-10 实验室背景的高 AUC 是"接收机"还是"内容"（突发/窄带）差异
# 只读特征缓存和 IQ 缓存；不写任何受保护目录。
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False

CACHE = Path('E:/fusionB_features_20261110')
OUT = Path('C:/Users/Administrator/Desktop/数据集处理/work/fusionB_20261110/check_a_content')
NAMES = json.loads((CACHE / 'feature_names.json').read_text(encoding='utf-8'))
NOTCH = pd.read_csv('C:/Users/Administrator/Desktop/数据集处理/work/crop_slice_20261107/notch_table.csv', encoding='utf-8-sig')
GAUSS = {'shape_factor': np.sqrt(4 / np.pi), 'c42n': 0.0, 'spec_flatness': 1.0}  # 平稳复高斯理论值（谱平坦度要求白谱）


def load(band='2.4GHz'):
    parts = []
    for s in ['train', 'val', 'test']:
        m = pd.read_csv(CACHE / f'{band}_{s}_spec_meta_features_meta.csv', encoding='utf-8')
        x = np.load(CACHE / f'{band}_{s}_spec_meta_features.npy')
        parts.append(pd.concat([m, pd.DataFrame(x, columns=NAMES)], axis=1))
    if band == '2.4GHz':
        m = pd.read_csv(CACHE / 'bgA_meta_features_meta.csv', encoding='utf-8')
        x = np.load(CACHE / 'bgA_meta_features.npy')
        parts.append(pd.concat([m, pd.DataFrame(x, columns=NAMES)], axis=1))
    df = pd.concat(parts, ignore_index=True)
    df = df[df.keep.astype(str) == 'True'].copy()
    df['snr_proxy_db'] = df.ctl_abs_power_db - df.ctl_floor_db  # 片段内"平均功率/底噪"，与接收机增益无关
    g = np.where(df.dataset == 'dataA_mirror', 'bgA',
        np.where((df.dataset == 'v3') & (df.label == 'background') & df.group_id.str.contains('20260710'), 'v3bg_0710',
        np.where((df.dataset == 'v3') & (df.label == 'background'), 'v3bg_other',
        np.where(df.dataset == 'data', 'A_drone', 'B_drone'))))
    df['grp'] = g
    return df


def sym_auc(y, v):
    ok = np.isfinite(v)
    if len(np.unique(y[ok])) < 2: return np.nan
    a = roc_auc_score(y[ok], v[ok]); return max(a, 1 - a)


KEY = ['shape_factor', 'log_kurt_a', 'papr999_db', 'c42n', 'log_gamma_max', 'sigma_f', 'spec_flatness',
       'c40n_abs', 'c41n_abs', 'occ_frac', 'duty', 'log1p_n_bursts', 'c20n_abs', 'snr_proxy_db']


def stats(df):
    rows = []
    for (grp, gid), d in df.groupby(['grp', 'group_id']):
        if grp not in ('bgA', 'v3bg_0710', 'v3bg_other'): continue
        r = dict(grp=grp, session=gid, n=len(d), split=d.split.iloc[0])
        for k in KEY: r[k] = d[k].median()
        rows.append(r)
    for grp, d in df.groupby('grp'):
        r = dict(grp=grp, session='__ALL__', n=len(d), split='all')
        for k in KEY: r[k] = d[k].median()
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(OUT / 'session_medians.csv', index=False, encoding='utf-8-sig')
    # (a) AUC：全部 / 只用"未检出突发"的片段（duty==0）
    a = df[df.grp.isin(['bgA', 'v3bg_0710'])]
    y = (a.grp == 'v3bg_0710').values.astype(int)
    q = a[(a.duty == 0)]
    yq = (q.grp == 'v3bg_0710').values.astype(int)
    rr = []
    for k in KEY:
        rr.append(dict(feature=k, auc_all=sym_auc(y, a[k].values), n0=int((y == 0).sum()), n1=int(y.sum()),
                       auc_duty0=sym_auc(yq, q[k].values), n0_duty0=int((yq == 0).sum()), n1_duty0=int(yq.sum()),
                       med_bgA=a[k][y == 0].median(), med_v3=a[k][y == 1].median(),
                       med_bgA_duty0=q[k][yq == 0].median(), med_v3_duty0=q[k][yq == 1].median()))
    pd.DataFrame(rr).to_csv(OUT / 'auc_a_by_content.csv', index=False, encoding='utf-8-sig')
    return t, pd.DataFrame(rr)


FS, FC = 61.44e6, 2430.0
FREQ = FC + (np.arange(1024) - 512) * FS / 1024 / 1e6


def spec(r):
    z = np.array(np.load(r.block_file, mmap_mode='r')[int(r.start):int(r.start) + int(r.n_samples)], dtype=np.complex128)
    z = z - z.mean(); z = z / np.sqrt(np.mean(np.abs(z) ** 2))  # 与 B1 相同的去均值+功率归一化
    fr = z[:len(z) // 1024 * 1024].reshape(-1, 1024) * np.hanning(1024)
    p = np.fft.fftshift(np.abs(np.fft.fft(fr, axis=1)) ** 2, axes=1)
    p = p[:len(p) // 12 * 12].reshape(-1, 12, 1024).mean(1)  # 100 行 × 0.2 ms
    return 10 * np.log10(p + 1e-30)


def figures(df):
    pb = (FREQ >= 2405) & (FREQ <= 2455)
    rng = np.random.default_rng(2026)
    v3 = df[df.grp == 'v3bg_0710']; sess = sorted(v3.group_id.unique())
    bga = df[df.grp == 'bgA']; bsel = bga.iloc[rng.choice(len(bga), 4, replace=False)]
    oth = df[df.grp == 'v3bg_other']; osel = [oth[oth.group_id == g].iloc[10] for g in sorted(oth.group_id.unique())]
    # 1) 时频图：每个 07-10 会话 1 片（第 10 片）+ bgA 4 片 + 其他 v3 背景
    items = [('bgA ' + r.group_id.split(':')[1][:22], r) for _, r in bsel.iterrows()]
    items += [('v3 ' + g.split(':')[1].replace('_ambient', ''), v3[v3.group_id == g].iloc[10]) for g in sess]
    items += [('v3 ' + r.group_id.split(':')[1][:24], r) for r in osel]
    nc = 6; nr = int(np.ceil(len(items) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.6 * nr), sharex=True)
    for ax, (t, r) in zip(axes.ravel(), items):
        s = spec(r); s -= np.median(s[:, pb])
        ax.imshow(s, aspect='auto', origin='lower', extent=[FREQ[0], FREQ[-1], 0, 20], vmin=-3, vmax=25, cmap='viridis')
        ax.set_title(f'{t}\nshape={r.shape_factor:.2f} 功率/底噪={r.snr_proxy_db:.1f}dB', fontsize=7)
        for f in (2412, 2437): ax.axvline(f, color='w', lw=.5, ls=':')
    for ax in axes.ravel()[len(items):]: ax.axis('off')
    for ax in axes[-1]: ax.set_xlabel('MHz', fontsize=7)
    for ax in axes[:, 0]: ax.set_ylabel('ms', fontsize=7)
    fig.suptitle('图16a (a) 组两边到底是什么：bgA（接收机 A 镜像背景） vs v3 07-10 "实验室背景"，每片 20 ms，色标 = 相对片段通带中位数 dB（−3…25）；白虚线 = WiFi 信道 1/6 中心', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / '图16a_a组两边的时频图.png', dpi=150); plt.close(fig)
    # 2) 平均功率谱：每个会话 20 片平均
    fig, ax = plt.subplots(figsize=(12, 5))
    rows = []
    for g in sess:
        ss = [spec(r).mean(0) for _, r in v3[v3.group_id == g].iterrows()]
        m = 10 * np.log10(np.mean([10 ** (s / 10) for s in ss], 0)); m -= np.median(m[pb])
        ax.plot(FREQ, m, lw=.8, label='v3 ' + g.split(':')[1].replace('_ambient', ''))
        rows.append(dict(session=g, wifi_ch1_excess_db=float(np.median(m[(FREQ > 2405) & (FREQ < 2422)])), max_excess_db=float(m[pb].max())))
    ss = [spec(r).mean(0) for _, r in bga.iloc[rng.choice(len(bga), 40, replace=False)].iterrows()]
    m = 10 * np.log10(np.mean([10 ** (s / 10) for s in ss], 0)); m -= np.median(m[pb])
    ax.plot(FREQ, m, 'k', lw=2, label='bgA（40 片平均）')
    rows.append(dict(session='bgA', wifi_ch1_excess_db=float(np.median(m[(FREQ > 2405) & (FREQ < 2422)])), max_excess_db=float(m[pb].max())))
    ax.axvspan(FREQ[0], 2405, color='.9'); ax.axvspan(2455, FREQ[-1], color='.9')
    ax.set(xlabel='MHz', ylabel='相对通带中位数 (dB)', title='图16b 平均功率谱（功率归一化后、每会话 20 片平均）；灰色 = 通带外')
    ax.legend(fontsize=7, ncol=3); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / '图16b_a组平均功率谱.png', dpi=150); plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / 'psd_summary.csv', index=False, encoding='utf-8-sig')
    # 3) 第 2 点：包络统计量 vs 片段内功率/底噪
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    col = {'bgA': 'k', 'v3bg_0710': 'tab:orange', 'v3bg_other': 'tab:red', 'A_drone': 'tab:blue', 'B_drone': 'tab:green'}
    lab = {'bgA': '接收机A 背景(bgA)', 'v3bg_0710': '接收机B 背景 07-10', 'v3bg_other': '接收机B 背景 其他', 'A_drone': '接收机A 无人机(data)', 'B_drone': '接收机B 无人机(v3)'}
    for ax, k in zip(axes, ['shape_factor', 'c42n', 'spec_flatness']):
        for gname in ['A_drone', 'bgA', 'B_drone', 'v3bg_0710', 'v3bg_other']:
            d = df[df.grp == gname]; d = d.iloc[rng.choice(len(d), min(len(d), 800), replace=False)]
            ax.scatter(d.snr_proxy_db, d[k], s=4, alpha=.5, c=col[gname], label=lab[gname])
        ax.set(xlabel='片段内 平均功率/底噪 (dB)（底噪=0.1ms帧功率 p10）', ylabel=k); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, markerscale=3)
    fig.suptitle('图16c 功率归一化后，包络/谱形特征仍随"片段内信号高出底噪多少"单调变化（2.4G，全部 split，keep=True，每组最多抽 800 点）', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / '图16c_特征随片段内信噪比变化.png', dpi=150); plt.close(fig)


def matched_snr(df):
    # 同为无人机：接收机 A(data) vs B(v3)，在片段内信噪比相近的分箱里比较
    d = df[df.grp.isin(['A_drone', 'B_drone'])].copy(); y = (d.grp == 'B_drone').astype(int).values
    bins = [-1, 3, 6, 9, 12, 15, 18, 21, 24, 30, 60]; d['bin'] = pd.cut(d.snr_proxy_db, bins)
    rr = []
    for k in ['shape_factor', 'c42n', 'papr999_db', 'log_gamma_max', 'spec_flatness', 'duty', 'log1p_n_bursts', 'occ_frac', 'snr_proxy_db']:
        r = dict(feature=k, pooled=sym_auc(y, d[k].values))
        for b, q in d.groupby('bin', observed=True):
            yy = (q.grp == 'B_drone').astype(int).values
            if yy.sum() >= 10 and (1 - yy).sum() >= 10: r[str(b)] = round(sym_auc(yy, q[k].values), 3)
        rr.append(r)
    cnt = d.groupby(['bin', 'grp'], observed=True).size().unstack(fill_value=0)
    t = pd.DataFrame(rr); t.to_csv(OUT / 'drone_A_vs_B_matched_snr_auc.csv', index=False, encoding='utf-8-sig')
    cnt.to_csv(OUT / 'drone_snr_bin_counts.csv', encoding='utf-8-sig')
    return t, cnt


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    df = load()
    if len(sys.argv) > 1 and sys.argv[1] == 'fig':
        figures(df); t, c = matched_snr(df); print(c.to_string()); print(t.round(3).to_string())
        print(pd.read_csv(OUT / 'psd_summary.csv').round(2).to_string()); sys.exit()
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30); pd.set_option('display.max_rows', 200)
    t, r = stats(df)
    print(t.round(4).to_string())
    print(r.round(4).to_string())
    print('Gaussian theory', GAUSS)
