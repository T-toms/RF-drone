import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
import hashlib
def pct(x):
    x=float(x)
    return f'{100*x:.2f}' if np.isfinite(x) else 'N/A'
def ms(r,key):return pct(r[key+'_mean'])+'±'+pct(r[key+'_std'])
def table(headers,rs):return ['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(v) for v in r)+'|' for r in rs]
def run():
    main=rows(OUT/'main_metrics.csv');mean=rows(OUT/'main_metrics_summary.csv');src=rows(OUT/'recall_by_source_summary.csv');same=rows(OUT/'recall_same_model_summary.csv');mix=rows(OUT/'mix_summary.csv');cam=rows(OUT/'gradcam_summary.csv');strength=rows(OUT/'test_power_proxy.csv')
    # Full numeric tables remain local linked artifacts, with no interpretive text.
    for name in ['miss_by_model_state_bw','fa_by_session','recall_same_model','recall_by_source','rejected_test_recall','confusion_matrices','main_metrics','mix_results','gradcam_per_sample']:
        rr=rows(OUT/f'{name}.csv');headers=list(rr[0]);body=table(headers,[[r[k] for k in headers] for r in rr])
        if (OUT/f'{name}_summary.csv').exists():
            ar=rows(OUT/f'{name}_summary.csv');ah=list(ar[0]);body+=['','均值与样本标准差（原始比例，非百分数）：','']+table(ah,[[r[k] for k in ah] for r in ar])
        (OUT/f'{name}.md').write_text('\n'.join(body)+'\n',encoding='utf-8')
    lines=['# 一阶段 L20 第一轮训练数字报告','', '输入：原 index_L20.csv 的 split；各 split 仅 keep=True，background=0、drone=1。Hann1024/hop512，center=False、双边 FFT；先裁 50/90 MHz 通带，在功率域池化 256×256，再加中位功率×10⁻³ 转 dB、逐片标准化、float16 缓存。', '模型直接导入原 ResNet18(1,2)，随机初始化；种子 2026/2027/2028；40 epoch、batch64、AdamW(1e-3,1e-4)、cosine→1e-5、BF16；训练频率倒数权重均值=1，无增强/重采样。选 val(FA+Miss,loss) 字典序最小。以下率单位均为 %，± 为 3 种子的样本标准差。','', '**训练与 val（FA/Miss 为 argmax）**']
    train=[];resources=[]
    for b in BANDS:
        for s in SEEDS:
            c=torch.load(CACHE/f'{b}_seed{s}_best.pt',map_location='cpu',weights_only=False);d=load(OUT/f'{b}_seed{s}_done.json');resources.append(d);train.append([b,s,c['epoch'],pct(c['val_fa']),pct(c['val_miss']),f'{c["val_loss"]:.5f}',c['val_background_n'],f'{c["thr_fa1"]:.9g}'])
    lines+=table(['频段','种子','best epoch','val FA','val Miss','val loss','val背景N','thr_fa1'],train)
    lines+=['','**test 主指标**']
    rr=[]
    for b in BANDS:
        for s in SEEDS:
            a=next(r for r in main if r['band']==b and int(r['seed'])==s and r['mode']=='argmax');t=next(r for r in main if r['band']==b and int(r['seed'])==s and r['mode']=='thr_fa1');rr.append([b,s,pct(a['fa']),pct(a['miss']),pct(t['fa']),pct(t['miss'])])
        a=next(r for r in mean if r['band']==b and r['mode']=='argmax');t=next(r for r in mean if r['band']==b and r['mode']=='thr_fa1');rr.append([b,'均值±SD',ms(a,'fa'),ms(a,'miss'),ms(t,'fa'),ms(t,'miss')])
    lines+=table(['频段','种子','FA arg','Miss arg','FA thr','Miss thr'],rr)
    lines+=['test 背景 N=80/60；零错误时 95% 上界近似 3/N=3.75%/5.00%；此样本量不能验证 FA≤1%。val 背景 N=80/60，1% 容许错误数 floor(0.01N)=0；thr 取最大背景概率的下一个 float32 数，判断 p≥thr。', '分组漏检：[型号/状态/带宽表](miss_by_model_state_bw.md)；背景：[逐会话 FA 表](fa_by_session.md)；[混淆矩阵](confusion_matrices.md)；[被剔除 test 召回附表](rejected_test_recall.md)，均含 3 种子及两种判断。','', '**来源召回（均值±SD）**']
    source_rows=[]
    for b in BANDS:
        for ds in ['data','v3']:
            a=next(r for r in src if r['band']==b and r['dataset']==ds and r['mode']=='argmax');t=next(r for r in src if r['band']==b and r['dataset']==ds and r['mode']=='thr_fa1');source_rows.append([b,ds,a['n'],ms(a,'recall'),ms(t,'recall')])
    lines+=table(['频段','来源','N','recall arg','recall thr'],source_rows)
    lines+=['**同型号（按原 bw_label，v3 unknown 不映射为 data 带宽）**']
    sr=[]
    for a in same:
        if a['mode']!='argmax':continue
        t=next(r for r in same if r['mode']=='thr_fa1' and all(r[k]==a[k] for k in ['band','canonical_model','dataset','bw_label']));sr.append([a['band'],a['canonical_model'],a['dataset'],a['bw_label'],a['n'],ms(a,'recall'),ms(t,'recall')])
    lines+=table(['频段','型号','来源','bw','N','arg','thr'],sr)
    lines+=['5.8GHz 的 data EVO II Pro V3 N=0，v3 单列；所有逐片功率代理见 [CSV](test_power_proxy.csv)。**叠加检验（A=data drone+B；B=v3 drone+B；C=另一 v3 背景+B，C 报 FA）**']
    mr=[]
    for b in BANDS:
        for x in [-10,-5,0,5,10,15,20]:
            vals=[]
            for mode in ['argmax','thr_fa1']:
                for group in ['data_drone','v3_drone','v3_background']:
                    r=next(r for r in mix if r['band']==b and r['group']==group and r['mode']==mode and int(r['x_db'])==x);vals.append(ms(r,'detection_rate'))
            mr.append([b,x]+vals)
    lines+=table(['频段','X dB','A arg','B arg','C arg','A thr','B thr','C thr'],mr)
    lines+=['叠加 N、逐种子率见 [表](mix_results.md)，逐对样本与 g、实测功率比见 [CSV](mix_pairs.csv)。mean|d|² 含 A 的底噪；20 个 1ms 功率的 P90−P10 是代理，不是独立测得的 SNR。', '**Grad-CAM（seed2026、layer4、目标=预测类，逐片最大值归一化后平均；占比均值±片间SD）**']
    lines+=table(['频段','来源/类别','N','有效N','零热图N','区域能量占比'],[[r['band'],r['dataset']+'/'+r['label'],r['n'],r['valid_energy_n'],r['zero_heat_n'],pct(r['energy_fraction_mean'])+'±'+pct(r['energy_fraction_std'])] for r in cam])
    sizes=sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file());secs=sum(r['seconds'] for r in resources);peak=max(r['peak_gpu_bytes'] for r in resources)
    lines+= [f'耗时：6 次训练合计 {secs:.1f}s；缓存 {sum(load(p)["seconds"] for p in CACHE.glob("*_spec.done.json")):.1f}s；评估 {load(OUT/"evaluate_done.json")["seconds"]:.1f}s；叠加 {load(OUT/"mix_done.json")["seconds"]:.1f}s；Grad-CAM {load(OUT/"gradcam_done.json")["seconds"]:.1f}s。PyTorch 已分配显存峰值 {peak/2**30:.3f} GiB，E 缓存/checkpoint/派生数组合计 {sizes/2**30:.3f} GiB。', '修正/小事项：协调者答复 msg_38e19018543d 要求先按实际 fs 裁通带再池化；原索引/模型 SHA-256 验证见 verification.json。冒烟独立重启正式训练；-I 显式加入脚本目录。', '阈值以 float32 nextafter 严格越过 val 最大背景概率；标准差 ddof=1；cuDNN benchmark=True，未启用逐位确定性模式。背景配对按 test 缓存顺序循环，背景+背景错开 1 片；均只用 keep=True test；v3 未知带宽保留 unknown。', '5.8GHz 背景指定 WiFi 区间 2401–2423/2426–2448 MHz 与输入无交集，能量占比 N/A；零热图不除零。Grad-CAM 使用 BF16 前向/反向、float32 热图计算；双线性仅用于热图放大。']
    for b in BANDS:
        v=np.array([float(r['power_proxy_db']) for r in strength if r['band']==b and r['dataset']=='data' and r['label']=='drone' and r['keep']=='True']);lines.append(f'{b} data test 功率代理 P10/中位/P90={np.quantile(v,.1):.2f}/{np.median(v):.2f}/{np.quantile(v,.9):.2f} dB（N={len(v)}）。')
    lines+=['图：'+ '；'.join(f'[{b} 输入](fig0_input_examples_{b}.png) / [来源](fig2_recall_by_source_{b}.png) / [叠加](fig3_mix_test_{b}.png) / [叠加样本](fig3b_mix_examples_{b}.png)' for b in BANDS), '图：'+'；'.join(f'[{b} CAM](fig4_gradcam_{b}.png) / [均值 CAM](fig4b_gradcam_mean_{b}.png) / [训练](fig5_curves_{b}.png)' for b in BANDS), '**事实陈述，不下结论**', '来源：'+'；'.join('/'.join(map(str,r)) for r in source_rows), '叠加 X=10dB：'+'；'.join('/'.join(map(str,r)) for r in mr if r[1]==10), 'Grad-CAM：'+'；'.join(f'{r["band"]}/{r["dataset"]}/{r["label"]}={pct(r["energy_fraction_mean"])}%' for r in cam)]
    assert len(lines)<=90,len(lines)
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('report_lines',len(lines),flush=True)
if __name__=='__main__':run()

