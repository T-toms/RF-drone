import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from support import *
from preflight import hashes
from evaluate import cp
import difflib
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
def run():
    rr=rows(OUT/'metrics_all.csv');mix=rows(OUT/'mix_results.csv');groups=collections.defaultdict(dict)
    for r in rr:groups[tuple(r[k] for k in ['category','group','mode'])].setdefault(r['round'],[]).append(r)
    paired=[];lines=['# 第一轮 H 与第二轮 J 并列表','每项顺序为种子 2026、2027、2028；标准差 ddof=1；虚警逐种子给出 k/N、95% 单侧 Clopper–Pearson 上界。bgA val 参与了模型选择。','|类别/分组|判决|第一轮 H|第二轮 J|','|---|---|---|---|']
    def textcell(rs):
        vals=[float(r['rate']) for r in rs];fa=rs[0]['fa']=='True'
        return '; '.join(f"{r['seed']}: {r['k']}/{r['n']}={float(r['rate'])*100:.3f}%"+(f" [U95={float(r['cp95_upper'])*100:.3f}%]" if fa else '') for r in rs)+f"; 均值±SD={np.mean(vals)*100:.3f}±{np.std(vals,ddof=1)*100:.3f}%"
    for (cat,g,mode),rounds in groups.items():
        h=sorted(rounds['H'],key=lambda r:r['seed']);j=sorted(rounds['J'],key=lambda r:r['seed']);z=dict(category=cat,group=g,mode=mode,n=h[0]['n'])
        for tag,rs in [('H',h),('J',j)]:
            for r in rs:
                for key in ['k','rate','cp95_upper']:z[f'{tag}_{r["seed"]}_{key}']=r[key]
            z[tag+'_mean']=float(np.mean([float(r['rate']) for r in rs]));z[tag+'_std']=float(np.std([float(r['rate']) for r in rs],ddof=1))
        paired.append(z);lines.append(f'|{cat}: {g.replace("|","/")}|{mode}|{textcell(h)}|{textcell(j)}|')
    # CSV union keys for FA and recall alike.
    writecsv(OUT/'metrics_paired.csv',paired);(OUT/'metrics_paired.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for cat in sorted(set(z['category'] for z in paired)):writecsv(OUT/(cat+'_paired.csv'),[z for z in paired if z['category']==cat])
    def summary(cat,g,mode,tag):
        z=next(z for z in paired if z['category']==cat and z['group']==g and z['mode']==mode);return z[tag+'_mean'],z[tag+'_std']
    for mode in ['argmax','thr_fa1']:
        fig,axs=plt.subplots(1,2,figsize=(16,6))
        for ax,labels,queries in [(axs[0],['v3测试背景','接收机A测试背景','013真实安静','白噪声','陷波白噪声'],[('main',g) for g in ['v3_test_FA','bgA_test_FA','quiet013_FA','white_noise_FA','notched_noise_FA']]),(axs[1],['data无人机（非G）','v3无人机','G: EVO Lite 6K','G: EVO II Pro V3'],[('main','data_nonG_recall'),('main','v3_drone_recall'),('G_recall','evo lite 6k'),('G_recall','evo ii pro v3')])]:
            for tag,off,label in [('H',-.2,'第一轮'),('J',.2,'加入接收机A背景')]:
                v=[summary(cat,g,mode,tag) for cat,g in queries];ax.bar(np.arange(len(labels))+off,[a*100 for a,b in v],width=.38,yerr=[b*100 for a,b in v],capsize=3,label=label)
            ax.set_xticks(range(len(labels)),labels,rotation=20,ha='right');ax.set_ylim(0,110);ax.set_ylabel('虚警率（%）' if ax==axs[0] else '召回率（%）');ax.legend()
        fig.suptitle('加入接收机A背景前后：2.4 GHz，三种子均值±标准差；'+mode);fig.tight_layout();fig.savefig(OUT/('图12_加入接收机A背景前后_2.4GHz.png' if mode=='argmax' else '图12_thr_fa1_2.4GHz.png'),dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5));ms=[]
    for ax,mode in zip(axs,['argmax','thr_fa1']):
        for gi,group in enumerate(['data_drone','v3_drone','v3_background']):
            for tag in ['H','J']:
                means=[];stds=[]
                for x in [-10,-5,0,5,10,15,20]:
                    r=[r for r in mix if r['round']==tag and r['mode']==mode and r['group']==group and int(r['x_db'])==x];v=[float(r['rate']) for r in r];means.append(np.mean(v));stds.append(np.std(v,ddof=1));ms.append(dict(round=tag,mode=mode,group=group,x_db=x,n=r[0]['n'],rate_mean=means[-1],rate_std=stds[-1]))
                ax.errorbar([-10,-5,0,5,10,15,20],means,yerr=stds,ls='--' if tag=='H' else '-',color=f'C{gi}',marker='o' if tag=='J' else None,label=tag+' '+group)
        ax.set(xlabel='叠加分量/背景功率（dB）',ylabel='检出率 / 虚警率',title=mode,ylim=(-.03,1.03));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'fig3_mix_test_2.4GHz.png',dpi=160);plt.close(fig);writecsv(OUT/'mix_summary.csv',ms)
    fig,axs=plt.subplots(1,4,figsize=(18,4))
    for s in SEEDS:
        log=rows(OUT/f'train_log_2.4GHz_seed{s}.csv')
        for ax,key in zip(axs,['val_fa_v3','val_fa_bgA','val_miss','val_loss']):
            ax.plot([int(r['epoch']) for r in log],[float(r[key]) for r in log],label=str(s));ax.set(xlabel='Epoch',ylabel=key);ax.legend()
        axs[-1].plot([int(r['epoch']) for r in log],[float(r['train_loss']) for r in log],ls=':',label=str(s)+' train')
    axs[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'fig5_curves_2.4GHz.png',dpi=150);plt.close(fig)
    before=load(OUT/'protected_sha256_before.json');after=hashes();dump(OUT/'protected_sha256_after.json',after);assert before==after,'protected file changes'
    dump(OUT/'integrity.json',dict(protected_file_count=len(after),identical=True))
    oldsrc=(ROOT/'work/train_stage1_L20_20261108/train.py').read_text(encoding='utf-8');newsrc=(OUT/'train.py').read_text(encoding='utf-8');(OUT/'train_vs_H.diff').write_text(''.join(difflib.unified_diff(oldsrc.splitlines(True),newsrc.splitlines(True),fromfile='TaskH/train.py',tofile='TaskJ/train.py')),encoding='utf-8')
    cks=[torch.load(DEST/f'2.4GHz_seed{s}_best.pt',map_location='cpu',weights_only=False) for s in SEEDS];dones=[load(DEST/f'2.4GHz_seed{s}_done.json') for s in SEEDS];checks=load(OUT/'preflight.json');smoke=load(DEST/'2.4GHz_seed2026_smoke_done.json')
    report=['# 任务J：2.4 GHz 加入接收机A背景训练（2026-11-09）',
    '设置：沿用 H 的 ResNet18(1,2)、AdamW(lr=1e-3, wd=1e-4)、cosine(T_max=40, eta_min=1e-5)、40 epoch、batch64、AMP bfloat16、seed 2026/2027/2028、原 seed()、cudnn.benchmark=True；无增强/加噪/调参。只训练2.4 GHz。',
    f"训练计数 N_d/N_v/N_a={cks[0]['train_counts']}；缩放后逐样本权重（无人机/v3背景/bgA）={cks[0]['sample_weights']}；原始为1/N_d、0.5/N_v、0.5/N_a，乘总样本数/2使均值=1。",
    'train每batch损失=Σ(w·CE)/Σw；val沿用train的权重并按全val加权分母归一；背景总权重中v3/bgA各占一半。',
    'H train.py复制修改：[完整diff](train_vs_H.diff)。变更仅训练/验证数据拼接、样本权重、选模FA_v3+FA_bgA+Miss、双背景阈值取大、输出路径；附加断点保存optimizer/scheduler/scaler/全部RNG和Windows资源计数，不改变训练参数。',
    f"开工检查：keep=True的train/val/test=2861/990/1287（排除492行）；split不一致0，跨split重叠0；9片缓存重算最大差0；H2026对bgA test前200片最大预测差={checks['prediction_max_error']:.9f}<1e-3。",
    f'保护目录F/H/I共{len(after)}文件，前后SHA256完全一致：[检查](integrity.json)，[前](protected_sha256_before.json)/[后](protected_sha256_after.json)。既有缓存只读，不使用common.model()，checkpoint与done在E:/drone_lab_stage1_L20_bgA_20261109/。',
    '|seed|best epoch|val FA_v3|val FA_bgA|val Miss|thr_v3|thr_bgA|thr_fa1|','|---|---|---|---|---|---|---|---|']
    for s,c in zip(SEEDS,cks):
        kv=int(round(c['val_fa_v3']*80));ka=int(round(c['val_fa_bgA']*990))
        report.append(f"|{s}|{c['epoch']}|{kv}/80={c['val_fa_v3']:.6f},U95={cp(kv,80):.6f}|{ka}/990={c['val_fa_bgA']:.6f},U95={cp(ka,990):.6f}|{c['val_miss']:.6f}|{c['thr_v3']:.9f}|{c['thr_bgA']:.9f}|{c['thr_fa1']:.9f}|")
    report+=['val背景片数v3=80、bgA=990；阈值各按nextafter(sorted_bg[-1-floor(.01*N)],+inf)，最终取max；bgA val参与选模，其FA不是独立测试结果。',
    '全部逐种子并列表：[metrics_paired.md](metrics_paired.md) / [CSV](metrics_paired.csv)（虚警每项含k/N、单侧95% CP上界；均值±SD使用ddof=1）。以下为主表，单位%。',
    '|判决/指标|第一轮H 均值±SD|第二轮J 均值±SD|','|---|---|---|']
    for mode in ['argmax','thr_fa1']:
        for g in ['v3_test_FA','bgA_test_FA','bgA_val_FA','quiet013_FA','white_noise_FA','notched_noise_FA','nonG_miss','all_drone_miss_with_G','data_nonG_recall','v3_drone_recall']:
            h,hs=summary('main',g,mode,'H');j,js=summary('main',g,mode,'J')
            extra=[]
            for tag in ['H','J']:
                rs=[r for r in rr if r['category']=='main' and r['group']==g and r['mode']==mode and r['round']==tag]
                extra.append('；'+', '.join(f"{r['seed']}:{r['k']}/{r['n']},U95={float(r['cp95_upper'])*100:.3f}%" for r in rs) if rs[0]['fa']=='True' else '')
            report.append(f'|{mode}/{g}|{100*h:.3f}±{100*hs:.3f}{extra[0]}|{100*j:.3f}±{100*js:.3f}{extra[1]}|')
    report+=['主漏检不含G（data evo lite 6k、data evo ii pro v3）；G各项单列：[G召回](G_recall_paired.csv)。含G原口径已单列。',
    '分表：[v3背景会话](fa_by_session_paired.csv)、[bgA test源型号/状态/带宽](bgA_test_by_source_paired.csv)、[bgA val源分组](bgA_val_by_source_paired.csv)、[无人机型号/状态/带宽漏检](miss_by_model_state_bw_paired.csv)。',
    '来源与同型号：[data/v3召回](source_recall_paired.csv)、[Mini5及EVOII按来源/带宽召回](same_model_recall_paired.csv)；[keep筛除测试片段](rejected_paired.csv)与[分项](rejected_detail_paired.csv)。',
    '013共19片：[切片与split](quiet013_metadata.csv)、[按group/split虚警](quiet013_by_group_split_paired.csv)。train组的013片段曾在训练中以“无人机”标签出现，不能当作独立测试背景。',
    '纯白噪声80片：numpy default_rng(20261109)，复高斯real+i·imag，经common.spec()；陷波白噪声80片读取协调者输入，不重做FIR。',
    '叠加检验只读取H三组mix缓存；[逐种子含背景k/N、CP上界](mix_results.csv)、[均值±SD](mix_summary.csv)。',
    '![图12](图12_加入接收机A背景前后_2.4GHz.png)',
    '![叠加检验（H虚线/J实线）](fig3_mix_test_2.4GHz.png)',
    '另附[阈值图12](图12_thr_fa1_2.4GHz.png)、[三种子训练曲线](fig5_curves_2.4GHz.png)。',
    'test_predictions_2.4GHz_seed<s>.csv与bgA_test_predictions_seed<s>.csv均含逐片预测；H_前缀是第一轮重推对照；原TaskH/I文件未改。',
    '5.8 GHz没有空闲频带/bgA，未训练，结果沿用任务H：[原5.8GHz结果](../train_stage1_L20_20261108/main_metrics.csv)。']
    peak=max([d['peak_host_bytes'] for d in dones]+[json.loads((OUT/'memory_monitor.json').read_text(encoding='utf-8-sig'))['peak_training_process_bytes']])
    report+=[f"冒烟3epoch耗时{smoke['seconds']:.2f}s；正式三次耗时分别={[round(d['seconds'],2) for d in dones]}s，总计{sum(d['seconds'] for d in dones):.2f}s；评估{load(OUT/'evaluation_done.json')['seconds']:.2f}s；GPU张量分配峰值{max(d['peak_gpu_bytes'] for d in dones)/1024**3:.3f}GiB；训练进程WorkingSet峰值{peak/1024**3:.3f}GiB。",
    '小事项：Windows资源计数原接口返回0，修为64位句柄并补充外部峰值监测；2026及冒烟done内0不作为真实峰值。报告生成修复了PowerShell JSON的BOM读取，并恢复J目录导入优先级，避免验收误导入H的同名模块。前200预测通过1e-3门限（差异原因未单独验证）。bgA源分组从F完整index映射：306片val/test的配对无人机不在H的keep缓存；bgA自身严格keep=True。全程float16缓存，无整库float32复制。',
    '断点：last.pt含优化器、调度器、scaler、best及全部RNG状态；重跑run_all.py可续。冒烟独立从头，正式40epoch重新按同种子开始。',
    '**事实陈述，不下结论**（argmax，三种子均值±SD，%）：']
    for g in ['bgA_test_FA','v3_test_FA','nonG_miss','v3_drone_recall','notched_noise_FA']:
        h,hs=summary('main',g,'argmax','H');j,js=summary('main',g,'argmax','J');report.append(f'{g}：{h*100:.3f}±{hs*100:.3f} → {j*100:.3f}±{js*100:.3f}。')
    assert len(report)<=90,len(report);(OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8');dump(OUT/'finished.json',dict(report_lines=len(report),protected_files=len(after)))
if __name__=='__main__':run()
