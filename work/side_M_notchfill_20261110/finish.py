import sys
sys.dont_write_bytecode=True
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from support import *
from evaluate import cp
from preflight import hashes
import fill,difflib
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
TAGS=['Lb','M'];XS=[-10,-5,0,5,10,15,20]
def run():
    import shutil
    for tag in TAGS:
        for s in SEEDS:shutil.copyfile(OUT/f'C1_predictions_{tag}_seed{s}.csv',OUT/f'{tag}_C1_predictions_seed{s}.csv')
    rr=rows(OUT/'metrics_all.csv');mix=rows(OUT/'mix_results.csv');groups=collections.defaultdict(dict)
    for r in rr:groups[tuple(r[k] for k in ['category','group','mode'])].setdefault(r['round'],[]).append(r)
    paired=[];lines=['# L-b / M 逐种子并列表','','单位为比例；SD使用ddof=1；虚警含k/N和95%单侧Clopper–Pearson上界。','','|类别/分组|判决|L-b|M|','|---|---|---|---|']
    def cell(rs):
        vals=[float(r['rate']) for r in rs]
        return '; '.join(f"{r['seed']}:{r['k']}/{r['n']}={float(r['rate'])*100:.3f}%"+(f" [U95={float(r['cp95_upper'])*100:.3f}%]" if r['fa']=='True' else '') for r in rs)+f'; 均值±SD={np.mean(vals)*100:.3f}±{np.std(vals,ddof=1)*100:.3f}%'
    for (cat,g,mode),rounds in groups.items():
        assert set(rounds)==set(TAGS),(cat,g,mode)
        z=dict(category=cat,group=g,mode=mode)
        for tag in TAGS:
            rs=sorted(rounds[tag],key=lambda r:r['seed']);assert len(rs)==3
            for r in rs:
                for k in ['n','k','rate','cp95_upper']:z[f'{tag}_{r["seed"]}_{k}']=r[k]
            vals=[float(r['rate']) for r in rs];z[tag+'_mean']=float(np.mean(vals));z[tag+'_std']=float(np.std(vals,ddof=1))
        paired.append(z);lines.append('|'+cat+': '+g.replace('|','/')+'|'+mode+'|'+'|'.join(cell(sorted(rounds[tag],key=lambda r:r['seed'])) for tag in TAGS)+'|')
    writecsv(OUT/'metrics_paired.csv',paired);(OUT/'metrics_paired.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for cat in sorted({r['category'] for r in paired}):writecsv(OUT/(cat+'_paired.csv'),[r for r in paired if r['category']==cat])
    writecsv(OUT/'mix_data_model_paired.csv',[dict(x_db=int(r['category'].replace('mix_data_model_','')),**{k:v for k,v in r.items() if k!='category'}) for r in paired if r['category'].startswith('mix_data_model_')])
    def summary(cat,g,mode,tag):
        z=next(z for z in paired if z['category']==cat and z['group']==g and z['mode']==mode);return z[tag+'_mean'],z[tag+'_std']
    def bars(ax,queries,labels,mode):
        for j,tag in enumerate(TAGS):
            v=[summary(c,g,mode,tag) for c,g in queries];ax.bar(np.arange(len(v))+(j-.5)*.36,[100*a for a,b in v],width=.35,yerr=[100*b for a,b in v],capsize=3,label='L-b' if tag=='Lb' else 'M')
        ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.set_ylim(0,110);ax.set_ylabel('比例（%）');ax.legend()
    qleft=[('main',g) for g in ['v3_test_FA','bgA_test_FA','C1_FA','quiet013_FA','white_noise_FA','notched_noise_FA']]
    qright=[('main','data_nonG_recall'),('main','v3_drone_recall'),('v3_model_recall','mavic2'),('v3_model_recall','mini3pro')]+[('G_recall',r['group']) for r in paired if r['category']=='G_recall' and r['mode']=='argmax']
    for mode in ['argmax','thr_fa1']:
        fig=plt.figure(figsize=(20,12));gs=fig.add_gridspec(2,4,height_ratios=[1,1.2])
        bgrows_test=bgrows('test');first=bgrows_test[0][0]
        # Same exact IQ segment before/after; quiet background chosen deterministically.
        from preflight import iq
        bgz=iq(bgrows_test[0][1]);nn=np.load(ROOT/'work/review_20261109/notched_noise_2.4GHz.npy',mmap_mode='r')[0]
        for j,(name,z) in enumerate([('bgA纯噪声',bgz),('陷波白噪声',np.array(nn))]):
            for k,(tag,im) in enumerate([('修补前',spec_original(z,'2.4GHz')),('修补后',spec_fill(z))]):
                ax=fig.add_subplot(gs[0,j*2+k]);image(ax,im,'2.4GHz',name+'：'+tag);ax.set_ylim(2420,2455);ax.set_xlabel('时间（毫秒）');ax.set_ylabel('频率（MHz）')
        ax=fig.add_subplot(gs[1,:2]);bars(ax,qleft,['v3测试背景','bgA测试','C1七档合并','013安静','白噪声','陷波白噪声'],mode);ax.set_title('虚警')
        ax=fig.add_subplot(gs[1,2:]);bars(ax,qright,['data非G','v3无人机','mavic2','mini3pro']+[g for c,g in qright[4:]],mode);ax.set_title('召回')
        fig.suptitle('陷波凹坑修补前后：三种子均值±标准差；'+mode);fig.tight_layout();fig.savefig(OUT/('图15_陷波修补前后_2.4GHz.png' if mode=='argmax' else '图15_thr_fa1_2.4GHz.png'),dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,5));ms=[]
    for ax,mode in zip(axs,['argmax','thr_fa1']):
        for gi,group in enumerate(['data_drone','v3_drone','v3_background']):
            for ti,tag in enumerate(TAGS):
                means=[];stds=[]
                for x in XS:
                    rs=[r for r in mix if r['round']==tag and r['mode']==mode and r['group']==group and int(r['x_db'])==x];assert len(rs)==3
                    v=[float(r['rate']) for r in rs];means.append(np.mean(v));stds.append(np.std(v,ddof=1));ms.append(dict(round=tag,mode=mode,group=group,x_db=x,rate_mean=means[-1],rate_std=stds[-1]))
                ax.errorbar(XS,means,yerr=stds,ls='--' if tag=='Lb' else '-',marker='o' if ti else 's',color=f'C{gi}',label=tag+' '+group)
        ax.axvspan(-5,15,color='gray',alpha=.1);ax.set(xlabel='功率比X（dB）；灰区=训练范围',ylabel='检出率 / 虚警率',title=mode,ylim=(-.03,1.03));ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'fig3_mix_test_2.4GHz.png',dpi=150);plt.close(fig);writecsv(OUT/'mix_summary.csv',ms)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,mode in zip(axs,['argmax','thr_fa1']):
        for tag in TAGS:
            v=[summary('C1',str(x),mode,tag) for x in XS];ax.errorbar(XS,[a for a,b in v],yerr=[b for a,b in v],marker='o',label=tag)
        ax.set(xlabel='功率比X（dB）',ylabel='背景判为无人机比例',title=mode,ylim=(-.03,1.03));ax.legend()
    fig.tight_layout();fig.savefig(OUT/'fig_C1_2.4GHz.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(23,4))
    for s in SEEDS:
        rs=rows(OUT/f'train_log_Lb_seed{s}.csv');assert len(rs)==40
        for ax,key in zip(axs,['train_loss','val_loss','val_fa_v3','val_fa_bgA','val_miss']):ax.plot([int(r['epoch']) for r in rs],[float(r[key]) for r in rs],label=f'M/{s}');ax.set(xlabel='Epoch',ylabel=key)
    axs[-1].legend();fig.tight_layout();fig.savefig(OUT/'fig5_curves_2.4GHz.png',dpi=150);plt.close(fig)
    diff=[]
    for fn in ['train.py','support.py']:
        diff+=list(difflib.unified_diff((L/fn).read_text(encoding='utf-8').splitlines(True),(OUT/fn).read_text(encoding='utf-8').splitlines(True),fromfile='L/'+fn,tofile='M/'+fn))
    (OUT/'vs_L.diff').write_text(''.join(diff),encoding='utf-8')
    before=load(OUT/'protected_before.json');after=hashes();dump(OUT/'protected_after.json',after)
    changed=[p for p in before if before.get(p)!=after.get(p)]
    added=[p for p in after if p not in before]
    allowed=all(p.startswith('work\\review_20261109\\') or p.startswith('work/review_20261109/') for p in added)
    dump(OUT/'integrity.json',dict(protected_files=len(after),baseline_files=len(before),changed=changed,external_added=added,baseline_identical=not changed));assert not changed and allowed,(changed,added)
    cks={s:torch.load(DEST/f'2.4GHz_Lb_seed{s}_best.pt',map_location='cpu',weights_only=False) for s in SEEDS};dones=[load(DEST/f'2.4GHz_Lb_seed{s}_done.json') for s in SEEDS]
    checks=load(OUT/'preflight.json');cross=rows(OUT/'cross_metrics.csv')
    cg=collections.defaultdict(list)
    for r in cross:cg[(r['model'],r['input'],r['mode'])].append(float(r['rate']))
    writecsv(OUT/'cross_summary.csv',[dict(model=t,input=n,mode=m,mean=float(np.mean(v)),std=float(np.std(v,ddof=1))) for (t,n,m),v in cg.items()])
    report=['# 任务M：2.4GHz陷波凹坑镜像修补后重训','',
    '完成：3epoch冒烟、三种子各40epoch、全部缓存重算及L-b/M并列评估、交叉对照和行压低实验。',
    '设置：ResNet18(1,2)，AdamW lr=1e-3/wd=1e-4，cosine T_max40/eta_min1e-5，batch64，AMP bfloat16，seed2026/2027/2028，cudnn.benchmark=True。',
    '与L-b差异：[diff](vs_L.diff)；train.py逐字相同（内部variant与checkpoint仍标Lb）；support只改输出/缓存路径及spec→spec_fill；IQ、配对、权重、阈值、选模、超参不变。',
    '原始频点：'+ '；'.join(f"{n}: "+', '.join(f'{f:.2f}' for f in fill.fk[idx])+' MHz' for n,idx in zip(fill.HOLES,fill.hole_idx))+'。',
    '镜像规则：左半复制左侧坑外对称点，右半复制右侧坑外对称点；池化前实际替换5+7+7=19个频点；后续流程相同。',
    '规格更正（协调者确认）：任务书文字16点有误，以pilot代码为准；闭区间实际含19点，沿用pilot全部频点，不改、不重训。',
    '背景更正（协调者确认）：任务K中陷波白噪声99.5%为任务J seed2026结果，不是L-b；L-b在任务L已有CSV中三种子均0%，本任务直接沿用。',
    f"开工计数={checks['counts']}（data无人机/v3无人机/v3背景/bgA train/val/test/叠加无人机/叠加背景）；叠加6片最大差={max(checks['mix_recompute'])}。",
    f"关闭修补6片逐像素完全一致；非陷波行回归float32最大残差={max(r['residual_float32'] for r in checks['consistency']):.9f}，float16={max(r['residual_float16'] for r in checks['consistency']):.9f}。",
    '异常：原float16≤0.01检查在H样本L20_00009388失败0.015210391；协调者确认量化舍入，授权float32≤0.001、float16≤0.02；补测全部通过。']
    for r in checks['row_stats']:report.append(f"修补行统计 {r['source']}/{r['freq']}MHz：均值差={r['mean_difference']:.6f}，标准差比={r['std_ratio']:.6f}（40片，均合格）。")
    report+=['GPU开始无其他Python计算进程；[开工检查](preflight.json)。所有IQ按block_file/start顺序mmap读取、按原行索引写回；bgA缓存保留原meta索引，仅keep行有效，另存keep_indices.json。',
    '两组训练pairs和C1 pairs逐字复制L文件，沿用g直接重建b+g*d；v3背景IQ只读预读；80片白噪声seed20261109；013按L同19片。',
    '|seed|best epoch|val FA_v3|val FA_bgA|val Miss|val loss|thr_fa1|','|---|---|---|---|---|---|---|']
    for s,c in cks.items():report.append(f"|{s}|{c['epoch']}|{c['val_fa_v3']:.6f}|{c['val_fa_bgA']:.6f}|{c['val_miss']:.6f}|{c['val_loss']:.6f}|{c['thr_fa1']:.9f}|")
    report+=['逐片预测以Lb_/M_开头；L-b全部标准指标和预测取L已有CSV，无重训；[逐种子并列表](metrics_paired.md) / [CSV](metrics_paired.csv)含argmax、thr_fa1、均值±SD、虚警k/N和95%单侧CP上界。',
    '[型号/状态/带宽漏检](miss_by_model_state_bw_paired.csv)、[含G](main_paired.csv)、[G单列](G_recall_paired.csv)、[v3型号召回](v3_model_recall_paired.csv)、[同型号](same_model_recall_paired.csv)、[来源召回](source_recall_paired.csv)。',
    '[v3会话虚警](fa_by_session_paired.csv)、[bgA test源分组](bgA_test_by_source_paired.csv)、[bgA val源分组](bgA_val_by_source_paired.csv)、[筛除片段](rejected_detail_paired.csv)、[013分组](quiet013_by_group_split_paired.csv)。',
    '[叠加逐种子](mix_results.csv)、[均值±SD](mix_summary.csv)、[按型号叠加](mix_data_model_paired.csv)、[C1](C1_paired.csv)；训练X[-5,15]dB，检验-10至20七档。',
    'C1合并1400项为200源片×7档，CP采用逐片二项式口径；013中train组曾作为无人机训练，非独立背景；val参与选模与定阈值。',
    '![图15](图15_陷波修补前后_2.4GHz.png)',
    '[阈值图15](图15_thr_fa1_2.4GHz.png)、[叠加检验图](fig3_mix_test_2.4GHz.png)、[C1图](fig_C1_2.4GHz.png)、[训练曲线](fig5_curves_2.4GHz.png)。',
    '[交叉逐种子k/N及CP](cross_metrics.csv)、[全部交叉均值±SD](cross_summary.csv)：L-b在修补后test/bgA/陷波白噪声；M在原bgA/原陷波白噪声；test背景FA与无人机召回分别列出。',
    '行压低实验：bgA test前200片，near=|池化行中心频率−陷波频率|≤0.25MHz；各处减2.0/5.2/6.9，不重标准化，行号见attention_rows.json；仅描述。']
    for tag,names in [('Lb',['filled_test','filled_bgA_test','filled_notched_noise']),('M',['old_bgA_test','old_notched_noise','attention_before','attention_depressed'])]:
        for name in names:
            cells=[]
            for mode in ['argmax','thr_fa1']:
                rs=[r for r in cross if r['model']==tag and r['input']==name and r['mode']==mode];v=[float(r['rate'])*100 for r in rs];cells.append(f'{mode}={np.mean(v):.3f}±{np.std(v,ddof=1):.3f}%')
            report.append(f"交叉/关注度 {tag}/{name}："+'；'.join(cells)+'。')
    disk=sum(p.stat().st_size for d in [OUT,DEST] for p in d.rglob('*') if p.is_file())
    report += [f"生成{load(OUT/'generation_done.json')['seconds']:.1f}s；3epoch冒烟{load(DEST/'2.4GHz_Lb_seed2026_smoke_done.json')['seconds']:.1f}s；三次训练秒={[round(d['seconds'],1) for d in dones]}；评估{load(OUT/'evaluation_done.json')['seconds']:.1f}s；交叉{load(OUT/'cross_done.json')['seconds']:.1f}s。",
    f"训练GPU张量峰值{max(d['peak_gpu_bytes'] for d in dones)/2**30:.3f}GiB；训练进程WorkingSet峰值{max(d['peak_host_bytes'] for d in dones)/2**30:.3f}GiB；生成进程中途读取的系统WorkingSet历史峰值{json.loads((OUT/'generation_host_peak.json').read_text(encoding='utf-8-sig'))['PeakWorkingSet64']/2**30:.3f}GiB（非全程监测）；总新占磁盘{disk/2**30:.3f}GiB。",
    f"保护目录SHA256：基线{len(before)}文件，已有文件差异0；协调者外部新增={added}（授权单列、不改基线）。无DSH、crop.py/pilot模块import、common.model调用；无安装/提交/改划分/5.8GHz训练。",
    '自行决定：冒烟独立运行，正式训练同种子从头；L训练脚本保留内部Lb标签，报告与推理输出统一为M；关闭pyc写入。',
    '**事实陈述，不下结论**（argmax，L-b → M，三种子均值±标准差，%）：']
    for cat,g in [('main','notched_noise_FA'),('main','white_noise_FA'),('main','bgA_test_FA'),('main','v3_test_FA'),('main','C1_FA'),('main','v3_drone_recall'),('v3_model_recall','mavic2'),('v3_model_recall','mini3pro'),('main','nonG_miss')]:report.append(g+'：'+' → '.join(f'{a*100:.3f}±{b*100:.3f}' for a,b in [summary(cat,g,'argmax',t) for t in TAGS])+'。')
    for tag,name in [('Lb','filled_test'),('Lb','filled_bgA_test'),('Lb','filled_notched_noise'),('M','old_notched_noise')]:
        v=[float(r['rate'])*100 for r in cross if r['model']==tag and r['input']==name and r['mode']=='argmax'];report.append(f'{tag}模型在{name}输入上的FA：{np.mean(v):.3f}±{np.std(v,ddof=1):.3f}%。')
    formatted=[]
    for i,line in enumerate(report):
        if i and (line.startswith('|')!=report[i-1].startswith('|') or line.startswith('![') or report[i-1].startswith('![') or line.startswith('**事实')):formatted.append('')
        formatted.append(line)
    report=formatted
    assert len(report)<=90,len(report)
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8');dump(OUT/'finished.json',dict(report_lines=len(report),protected_files=len(after),training_runs=3))
if __name__=='__main__':run()
