import sys
sys.dont_write_bytecode=True
from pathlib import Path
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O))
from support import *
from evaluate import cp
from preflight import hashes
import shutil
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
TAGS=['J','La','Lb'];XS=[-10,-5,0,5,10,15,20]
def run():
    rr=rows(OUT/'metrics_all.csv');mix=rows(OUT/'mix_results.csv');groups=collections.defaultdict(dict)
    # Keep J's published aggregate mix numbers; newly inferred per-model results are separate.
    jm=[r for r in rows(ROOT/'work/train_stage1_L20_bgA_20261109/mix_results.csv') if r['round']=='J']
    mix=[r for r in mix if r['round']!='J']+jm;writecsv(OUT/'mix_results.csv',mix)
    for r in rr:groups[tuple(r[k] for k in ['category','group','mode'])].setdefault(r['round'],[]).append(r)
    paired=[];lines=['# J / L-a / L-b 逐种子并列表','各项为种子2026/2027/2028；均值±标准差ddof=1；虚警含k/N及95%单侧Clopper-Pearson上界。','|类别/分组|判决|J|L-a|L-b|','|---|---|---|---|---|']
    def cell(rs):
        vals=[float(r['rate']) for r in rs];fa=rs[0]['fa']=='True'
        return '; '.join(f"{r['seed']}:{r['k']}/{r['n']}={float(r['rate'])*100:.3f}%"+(f" [U95={float(r['cp95_upper'])*100:.3f}%]" if fa else '') for r in rs)+f'; 均值±SD={np.mean(vals)*100:.3f}±{np.std(vals,ddof=1)*100:.3f}%'
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
    permodel=[dict(x_db=int(r['category'].replace('mix_data_model_','')),**{k:v for k,v in r.items() if k!='category'}) for r in paired if r['category'].startswith('mix_data_model_')]
    writecsv(OUT/'mix_data_model_paired.csv',permodel)
    def summary(cat,g,mode,tag):
        z=next(z for z in paired if z['category']==cat and z['group']==g and z['mode']==mode);return z[tag+'_mean'],z[tag+'_std']
    # Frequency profile descriptors: cached spectra only, no filtering.
    freq=2405+(np.arange(256)+.5)*50/256;wifi=[]
    for split in ['train','val','test']:
        rs=rows(OLD/f'2.4GHz_{split}_spec_meta.csv');a=np.load(OLD/f'2.4GHz_{split}_spec.npy',mmap_mode='r');wg=collections.defaultdict(list)
        for i,r in enumerate(rs):
            if r['dataset']=='v3' and r['label']=='background':
                p=np.median(a[i].astype(np.float32),axis=1);ref=p[(freq>=2451)&(freq<=2454)].mean()
                wg[r['group_id']].append([p[(freq>=2408)&(freq<=2416)].mean()-ref,p[(freq>=2433)&(freq<=2441)].mean()-ref])
        for g,v in wg.items():wifi.append(dict(split=split,group_id=g,n=len(v),ch1_minus_ref=float(np.median(np.array(v)[:,0])),ch6_minus_ref=float(np.median(np.array(v)[:,1]))))
    writecsv(OUT/'wifi_group_descriptors.csv',wifi)
    for mode in ['argmax','thr_fa1']:
        fig,axs=plt.subplots(1,2,figsize=(24,7))
        qleft=[('main',g) for g in ['v3_test_FA','bgA_test_FA','C1_FA','quiet013_FA','notched_noise_FA']]
        qright=[('main','data_nonG_recall'),('main','v3_drone_recall'),('v3_model_recall','mavic2'),('v3_model_recall','mini3pro')]
        qright += [('same_model_recall',r['group']) for r in paired if r['category']=='same_model_recall' and r['mode']==mode]
        qright += [('G_recall',r['group']) for r in paired if r['category']=='G_recall' and r['mode']==mode]
        for ax,queries,labels in [(axs[0],qleft,['v3测试背景','bgA测试','C1七档合并','013安静','陷波白噪声']),(axs[1],qright,['data非G','v3无人机','v3 mavic2','v3 mini3pro']+[g.replace(' | ','/') for c,g in qright[4:]])]:
            for j,tag in enumerate(TAGS):
                v=[summary(cat,g,mode,tag) for cat,g in queries];ax.bar(np.arange(len(labels))+(j-1)*.26,[100*a for a,b in v],width=.25,yerr=[100*b for a,b in v],capsize=3,label=tag)
            ax.set_xticks(range(len(labels)),labels,rotation=40,ha='right');ax.set_ylim(0,110);ax.set_ylabel('虚警率（%）' if ax==axs[0] else '召回率（%）');ax.legend()
        fig.suptitle('叠加WiFi训练前后：三种子均值±标准差；'+mode);fig.tight_layout();fig.savefig(OUT/('图14_叠加WiFi训练前后_2.4GHz.png' if mode=='argmax' else '图14_thr_fa1_2.4GHz.png'),dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,5));ms=[]
    for ax,mode in zip(axs,['argmax','thr_fa1']):
        for gi,group in enumerate(['data_drone','v3_drone','v3_background']):
            for ti,tag in enumerate(TAGS):
                means=[];stds=[]
                for x in XS:
                    rs=[r for r in mix if r['round']==tag and r['mode']==mode and r['group']==group and int(r['x_db'])==x];assert len(rs)==3
                    vals=[float(r['rate']) for r in rs];means.append(np.mean(vals));stds.append(np.std(vals,ddof=1));ms.append(dict(round=tag,mode=mode,group=group,x_db=x,rate_mean=means[-1],rate_std=stds[-1]))
                ax.errorbar(XS,means,yerr=stds,ls='--' if tag=='J' else '-',marker=[None,'o','s'][ti],color=f'C{gi}',label=tag+' '+group)
        ax.axvspan(-5,15,color='gray',alpha=.1);ax.set(xlabel='功率比X（dB）；训练[-5,15]，-10/+20范围外',ylabel='检出率 / 虚警率',title=mode,ylim=(-.03,1.03));ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'fig3_mix_test_2.4GHz.png',dpi=150);plt.close(fig);writecsv(OUT/'mix_summary.csv',ms)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,mode in zip(axs,['argmax','thr_fa1']):
        for tag in TAGS:
            v=[summary('C1',str(x),mode,tag) for x in XS];ax.errorbar(XS,[a for a,b in v],yerr=[b for a,b in v],marker='o',ls='--' if tag=='J' else '-',label=tag)
        ax.set(xlabel='功率比X（dB）',ylabel='背景判为无人机比例',title=mode,ylim=(-.03,1.03));ax.legend()
    fig.tight_layout();fig.savefig(OUT/'fig_C1_2.4GHz.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,5,figsize=(23,4))
    for tag in ['La','Lb']:
        for s in SEEDS:
            rs=rows(OUT/f'train_log_{tag}_seed{s}.csv');assert len(rs)==40
            for ax,key in zip(axs,['train_loss','val_loss','val_fa_v3','val_fa_bgA','val_miss']):ax.plot([int(r['epoch']) for r in rs],[float(r[key]) for r in rs],label=f'{tag}/{s}',ls='-' if tag=='La' else '--');ax.set(xlabel='Epoch',ylabel=key)
    axs[-1].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'fig5_curves_2.4GHz.png',dpi=150);plt.close(fig)
    for tag in ['La','Lb']:
        for s in SEEDS:
            shutil.copyfile(OUT/f'{tag}_test_predictions_2.4GHz_seed{s}.csv',OUT/f'test_predictions_2.4GHz_{tag}_seed{s}.csv')
            shutil.copyfile(OUT/f'{tag}_bgA_test_predictions_seed{s}.csv',OUT/f'bgA_test_predictions_{tag}_seed{s}.csv')
    before=load(OUT/'protected_before.json');after=hashes();dump(OUT/'protected_after.json',after)
    changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)];dump(OUT/'integrity.json',dict(protected_files=len(after),changed=changed,identical=before==after));assert not changed
    cks={(tag,s):torch.load(DEST/f'2.4GHz_{tag}_seed{s}_best.pt',map_location='cpu',weights_only=False) for tag in ['La','Lb'] for s in SEEDS}
    dones=[load(DEST/f'2.4GHz_{tag}_seed{s}_done.json') for tag in ['La','Lb'] for s in SEEDS];smoke=load(DEST/'2.4GHz_La_seed2026_smoke_done.json');gen=load(MIXCACHE/'generation_done.json')
    report=['# 任务L：2.4GHz无人机叠加WiFi训练','续跑自 ctx_66fa10ede724，§3/§4 试算沿用上一轮。',
    '状态：完成两组共6次40epoch、3epoch冒烟、全部评估；只训练2.4GHz。',
    '设置：ResNet18(1,2)，AdamW lr=1e-3/wd=1e-4，cosine T_max=40/eta_min=1e-5，batch64，AMP自动选bfloat16，seed2026/2027/2028，原seed()、cudnn.benchmark=True；无其他增强、调参。',
    '与J的[训练diff](train_vs_J.diff)：只改数据拼接、训练权重、variant及路径命名；val权重仍用J原训练计数，选模FA_v3+FA_bgA+Miss、双背景阈值取大及断点规则均沿用J。',
    '训练子类顺序：原无人机/v3背景/bgA/叠加无人机/叠加背景；类总权重各0.5，整体均值1；batch损失ΣwCE/Σw。']
    for tag in ['La','Lb']:
        c=cks[tag,2026];report.append(f"{tag}计数={c['train_counts']}；实际逐样本权重={c['sample_weights']}。")
    report += [f'预检沿用preflight.json：计数2832/152/234和bgA2861/990/1287；split冲突0；9片重算及3片mix重算差0；J2026复推最大差0.000931948<0.001；保护目录{len(after)}文件SHA256变化0。',
    '审图回复已接受X∈[-5,15]dB；[试算](fig_L_pilot_2.4GHz.png)沿用。atom se原片突发占空比0.003；air 3s占空比0因宽带抬高中位数，图上信号可见，保留原指标。',
    '功率比按整片平均功率计算，包含接收机A底噪；X高时WiFi会被A底噪盖住（试算15dB列可见）。与mix_test.py一致，本任务不改。',
    'rng20261110按原行序交替抽X和背景下标，先2832无人机再2861背景；读取源按block_file/start排序，按原索引写回；234片train背景IQ预读，输出float16。']
    for name,v in gen['summary'].items():report.append(f"叠加{name}：N={v['n']}；X的0/25/50/75/100%分位={v['X_quantiles']}dB；实际功率比与X最大差={v['max_power_ratio_error']:.3g}dB；[配对](mix_train_{name}.pairs.csv)。")
    report+=['|组/seed|best epoch|val FA_v3 k/80,U95|val FA_bgA k/990,U95|val Miss|thr_fa1|','|---|---|---|---|---|---|']
    for (tag,s),c in cks.items():
        kv=round(c['val_fa_v3']*80);ka=round(c['val_fa_bgA']*990)
        report.append(f"|{tag}/{s}|{c['epoch']}|{kv}/80={c['val_fa_v3']:.6f}, {cp(kv,80):.6f}|{ka}/990={c['val_fa_bgA']:.6f}, {cp(ka,990):.6f}|{c['val_miss']:.6f}|{c['thr_fa1']:.9f}|")
    report+=['best.pt还保存thr_v3/thr_bgA、背景各自片数及全部权重；val参与选模，不是独立测试。last.pt含优化器/调度器/scaler与全部RNG，可用run_all.py续跑。',
    '全部逐种子、均值±SD及虚警k/N、单侧95% CP上界：[并列表](metrics_paired.md) / [CSV](metrics_paired.csv)；主指标如下，单位%。',
    '|判决/指标|J|L-a|L-b|','|---|---|---|---|']
    for mode in ['argmax','thr_fa1']:
        for g in ['v3_test_FA','bgA_test_FA','bgA_val_FA','quiet013_FA','white_noise_FA','notched_noise_FA','C1_FA','nonG_miss','all_drone_miss_with_G','data_nonG_recall','v3_drone_recall']:
            cells=[]
            for t in TAGS:
                a,b=summary('main',g,mode,t);cell=f'{a*100:.3f}±{b*100:.3f}'
                rates=[r for r in rr if r['category']=='main' and r['group']==g and r['mode']==mode and r['round']==t]
                if rates[0]['fa']=='True':cell+='；'+', '.join(f"{r['seed']}:{r['k']}/{r['n']},U95={float(r['cp95_upper'])*100:.3f}%" for r in rates)
                cells.append(cell)
            report.append('|'+mode+'/'+g+'|'+'|'.join(cells)+'|')
    report+=['主表漏检不含G；含G原口径单列，[G召回](G_recall_paired.csv)；[型号/状态/带宽漏检](miss_by_model_state_bw_paired.csv)、[来源召回](source_recall_paired.csv)、[同型号](same_model_recall_paired.csv)、[v3型号召回](v3_model_recall_paired.csv)。',
    '[v3会话FA](fa_by_session_paired.csv)、[bgA test源分组](bgA_test_by_source_paired.csv)、[bgA val源分组](bgA_val_by_source_paired.csv)、[筛除片段](rejected_detail_paired.csv)、[013分组](quiet013_by_group_split_paired.csv)。013共19片，train组曾以无人机标签训练，非独立背景。',
    'J标准评估和mix总数取J已有CSV；新加按型号mix及C1由J checkpoint推理。H三组mix测试缓存只读，均与训练split隔离。',
    '[mix逐种子及背景CP上界](mix_results.csv)、[均值±SD](mix_summary.csv)、[data各型号每X并列](mix_data_model_paired.csv)，含atom se/mavic 4。',
    'C1：rng20261111从bgA test无放回200片，按抽样顺序i配v3 test背景i%80，七档X；[逐种子及CP](C1_paired.csv)，[配对](C1.pairs.csv)。合并N=1400是200源片重复7档，并非1400独立源片，CP为逐片二项式口径。',
    '[WiFi粗指标](wifi_group_descriptors.csv)：各背景group的ch1−参考/ch6−参考中位数；行频率2405+(i+0.5)×50/256MHz，先按时间取中位数。参考带2451–2454MHz靠近陷波和滤波器边缘，仅描述、不筛选。',
    '![图14](图14_叠加WiFi训练前后_2.4GHz.png)',
    '![mix检验，J虚线/L-a和L-b实线；灰区训练范围](fig3_mix_test_2.4GHz.png)',
    '![C1](fig_C1_2.4GHz.png)',
    '[阈值图14](图14_thr_fa1_2.4GHz.png)、[6次训练曲线](fig5_curves_2.4GHz.png)；逐片test_predictions_2.4GHz_{La,Lb}_seed<s>.csv、bgA_test_predictions_{La,Lb}_seed<s>.csv与C1_predictions_<组>_seed<s>.csv。']
    report += ['data按型号mix重点项（argmax，三种子均值%，每格J/L-a/L-b）：','|X dB|atom se|mavic 4|','|---|---|---|']
    for x in XS:
        report.append('|'+str(x)+'|'+'|'.join('/'.join(f"{summary('mix_data_model_'+str(x),model,'argmax',t)[0]*100:.3f}" for t in TAGS) for model in ['atom se','mavic 4'])+'|')
    disk=sum(p.stat().st_size for d in [OUT,DEST] for p in d.rglob('*') if p.is_file())
    report += [f"生成{gen['seconds']:.1f}s；3epoch冒烟{smoke['seconds']:.1f}s；6次训练耗时秒={[round(d['seconds'],1) for d in dones]}；评估{load(OUT/'evaluation_done.json')['seconds']:.1f}s；训练GPU张量峰值{max(d['peak_gpu_bytes'] for d in dones)/2**30:.3f}GiB，进程WorkingSet峰值{max(d['peak_host_bytes'] for d in dones)/2**30:.3f}GiB；本任务产物占磁盘{disk/2**30:.3f}GiB（含沿用试算）。",
    '执行决定：冒烟独立3epoch后正式同种子从头40epoch；文本UTF-8，关闭pyc写入；保留J基础算法，仅修改上述数据/权重/路径。未调用DSH、common.model或crop.py，未写入H缓存/J DEST，未改F/H/I/J；GPU开始无其他Python计算进程。',
    '**事实陈述，不下结论**（argmax，J → L-a → L-b，三种子均值±SD，%）：']
    for cat,g in [('main','v3_drone_recall'),('v3_model_recall','mavic2'),('v3_model_recall','mini3pro'),('main','nonG_miss'),('main','v3_test_FA'),('main','bgA_test_FA'),('main','C1_FA'),('main','notched_noise_FA')]:report.append(g+'：'+' → '.join(f'{a*100:.3f}±{b*100:.3f}' for a,b in [summary(cat,g,'argmax',t) for t in TAGS])+'。')
    report.append('mix 0dB data检出：'+' → '.join(f"{r['rate_mean']*100:.3f}±{r['rate_std']*100:.3f}" for t in TAGS for r in ms if r['round']==t and r['mode']=='argmax' and r['group']=='data_drone' and r['x_db']==0)+'。')
    formatted=[]
    for i,line in enumerate(report):
        if i and (line.startswith('#') or line.startswith('|')!=report[i-1].startswith('|') or report[i-1].startswith('#')):formatted.append('')
        formatted.append(line)
    report=formatted
    assert len(report)<=100,len(report)
    (OUT/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8');dump(OUT/'finished.json',dict(report_lines=len(report),protected_files=len(after),training_runs=6))
if __name__=='__main__':run()
