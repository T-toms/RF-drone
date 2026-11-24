from pathlib import Path
import csv,json,time,hashlib
import numpy as np
from scipy.stats import beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent; P=O.parents[1]
def load(name):return json.loads((O/name).read_text(encoding='utf-8'))
def csvout(name,rs):
 with (O/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
with (O/'inference.csv').open(encoding='utf-8') as f:results=list(csv.DictReader(f))
for r in results:
 for k in ['seed','k','n']:r[k]=int(r[k])
 r['rate']=float(r['rate']);r['cp95_upper']=float(beta.ppf(.95,r['k']+1,r['n']-r['k'])) if r['k']<r['n'] else 1.
csvout('results_with_ci.csv',results)
index={(r['band'],r['model'],r['seed'],r['input'],r['decision']):r for r in results}
proxy=[]
for model in ['H','J']:
 for seed in [2026,2027,2028]:
  for dec in ['argmax','thr_fa1']:
   for who,real in [('A','bgA_test'),('B','bgB_0710')]:
    a=index['2.4GHz',model,seed,who,dec];b=index['2.4GHz',model,seed,real,dec]
    proxy.append(dict(model=model,seed=seed,decision=dec,proxy=who,proxy_k=a['k'],proxy_n=a['n'],real_k=b['k'],real_n=b['n'],difference_pp=100*abs(a['rate']-b['rate']),passed=abs(a['rate']-b['rate'])<=.1))
csvout('proxy_checks.csv',proxy);gate=all(r['passed'] for r in proxy)
shapes=np.load(O/'shapes.npz');summ=load('shape_summary.json');cal=load('calibration.json');final=load('final_validation.json');stats=load('input_statistics.json');pilot=load('pilot_validation.json')
bins=[]
for s in summ:
 key=s['band']+'_'+s['receiver']
 for i,(f,v,c,bad) in enumerate(zip(shapes[key+'_freq'],shapes[key],shapes[key+'_counts'],shapes[key+'_bad'])):bins.append(dict(band=s['band'],receiver=s['receiver'],bin=i,freq_mhz=float(f),s_db=float(v),participating_blocks=int(c),interpolated=bool(bad)))
csvout('shape_bins.csv',bins)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(2,2,figsize=(21,11),gridspec_kw={'width_ratios':[1,1.7]})
notches={'2.4GHz':[2426.88,2440.,2450.],'5.8GHz':[5800.,5795.5009375]}
for ax,band in zip(axs[:,0],['2.4GHz','5.8GHz']):
 for who,name in [('A','接收机 A（无人机块估计）'),('B','接收机 B（07-10 背景）')]:
  key=band+'_'+who;ax.plot(shapes[key+'_freq'],shapes[key],label=name,lw=1.3)
 for f in notches[band]:ax.axvline(f,color='gray',ls=':',lw=.9)
 ax.set(title=band+' 底噪形状 S(f)',xlabel='频率（MHz）',ylabel='功率（dB）');ax.legend(fontsize=9);ax.grid(alpha=.2)
kinds=['a','b','A','B','e_bg','e_drone','f_b','f_A','f_e_bg','f_e_drone']
names=['白噪声','FIR+陷波','代理 A','代理 B','真实背景','无人机召回','抹平：白噪','抹平：代理A','抹平：背景','抹平：召回']
combos=[('2.4GHz','H'),('2.4GHz','J'),('5.8GHz','H')]
for ax,dec,title in zip(axs[:,1],['argmax','thr_fa1'],['argmax 判决','虚警阈值 thr_fa1 判决']):
 for j,(band,model) in enumerate(combos):
  rates=np.array([[index[band,model,seed,k,dec]['rate']*100 for seed in [2026,2027,2028]] for k in kinds])
  ax.bar(np.arange(len(kinds))+(j-1)*.25,rates.mean(1),.24,yerr=rates.std(1,ddof=1),capsize=2,label=band+'-'+model)
 ax.set_xticks(range(len(kinds)));ax.set_xticklabels(names,rotation=28,ha='right');ax.set_ylim(0,110);ax.set(title=title,ylabel='判为无人机比例（%）');ax.legend(ncol=3);ax.grid(axis='y',alpha=.2)
fig.suptitle('图13  空频段与底噪代理对照（误差棒：3 个种子样本标准差）'+('' if gate else '\n代理未通过 2.4G 检验，5.8G 代理数字仅供参考'),fontsize=16)
fig.tight_layout();fig.savefig(O/'图13_空频段与底噪代理对照.png',dpi=180);plt.close(fig)
before=load('hash_before.json');after={}
for rel in before:
 p=P/rel;h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 after[rel]=h.hexdigest()
current={str(p.relative_to(P)) for name in ['crop_slice_20261107','train_stage1_L20_20261108','bgA_20261109','train_stage1_L20_bgA_20261109'] for p in (P/'work'/name).rglob('*') if p.is_file()}
same=before==after and current==set(before)
(O/'hash_after.json').write_text(json.dumps(after,ensure_ascii=False,indent=2),encoding='utf-8');(O/'hash_verification.json').write_text(json.dumps(dict(files=len(before),unchanged=same,added=sorted(current-set(before)),changed=[r for r in before if before[r]!=after[r]]),ensure_ascii=False,indent=2),encoding='utf-8')
lines=['# 任务 K：空频段、陷波与底噪代理对照（2026-11-09）',
'仅推理；H=第一轮，J=加入 A 背景的第二轮；每类合成 200 片。无训练、无 DSH、无 heartbeat、无提交。',
'频率核对：注入已知正频率复单音，2.4G 的 2442.000 MHz→bin356→2442.000 MHz；5.8G 的 5812.500 MHz→bin640→5812.500 MHz；频率随 bin 递增，无偏移。',
'真实cell再核对：2426.88/2440/2450 MHz陷波谷分别落在2426.88/2439.96/2450.04 MHz；5795.5009375/5800 MHz分别落在5795.5078125/5800 MHz，均在半个bin内。',
'cell 照抄 Task F 的 Hann/分帧/功率平均计算，未导入 crop.py。两频段 bin 宽分别 120 / 97.65625 kHz。',
'协调者批准修正原规范：原 S 已含 FIR 滚降，直接再 FIR 的全频格误差为 '+ ' / '.join(f"{r['max_all_db']:.3f}" for r in pilot)+' dB；通带外改常数外延再 FIR，通带内允许最多三次校准。',
'验收仅取 2405–2455 / 5755–5845 MHz 且排除陷波 ±0.3 MHz；全频格/通带外误差均保留。50 个 1MHz 噪声底子带统一取 FC±25MHz，陷波不参与跨块拟合。',
'|频段-接收机|块/组|参与块数 min/Q1/中位/Q3/max|插值 bin（边缘外延）|校准通带最大误差序列 dB|200片误差 通带/全格/外带 dB|',
'|---|---:|---|---:|---|---|']
for s,v in zip(summ,final):
 key=s['band']+'_'+s['receiver']; seq=[r['max_passband_db'] for r in cal if r['key']==key]
 lines.append(f"|{key}|{s['blocks']}/{s['groups']}|"+'/'.join(f'{x:g}' for x in s['counts_percentiles'])+f"|{s['interpolated']}（{s['edge_extrapolated']}）|"+'→'.join(f'{x:.3f}' for x in seq)+f"|{v['max_passband_db']:.3f}/{v['max_fullgrid_db']:.3f}/{v['max_outside_db']:.3f}|")
lines+=['2.4G-B 三次校准后仍超0.3dB门槛；按协调者补充要求保留真实误差，继续代理检验，不将谱形验收宣称全部通过。',
'逐 bin 的计数、插值位置和 S 数字见 `shape_bins.csv`；逐块底噪/来源见 `shape_blocks.csv`；插值只用邻近有效 bin，两端采用最近值外延。',
'代理可信性（各格为 proxy k/N、真实 k/N、差值百分点；门槛≤10）：',
'|模型-种子-判决|代理 A vs bgA test|代理 B vs v3 07-10 test|', '|---|---|---|']
for model in ['H','J']:
 for seed in [2026,2027,2028]:
  for dec in ['argmax','thr_fa1']:
   rr=[r for r in proxy if r['model']==model and r['seed']==seed and r['decision']==dec]
   lines.append(f'|{model}-{seed}-{dec}|'+ '|'.join(f"{r['proxy_k']}/{r['proxy_n']}，{r['real_k']}/{r['real_n']}，{r['difference_pp']:.2f}（{'过' if r['passed'] else '未过'}）" for r in rr)+'|')
lines.append('代理总体：'+('通过。' if gate else '**未通过；所有 5.8G proxy 数字：代理未通过 2.4G 检验，仅供参考。**'))
for r in stats:lines.append(f"输入统计 proxy{r['proxy']}：频率剖面相关={r['correlation']:.5f}，最大差={r['maxdiff_std']:.4f} 标准化单位，像素>+3：proxy={100*r['proxy_pixels_gt3']:.4f}%、真实={100*r['real_pixels_gt3']:.4f}%。")
cols=[(band,model,seed,dec) for band,model in combos for seed in [2026,2027,2028] for dec in ['argmax','thr_fa1']]
lines+=['总表每格为 **k/N；比例%；95%单侧 Clopper–Pearson 上界%**，A=argmax、T=thr_fa1；e背景按全部会话汇总，逐会话原值在 `results_with_ci.csv`。',
'|输入|'+ '|'.join(f'{b[:3]}-{m}-{s}-{d[0].upper()}' for b,m,s,d in cols)+'|','|---|'+ '|'.join(['---']*len(cols))+'|']
for kind,label in zip(kinds,['(a)白噪声','(b)FIR+陷波','(c)proxyA','(d)proxyB','(e)真实v3背景','(e)真实data无人机','(f)抹平(b)','(f)抹平(c)','(f)抹平(e)背景','(f)抹平(e)召回']):
 vals=[index[b,m,s,kind,d] for b,m,s,d in cols]
 lines.append('|'+label+'|'+'|'.join(f"{r['k']}/{r['n']};{r['rate']*100:.2f};{r['cp95_upper']*100:.2f}" for r in vals)+'|')
lines+=['阈值比较采用 p(drone)≥thr_fa1；argmax 取两类最大值。抹平行严格由 STFT 裁剪和 adaptive pooling 的实际行中心映射，±0.45MHz 线性插值后逐片标准化。',
'bgA 仅 keep==True 且 test 的全部1287片；B可信性仅07-10 test；data召回不含泛化集。真实时频缓存直接读取，未重算。',
'合成随机种子：2026110900 + band_index×100000 + kind_index×1000 + i；band_index=0/1，kind_index=a/b/A/B→0/1/2/3，i=0..199；校准i=50000..50039。',
'独立校准每组40片，最终检验另取200片；最终 S′为每片20个cell的p10再跨片中位数，噪声片均参与；GPU输入严格 common.spec，未调用 common.model。',
'合成 IQ 通过内存管道传GPU，无IQ落盘；只将本任务时频缓存写入 E:/side_5p8_controls_20261109，含保留的早期缓存约300 MiB。']
runtime=load('synthesis_runtime.json');gpu=load('gpu_eval.json');mins=(time.time()-(O/'hash_start.py').stat().st_mtime)/60
regen=load('control_regeneration.json')
lines += [f"资源：总墙钟约{mins:.1f}分钟（含协调等待），首次合成+谱转换{runtime['seconds']/60:.2f}分钟，白噪声缓存修复重算{regen['seconds']/60:.2f}分钟，4个CPU进程，推理峰值显存{gpu['peak_mb']:.1f} MiB（RTX4060 Laptop 8GB）。",
f"保护核验：四个指定目录逐文件 SHA256，共{len(before)}文件，开工/交付完全一致={same}；见 hash_before/after.json 与 hash_verification.json。",
'小事项：PowerShell默认编码会乱码，统一UTF-8；rg不可用改用Select-String；未安装包；原规范双重FIR问题已获协调者明确修正，未改变原始数据与模型。',
'缓存修复：Windows把早期a/A、b/B文件名视为同名，已改为white_raw/white_filtered/proxy_A/proxy_B；保留代理缓存，重新生成全部白噪对照并重跑所有推理，早期表图已替换。',
'![图13](图13_空频段与底噪代理对照.png)', '**事实陈述，不下结论**']
for band,model in combos:
 lines.append(f"{band}-{model}：白噪/FIR陷波/proxyA/proxyB 的 argmax 三种子比例分别为 "+'；'.join('/'.join(f"{index[band,model,s,k,'argmax']['rate']*100:.1f}%" for s in [2026,2027,2028]) for k in ['a','b','A','B'])+'。')
lines.append(f"代理检验共{len(proxy)}项，通过{sum(r['passed'] for r in proxy)}项；最终四组通带最大误差为"+'/'.join(f"{r['max_passband_db']:.3f}" for r in final)+' dB。')
assert len(lines)<=80,len(lines)
(O/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('report lines',len(lines),'hash unchanged',same,'proxy overall',gate,flush=True)
assert same,'protected files changed'
