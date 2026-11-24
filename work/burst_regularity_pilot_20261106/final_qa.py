from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pilot import *
notes={
1:'三段主要宽带/跳频块均有对应框；5800 MHz细线及边缘形成大量碎片，1107/1105/1089为程序计数，未完成独立逐个肉眼计数。',
2:'三段可见多个固定频点反复短突发，主块有框；低频镜像/弱边缘被分成多碎片，384/369/365不是人工确认的物理突发数。',
3:'三段宽带脉冲上下边缘碎裂明显，4433/4520/3960显著多于肉眼可辨主脉冲；计数一致性不通过。',
4:'三段固定中心主突发与高频跳变块有框；镜像边缘及5800 MHz细线产生密集小框，1454/1414/1198未逐个独立确认。',
5:'三段约2400–2410 MHz长突发与窄带跳变有框；底边有碎片小框，357/377/359未逐个人工确认。',
6:'三段主要宽带块及窄带跳变有框；掩膜边缘外信号有意不检测，部分窄脉冲被分开，133/120/137未逐个人工确认。',
7:'三段约5767 MHz或5807 MHz弱脉冲被碎片化，985/1091/685明显高于物理主突发数；计数一致性不通过。',
8:'三段原始谱主要为5800 MHz稳定细线；检测仅有中心屏蔽边缘瞬态小框，程序计数2/2/0，未发现宽带环境通信。'}
v=json.loads((OUT/'verification.json').read_text(encoding='utf8'));results=json.loads((OUT/'results.json').read_text(encoding='utf8'))
for entry in v['random_views']:
    idx=RECS.index(entry['recording'])+1;entry['visual_check']='macro_alignment_reviewed; exact_count_not_independently_verified';entry['conclusion']=notes[idx];entry['count_agreement_pass']=False
v['manual_count_acceptance_pass']=False;v['manual_review_scope']='已视觉查看8张三联放大图（24个50ms窗口）；仅确认主块/碎片现象，不伪造精确逐个人工计数。'
v['cross_boundary_confirmation']=json.loads((OUT/'boundary_evidence.json').read_text(encoding='utf8'));v['cross_boundary_confirmation']['visual_pass']=True;v['cross_boundary_confirmation']['conclusion']='首录制49.396–50.308 ms主突发跨50 ms虚线连续，输出单行parts=2，已看boundary_merge.png确认。'
v['source_after']=snapshot();v['source_unchanged']=v['source_before']==v['source_after'];dump(OUT/'source_after.json',v['source_after'])
required=[OUT/f'{prefix}_{rec}.png' for prefix in ['annot','zoom','clusters'] for rec in RECS];v['png_sizes']={p.name:p.stat().st_size for p in OUT.glob('*.png')};v['pngs_complete']=all(p.exists() and p.stat().st_size>30000 for p in required);v['required_png_count']=len(required)
v['implementation_notes']=['DBSCAN使用0.05标准差网格加权近似，非未量化的精确邻域聚类。','两两差直方图通过0.05ms量化时间自相关累积；未跨NN块计算。','data逐文件STFT，边界接触按3hop时间容差及频率重叠合并；未计算跨文件的完整滑动窗。','v3 floor使用全会话1ms单元线性均值功率的各频点第10百分位、可用频带中位数。','100ms覆盖与10/20/30/40/50ms出现率均按事件持续区间与实际观测窗相交计算。']
v['acceptance_status']='failed_manual_count_requirement';v['artifacts_complete']=all((OUT/f'bursts_{rec}.csv').exists() for rec in RECS) and v['pngs_complete']
dump(OUT/'verification.json',v)
p=OUT/'REPORT.md';s=p.read_text(encoding='utf8');s=s.replace('随机三段/录制放大图已生成；人工复核结论见 verification.json。','已查看8张三联放大图（24窗）：主突发基本有框；#3/#7碎片计数远多于物理主突发，其他录制未完成独立逐个计数，人工计数一致性验收不通过。')
s=s.replace('背景里有规律的环境通信也可满足 featured；also_in_background 仅做形状和频率匹配，不代表无人机来源。','背景#8输出10个孤立中心边缘瞬态、0 featured、0 review；5800 MHz稳定细线按任务E屏蔽，不作为有特征信号。')
s+='\n逐录制视觉复核：'+ '；'.join(f'#{i} '+notes[i] for i in notes)+'\n'
s+='跨段实证：#1在50 ms边界的一次主突发已合为单行parts=2，boundary_merge.png经视觉确认；原始文件快照一致。\n'
s+='方法偏差：逐文件STFT未补跨文件滑动帧；边界用3hop时间容差+频率重叠合并；网格DBSCAN和时间直方图均为量化近似。\n'
s+='结论：产物齐全，但严格人工计数一致性未通过；不得把本试点视为已通过自动截频验收。\n';p.write_text(s,encoding='utf8')
assert len(s.splitlines())<=60
print(json.dumps(dict(recordings=len(results),signals=sum(len(r['signals']) for r in results),pngs=len(v['png_sizes']),pngs_complete=v['pngs_complete'],source_unchanged=v['source_unchanged'],boundary_pass=v['cross_boundary_confirmation']['visual_pass'],manual_count_pass=False,report_lines=len(s.splitlines())),ensure_ascii=False))
