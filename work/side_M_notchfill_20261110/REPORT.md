# 任务M：2.4GHz陷波凹坑镜像修补后重训

完成：3epoch冒烟、三种子各40epoch、全部缓存重算及L-b/M并列评估、交叉对照和行压低实验。
设置：ResNet18(1,2)，AdamW lr=1e-3/wd=1e-4，cosine T_max40/eta_min1e-5，batch64，AMP bfloat16，seed2026/2027/2028，cudnn.benchmark=True。
与L-b差异：[diff](vs_L.diff)；train.py逐字相同（内部variant与checkpoint仍标Lb）；support只改输出/缓存路径及spec→spec_fill；IQ、配对、权重、阈值、选模、超参不变。
原始频点：2426.88: 2426.76, 2426.82, 2426.88, 2426.94, 2427.00 MHz；2440.0: 2439.84, 2439.90, 2439.96, 2440.02, 2440.08, 2440.14, 2440.20 MHz；2450.0: 2449.80, 2449.86, 2449.92, 2449.98, 2450.04, 2450.10, 2450.16 MHz。
镜像规则：左半复制左侧坑外对称点，右半复制右侧坑外对称点；池化前实际替换5+7+7=19个频点；后续流程相同。
规格更正（协调者确认）：任务书文字16点有误，以pilot代码为准；闭区间实际含19点，沿用pilot全部频点，不改、不重训。
背景更正（协调者确认）：任务K中陷波白噪声99.5%为任务J seed2026结果，不是L-b；L-b在任务L已有CSV中三种子均0%，本任务直接沿用。
开工计数=[2832, 152, 234, 2861, 990, 1287, 2832, 2861]（data无人机/v3无人机/v3背景/bgA train/val/test/叠加无人机/叠加背景）；叠加6片最大差=0.0。
关闭修补6片逐像素完全一致；非陷波行回归float32最大残差=0.000103284，float16=0.015210391。
异常：原float16≤0.01检查在H样本L20_00009388失败0.015210391；协调者确认量化舍入，授权float32≤0.001、float16≤0.02；补测全部通过。
修补行统计 bgA/2426.88MHz：均值差=-0.056508，标准差比=1.072817（40片，均合格）。
修补行统计 bgA/2440.0MHz：均值差=-0.118702，标准差比=0.940888（40片，均合格）。
修补行统计 bgA/2450.0MHz：均值差=-0.070331，标准差比=0.918619（40片，均合格）。
修补行统计 notched/2426.88MHz：均值差=-0.074968，标准差比=1.075589（40片，均合格）。
修补行统计 notched/2440.0MHz：均值差=-0.112398，标准差比=0.952640（40片，均合格）。
修补行统计 notched/2450.0MHz：均值差=-0.120919，标准差比=0.953047（40片，均合格）。
GPU开始无其他Python计算进程；[开工检查](preflight.json)。所有IQ按block_file/start顺序mmap读取、按原行索引写回；bgA缓存保留原meta索引，仅keep行有效，另存keep_indices.json。
两组训练pairs和C1 pairs逐字复制L文件，沿用g直接重建b+g*d；v3背景IQ只读预读；80片白噪声seed20261109；013按L同19片。

|seed|best epoch|val FA_v3|val FA_bgA|val Miss|val loss|thr_fa1|
|---|---|---|---|---|---|---|
|2026|13|0.000000|0.000000|0.000978|0.006348|0.216673806|
|2027|3|0.000000|0.000000|0.005865|0.009715|0.259825677|
|2028|26|0.000000|0.000000|0.002933|0.007179|0.264738411|

逐片预测以Lb_/M_开头；L-b全部标准指标和预测取L已有CSV，无重训；[逐种子并列表](metrics_paired.md) / [CSV](metrics_paired.csv)含argmax、thr_fa1、均值±SD、虚警k/N和95%单侧CP上界。
[型号/状态/带宽漏检](miss_by_model_state_bw_paired.csv)、[含G](main_paired.csv)、[G单列](G_recall_paired.csv)、[v3型号召回](v3_model_recall_paired.csv)、[同型号](same_model_recall_paired.csv)、[来源召回](source_recall_paired.csv)。
[v3会话虚警](fa_by_session_paired.csv)、[bgA test源分组](bgA_test_by_source_paired.csv)、[bgA val源分组](bgA_val_by_source_paired.csv)、[筛除片段](rejected_detail_paired.csv)、[013分组](quiet013_by_group_split_paired.csv)。
[叠加逐种子](mix_results.csv)、[均值±SD](mix_summary.csv)、[按型号叠加](mix_data_model_paired.csv)、[C1](C1_paired.csv)；训练X[-5,15]dB，检验-10至20七档。
C1合并1400项为200源片×7档，CP采用逐片二项式口径；013中train组曾作为无人机训练，非独立背景；val参与选模与定阈值。

![图15](图15_陷波修补前后_2.4GHz.png)

[阈值图15](图15_thr_fa1_2.4GHz.png)、[叠加检验图](fig3_mix_test_2.4GHz.png)、[C1图](fig_C1_2.4GHz.png)、[训练曲线](fig5_curves_2.4GHz.png)。
[交叉逐种子k/N及CP](cross_metrics.csv)、[全部交叉均值±SD](cross_summary.csv)：L-b在修补后test/bgA/陷波白噪声；M在原bgA/原陷波白噪声；test背景FA与无人机召回分别列出。
行压低实验：bgA test前200片，near=|池化行中心频率−陷波频率|≤0.25MHz；各处减2.0/5.2/6.9，不重标准化，行号见attention_rows.json；仅描述。
交叉/关注度 Lb/filled_test：argmax=2.500±1.250%；thr_fa1=2.917±0.722%。
交叉/关注度 Lb/filled_bgA_test：argmax=0.000±0.000%；thr_fa1=0.000±0.000%。
交叉/关注度 Lb/filled_notched_noise：argmax=8.750±15.155%；thr_fa1=16.250±28.146%。
交叉/关注度 M/old_bgA_test：argmax=0.000±0.000%；thr_fa1=0.000±0.000%。
交叉/关注度 M/old_notched_noise：argmax=0.000±0.000%；thr_fa1=0.000±0.000%。
交叉/关注度 M/attention_before：argmax=0.000±0.000%；thr_fa1=0.000±0.000%。
交叉/关注度 M/attention_depressed：argmax=65.500±56.738%；thr_fa1=66.667±57.735%。
生成775.1s；3epoch冒烟58.1s；三次训练秒=[671.3, 671.8, 677.3]；评估26.1s；交叉10.7s。
训练GPU张量峰值1.178GiB；训练进程WorkingSet峰值3.416GiB；生成进程中途读取的系统WorkingSet历史峰值4.636GiB（非全程监测）；总新占磁盘4.095GiB。
保护目录SHA256：基线3586文件，已有文件差异0；协调者外部新增=['work\\review_20261109\\check_M_precheck_residual.py']（授权单列、不改基线）。无DSH、crop.py/pilot模块import、common.model调用；无安装/提交/改划分/5.8GHz训练。
自行决定：冒烟独立运行，正式训练同种子从头；L训练脚本保留内部Lb标签，报告与推理输出统一为M；关闭pyc写入。

**事实陈述，不下结论**（argmax，L-b → M，三种子均值±标准差，%）：
notched_noise_FA：0.000±0.000 → 0.000±0.000。
white_noise_FA：5.000±8.660 → 0.000±0.000。
bgA_test_FA：0.000±0.000 → 0.000±0.000。
v3_test_FA：2.500±1.250 → 2.500±2.500。
C1_FA：1.857±0.446 → 1.786±0.786。
v3_drone_recall：80.635±9.540 → 86.032±8.536。
mavic2：53.333±14.434 → 56.667±15.275。
mini3pro：77.083±15.729 → 89.583±13.010。
nonG_miss：5.576±4.061 → 5.699±1.530。
Lb模型在filled_test输入上的FA：2.500±1.250%。
Lb模型在filled_bgA_test输入上的FA：0.000±0.000%。
Lb模型在filled_notched_noise输入上的FA：8.750±15.155%。
M模型在old_notched_noise输入上的FA：0.000±0.000%。
