# DroneRFa 项目共享摘要

真实项目：C:/Users/Administrator/Desktop/数据集处理。采用 deepseek-workflow：Codex定范围、架构、冻结规则并验收，DSH Desktop 的 DeepSeek 执行具体代码、报告与必要测试。不能自动提交、推送、发布或更改权限。审批提示交用户，不代批准。

## 最新授权与当前阶段

2026-09-18用户明确“文件下载好了，进入下一步”，恢复下载后的工作。下载和数据准备已验收，当前只执行冻结的六模型未使用源文件评价。范围与公式见work/deepseek-plan.md、work/heldout-eval-config-20260918.json及work/deepseek-tasks/20260918-heldout-six-model-evaluation.md。本任务不重训、调阈值或进入第二阶段。

DSH Desktop 原数据准备任务 sess_dsh_5efa784d1fab1faeae088d6e，external session-bf6925d4-73d7-401e-aafb-3a616b52d658，已空闲完成；旧run work/deepseek-runs/20260917-214847-30503d91已验收归档。不能重复派发未完任务，以workflow实际activeRunDirectory为准。桌面会话可见，公开RuntimeProtocol认证仅内存使用，不复制发现文件或凭据，不启动另一服务器。

## 研究目标与冻结旧实验

两阶段主线：第一阶段背景/无人机存在性检测，第二阶段机型分类。第二阶段及串联评价未完成。原12个MAT只读 C:/Users/Administrator/Documents/DroneRFa数据集，T0000/T0010/T0101 × D00 × S0000/S0001/S0010/S0011。旧索引训练920、验证450、测试440，共1810窗，源文件分离不证明采集会话独立。

前端固定RF0 I/Q，100MS/s，10ms=1000000点，每10000000点采集块内切窗；去均值/RMS归一化/复AWGN，Hamming STFT n_fft1024 hop512，log power/fftshift/逐样本标准化，1x256x256，ResNet18二分类。模型CUDA AMP float16，frontend float32，logits转float32后差>0判Drone。

A训练噪声 clean,10,5,0,-5,-10,-12；B增加-15，各档等概率。训练种子2026/2027/2028各A/B训练5轮，验证噪声seed12026；按四档clean/-10/-12/-15平均BA选模，同分最早。六模型best_model.pt对应multi_noise：2026 A3/B5，2027 A2/B5，2028 A5/B3。旧验证综合A→B分别95.73%→98.67%、98.10%→98.45%、98.73%→98.09%，尚未证明稳定提升。累计54000预测统计、480原始IQ抽查通过。不要按新结果重新选种子或轮次。

原代码、旧索引、六个best_model.pt SHA256冻结于work/codex-preserved-inputs-20260917.json，共14项；2026-09-18 Codex复核全部匹配。

## 已验收数据准备

新12个MAT只读 F:/DroneRFa数据集/heldout_20260917：T0000/T0010/T0101 × D00 × S0100/S0101/S0110/S0111。大下载与缓存始终F盘，下载于12:02前后完成，PID17312已结束。实际合计12382967102字节=11.53GiB，各文件大小匹配manifest。

准备产物 dronerfa_baseline/outputs/heldout_preparation_20260917/：report.md、verification.json、heldout_dataset_summary.json、heldout_window_index.csv、duplicate_check.json、activity_band_check.json。1820窗：背景560、Phantom610、Air2S650；长度140M/150M/160M/180M各不相同。四通道float64向量长度一致，全量扫描非有限值0，尾部样本0。新旧文件完整SHA256重复0对，固定IQ指纹匹配0；抽样指纹不排除全部重录/重编码。

Codex验收证据 work/heldout_prep_20260917/codex-acceptance-20260918.json：实际大小、四通道形状/固定短切片有限性、报告全扫描覆盖、1820窗逐行边界/覆盖/标签/零重叠、旧索引零文件交集、索引与summary哈希、原14输入hash。未重复全量扫描或MAT全hash。索引SHA256固定6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf。

小幅记账修正交本次执行器先处理：download_records顶层status残留running、verification将所有通道扫描长度写为140M、report把最后r3请求/重试数称整轮。不得改索引、已有数据哈希和信号统计。

## 活动频段观察与判断

固定首/中/末各10ms窗：Phantom和Air2S RF0活动各12/12，背景5/12；RF1新旧72固定窗均未检出活动。只说明固定窗观察，不能宣称整个RF1绝无信号。Phantom RF0 95%占用带宽中位数新6.25MHz、旧24.56MHz，Codex保留为分布差异，证据不足判定协议不匹配/设备混入，不排除预先指定文件、不变更输入RF0或加入RF1评价档。

MAT没有中心频率/增益/采集日期会话地点元数据，绝对频段及采集会话独立性均未确认。频谱仅相对通道中心偏移，不依据模型表现挑文件。不称跨会话/跨地点泛化，附加AWGN档不是现场绝对SNR。

## 本次冻结评价与验收

完整1820窗，同一CSV顺序，clean/-10/-12/-15；AWGN seed22026，epoch0，完整行索引i对应RNG22026+i，六模型共享每档float32特征。每档内存缓存约455MiB；若大缓存落盘，仅F:/DroneRFa数据集/heldout_eval_20260918/feature_cache。

主口径：每类别4源文件的误报/检出率等权；两机型检出率再等权为无人机召回；BA=0.5*(1-背景文件等权误报率+两机型等权召回)。模型综合是四档BA均值，三个训练种子全部给B-A配对差值。同时保留pooled窗口统计，不能混淆两口径。

新增实现/报告允许范围见任务；产物 dronerfa_baseline/outputs/heldout_evaluation_20260918/。43,680预测、288逐文件统计、24模型条件统计、3种子配对差值；数据原始定位、hash、特征共享证据、必要失败用例与真实窗独立复核齐全，才由Codex验收。原代码/索引/权重保持冻结。完成后不自行转第二阶段。

## 参考

work/deepseek-plan.md；work/heldout-eval-config-20260918.json；work/codex-preserved-inputs-20260917.json；dronerfa_baseline/outputs/noise_ab_replicates_20260917/后续留出数据方案.md；dronerfa_baseline/run_noise_ab.py、dronerfa_data.py、model.py、train_resnet18.py、verify_noise_ab_inputs.py。更早下载过程见work/heldout_prep_20260917/context-before-evaluation-20260918.md，不用继承已失效的暂停或running状态。
