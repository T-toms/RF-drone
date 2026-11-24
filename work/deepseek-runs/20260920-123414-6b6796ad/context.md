# DroneRFa 项目共享摘要

真实项目：C:/Users/Administrator/Desktop/数据集处理。采用 deepseek-workflow：Codex定范围、架构、冻结规则并验收，DSH Desktop 的 DeepSeek 执行具体代码、报告与必要测试。不能自动提交、推送、发布或更改权限。审批提示交用户，不代批准。

## 最新授权与当前阶段

截至2026-09-19，下载与数据准备、六模型留出评价EVAL6、传统基线BASE2和四档STFT点数消融NFFT4均已验收。当前没有执行中的DSH任务，也没有自动进入噪声方式实验或第二阶段。下一阶段须由用户决定；所有定时唤醒继续保持取消。

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

本次已派发并确认running：work/deepseek-runs/20260918-134154-1d5b2b1d，session-e063be94-0447-4012-b6f6-bcc8bebced11，external aa_821c4d8afc4c5d03_session-e063be94-0447-4012-b6f6-bcc8bebced11。DSH Desktop自动标题：Codex delegated task: DroneRFa：六模，cwd为本项目。普通DeepSeek flash/max、standard、workspace-write配置继承不变。当前reviewRequired=true，不能重复提交。

2026-09-18 15:55恢复：原评价任务已停止（中断后没有完整评价产物，reviewRequired保持true）。Codex已对evaluate_heldout_models.py作必要接口修正：checkpoint训练索引hash对原preserved inputs中的window_index.csv，新heldout索引hash独立对config；真实六权重核对无errors。GPU可识别RTX4060Laptop。F盘首908MB MAT全SHA256核对25.25秒且匹配，表明此前没有及时输出至少与逐文件全量hash读取慢有关，不能宣称整个执行器或GPU故障。普通实现/测试/正式评价将继续交同一DeepSeek任务。全量校验须显示逐文件进度，避免重复大数据冒烟。当前代码root修正只限索引接口，原14冻结输入不改。

当前任务新增唯一交接文件work/deepseek-handoffs/20260918-134154-1d5b2b1d.json，已登记原run metadata，不变更会话/transport。原生heartbeat dronerfa此前每2分钟运行，现已按用户取消要求通过原生工具删除，不再定时唤醒；具体owner已从真实原生配置target_thread_id核对，create/view工具不返回owner字段，已注明来源。work/deepseek-continuation.json限定仅EVAL6。首轮原生heartbeat已于2026-09-18T08:02:26.007Z实际触发；自动检查单独记账于work/heldout_eval_20260918/heartbeat-observations.jsonl，不把前台工作当自动执行。当前用户已取消定时唤醒，不能自动恢复，不能扩大阶段。
正式评价已确认启动：原DSH任务继续running，full_run.log已核对6个新MAT完整hash，观察到主Python26192运行；完整预测/汇总尚未完成，reviewRequired保持true。已将两个启动尝试的精确进程观察发回执行器核实，禁止按名称结束进程。正式执行日志work/heldout_eval_20260918/full_run.log；Codex核心修正检查证据work/heldout_eval_20260918/codex-core-fix-verification.json。下一自动轮次先查原会话与实际产物，不重复派发。

16:03首轮心跳实际核对：12个新MAT完整SHA256已通过；当前evaluation_manifest明确truncated_debug_run=true，仅clean条件8窗×6模型=48条预测。这是小规模调试产物，完整43680条/288逐文件/24模型条件统计尚未完成。原DSH会话仍running、coordinationAction=wait，保持待验收，不新派任务、不并行改执行器代码，下一心跳继续查询同一任务。


2026-09-18 16:09 第二次真实心跳（计划时间08:06:26.026Z）：原EVAL6会话仍running/wait，无交接或待批准。实际工具记录已启动无max-windows/conditions限制的正式调用，日志核对完整1820窗及六个冻结epoch，进入clean共享float32特征缓存（观察100/1820）。当前结果目录仍为上一轮8窗clean调试的48条预测，不能据此验收。保持reviewRequired=true，等待原执行器完成；未重新派发、未修改评价代码。
当前调度规则（2026-09-18用户明确确认）：已停止自动唤醒。原生工具已删除旧dronerfa心跳，continuation.enabled=false/status=stopped_by_user；后续普通继续或新目标都不能自动恢复任何定时器，需要用户新的明确授权。此次只停止定时唤醒，没有中断原DSH评价会话、修改评价代码或标记EVAL6通过，activeRunDirectory及reviewRequired=true继续保留。Harness向本Codex桌面任务主动发消息并启动下一轮的入口尚未验证，交接文件本身不能唤醒Codex；后续前台恢复时查询原会话及实际交接，禁止重复派发。

2026-09-18 16:36 前台继续检查：首轮正式推理16:26结束，完整43680预测、288逐文件、24模型条件已生成并保存在heldout_evaluation_20260918/run1_evidence等产物；不是48行调试。Codex独立CSV检查原始索引身份、24组合全部1820行顺序/覆盖、有限logits/阈值、逐文件混淆矩阵及文件等权和pooled指标回算、14冻结原输入SHA256均通过。配对CSV存在30行/15唯一键（每个种子重复两遍），已通过同一桌面会话发回修正，不改变冻结设计或重复创建任务。
首轮独立验证未通过：每档window_digests仅32字节而非1820*32，抽样logits最大差0.00390625超过1e-4但抽样分类标签未变。Codex经公开入口确认执行器停止后修正必要核心摘要记录：逐窗SHA256原始32字节串接，另留总摘要；原索引接口修正保留，14原文件不动。已核实并停止仍执行旧错误逻辑的第二轮Python24168，其启动器19576随之退出，不按名称杀其他进程。执行器被已接收的原会话反馈恢复，目前第二轮实际full_run2.log逐文件hash进展，不能同时修改其代码。Codex独立原批次边界数值复核正在运行，证据将写work/heldout_eval_20260918/codex-batch-parity-probe.json。EVAL6保持reviewRequired=true、activeRunDirectory不清空；第二轮、配对修正、真实IQ/精度验证须通过后再验收。所有旧timer保持删除，Harness主动通知入口仍未验证。

16:39核心复核：codex-digest-regression-probe.json五窗（含末尾不满批次）逐窗160字节、每窗hash、concat hash、完整feature hash均通过。codex-batch-parity-probe.json独立raw IQ实测2028 A clean的36目标：原全局边界batch4重算logits与首轮CSV差0/分类差0；batch36前端11窗SHA不同、最大feature差9.5367431640625e-07；保留原batch4特征重新分组做网络则logits仍差0。这证实验证应沿正式batch4边界重建前端/AMP推理，不能放宽1e-4容差。已按同一公开会话回传方案及证据，须继续所有6模型4档36目标实际复核。首轮失败报告已保留work/heldout_eval_20260918/independent-verification-first-failed.json；核心修正检查codex-digest-core-fix-verification.json。PREP12三项记账实际核对通过，summary哈希更新匹配。最终仍未验收。

16:54实际进展：同一原EVAL6会话在16:43通过公开入口恢复并处理反馈，summarize配对循环已按3个唯一种子修正；verify已改原全局batch4边界重建共享特征、保留1e-4严格容差、去掉冗余CPU整网，加入15唯一完整配对键检查；compare_evaluation_runs.py补两轮证据比较。尚未重生成最终汇总，原30条重复配对产物不能验收。第二轮full_run2.log clean/-10六模型推理已完成，两档完整feature SHA与首轮完全一致；各逐窗摘要58240bytes正确。当前-12特征800/1820观察，之后-15、回算/图/报告、全部模型条件原批次真实IQ复核仍待完成。Root26必要用例全部通过，证据codex-required-tests-verification.json；PREP12逐文件140M/150M/160M/180M记账及summary哈希实际核对通过，证据codex-prep-record-corrections-review.json。前台继续等待原会话，禁止新建重复任务和任何定时唤醒。

2026-09-18 17:10 EVAL6 已由 Codex 验收通过。DeepSeek 交接文件为 work/deepseek-handoffs/20260918-134154-1d5b2b1d.json，执行器已停止。最终产物 dronerfa_baseline/outputs/heldout_evaluation_20260918/：43,680 条窗口预测、288 条逐文件、24 条模型条件、15 条唯一配对；run_parity.json 显示两轮四档 feature SHA、预测 CSV（SHA256 3e7be3f9…）和决策完全一致；independent_verification.json 的 15 项检查全部通过，四档每档 36 个固定目标窗按原 batch-4 重建无摘要不符、同窗噪声重读一致、6×4 AMP logits 与 CSV 最大差0、阈值/有限性/汇总回算无违规。Codex 独立 CSV 回算、14 个冻结输入哈希和 26 项拒绝/聚合/阈值用例也通过。最终文件等权四档均值 BA：2026 A/B 96.41%/99.28%，2027 99.02%/99.07%，2028 98.79%/99.40%；B-A 分别+2.87/+0.05/+0.62个百分点，仅作描述性结果；-15dB 三个种子均为正，其余条件有负值。首轮摘要写入缺陷和错误批次抽样已留在 run1_evidence 与 independent-verification-first-failed.json，第二轮修正后通过。PREP12 三项记账修正已核对，12 文件仍在 F:/DroneRFa数据集/heldout_20260917。EVAL6 完成后不自动进入第二阶段，也不恢复定时唤醒；如要进入后续阶段需用户另行明确要求并重新规划。

2026-09-19 用户新授权：噪声方式实验暂不做；先做两个传统对照（能量阈值、PSD+SVM），验收后再比较 n_fft=512/1024/2048/4096，统一采用50% hop。用户已在开始前获知：STFT训练输入动态在内存/GPU生成，不批量保存逐窗PNG；仅NFFT4保存同一代表窗的原生STFT和实际256×256模型输入对比图。代码保存在 dronerfa_baseline/，结果统一保存在 dronerfa_baseline/outputs/stage1_method_comparison_20260919/。BASE2只做clean，旧train/val用于拟合/选择，旧test和新heldout用于冻结评价；新heldout不得参与阈值、标准化或SVM超参数选择。任务文件：work/deepseek-tasks/20260919-stage1-classical-baselines.md。NFFT4必须等待BASE2由Codex验收，不恢复任何定时唤醒。

2026-09-19 BASE2 已由 Codex 验收。代码：dronerfa_baseline/run_classical_baselines.py、test_stage1_method_comparison.py；产物：dronerfa_baseline/outputs/stage1_method_comparison_20260919/baselines/。能量阈值仅由旧 train+val 选为 -47.336378dB，旧 test/heldout 文件等权 BA 为96.96%/98.30%。PSD 采用RF0去均值、RMS归一化、周期Hamming、n_fft1024/hop512、双边功率时间平均和fftshift，StandardScaler只拟合旧train，18个SVM候选只用旧val选中linear-C0.01；旧test/heldout文件等权BA为98.21%/100%。20项单元测试、38项产物校验、执行器独立回算和Codex 27项独立复算通过，原14冻结输入hash不变。heldout仅12个源文件；100%不能称跨会话或跨场景泛化。项目.venv缺sklearn，SVM阶段显式从本机D:/anaconda/Lib/site-packages导入sklearn1.7.2/joblib1.5.2，来源已记manifest。

下一项 NFFT4 已冻结于 work/deepseek-tasks/20260919-stage1-nfft-comparison.md：n_fft 512/1024/2048/4096，hop各为一半，image_size256；种子2026/2027/2028，5轮，固定A_original训练SNR与验证/留出噪声配对；同种子四组初权重和训练顺序一致。只比较STFT点数，不比较噪声生成方式。主评价为新heldout四档，补充旧test clean；保存固定代表窗的原生STFT和实际256x256输入图，不保存逐窗PNG。所有timer继续保持取消。

2026-09-19 NFFT4 已由 Codex 验收通过。执行器已停止，交接为 work/deepseek-handoffs/20260919-121436-0fa1a0c1.json；产物位于 dronerfa_baseline/outputs/stage1_method_comparison_20260919/nfft/。12个模型全部就位：1024的三个旧A模型经完整等价性验证复用，其余9个重训5轮。heldout 87,360条预测、旧test clean 5,280条预测均主键唯一且覆盖完整。文件等权四条件BA跨种子均值：512为99.04%±0.21pp、1024为98.07%±1.44pp、2048为97.40%±1.00pp、4096为97.98%±0.96pp；512相对1024的三个配对差值+2.41/+0.02/+0.46pp，方向一致，但n=3且窗口/条件相关，只作描述性结果。Codex独立复跑29项测试、验证器、CSV主键/行数及均值/样本标准差回算，全部通过；两张代表图可读，14个冻结输入不变。验收证据：work/nfft4-codex-acceptance-20260919.json。未开始噪声方式对照、第二阶段、定时器、提交或发布。

2026-09-19 用户授权开始第二阶段第一步 STAGE2-BASE。任务冻结于 work/deepseek-tasks/20260919-stage2-model-classification.md：只保留 Phantom(0010) 与 Air2S(0101)，重映射为0/1；旧 train/val/test 为620/290/300窗，新heldout 1260窗。比较 n_fft512/1024，三个训练种子、5轮、原A噪声协议；模型从头训练，同seed两档共享初始权重和训练顺序。每模型只用旧val四条件选epoch，n_fft也只用旧val跨seed均值选择；heldout不参与选择。当前范围仅oracle机型分类，不加入背景、不串联第一阶段、不做噪声方式实验或定时器。
2026-09-19 STAGE2-BASE 已验收，run 20260919-180356-9621af9a 已停止。Codex复跑独立验证器零失败（work/stage2-codex-verification.json），29项测试全部通过；审查标签重映射、验证集选epoch/n_fft与选择锁定逻辑，两张代表图可读。六模型完成，heldout30240预测、test1800预测；旧val选择512（99.2619% vs 98.8135%），新heldout四条件文件等权BA为512 99.78%±0.09pp、1024 99.46%±0.36pp。旧test的512 seed2027为96.79%，保留波动。14冻结输入不变。当前仅完成已知无人机输入的两机型分类；级联三分类尚未开始，噪声方式实验和定时器均未恢复。

2026-09-19 用户授权开始 CASCADE3，直接连接已验收的Stage1与Stage2。冻结配置 `work/cascade3-config-20260919.json`，任务 `work/deepseek-tasks/20260919-cascade3-evaluation.md`。使用n_fft512/hop256，2026/2027/2028同seed配对；每个完整split行号按seed22026生成一次噪声和同一STFT张量，Stage1判Drone后才将原张量交给Stage2。最终三类为Background、Phantom 4 Pro、Air 2S。正式规模heldout 21840行、旧test clean 1320行；不重训、不调阈值、不连接旧预测CSV、不做未知类/噪声方式实验/任意信号UI，不恢复定时器。Stage1和Stage2的512配置均由旧validation选择，新heldout只作描述性评价。

CASCADE3 已派发到 DSH Desktop：run `work/deepseek-runs/20260919-225455-88f703c3`，session `session-cac1ac4a-1ba8-4b24-9735-56311199d9da`，external `aa_821c4d8afc4c5d03_session-cac1ac4a-1ba8-4b24-9735-56311199d9da`。22:55状态为running、coordinationAction=wait、reviewRequired=true，桌面标题 `Codex delegated task: DroneRFa：三级`。不得重复派发；完成后检查同一run与交接文件再验收。定时器保持取消。

2026-09-20 首次 CASCADE3 验收未通过。原执行器已 completed/idle，正式23,160行预测及静态身份、决策规则、汇总回算、冻结哈希、Stage1发布结果parity均通过；但Codex在全新普通Python进程运行 `verify_cascade3_evaluation.py --no-write --report work/cascade3-codex-verification.json` 得到passed=false：目标窗重建255个feature digest不符，Stage1/Stage2最大logit差0.00252485/0.007019997，rule mismatch合计242。不能用原交接中同进程/旧日志的零差替代独立复验，也不能直接标记完成。返工任务 `work/deepseek-tasks/20260920-cascade3-rework1.md` 要求查明GPU STFT/AMP跨进程复现差异、拆分数值误差与真实决策变化、加入原始输入精确证据，并让两个全新进程直接验证都通过。正式预测先保持不变；定时器仍取消。

返工1已派发并确认running：run `work/deepseek-runs/20260920-105246-475b889f`，session `session-d1a06e50-6be6-4dd9-8db2-179c39b7528a`，external `aa_821c4d8afc4c5d03_session-d1a06e50-6be6-4dd9-8db2-179c39b7528a`，桌面标题仍显示 `Codex delegated task: DroneRFa：三级`。reviewRequired=true；不得重复派发，等待本run完成后复验。定时器保持取消。

返工1核心问题已解决并由Codex复验：根因是上次误用PATH中的D:/anaconda CPU PyTorch，而正式推理用项目.venv的CUDA PyTorch；旧验证器静默回退CPU并把数值漂移混称rule mismatch。最终验证器记录环境并拆分计数。Codex用 `.venv/Scripts/python.exe` 复跑得到 work/cascade3-codex-verification-after.json：passed=true、formal_environment、156目标窗输入/feature/logits零差、执行门/机型/batch错误全0；48项测试通过，window_predictions SHA256仍为59321de7...。视觉抽查另发现 recall_by_condition.png 的 heldout四档点存在但横轴只显示clean，原因是old_test单条件在循环末覆盖xticks。返工2仅修图和清单哈希，不重跑推理，任务 `work/deepseek-tasks/20260920-cascade3-rework2-figure.md`。定时器保持取消。
