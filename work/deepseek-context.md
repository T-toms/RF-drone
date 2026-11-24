# DroneRFa 项目共享上下文（Codex维护）

当前有效进度截至 2026-09-29（Asia/Shanghai）。历史完整记录已原样归档 `work/deepseek-context-archive-20260928.md`；详尽旧路线见 `work/deepseek-plan.md`，不能把计划中的“待办”当已执行。

**用户于 2026-09-29 明确要求检查 DSH 结果，此后由 Codex 自己实施，不再使用 deepseek-workflow。** `work/deepseek-workflow.json` 已设为 disabled，旧 DSH run `20260928-133637-058080b6` 已结束并验收；不得自动向 DSH 派发后续任务。

## 目标与协作边界

用户的目标是用现有 **12种无人机＋背景** 数据完成两阶段基线：Stage1 背景/无人机检测，Stage2 已知无人机的12机型识别，然后锁定模型做串联开发评价。旧 Phantom 4 Pro / Air 2S＋背景三类基线已完成但范围过窄，不能代替当前任务。用户已于2026-09-27授权路线：R1/R2加载验收→一次seed2027损失归一化对照→冻结Stage1→十二类Stage2→串联。Stage1 对照已验收并冻结，Stage2 正由 Codex 直接实施。

旧 DSH 协作流程已由用户明确取消用于本项目当前后续工作。用户2026-09-18取消定时唤醒：不得创建heartbeat/automation。不自动提交、推送、发布或改变权限。

## 冻结数据与边界

G正式清单 `G:\DroneRFa数据集\dataset_manifest.csv` SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`，冻结开发划分 `work/meta_split2/split_candidate_104_20260926.csv` SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`，全窗口索引 `dronerfa_baseline/outputs/stage1_12drones_20260926/window_index_stage1_12drones.csv` SHA256 `5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6`。104个MAT，13类各8文件；源文件级62 train/16 val/20 development_test/6 development_test_exposed，索引总数 **15790**（9590/2360/2950/890）；任何任务以冻结索引精确值15790为准。训练/验证仅读对应源文件IQ，测试数据不能参与选型。逐文件清单见 `reports/DroneRFa_当前12机型与背景_距离及文件级划分_20260926.md`。

20个新的开发测试文件只有新十种无人机，无法评价背景误报；6个旧文件为已暴露对照（背景＋旧两机型）。当前源文件处于各类的短时间簇，文件不交叉不等于采集环境独立；不得称13类fresh final、跨采集会话或广距离泛化。九种机型的D00代表论文所列20–40m，T1010/T1011/T1100无距离字段不可推定。背景仅4train/2val源文件。

## Stage1已接受结果及问题

Stage1 RF0 I/Q、1M非重叠窗、512/hop256 Hamming STFT→256×256、单通道ResNet18、batch4、Adam lr1e-3 wd0、FP16 AMP、5epoch、加权CE；训练clean/10/5/0/-5/-10/-12，验证clean/-10/-12/-15，验证噪声seed12026；固定float32无人机与背景logit差>0，按四条件pooled BA均值选epoch（同分最早）。项目 `.venv` Python3.13.9 / torch2.12.1+cu130，GPU RTX4060 Laptop 8GB。

seed2026最佳epoch3、四条件均值BA=0.900137363、checkpoint SHA256 `37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b`。seed2027最佳epoch3、均值BA=0.799845467、checkpoint SHA256 `63e611e797d5e189662bbcb884fa9edc040692b699092e97ba04845e240cfbcc`。两次正式运行已验收、均旧串行0worker；它们只是开发验证，不是12类识别率。见 `reports/DroneRFa_一阶段12机型_seed复跑对照_20260927.md`。新诊断 `work/stage1-batch-balance-diagnostic-20260927.json`：约77% batch4没有背景，weighted CE在纯无人机批次的类别权重抵消；这只是待检验机制，未证明seed波动原因。

## 加载R1/R2验收与当前决策

R1 run `20260927-190015-e514cf21` 与R2 run `20260928-131620-dd6ebd5f` 已停且Codex验收。R1同版本legacy/0/2/4的2epoch、24batch完整训练轨迹逐位一致；并行顶层核验41/41、负向11/11。R2真实训练步有界48窗×2epoch×3轮基准0/2/4 workers为8.272/6.568/4.457窗/s；各配置读取/完成/未消费288/288/0，父进程内存各21次有效采样、CSV与JSON相符、R2证据21/21；旧正式checkpoint哈希未变。**后续正式训练采用num_workers=0，父torch线程保持seed2027原默认（记录为10），不做额外worker搜索。**短基准不等于全9590窗精确吞吐。证据 `work/stage1-io-r2-codex-acceptance-20260928.json`、`reports/DroneRFa_一阶段数据加载优化返工R1_20260927.md`。

R2曾按进程名停止Python PID13348/18884，二者启动于本run前，归属未核实；后来观察8765无监听，但因果未证实。Codex已中断纠正并恢复同会话，禁止全局进程清理，只能处理本轮自己持有句柄的子进程。服务状态仅作观察，不能据此声称被杀进程身份。后续任务不应再使用名称范围停止进程。

## 当前执行及后续任务

DSH `STAGE1-LOSS-NORM` run `20260928-133637-058080b6` 已结束，Codex 查验核验器和三份 checkpoint 哈希后接受。候选 seed2027/global_train_mean 四条件 BA=0.895990728，配对原 seed2027 为0.799845467，但原 seed2026 为0.900137363。候选的背景误报显著下降，同时无人机漏检上升、12机型逐类召回全降；只说明固定零门限操作点移动，不代表检测可分性全面提升。核验器潜在 NameError 已修复，旧结果回归通过；候选 objective_loss.json 预期尾批分母说明字段 float64 2.0000000603 与实际 float32 2.0 有微小差异，不影响训练。按预声明规则在测试 IQ 打开前冻结 **Stage1 原 seed2026 epoch3**，见 `work/stage1-12drones-freeze-20260929.json`。

Codex 已建 `dronerfa_baseline/stage2_12class_data.py`，由冻结 CSV 生成十二机型连续标签视图，train8990/val2080/newtest2950/exposed610，无背景 Stage2 训练。索引与来源哈希、104源文件、15790全窗和4角色计数核对；合同在 `dronerfa_baseline/outputs/stage2_12drones_20260927/contract/`。抽查 train 和 val 加噪窗与 Stage1 同坐标张量逐位相同。`run_stage2_12drones.py` GPU 冒烟通过（24 train/96 val 读取，测试零读），4项定向测试通过。正式 seed2027 五轮训练使用 RTX4060、0 workers、10 torch 线程；用户要求第一轮后暂停重启，Codex 保存完整续训状态再恢复，从第2轮直至第5轮完成。2026-09-29 19:40:35 全部五轮完成，独立核验 `seed_2027/independent_verification.json` passed=true、issues=[]：选中 epoch4，预声明四条件文件/类等权分数0.8504476585；最佳 checkpoint SHA256 `44813b527b3517a8062e25203d4e21c26ca99ae45239679fc1adacea91f2f0e0`。训练实际读取44950窗；中断前/恢复后各重建一次验证缓存，验证物理读取总16640窗；测试 IQ 在锁模前零读取。证据 `work/codex-direct-run-20260929.json`。

2026-09-29 20:21:42 已写 `dronerfa_baseline/outputs/cascade12_development_20260927/selection_lock.json`，Stage1原seed2026 epoch3、Stage2 seed2027 epoch4、映射/索引/代码/噪声种子均锁定；之后两阶段开发评价于21:08:48完成。共15360串联预测、14240无人机oracle记录；独立逐窗复算 `cascade12_development_20260927/independent_verification.json` 通过，12个原始RF0 IQ抽样重放输入摘要与判决也通过。新十类20文件的类/文件等权最终召回 clean/−10/−12/−15 dB 为 **85.05%/77.90%/70.69%/51.48%**；旧暴露三类6文件按条件单独报告。完整表与边界见 `reports/DroneRFa_十二机型两阶段基线_开发评价_20260929.md`。这完成当前已授权的12机型＋背景两阶段基线；不依据开发测试再调参或扩展训练。

Stage2仅一 seed2027/5epoch、512前端、传统加权CE，依四条件类等权且类内文件等权 recall 选 epoch。训练全部完成并核验后，由 `evaluate_cascade12.py --lock-only` 写入不可覆盖模型锁，再执行评价。串联评价20新test2950无人机窗与6旧exposed890窗分开报，背景FPR仅旧暴露有定义。用户目标完成后停止自动扩展实验。


## 2026-10-23 当前任务覆盖旧路线

用户明确要求不用DeepSeek，由Codex直接复核G:/drone，解决Claude遗留问题，并为五条改进建议查文献。原始目录只读；当前没有训练、删除或转换授权目标。新证据见work/audit_drone_20261023；报告见reports/实验室数据复核与改进方案_20261023.md。全目录6629文件，raw除BIN/JSON外还有14MAT/902PNG；data参数仍缺直接证据，不能用raw参数替填。完整IQ数值扫描进行中，发现全零与零前缀，最终结论以data_values.jsonl和values_complete.json为准。未派发DSH任务，不创建定时唤醒。

全量数值复核已完成：3836个去重MAT，15个全零文件，25个含整块零且仍有非零数据的文件，读取错误0。详情见报告第8节；不推断补零原因。仅采集参数等外部事实仍待确认。


2026-10-23用户转述导师补充确认：data以文件名为准，20M是信号带宽20 MHz，信号连续可拼接；combination仍未知。同录制按epoch/seg排序拼接，缺段保留断点，不能据带宽推断fs。证据与逻辑索引见audit_drone_20261023/user_confirmations.json和data_continuous_runs.json。


### 2026-10-23 续查完成

全部890个raw BIN完整读取并核对JSON哈希；1个不一致，1个全零。13个IQ MAT共390M复数点与BIN/32768全量比对已完成。缓存MATLAB解码仅含2440000000键与1024点向量，无data采样参数。25次受零前缀影响录制均为5.8GHz/epoch0。文献适配复核及剩余边界见报告第9节；未训练、未修改原始数据，继续保持DSH与定时唤醒关闭。

### 2026-10-24 已知问题处理与复查完成

已生成data_preprocessing_manifest.csv和data_usable_runs.json：97副本、15全零排除，25零前缀截取，3821文件/100连续区间。raw_preprocessing_manifest.csv排除10副本及1异常，879文件；214个drone配45个背景。14项复查通过，报告重跑不会重复或丢失后续章节。证据见work/audit_drone_20261023/issue_resolution_verification.json；报告第10节。data采样率、精确接收中心频率和combination仍待采集信息。



2026-10-24用户进一步转述导师：data/train和data/test不是已划分好的数据集，应自行划分。source_split仅表示原存放目录；training_split仍unassigned。两目录全部纳入源数据池，尚未创建新的研究划分。


2026-10-24当前研究阶段：用户要求先调研清楚、不着急训练。下一步优先长时间信号摸底，采集参数与背景/干扰来源同步核对，再确定子带、窗长、IQ表示及候选特征。具体完成标准见现有报告第11节；该节为工作安排，长时图谱尚未执行。文件完整性检查已完成，不重复全量扫描。

### 2026-10-24 长信号首轮观察完成

用户要求先放下精确中心频率问题，先观察信号。4次录制已全段读取：009/2.4G/10M、009/2.4G/60M、007/5.8G/10M、F009/2.4G/60M，共157源段、964608000复数点，8图已目视检查、26项核对通过。横轴样点，纵轴归一化频率，未假定fs/fc；缺段原位留空。报告第12节和work/long_signal_pilot_20261024保留图与统计。后续扩展其余机型并对照背景，不训练。参数探测产物保留在work/parameter_investigation_20261024，尚未写入正式采集参数。

### 2026-10-24 当前本地观察脚本

用户需要本地选MAT观察程序，已建signal_observer/和根目录“打开本地信号观察.cmd”。用户反复修正：默认保留原局部图清晰尺度（262144点），通过按钮/时间滑块浏览整段，不将整段挤在主图，也不拖图片。现已增加只读IQ/局部结果缓存与复用绘图对象；同文件预热Agg基准555→191 ms。最新要求为连续播放整个片段，已加播放/暂停、停止回头、速度上限及步长，后台预计算下一视窗，缓冲不跳区间，末尾停止，手动定位暂停。18数值+10播放事件检查通过；两个真实连续MAT全12.288M点按半窗遍历93视窗、2跨界，原始IQ只读。证据output/signal_observer/real_playback_verification.json，操作说明signal_observer/README.md，报告第13节。原生播放按钮操作验收待检查；不宣称已测GUI帧率。保持Codex直接执行、DSH/定时唤醒关闭、不训练。

