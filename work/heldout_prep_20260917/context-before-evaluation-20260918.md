# DroneRFa 项目共享摘要

实际项目：C:/Users/Administrator/Desktop/数据集处理。用户授权继续下一项，按 DeepSeek 网页工作流执行；Codex 负责范围、规则和验收，Harness 负责具体实现及运行。git status 未识别为仓库，Harness目录清单显示有.git目录，未确认其有效性；本次未找到适用 AGENTS.md。不要提交、推送。

## 目标与已知状态

两阶段主线不变：第一阶段存在性检测（背景/无人机），第二阶段机型分类。当前第一阶段已完成基线、验证集门槛诊断、Air 2S 窗口和信号特征分析，以及三个训练种子的 A/B 对照。第二阶段及串联评价尚未完成。

原数据只读位置：C:/Users/Administrator/Documents/DroneRFa数据集。12 个 MAT 文件：T0000 背景、T0010 Phantom 4 Pro、T0101 Air 2S，各 S0000/S0001/S0010/S0011，均 D00。RF0 I/Q，100 MS/s，10 ms=1,000,000 点，每个 10,000,000 点采集块内切窗，不跨块。训练920、验证450、测试440窗；三个集合不共享源文件，跨会话独立性尚无元数据证明。

前端：去均值/RMS归一化/复AWGN → Hamming STFT n_fft1024 hop512 → log power/频率居中/逐样本标准化 → 1×256×256 → ResNet18背景/无人机。附加噪声 dB 不是现场绝对 SNR。

A噪声档 clean,10,5,0,-5,-10,-12；B增加-15，各档等概率。训练种子2026/2027/2028，各A/B训练5轮；验证噪声种子12026固定。按clean/-10/-12/-15四档平均平衡准确率选模型，同分最早轮次；logit差>0判Drone。

A→B综合结果：2026 95.73%→98.67%；2027 98.10%→98.45%；2028 98.73%→98.09%。B尚未证明稳定提升。累计54000条预测统计、480条原始IQ抽查通过。不要重新挑种子或轮次，不把重复窗口当独立采集。

## 本次下一项与冻结规则

准备尚未使用的12个同距离源文件：T0000/T0010/T0101 × S0100/S0101/S0110/S0111。候选压缩11929877555字节，解压12382967102字节。用户随后明确指定所有下载内容放F盘，F盘可用512075104256字节（约477GiB）；Codex已创建F:/DroneRFa数据集/heldout_20260917及其中downloads目录。采用逐文件下载/校验/解压，校验成功后仅清理本次生成的临时压缩包，保留日志及断点信息，保持至少5GiB空闲。

新MAT和本次临时RAR、.part、断点缓存只写入F:/DroneRFa数据集/heldout_20260917/，临时包位于其中downloads/。代码、索引、核查报告、运行日志仍保存在实际项目内。不得向C或D盘下载大文件，不写原Documents数据目录。当前范围仅下载与必要完整性校验；索引、数据分析和模型评价由用户暂停。用户路径指令已通过同一Harness会话插话送达，覆盖首次派发快照的C盘路径；未创建重复会话。

预先记录的后续评价规则保持：六个所选A/B模型全部评价，原网络/前端/门槛不变；clean/-10/-12/-15；独立AWGN种子22026；所有模型共享相同窗口和噪声实现；主汇总按每机型源文件等权、保留计数及窗口汇总；不利用新数据调整门槛、轮次或噪声训练范围。不同源文件不自动称跨会话泛化。

## 参考文件

- 已完成工作与问题汇总_2026-09-17.md
- 项目现状核对与续做依据_2026-09-17.md（以末尾更新为准）
- dronerfa_baseline/outputs/noise_ab_replicates_20260917/后续留出数据方案.md
- dronerfa_baseline/outputs/noise_ab_replicates_20260917/candidate_heldout_files.json
- dronerfa_manifest.json（ScienceDB非solid RAR清单，有分块offset及大小）
- download_dronerfa_subset.py（现有Range选择下载/RAR校验/解压/断点脚本，可复用，默认输出位置需显式覆盖）
- dronerfa_baseline/prepare_index.py、dronerfa_data.py、audit_dataset.py、model.py（读取语义和前端）

现有 .venv/Scripts/python.exe 可用，已确认 h5py3.16.0/numpy2.5.1。优先使用现有运行时，不创建另一套环境，不安装或启动另一个Harness服务器。Harness health成功，127.0.0.1:3080共有13个可见会话。

## 当前问题及接下来验收

Harness已观察到沙箱内TLS失败，按自身规则申请更广权限后curl返回206、正确档案大小及RAR5签名；http.client后端也有成功记录。12个候选块头已通过核查，见remote_probe.json。采用已经验证工作的curl或严格验证证书的http.client后端，不关闭TLS验证，不再追加无关环境探测。审批说明和回复均用中文。完成下载后检查完整性、MAT字段/块数、旧/新文件内容重复及RF0活动频段。频段仅按实际可读元数据与固定信号统计判断；不要依据模型表现删除困难文件。

Codex随后检查实际新增文件、数据完整性、冻结规则和必要验证，更新此摘要及work/deepseek-workflow.json。Harness不得编辑本摘要或该workflow状态，不自动提交、推送或发布。

## 用户最新优先级（2026-09-17，覆盖上述执行范围）

先暂停其他工作，先下载数据。本轮仅完成12个候选MAT文件下载到F:/DroneRFa数据集/heldout_20260917，所有大文件和临时下载缓存放其中downloads/。保留断点恢复、RAR完整性及解压文件大小校验。暂缓索引、频段、重复内容分析、模型评价、额外测试和环境优化。远端12个候选块头及RAR签名核查已通过；F盘第一个.part已产生，实际下载已启动。此前requests安装审批记录已过时，现有真实requests可复用。其他计划等待用户恢复后再执行。

下载状态（2026-09-17最终）：用户在Harness要求暂停至次日。旧下载PID4012和curl已结束，首文件.part冻结603979792字节，state current_block_bytes=603979776，单文件69.37%。暂停记录见work/heldout_prep_20260917/下载暂停记录_2026-09-17.md。


## 2026-09-18恢复下载

用户明确要求继续昨天工作。Codex08:40恢复同一run/session并核对F盘首文件576MiB断点一致，旧PID不存在且无curl。Harness健康、原会话空闲未验收；08:41已在原网页会话要求只续传12文件，其他工作仍暂停。沿用all/4workers/16MiB和F盘原目录，今天用独立日志，保存PID及退出状态；不重复提交会话、不重装环境、不复跑已通过检查。等待实际启动与增长核查。


08:45首文件完整下载并解压成功，908097315字节、四通道140000000采样。脚本两处source_samples==unpacked_size//4错误地从HDF5字节数推算采样数，误报失败；Harness已修复，Codex真实文件校验existing_mat_is_valid=True。旧PID18080仍在下载第2文件，旧代码尚未重启。Harness待批准命令误用按名称结束全部curl，可能影响无关下载；Codex提醒用户先拒绝，自动审批拒绝代点拒绝（技能要求审批交用户），网页审批期间无消息框，更正指令未送达。后续精确PID修复步骤保存在work/deepseek-tasks/20260918-download-validation-restart-fix.md，用户处理后继续同一会话，不创建新任务。尚未验收完成，其他分析仍暂停。

08:56实际状态覆盖前述等待审批记录：用户已在Harness允许进程重启命令及后台启动，审批已完成。旧PID18080结束，修复版PID10816运行。下载记录第1文件skipped_existing（908097315字节、140000000采样），第2文件completed（907985834字节、140000000采样、SHA256 c2839e75850a5081dfb61f060ab8b3eeb2dbdc756ce4f201e6039bff9cf68248），08:55开始第3文件。新日志F:/DroneRFa数据集/heldout_20260917/downloads/download_heldout_20260918_resume.log；错误日志为空。下载未全部完成，不清空原run，不派发索引/分析。用户提出将自动审批审查改为同意；Codex未修改平台判定或权限配置，实际下载当前无待审批阻塞。

2026-09-18约11:00检查实际下载：9/12 MAT已落位，9个大小全部匹配候选manifest，合计9041885393字节（8.42GiB）。第10个T0101_D00_S0101.mat的.part=939524112，state current_block_bytes=939524096，干净断点79.40%；总压缩下载进度80.86%，剩余2283755240字节（2.13GiB）。最近r2日志末写10:56:34；PID20964不存在，无curl，错误日志为空，退出原因未知。当前并未下载中，已更正workflow。技能接入更新为DSH Desktop；旧启动器status连接失败，默认desktop health正常。通过公开session.list确认同一externalSessionId对应桌面sess_dsh_5efa784d1fab1faeae088d6e，getState idle、cwd本项目、workspace-write，近期snapshot保存在原run/desktop-progress-20260918.json。没有重复派发或改写旧运行记录transport，没有恢复下载。用户询问为何侧边未打开DSH：已说明桌面版是独立窗口，接口读取不会自动打开Codex右侧网页。其他分析仍暂停。

用户明确要求继续下载。已核实安装插件的公开session.startTurn支持已有会话续聊；在DSH Desktop公开映射的同一原任务sess_dsh_5efa784d1fab1faeae088d6e/externalSessionId session-bf6925d4-73d7-401e-aafb-3a616b52d658派发最后3文件续传，返回accepted，未新建任务、未变更配置。任务与回执见work/deepseek-tasks/20260918-resume-final-three-downloads.md及原run/desktop-resume-request-20260918.json。仅处理T0101_D00_S0101/S0110/S0111，F盘原目录，其他分析暂停。需要查原桌面任务状态与实际进程，不能以accepted宣称已开始。

11:04已确认实际续传恢复：Harness原任务r3启动已获allowed-once；PID17312存活，第10文件断点新增67108864字节（64MiB）至1006632960（85.07%），curl有传输，F盘scratch持续增长，错误日志为空。已更新workflow为downloading，保留原任务待验收，其他分析仍暂停。

11:09检查：第10个T0101_D00_S0101.mat于11:07:18完成，1226409950字节、四通道各180000000采样，SHA256 6967c201a2c8a95464274f5e746fe9fdeacc613eae619b7dd155c960e81f934b；download/rar_structure/unrar_test/extract/hdf5_read全部passed。实际10个MAT大小匹配候选manifest。PID17312继续第11个T0101_D00_S0110.mat，自动随后下载第12个T0101_D00_S0111.mat；错误日志为空，其他工作暂停。整个12文件下载尚未完成，不清空activeRunDirectory或reviewRequired。
