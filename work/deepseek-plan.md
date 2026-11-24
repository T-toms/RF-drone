# DroneRFa 后续工作计划

用户于 2026-09-18 明确要求下载完成后进入下一步。沿用 deepseek-workflow，串行验收和派发。

| 标识 | 目标 | 依赖 | 状态 | 实际依据 |
|---|---|---|---|---|
| PREP12 | 12 文件下载、结构/重复/频谱核查及固定索引 | 无 | 已验收，附记录修正 | work/heldout_prep_20260917/codex-acceptance-20260918.json；heldout_preparation_20260917/verification.json |
| EVAL6 | 六个已选检查点在 1820 窗、四档条件上的冻结评价 | PREP12 | 已验收 | work/heldout-eval-config-20260918.json；dronerfa_baseline/outputs/heldout_evaluation_20260918/independent_verification.json；work/deepseek-handoffs/20260918-134154-1d5b2b1d.json |
| BASE2 | clean 条件的能量阈值与 PSD+SVM 两个传统对照 | EVAL6 | 已验收 | baselines/report.md；baselines/verification.json；work/base2-codex-independent-verification-20260919.json；work/deepseek-handoffs/20260919-112858-71655f1d.json |
| NFFT4 | 固定协议下比较 n_fft=512/1024/2048/4096，50% hop | BASE2 | 已验收 | nfft/report.md；nfft/verification.json；work/nfft4-codex-acceptance-20260919.json；work/deepseek-handoffs/20260919-121436-0fa1a0c1.json |
| STAGE2-BASE | Phantom 4 Pro / Air 2S 独立分类；比较 n_fft=512/1024，只用旧 val 选配置 | NFFT4 | 已验收 | work/deepseek-tasks/20260919-stage2-model-classification.md；work/deepseek-runs/20260919-180356-9621af9a |
| CASCADE3 | 第一阶段检测与第二阶段分类串联，评价 Background/Phantom/Air2S | STAGE2-BASE | 已验收 | 23,160条预测；最终CUDA验证0失败，52项执行器测试通过，召回率图四档标签已目视验收 |
| INFER-UI1 | 本地浏览器任意IQ信号分析、可视化和结果导出 | CASCADE3 | 已验收，服务运行中 | run 20260920-133122-fd9d7b9f；work/infer-ui1-codex-acceptance-20260920.json；http://127.0.0.1:8765/ |
| LAUNCHER1 | Windows桌面一键打开信号分析界面 | INFER-UI1 | 已验收 | 桌面“无人机信号分析.lnk”；work/launcher1-codex-acceptance-20260920.json |
| LIT-AUDIT1 | 至少45篇的前沿文献矩阵、证据审计及与当前实验对齐的路线图 | CASCADE3 | 已验收 | run 20260920-190804-183f7c6f；literature_review_from_scratch/frontier_update_20260920/；work/deepseek-handoffs/20260920-190804-183f7c6f.json |
| REPORT-DOC1 | 三页以内中文进度报告，说明已完成工作和主要结果 | LIT-AUDIT1 | 已验收 | reports/DroneRFa项目进展汇报_结果版_20260921.docx；work/progress-report-codex-acceptance-20260921.json |

说明：旧 `LAUNCHER1` 会话在用户处理审批后又运行了一轮回归，最后记录为 8 通过、1 失败、1 跳过，且未形成新的交接；其对 `open_signal_analyzer.ps1`、`test_open_signal_analyzer.py`、`signal_analyzer_README.md` 的后续修改暂不作为本次文献任务的一部分，也不在本轮宣称为重新验收。

PREP12：实际 12 文件共 12,382,967,102 字节；1820 窗（背景560、Phantom610、Air2S650），无跨块/重叠，与原索引无文件交集。Codex核对实际HDF5长度、固定短切片有限性、扫描覆盖、索引及summary哈希、原14个冻结代码/检查点哈希。未重复全量扫描与MAT哈希。

频谱观察：Phantom 95%占用带宽中位数新6.25MHz、旧24.56MHz，保留为分布差异；固定窗活动均在RF0，缺少中心频率元数据，现有证据不足以判定协议不匹配或设备混入。保留全部预先指定文件，评价采用既定RF0前端。RF1仅已有核查对照，不加入输入或评价档位。不同源文件不证明独立采集会话。

EVAL6允许新增评价脚本、汇总与有意义的失败用例；输出限heldout_evaluation_20260918。可修正PREP12的三项记录表述（总体running、单一140M扫描长度、将r3请求数称整轮）。不改heldout索引、MAT、原代码、训练配置或检查点。验收须有43,680条窗口预测、288条逐文件统计、24条模型/条件汇总、3个种子的配对差值，明确文件等权主指标及窗口汇总补充指标。原代码/权重/索引哈希保持不变。

本步只完成留出评价，不延伸到重训、调门槛、换RF通道、第二阶段分类或发布。

## 2026-09-19 新授权

用户暂不做 AWGN/真实背景等噪声方式对照，先串行执行两个传统基线，再比较四档 STFT 点数。BASE2 将 PSD+SVM 视为一条组合基线，另一条为能量阈值；只做 clean，不把新留出数据用于选择。NFFT4 采用 `hop_length=n_fft/2`，训练时动态计算 STFT，仅保存固定代表窗的原生谱图和实际 256×256 模型输入对比图，不批量落盘逐窗 PNG。程序位于 `dronerfa_baseline/`，产物统一位于 `dronerfa_baseline/outputs/stage1_method_comparison_20260919/`。

## BASE2 验收结果（2026-09-19）

两条基线均完成旧 test 440 窗和新 heldout 1820 窗评价。能量阈值只用旧 train+val 选择 `-47.336378 dB`：旧 test 文件等权 BA 96.96%，heldout 98.30%。PSD+SVM 的 StandardScaler 只拟合旧 train，18 个预声明候选只用旧 val 选择，最终为 `linear, C=0.01`：旧 test 文件等权 BA 98.21%，heldout 100.00%。20 项单元测试、执行器 38 项核查、独立输出复算以及 Codex 27 项独立核查全部通过；14 个冻结输入哈希不变。100% 仅来自 12 个新源文件，不解释为跨会话泛化。

## NFFT4 验收结果（2026-09-19）

四档 `n_fft=512/1024/2048/4096`、三个训练种子、四个 heldout 条件均完成。共 87,360 条 heldout 预测和 5,280 条旧 test clean 预测，主键全部唯一且每组覆盖完整。文件等权四条件 BA 跨种子均值分别为 99.04%、98.07%、97.40%、97.98%；512 相对 1024 的同种子差值为 +2.41、+0.02、+0.46 个百分点，三次同向，但只作描述性结论。Codex 独立运行 29 项测试、输出验证和直接 CSV 回算均通过，14 个冻结输入哈希不变；验收证据见 `work/nfft4-codex-acceptance-20260919.json`。当前没有自动进入噪声方式实验或第二阶段。

## STAGE2-BASE 冻结设计（2026-09-19）

用户已授权开始第二阶段第一步。只使用真实无人机窗训练 Phantom/Air2S 二分类器；旧 train/val/test 分别为620/290/300窗，新heldout为1260窗。比较 n_fft=512/hop256 与1024/hop512，三个种子、5轮、A_original训练噪声与四档验证协议不变；每个模型用旧val选epoch，并只根据旧val跨种子结果选n_fft。新heldout不参与任何选择。先完成oracle机型分类，不串联第一阶段，不加入背景，不恢复噪声方式实验。


2026-09-19 STAGE2-BASE 已验收，run 20260919-180356-9621af9a 已停止。Codex复跑独立验证器零失败（work/stage2-codex-verification.json），29项测试全部通过；审查标签重映射、验证集选epoch/n_fft与选择锁定逻辑，两张代表图可读。六模型完成，heldout30240预测、test1800预测；旧val选择512（99.2619% vs 98.8135%），新heldout四条件文件等权BA为512 99.78%±0.09pp、1024 99.46%±0.36pp。旧test的512 seed2027为96.79%，保留波动。14冻结输入不变。当前仅完成已知无人机输入的两机型分类；级联三分类尚未开始，噪声方式实验和定时器均未恢复。

## CASCADE3 冻结设计（2026-09-19）

用户已授权直接串联第一和第二阶段。采用两个阶段在旧validation均优选的n_fft512；三组同seed模型配对，不集成、不重训、不调阈值。每个完整split行号只生成一次噪声与STFT张量：Stage1 margin>0才执行Stage2，且Stage2使用完全相同张量；最终输出Background/Phantom/Air2S。heldout为1820窗×4条件×3种子=21840行，旧test clean为440×3=1320行。heldout曾用于此前描述性消融，不声称全新最终测试。所有定时器保持取消。

## CASCADE3 验收结果（2026-09-20）

三级串联已验收：heldout 21,840行、旧test clean 1,320行，共23,160条预测。heldout文件等权准确率三seed均值：clean 99.75%、-10dB 99.98%、-12dB 99.63%、-15dB 95.14%，四档均值98.62%；旧test clean 98.97%。-15dB主要错误是Air2S被第一阶段漏检为Background。Codex最终用项目.venv/CUDA复跑独立验证器，156个目标窗的输入、feature摘要和两阶段logits零差，0失败；图表四档标签目视通过。执行器52项测试通过，正式预测SHA256为59321de7be48aae7d8b135f258aea5287f7e853d3d0d0a726ccfba8bc313f052。两轮返工只修验证环境识别和图表刻度，未重跑正式推理、未改预测或阈值。结果为描述性留出评价，不声称全新未触碰最终测试或跨会话泛化。

## INFER-UI1 冻结设计（2026-09-20）

用户授权开发任意信号分析界面，并要求先报告方案、再用deepseek-workflow开始。界面采用127.0.0.1:8765的标准库本地HTTP服务与离线HTML/JS，不安装新UI框架。支持MAT v7.3、NPY、NPZ和显式参数RAW；只在确认100MS/s、至少1,000,000复数点时启用分类，不重采样、不补零、不加人工噪声。复用三套已验收级联并对三seed最终标签多数决，1-1-1显示不确定；文件级主导结论按超过50%窗口规则，同时展示任何无人机窗和比例。默认输出F:/DroneRFa数据集/analysis_outputs，代码留在dronerfa_baseline。任务与验收见 work/deepseek-tasks/20260920-arbitrary-signal-analyzer-ui.md。

## INFER-UI1 验收结果（2026-09-20）

任意IQ信号分析界面已验收并运行在 `http://127.0.0.1:8765/`。Codex用项目`.venv`复跑完整验证器为`VERIFICATION PASSED`，单独复跑19项测试全部通过，并在可见浏览器中对 `F:/DroneRFa数据集/original/T0010_D00_S0000.mat` 做真实端到端验收：识别为MAT v7.3、HDF5用户块偏移512、RF0/RF1各140,000,000点；前两个1,000,000点窗口在RTX 4060上0.83秒完成，均为DJI Phantom 4 Pro，三个种子均3/3一致。界面显示I/Q、幅度、STFT、逐种子margin和逐窗表；8项产物已写入 `F:/DroneRFa数据集/analysis_outputs/20260920-140719_T0010_D00_S0000`。工作流验收证据为 `work/infer-ui1-codex-acceptance-20260920.json`；冻结CASCADE3预测、权重、索引和配置未改，定时器仍保持取消。

## SELECT12-G3 验收与 META-SPLIT 前置（2026-09-25）

G 盘 104/104 个目标 MAT 通过结构和全量 SHA256 检查；正式 `G:\DroneRFa数据集\dataset_manifest.csv` 现为 104 行（新增 80 行 unassigned，旧 24 行历史用途保留），发布哈希见 `work/deepseek-context.md`。42 个已完成 `.restart` 冗余裸块清理完毕，两份旧 `.part` 保留，记录见 `work/sel12g/restart-cleanup-audit-20260925.json`。META-SPLIT 下一项只做来源/采集分组证据审计与划分提案；旧 24 文件已用于开发/评估，不能用其声明全新 final test，未得到额外未触碰数据前不冻结覆盖全部 13 类的 final test。

## META-SPLIT-1 验收与 META-HEADER1（2026-09-25）

`work/meta_split1/` 给出 104 文件时间和来源表：13 类各聚于一个短文件时间簇、两个夜，旧 24 文件已暴露，新 80 文件未见于原模型/数据产物；104 整文件 SHA256 和抽样片段无完全重复。Codex 接受时间簇作批次代理，但 MATLAB Created on 与 RAR HTIME 很可能同源于原文件时间，不能宣称独立采集日志或会话 ID。当前 104 文件不能构建 13 类批次独立或全新未参与选型的 final test。下一项 META-HEADER1 仅对旧三类的 88 个归档候选读取 ≤128 字节成员头并核对日期，无载荷下载；之后再决定最小新文件补充范围。夜级未见类开放集实验暂缓，任何文件级划分只称同批次文件留出。

## META-HEADER1 验收与 META-SPLIT2（2026-09-26）

88/88 归档头安全只读校验通过，总传输 6,812 字节。只有 T0000 有不同文件日期候选，而且同时换频段；T0010/T0101 的其他成员与现有文件同日、时间连续，归档不能提供旧三类全覆盖的新批次测试。决定不下载候选、不再花数小时探测其他十类的头；当前推进 13 类开发用途文件级分组，旧三类已暴露，结果不得叫全新 final 或 cross-session。META-SPLIT2 生成 104 文件候选划分及源文件零交叉验证，不训练、不改 G 正式清单。

## META-SPLIT2 验收（2026-09-26）

DSH run `20260926-105955-f160a628` 已停止并交接 completed。Codex 复跑 `verify_split_candidate.py` 为 21/21 通过、`negative_control.py` 为 13/13 通过；候选 CSV 为 104 行、62 train/16 val/20 development_test/6 development_test_exposed，SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`；G 清单哈希不变。通过并按 `work/meta_split2/CODEX-FREEZE-20260926.json` 正式冻结为开发划分，不改 G 清单。旧 source_table 的 `prior_use_role` 命名冲突以冻结表中的分列语义解决，旧审计表不改写。未来训练任务必须读取冻结 CSV 并确保源 MAT 的所有窗口同 split；新十类 test 是同时间簇文件级开发主指标，旧三类 test 是已暴露对照，不存在 13 类独立最终测试。当前无已派发下一任务。

## CASCADE3-NEWSPLIT（2026-09-26 用户授权）

1. A：从 META-SPLIT2 冻结 CSV 生成三类 24 MAT 的源文件级窗口索引，核验 4/2/2 每类角色、G 路径和零交叉。任务 `work/deepseek-tasks/20260926-cascade3-newsplit-index.md`；进行中。
2. B：依 A 通过后，复用旧 512/256 两阶段架构与三个种子，在新 train/val 上重训，只凭 val 选 epoch；输出独立检查点。
3. C：依 B 通过后，在新 `development_test_exposed` 上执行两阶段串联并报告文件等权/混淆矩阵及来源限制。不得称 fresh final。

## STAGE1-12DRONES（2026-09-26 用户纠正并授权）

- `STAGE1-12DRONES-A` 已于 2026-09-27 验收：依冻结 104 MAT 划分建全量窗口索引，并训练一阶段 Background vs 12 种 Drone，seed 2026、5 epochs、只用 train/val 选 checkpoint。任务书 `work/deepseek-tasks/20260926-stage1-12drones-background.md`。
- 允许写入新脚本与 `dronerfa_baseline/outputs/stage1_12drones_20260926/`；旧三类索引、训练和评价产物不变。验收需核对索引零跨角色、测试源文件原始 IQ 未读、checkpoint 可重载、验证选型可复算。
- 验收证据：DSH run `20260926-163613-a4ec4a96` completed/idle；索引 15,790 窗，train/val/test/test_exposed 为 9,590/2,360/2,950/890；读取审计 57,390 窗、78 个 train+val 文件，与冻结索引文件集一致，测试文件零打开。最佳 epoch=3，四条件验证 pooled BA 均值 0.900137；checkpoint SHA256 `37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b`，重载通过；独立核验 34/34、负向对照 8/8、单测 39/39。`dronerfa_baseline/outputs/stage1_12drones_20260926/verification.json` 与 `seed_2026/summary.json` 保存详情。此为二分类开发验证指标，不代表 12 机型识别率或独立最终测试。
- 第二阶段 12 机型分类和两阶段串联评价是后续工作；本轮只完成第一阶段。

## STAGE1-12DRONES-DIAG（2026-09-27）

- 用户追问一阶段是否应继续，决定先做已验收 seed 2026 验证结果的逐机型、逐文件、逐噪声与逐 epoch 错误诊断。任务书 `work/deepseek-tasks/20260927-stage1-12drones-diagnostics.md`；DSH run `20260927-135951-7bcd0383` 已完成并于 2026-09-27 验收。
- 诊断只分析既有验证评分和冻结索引，未新训练、未读测试 IQ。20 组指标复算最大差 0；epoch 3 的 −15 dB 漏检 394/2080、−10 dB 背景虚警 60/280；T0100_D00_S0101.mat 在 clean 漏检 39/160。第 4 轮 AUC 升而固定零门限 BA 降，显示门限漂移；第 5 轮 AUC 也下降。DSH 独立核验 18/18、负向对照 10/10、单测 23/23；Codex 抽查第 3 轮四条件 FP/FN、该 T0100 文件漏检及报告 SHA256 与交接一致。报告 `reports/DroneRFa_一阶段12机型验证错误诊断_20260927.md`。
- 下一项先保留固定零门限和原协议，以新训练种子复跑确认稳定性；不依据同一 val 集扫描得到的最佳门限改 checkpoint。第二阶段暂不启动。

## STAGE1-12DRONES-B（2026-09-27）

- 对训练脚本做最小种子参数化，并按原协议全量训练 seed 2027、5 epoch；默认 seed 2026 行为及产物须保持不变。任务书 `work/deepseek-tasks/20260927-stage1-12drones-seed2027.md`；DSH run `20260927-142720-1a3455e4` 已派发，待验收。
- 只在新 `seed_2027/` 写产物，保持验证噪声 seed 12026、固定零门限、四条件 pooled BA 选型规则；不读 26 个测试 MAT IQ，不做第二阶段。

## STAGE1-IO-OPT（2026-09-27 用户授权）

- 用户明确要求“继续优化”GPU 等待数据的问题。任务书 `work/deepseek-tasks/20260927-stage1-loader-optimization.md` 已追加到当前 run `20260927-142720-1a3455e4` 的原生 DSH 会话。
- 当前 epoch checkpoint 缺 optimizer/scaler/RNG，不能无损恢复。保留 seed2027 原训练继续，独立实现多进程读取/有界预取、跨 epoch 随机性与真实 worker 读取审计；正式训练结束后串行执行 0/2/4 workers 的等价性和真实性能对照，再启用验证过的路径。
- Codex 使用已核实的官方 session.interrupt 取消 agent 的 job_output 长等待（不是训练任务），确认训练 PID 21048 及日志继续后，使用 session.startTurn 给同一会话下达优化实施指令。原 queued 消息已取消，替代消息回执保存于原 run 目录。不能重复开启训练或把当前 seed2027 宣称为已优化。

## 2026-09-27 本轮验收与 STAGE1-IO-OPT-R1

- STAGE1-12DRONES-B训练通过：seed2027完整5轮，最佳epoch3 BA0.799845467；Codex独立复算20组指标和逐窗全集/标签通过，读取预算逐文件精确一致；证据 `work/stage1-codex-review-20260927.json`。
- STAGE1-IO-OPT不通过：预取消费坐标错误、并行顶层核验不兼容、smoke训练等价未证明、性能只测inference且报告/CSV不一致、启用线程参数缺失。不得发布17.6%为已验证训练提速。
- 第一轮返工按 `work/deepseek-tasks/20260927-stage1-io-rework1.md`，旧正式训练和其checkpoints保持不变，修审计并补完整训练步等价/真实基准；当前seed2026验证证据文件被覆盖的事实记录为版本替换，禁止再覆写。

## 2026-09-27 进一步审视与下一步顺序（Codex 架构决策）

1. **先收口R1正确性**：当前返工 run `20260927-190015-e514cf21` 运行中。对优化的通过标准是同输入/同协议的完整训练步等价、真实worker审计精确闭合、可重现的训练吞吐，而非GPU利用率必须达到某数值。性能收益不显著时保留可靠串行实现，不因追求满载拖延科研主线。
2. **优先检验损失归一化机制，而非盲跑新seed**：新证据 `work/stage1-batch-balance-diagnostic-20260927.json` 用实际epoch_order哈希逐一对齐两seed十轮，77.06%–77.86%的batch4无背景；默认weighted CE按批内权重和归一化，纯无人机batch类别权重抵消。每步背景损失项系数的均值仅约18.74%–19.32%（不是实际梯度量，也尚未证明是种子波动原因）。下一训练候选只对照当前归一化与固定归一化weighted loss，保留初始化、batch顺序/组成、噪声、BN配置、LR、5轮与验证选型；不能同时改采样/BN/门限/噪声档。该变化也会改变逐批梯度尺度，这是处理本身，不得解释成已单独证明类别均衡因果。
3. **实验前确定比较与停止规则**：先保留两已完成seed为冻结基线，候选用seed2027进行一次配对机制对照。主指标仍四条件固定零门限BA；同时给背景虚警率、无人机漏检率、12机型各自召回、文件等权、最差条件和描述性AUC。同一val反复用于选型的结果只称开发证据；任何阈值扫描不作为新验证成绩。一次对照后冻结阶段1开发版本并保留未解决问题，不设无依据的“必须稳定到某准确率”门槛。
4. **推进完整12机型两阶段目标**：R1和限定对照后建议进入Stage2真实无人机12类训练，先报告已知无人机输入下的分类能力，再做串联，区分Stage1漏检/背景虚警和Stage2错认。用户此前要求先做Stage1；本次计划讨论不等于已派发Stage2或seed2028，实际启动状态需单独记录。
5. **数据边界和可追溯性**：62/16/20/6源文件级划分保持；新20测试文件只有新十种无人机，不能据此评价背景虚警或声称13类全新最终测试；旧6文件为已暴露对照。背景仅4train/2val源文件，不能把相邻窗口当独立采集样本；跨环境/距离泛化需要以后独立新来源。后续新运行必须独立目录、记录代码/配置/数据/检查点哈希和实际启用加载配置，并保存optimizer/scaler/RNG与训练进度以支持真实续训，旧基线不改写。

## 2026-09-27 整体路线已获用户授权（当前执行表）

用户要求指挥DSH按上述路线推进。下表替代历史记录中“仅Stage1”“Stage2未授权”的后续限制；保留每阶段Codex验收门槛，不并发派DSH，不让DSH自行通过依赖。

| ID | 内容 | 依赖 | 当前状态 | 任务书/证据 |
|---|---|---|---|---|
| STAGE1-IO-OPT-R1 | 修真实读取审计、训练等价、完整训练测速 | 旧优化验收失败 | 技术结论已验收，报告证据经R2修正 | 20260927-stage1-io-rework1.md；run 20260927-190015-e514cf21 |
| STAGE1-IO-OPT-R2 | 补内存记录与报告数字 | R1审查 | 已验收，串行配置冻结 | 20260928-stage1-io-rework2.md；work/stage1-io-r2-codex-acceptance-20260928.json |
| STAGE1-LOSS-NORM | seed2027一次5轮损失归一化对照 | R2验收并确认0 worker配置 | DSH已派发、运行中 | 20260927-stage1-loss-normalization.md；run 20260928-133637-058080b6 |
| STAGE1-FREEZE | Codex验收对照并冻结开发模型 | LOSS-NORM结果验收或记录实际技术失败 | 待办 | 依据下述预声明规则写选择锁 |
| STAGE2-12-CONTRACT | 连续12类标签、原窗口坐标、共享噪声输入与有界预检 | STAGE1-FREEZE | 任务书备妥，未派发 | 20260927-stage2-12class-contract.md |
| STAGE2-12-BASE | seed2027/5轮真实无人机十二类训练与val验收 | CONTRACT验收 | 任务书备妥，未派发 | 20260927-stage2-12class-training.md |
| CASCADE12-DEV | 锁定后oracle/串联分角色开发测试 | Stage2验收、两模型锁定 | 任务书备妥，未派发 | 20260927-cascade12-development-evaluation.md |

所有任务书位于 `work/deepseek-tasks/`。路线已通过官方 `session.startTurn` 发送给当前DSH会话并收到accepted，回执 `work/deepseek-runs/20260927-190015-e514cf21/roadmap-instruction-receipt.json`。它只同步后续方向，当前任务范围仍为R1，不代表后续已开始。

### 预声明的模型冻结规则

- Stage1候选集合只含原seed2026、原seed2027、本次global_train_mean seed2027。各模型自身先依原四条件val pooled BA取最佳epoch，再按同一分数选择开发版；完全同分依原2026、原2027、归一化候选顺序优先。候选实际技术失败则记载失败，不能伪造分数。记录seed亦参与选择的开发偏差，保留所有候选，测试结果不得反向改变模型。
- Stage2只一seed2027，512/hop256、batch4、5轮，传统weighted CE（N/(12*n_c)，批内权重和归一化）。按每条件先类内文件等权、再12类等权、再四条件等权的召回选epoch，精确同分取最早。不能把Stage1损失结论直接移植成Stage2最优性结论。
- 测试噪声种子预声明new=22026/exposed=32026，val仍12026，使用含背景完整split位置的固定窗口身份。最终每窗口/条件共享一次输入与前端特征，oracle和cascade的比较不混入不同噪声。
- 不创建新定时器。前台由Codex查状态、验收和派发；前台结束后的交接文件不会主动唤醒Codex，不宣称无人值守阶段流转已实现。

## 2026-09-28 R1 交接审查与决策

R1 run `20260927-190015-e514cf21` completed/idle。Codex读取原始等价JSON、benchmark CSV、两份顶层核验JSON及新旧正式产物哈希清单：同版本2轮24batch的legacy/0/2/4训练轨迹逐位一致；实际并行41/41核验、11/11篡改对照检出；真实训练步0/2/4吞吐8.353/6.567/4.615窗/s，默认线程复测方向相同。**架构决策：后续正式训练保持0 worker串行。** 短样本基准不外推为精确全轮吞吐。

R1尚未完全验收：主基准和默认线程复测三档 `memory` 全是空对象，报告却称记录内存；报告第0节并行smoke写40/40＋10/10，与真实41/41＋11/11不一致；旧两份不同代码版本smoke的差异不能归因于具体原因。限定第二轮返工 `work/deepseek-tasks/20260928-stage1-io-rework2.md`，仅补内存证据、同范围新基准和报告修正。R2通过后再派已备好的损失归一化对照；不得重复R1等价性/核验大项或提前开始新正式训练。

R2于13:16在DSH Desktop启动。13:22审查DSH原生历史发现其按进程名停止PID 13348、18884，两者启动于13:10、早于R2，归属未核实；本地8765现无监听，但不能证明由该命令造成。Codex通过官方session.interrupt纠正并在同一会话恢复，禁止全局进程清理，只允许本轮自己持有句柄的子进程。随后再次收紧内存目标为主进程working set/peak可核验即可，worker进程树非必需；同会话已收到纠正指令。R2须如实交接这项潜在副作用。

13:35 R2完成并经验收通过。DSH交接completed/idle；Codex独立读R2 JSON/CSV确认同runId、0/2/4 workers 8.272/6.568/4.457窗/s、各档内存21个有效父进程样本、每档审计288/288/0；R2证据21/21及两旧正式checkpoint哈希未变。报告已纠正旧空内存、实际41/41＋11/11和旧smoke归因。详见 `work/stage1-io-r2-codex-acceptance-20260928.json`。正式新训练采用0worker并保持旧线程默认，接下来派损失归一化对照。误停两PID事实保留为未证实归属，8765无监听；桌面快捷方式可按需启动服务，正式模型和数据哈希未改。

13:36 `STAGE1-LOSS-NORM` 按预先任务书派到 DSH Desktop 原生会话 `session-3eafeda3-122a-4760-9377-829bb516da2f`，run `20260928-133637-058080b6`；status=running。只允许一次seed2027归一化5轮候选训练，0 worker、旧线程默认，待交接和Codex验收后才冻结Stage1与派Stage2接口任务。
