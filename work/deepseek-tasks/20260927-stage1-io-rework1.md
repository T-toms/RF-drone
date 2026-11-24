# STAGE1-IO-OPT-R1：Codex 验收返工（第一轮）

## 已接受与保留

原 run `20260927-142720-1a3455e4` 已 completed/idle。seed2027 的完整5轮训练通过 Codex 独立验收：20组验证指标由逐窗评分复算一致、每条件2360窗角色/标签全集正确、逐文件读取预算恰为 train×5+val×4、78个打开文件且测试零读、最佳epoch3/BA0.7998454670、模型哈希63e611e797d5e189662bbcb884fa9edc040692b699092e97ba04845e240cfbcc。不得重训或覆盖 seed2026/2027 产物。

优化尚未验收，失败复现及原始证据在 `work/stage1-codex-review-20260927.py` 和 `.json`。以下问题需实际修复并补证；请不要只改报告数字或绕过断言。

## R1：读取/消费审计坐标与次数错误（P1）

`LoaderAudit.prefetch_report` 用 range(windows_expected) 代替实际 consumption.positions；worker index 是底层 split 的位置，而 sampler position 对 IndexSubset 是槽位，两者不一致；set 还丢失重复读取/多epoch/多condition信息。Codex复现：已消费[10,20]两窗仍报告2个未消费；benchmark_large完整96窗报告88未消费；smoke完整40窗仍有大量未消费。报告“完整epoch最后天然有预取剩余”不成立：有限sampler遍历完整时应为0。

设计要求：统一事件和消费身份为 `(phase, epoch或validation condition, underlying split position)`，记录实际次数/有序消费而非推算 range。区分 delivered 与训练步完成；验证缓存也记录消费身份。按worker PID/唯一ID分组，不能全叫main。完整遍历读取消费精确闭合，提前退出仅真实未消费预取非零，跨epoch同位置与跨condition同位置不能被set吞掉。消费日志、事件退出/队列清理可靠，测试应含非连续subset、重复/漏窗、多epoch、val四condition和异常中断。

## R2：核验器不能验收实际并行产物（P1）

`check_read_audit` 已识别 parallel，但仍无条件执行串行 C1/C4（parent audit=0必失败）；新增独立helper的单测不能代表顶层核验通过。C8又要求val_cache unconsumed_examples为空，但提交的真实smoke中它非空。训练阶段只断言累计reads >= 预算，不能证明每个文件/窗口按epoch覆盖，漏A重复B可蒙混。

分支要真实：serial保持原预算检查；parallel从R1的实际事件/消费复算每epoch/文件/窗口预算、val每condition精确覆盖、阶段集合分别等于对应角色而非只并集。检查漏窗/重复窗/角色互换/未消费/丢事件，顶层对新的真实并行短运行成功，对篡改副本失败。不接受靠自报passed或预算下限通过。

## R3：完整训练步等价证据缺失（P1）

现有两份 smoke 同seed2027、initial_model_sha256和epoch_order_sha256相同，但serial train_loss=1.01445017、parallel4=0.93548022；Codex加载checkpoint发现102个状态张量不同，最大差5.255741。不能只比较val特征缓存就宣称训练等价。可能是历史代码不同/训练输入差异/数值非确定性，需定位而非猜测。

用当前同一版本、相同thread设置/同初始化/同batch次序/同epoch/同噪声做0/2/4短训练对照，逐批输入hash、labels、loss、参数、BN状态与GradScaler/优化器步骤对照。必要时只在等价测试中启用deterministic算法来分离CUDA非确定性，并保留真实模式对照和解释。至少覆盖两epoch。先验证真实推荐2-worker路径。保留历史smoke说明失效原因，产物写新目录。

## R4：性能指标和可复现配置不合格（P1/P2）

当前benchmark在torch.inference_mode/model.eval中，无backward、Adam或GradScaler步骤，不能称完整训练端到端收益。报告声称9.12→10.72，而对应benchmark_summary.csv是48窗8.39→7.60；benchmark_large.json缺environment/thread记录和逐轮raw/digests，memory为空；报告说96窗“两次独立同向”需要两份真实证据，不得把同一结果重复当两次。

修复：保存不可混淆runId/配置/时间与原始每轮记录、digest、进程内存、workers/prefetch/父torch线程、warmup/计时范围，JSON和CSV/报告必须从同一结果生成。对同一有界真实train样本顺序用真正的train模式+class-weighted CE+AMP+backward+Adam/GradScaler完整训练步对照0/2/4；每配置相同初始化，适度warmup并轮换执行顺序，批内数据hash对照置于计时外或明确单列开销，CUDA计时正确同步且清楚区分传输/前端/模型（不要把异步dispatch时间当kernel时间）。用nvidia-smi采样可选，CUDA event busy_fraction只能按自己的定义报告。不承诺GPU满载。

训练CLI目前没有线程数参数，报告启用命令也无法复制最佳配置。新增明确 --torch-threads（默认None保持旧行为），并在环境/run_config记录实际值。报告提供真实可复制且不覆写已有正式run的smoke/benchmark命令，不留<seed>或“手动调整线程”的空白步骤。默认0-worker兼容，测得无收益也如实报告。

## R5：证据文件保护及报告严谨性

seed2026原 verification.json 曾被覆盖，无法通过重新生成恢复原哈希。已验收checkpoint/指标没变，该修改需记录为证据版本替换，当前文件先保留，不得再次覆盖。所有本轮独立复核输出到新 `outputs/stage1_io_optimization_20260927/rework1/` 或新明确路径。既有两份正式训练的SHA256冻结前后对照。纠正原报告中“GPU忙占比=利用率”“4workers无收益的原因已定论”“验证审计空过”等无证据或不准确表述；区分已证实的实现问题、有限样本性能结论和后续假设。

## 范围和交接

不开始seed2028、第二阶段或新的全量训练；不改G盘数据/清单/冻结划分，不做几十GB格式迁移。允许修复加载模块、训练CLI及独立核验、必要测试、真实有界对照及报告。多进程若触发实际审批走正常审批，不能规避。完成后提交本轮唯一handoff，列出各R项证据、实测收益以及仍有限制。只对修改关联风险做必要复验，不重复跑无关长任务。
