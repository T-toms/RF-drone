# STAGE1-LOSS-NORM：一次 seed 2027 损失归一化对照

状态：用户于 2026-09-27 授权整体路线；本任务已备妥，依赖 R1 经 Codex 验收并冻结实际加载配置后单独派发。仅看到本文件不构成启动信号。DSH 不可自行越过验收启动。

## 目标、依据与停止规则

只验证一个明确假设：小批次下，按批内类别权重和归一化可能削弱全局类别权重的预期作用。`work/stage1-batch-balance-diagnostic-20260927.json` 已核对两种子十轮顺序哈希；约 77% 的 batch4 不含背景。此证据不证明种子差异的因果来源，本次改动同时改变逐批梯度尺度。

本任务只新增一份 seed2027、5 epoch 的候选正式训练。不得追加 seed2028、扩大轮数、调门限、换采样器、改 BN、学习率、噪声档或数据划分。保留全结果，不按验证成绩追加试验。完成后交接 Codex 验收；冻结开发版本与 Stage2 是下一任务。

## 数学接口与兼容性

令训练集 N=9590、n0=600、n1=8990，权重 w_c=N/(2*n_c)。仅使用 train 计数，mu_train=sum_c(n_c*w_c)/N=1。

- 原始模式 `batch_weight_sum`：L=sum_i(w_yi*CE_i)/sum_i(w_yi)，保持 PyTorch 原 weighted CE 的默认行为。
- 候选模式 `global_train_mean`：L=sum_i(w_yi*CE_i)/(B_actual*mu_train)。最后一批只有 2 个样本，必须用实际 B，不能固定除以 4，也不能再除以该批权重之和。
- 建议 CLI `--loss-normalization`，默认 `batch_weight_sum`；正式候选显式传 `global_train_mean`。保存公式、权重、计数、mu_train 和实际分母规则。
- 验证的原 weighted CE loss 作为独立辅助量维持原定义，模型选型仍只看既定 BA；可另存候选 objective loss，但不能混名比较训练损失数值。
- 不重写旧数据/模型基础模块；尽量在独立 criterion 模块与训练入口最小接入。默认行为须有兼容证据。

## 固定协议与运行前闸门

冻结输入、哈希和全量协议沿用 `work/deepseek-tasks/20260927-stage1-12drones-seed2027.md`：RF0、1M 不重叠窗、STFT512/hop256/256x256、单通道 ResNet18、batch4、Adam lr1e-3 wd0、FP16 AMP、5 epoch、seed2027、验证噪声 seed12026、训练 clean/10/5/0/-5/-10/-12、验证 clean/-10/-12/-15、margin>0、四条件 pooled BA 均值最大且同分取最早 epoch。

训练只有 62 train/16 val，26 个测试 MAT 的 IQ 不读。初始化哈希必须与原 seed2027 的 `d774eeb6e3f9ec15c2b17cf5ceeab1bce68c970481f9a4d96718bc2dc358aef9` 相同，五轮 order 哈希须逐一匹配旧 run_config；每窗噪声规则和底层 split position 不变。

加载配置由 Codex 于 2026-09-28 明确批准为 **`num_workers=0`**，预取不启用；父进程 `--torch-threads` 保持旧 seed2027 正式运行的默认行为（不显式改成2；在 run_config 中记录实际线程数，旧记录为10）。R1/R2 的0/2/4短训练表明同版本数学路径逐位一致，R2真实训练步基准0/2/4为8.27/6.57/4.46窗/s。若新实现改变旧串行输入或数学路径且不能证明兼容，先交 needs_codex；不能把加载差异混入损失因果对照。不得私自另跑完整控制臂；有必要时由 Codex 根据兼容证据决定。

允许必要的短测试：解析公式与梯度、纯单类/混合类/尾批、默认模式兼容、同初始模型和有界真实 train 输入的短训练 sanity、非有限数检测。短测试改变的 RNG 不得影响正式初始化。全部使用项目 .venv 和 CUDA；设备不满足时先报告，不静默改 CPU 正式训练。

## 输出与可恢复性

- 新根目录 `dronerfa_baseline/outputs/stage1_loss_normalization_20260927/`，正式候选 `seed_2027_global_train_mean/`；拒绝覆盖已有完成运行。旧 seed2026/seed2027 及其验证文件只读。
- 固化运行前 protocol.json（选择规则、停止条件、控制引用、候选参数、输入和代码哈希）；保存五轮完整逐窗 val logits、预测、指标、训练顺序哈希、实际读取审计、checkpoint 重载结果、耗时。
- 保存 `last_training_state.pt`：模型、前端、optimizer、GradScaler、Python/NumPy/torch CPU/CUDA RNG、已完成 epoch、下一 epoch、历史与最佳选择、协议/代码/索引身份。仅支持明确的 epoch 边界恢复，不宣称支持任意批次或断电时精确续训。恢复前检查身份，另记追加读取事件，不能伪造无中断预算。
- 用短确定性运行核对一次 epoch 边界中断/恢复，确认后续输入/步数及状态一致；真实 GPU 非确定模式不能承诺逐位相同。此功能只服务新运行，不修改旧 checkpoint。
- 中文报告及配对 CSV/JSON 放本次新目录或 `reports/DroneRFa_一阶段损失归一化对照_20260927.md`。不能宣称一次对照证实跨种子稳定性。

## 验收指标

主要比较：候选与既有 seed2027 各自按冻结规则选出 epoch 后的四条件平均 pooled BA，固定零门限。seed2026 仅提供另一随机种子的上下文，不把不同 seed 差异叫损失改动效应。

同时逐条件报告 TP/TN/FP/FN、背景 FPR、无人机 FNR/recall、12 机型各自检测召回、文件等权 BA、最差条件 BA，以及描述性 AUC。需说明 FPR/FNR 权衡，不能只报均值。文件等权 BA：背景源文件 specificity 均值与无人机源文件 recall 均值二者平均。各机型还应给源文件数/窗口数，避免伪造独立样本量。

核验逐窗 val 全集和标签、各 epoch/condition 唯一键、指标重算、选择规则、尾批与真实训练步数、逐窗/文件读取审计精确覆盖且测试 IQ 零读。正式结果不做门限扫描。旧控制权重、指标、冻结输入前后哈希不变。

完成唯一 handoff，列明真实改动、检查、训练结果、失败与限制。训练 NaN、读取越界、初始化/顺序不匹配、权限或协议冲突时停止依赖步骤并报告，不凭空调整多个参数。
