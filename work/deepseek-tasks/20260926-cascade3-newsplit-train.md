# CASCADE3-NEWSPLIT-B：新划分单种子两阶段重训

用户授权按新冻结的三类源文件划分重新训练旧两阶段基线；“跑一次”先固定一个种子 2026，完成一次可审计的端到端基线。此任务依赖 CASCADE3-NEWSPLIT-A 经 Codex 验收。

## 输入和角色

- 仅使用 `dronerfa_baseline/outputs/cascade3_newsplit_20260926/window_index_newsplit3.csv`（SHA256 `deed870f9693c5ebe93fb80a9647b452e90680dd5b1f21372e108ded4bf4d231`）及 `stage2_view_newsplit3.csv`（SHA256 `76001e671cf3ebe492b191af87a85b5368caed85e03c84c4c249ccc5081642de`）；源角色追溯到 `work/meta_split2/CODEX-FREEZE-20260926.json`。
- 训练：12 个 MAT、1810 窗，阶段 2 无人机 1210 窗。验证：6 个 MAT、930 窗，阶段 2 无人机 650 窗。测试：6 个 `development_test_exposed` MAT、890 窗；本任务训练期不得读取测试信号或标签选型。
- MAT 实际路径只用 G 盘，不修改 G 盘及旧实验工件。

## 冻结训练协议

1. Stage 1：Background vs Drone；Stage 2：Phantom 4 Pro vs Air 2S，只用真值为无人机的训练/验证窗。
2. 尽量复用 `dronerfa_baseline/train_resnet18.py`、`run_nfft_comparison.py`、`run_stage2_classification.py` 的已验收模型及输入处理：RF0、1M 采样、去均值、RMS 归一化、原 `A_original` 训练噪声、Hamming STFT、n_fft=512/hop=256、256×256、ResNet18、原优化器/学习率/batch size。两个阶段不得偷偷更改图像或噪声口径。
   新索引的 `window_index` 为源文件内全局窗序号，旧索引的同名列是采集块内序号；复用代码时以 `start` 为读取位置，不能沿用旧序号假设。
3. 固定 seed=2026、5 epochs；分别只根据新 val 选择最佳 epoch。训练前写入配置和输入哈希。不得用测试、旧 heldout 指标或测试窗口来选择 epoch、阈值、超参数。Stage 1 阈值仍为 drone logit − background logit > 0；Stage 2 阈值仍为 Air2S logit − Phantom logit > 0。
4. 将新 checkpoint、每轮训练/验证指标、选 epoch 依据、代码/配置/索引哈希和环境记录保存在 `dronerfa_baseline/outputs/cascade3_newsplit_20260926/seed_2026/`；支持中断后检查状态，绝不覆盖旧 `outputs/stage1_method_comparison_20260919` / `stage2_model_classification_20260919` / `cascade3_evaluation_20260919`。
5. 先做短的训练/验证读数 smoke（允许在 train/val 上），证明完整 1M 窗预处理和网络前后向可运行，再开始正式 5 epochs。若估计总耗时过长，仍按已授权的一次完整训练持续执行，不自行缩短 epochs 或抽样。技术问题报告 needs_codex。

## 验收与交接

- 两阶段各有 5 轮记录、val 选择证据和一个最佳 checkpoint；checkpoint 可重新加载，输入与源文件角色可追溯。
- 测试文件未出现在训练/验证读取日志或选型逻辑。
- 本任务不执行最终测试；Codex 验收后再派独立串联评价。
- 交接文件 `work/deepseek-handoffs/<runId>.json` 报 completed/needs_codex/needs_user、真实产物、检查和耗时。
