# NFFT4：第一阶段 STFT 点数消融

## 目标

在 BASE2 通过 Codex 验收后，固定现有 RF0 二分类、数据划分、训练噪声、网络结构、优化器、模型输入尺寸和选择规则，只比较四组 STFT 参数：

- `n_fft=512, hop_length=256`
- `n_fft=1024, hop_length=512`
- `n_fft=2048, hop_length=1024`
- `n_fft=4096, hop_length=2048`

本任务回答频率分辨率变化对当前 ResNet18 检测器的影响。它不是 AWGN 与真实背景等噪声生成方式对照；四组必须使用完全相同的既定 AWGN 方法。

## 冻结输入和公平性规则

- 原索引：`dronerfa_baseline/outputs/window_index.csv`，SHA256 `4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac`。
- 新留出索引：`dronerfa_baseline/outputs/heldout_preparation_20260917/heldout_window_index.csv`，SHA256 `6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf`。
- 输入仅 RF0_I/RF0_Q，窗口长度 1,000,000 点，Background=0、两类无人机合并为 Drone=1。
- 所有组保持 `image_size=256`。动态 STFT 在内存/GPU 中计算，不保存逐窗 PNG。
- 训练种子固定为 `2026,2027,2028`，每组每种子 5 epochs，batch size 4，Adam、learning rate `1e-3`、weight decay 0、AMP 开启。
- 训练 SNR 计划固定为现有 A_original：`clean,10,5,0,-5,-10,-12`。不得加入 `-15 dB` 训练，不得比较其他噪声方式。
- 验证条件固定为 `clean,-10,-12,-15`，固定验证噪声 seed `12026`。每一 n_fft、训练种子和 epoch 必须使用同一批原始验证窗及同一噪声实现。
- 每个 `(n_fft, training_seed)` 在 5 个 epoch 中，以四档验证 balanced accuracy 的平均值选择；并列取最早 epoch。test 和 heldout 不得参与模型或 epoch 选择。
- 同一 training seed 的四档 n_fft 使用相同 ResNet 初始权重、相同每轮训练窗口顺序以及相同每窗训练 SNR/RNG 规则；manifest 必须提供可核对摘要。
- 现有 `n_fft=1024` A_original 三个种子检查点允许复用，但必须逐项证明上述配置、源码、索引、初始权重、epoch 顺序和选择规则相同。若不能证明则重新训练。不得根据既有结果决定是否复用。

## 最终评价

- 主要评价：新 heldout 1820 窗，在 `clean,-10,-12,-15` 四档条件下评价全部 12 个选定模型。
- heldout 每个条件的噪声必须跨四档 n_fft 和三个训练种子配对一致；采用 `work/heldout-eval-config-20260918.json` 的冻结种子/哈希规则，或证明生成的逐窗噪声摘要与 EVAL6 一致。
- heldout 主指标为源文件等权；同时保留 pooled 窗口指标。至少报告 accuracy、balanced accuracy、precision、drone recall、background specificity、F1、TN/FP/FN/TP。
- 补充评价：旧 test clean（440 窗），不得用于选择。
- 汇总必须含每种子结果、四档 n_fft 的跨种子均值/标准差，以及相同种子下相对于 `n_fft=1024` 的配对差值。只有 3 个训练种子，不做夸大的显著性结论。

## 代码和输出

允许新增或修改：

- `dronerfa_baseline/run_nfft_comparison.py`
- `dronerfa_baseline/test_nfft_comparison.py`
- 必要时新增一个小型汇总/评价模块；优先复用现有 `dronerfa_data.py`、`model.py`、`run_noise_ab.py` 和 EVAL6 评价逻辑，不改它们的既有行为。
- 新输出仅限 `dronerfa_baseline/outputs/stage1_method_comparison_20260919/nfft/`。

至少输出：

- 每个 `(n_fft, seed)` 的 `run_config.json`、`training_history.csv`、逐 epoch 验证指标、`selection.json` 和选定检查点或已验证的复用引用；
- `heldout_window_predictions.csv`，预期 `4*3*4*1820 = 87360` 条唯一记录；
- `test_clean_window_predictions.csv`，预期 `4*3*440 = 5280` 条唯一记录；
- `model_condition_metrics.csv`、`paired_nfft_comparison.csv`、`nfft_summary.csv`；
- `native_stft_examples.png` 与 `model_input_examples.png`：在任务开始前固定的代表窗上展示四档 n_fft。原生图应保留各自时频网格或明确坐标，模型输入图展示实际 256×256 张量；
- `experiment_manifest.json`、`report.md`、`verification.json`。

不生成每个窗口的 STFT 图片。若为加速在运行期缓存特征，必须仅位于本输出目录，记录 dtype/shape/来源/哈希，并在报告中说明；避免留下无必要的大型临时缓存。

## 验收条件

1. CLI help、单元测试和有限窗口 smoke run 通过；至少覆盖四档前端 shape、50% hop、周期 Hamming、相同噪声配对、选择不泄漏以及图像输出。
2. 全部 12 个模型均有选择证据；复用 1024 检查点时有逐项等价验证和检查点 SHA256。
3. heldout 87360 行和 test clean 5280 行主键唯一、数值有限、标签/文件/起点与冻结索引一一对应。
4. 从逐窗预测独立复算所有指标，与 CSV/JSON 一致；混淆矩阵计数守恒。
5. heldout 没有参与拟合、epoch 选择、n_fft 选择或阈值调整。报告把四档完整呈现，不因结果挑选单一 n_fft 作为“最终模型”。
6. 固定代码、索引、BASE2、EVAL6 以及现有检查点不被修改；运行前后 SHA256 核对通过。
7. 代表图可读，图例明确区分 512/1024/2048/4096，并说明 256×256 resize 会压缩原生时间频率网格。
8. 结束前写唯一交接报告并停止执行器，等待 Codex 验收。

## 限制

- 不进行噪声生成方式实验，不增加 RF1，不做第二阶段机型分类。
- 不下载或改写原始 MAT，不修改旧实验输出，不创建 heartbeat 或定时任务。
- 不提交、推送、发布或改变权限。
