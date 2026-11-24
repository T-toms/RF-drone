# STAGE2-BASE：Phantom 4 Pro / Air 2S 独立机型分类

## 目标

在第一阶段检测实验已验收的基础上，建立独立的第二阶段二分类器。输入只包含真实标签为无人机的 RF0 IQ 窗口，输出：

- `0 = DJI Phantom 4 Pro`（原 target_code `0010`）
- `1 = DJI Air 2S`（原 target_code `0101`）

本任务先评价“已知输入为无人机时，模型能否区分两种机型”。不接入第一阶段预测，不做背景/无人机/机型的级联三分类。

## 数据与冻结边界

- 旧索引：`dronerfa_baseline/outputs/window_index.csv`，只保留 target_code `0010`、`0101`：
  - train：Phantom 310 + Air2S 310 = 620 窗；每类 2 个源文件。
  - val：Phantom 140 + Air2S 150 = 290 窗；每类 1 个源文件。
  - test：Phantom 140 + Air2S 160 = 300 窗；每类 1 个源文件。
- 新留出索引：`dronerfa_baseline/outputs/heldout_preparation_20260917/heldout_window_index.csv`：
  - Phantom 610（4 文件）+ Air2S 650（4 文件）= 1260 窗。
- 原 MAT、旧索引、新留出索引、`model.py`、`dronerfa_data.py`、`train_resnet18.py` 和既有输出只读；不得修改。
- 逐项核对 `work/codex-preserved-inputs-20260917.json` 的 14 个冻结输入，运行前后 SHA256 必须不变。
- heldout 不得参与拟合、选 epoch、选 n_fft、调阈值或任何配置决策。

## 实验设计

只比较以下两档：

| n_fft | hop_length | image_size |
|---:|---:|---:|
| 512 | 256 | 256 |
| 1024 | 512 | 256 |

其余设置冻结：

- 前端复用 `model.SpectrogramFrontEnd`：RF0 I/Q、每窗 1,000,000 点、去均值、RMS 归一化、周期 Hamming、双边 STFT、log power、fftshift、逐样本标准化、双线性缩放为 `1x256x256`。
- ResNet18 从头训练，`input_channels=1, num_classes=2`。不复用第一阶段分类头或已训练权重。
- training seeds：`2026, 2027, 2028`；同一个 seed 的 512/1024 模型必须使用相同初始 ResNet18 权重哈希和相同 epoch 窗口顺序。
- epochs：5；batch size：4；Adam，lr `1e-3`，weight_decay `0`；CUDA AMP float16（前端 float32），与第一阶段一致。
- 训练噪声沿用 A_original：`clean,10,5,0,-5,-10,-12` 等概率；不加入 `-15 dB` 训练，不比较噪声生成方式。
- 验证条件：`clean,-10,-12,-15 dB`，固定 validation noise seed `12026`。
- 每个 `(n_fft, seed)` 的选 epoch 规则：四档验证集文件等权平衡准确率算术平均最高；数值完全并列取最早 epoch。
- 在仅保留无人机行后，为 train/val/test/heldout 分别创建稳定、从 0 开始的 stage2 index position。相同 split/condition 的噪声必须在不同 n_fft 和不同训练 seed 间逐窗配对一致；验证沿用 seed 12026，heldout 沿用 seed 22026，epoch 固定 0；test clean 不加噪声。记录并测试 RNG 规则和逐窗特征摘要。
- n_fft 配置选择只能使用六个已选 epoch 的旧 val 四条件结果：分别对三个 seed 的四条件验证 BA 取平均，较高者记为 `validation_selected_nfft`；完全相同时选择计算量较小的 512。heldout 结果只能在配置选择文件写定后生成，且不得回写选择。

## 评价与产物

输出根目录：`dronerfa_baseline/outputs/stage2_model_classification_20260919/`

至少包含：

- 六个模型的 `best_model.pt`、`run_config.json`、`training_history.csv`、`validation_metrics.csv`、`selection.json`。
- `validation_nfft_selection.json`：只引用旧 val 证据，不含 test/heldout 指标、路径或字段。
- 新 heldout 四条件原始窗口预测：`1260 x 4 x 6 = 30,240` 行，主键 `(n_fft, training_seed, condition, index_position)` 唯一且每组完整覆盖 0..1259。
- 旧 test clean 原始窗口预测：`300 x 6 = 1,800` 行，主键唯一且每组完整覆盖 0..299。
- 每模型/条件完整指标：accuracy、balanced accuracy、precision、Phantom recall、Air2S recall、F1（明确 averaging）、TN/FP/FN/TP；主口径采用每机型内源文件等权 recall 后两类等权 BA，同时保留 pooled 窗口口径。
- 逐文件指标、跨 seed 的 n_fft 汇总、同 seed 的 512 相对 1024 配对差值。
- 代表性可视化：同一预先固定的 Phantom 窗和同一预先固定的 Air2S 窗，分别展示 512/1024 的原生 STFT 与实际 `1x256x256` 模型输入；不保存逐窗 PNG。
- 中文 `report.md`、`experiment_manifest.json`、独立 `verification.json`。

报告必须先给出 oracle 第二阶段结果，并明确：背景窗口不属于本任务；当前结论不是级联系统的三分类性能。说明旧训练集每类只有两个源文件、新留出每类四文件、模拟 AWGN 不等于现场绝对 SNR，不能声称跨会话或跨场景泛化。

## 实现文件

优先新增：

- `dronerfa_baseline/run_stage2_classification.py`
- `dronerfa_baseline/test_stage2_classification.py`
- `dronerfa_baseline/verify_stage2_classification.py`

可复用现有模块，但不修改冻结源文件。程序必须支持可重复的分步命令（输入核查、训练、选 n_fft、作图、评价、汇总、验证、全部执行），并支持有界 smoke/fixture 测试，不能让单元测试误跑完整 1260 窗评价。

## 验收条件

1. 六个模型完成，训练/选 epoch/选 n_fft 无 test 或 heldout 泄漏。
2. 30,240 heldout 行、1,800 test clean 行的身份、顺序、标签映射、长度、有限 logits、`margin > 0` 判 Air2S 规则、唯一键和覆盖全部通过。
3. 原始预测重算逐文件、模型条件、跨 seed 汇总与配对差值完全一致；混淆矩阵守恒。
4. 同 seed 的 512/1024 初始权重与 epoch 顺序相同；同条件噪声逐窗配对；hop 恒为 n_fft/2；两档前端输出均为 `1x256x256`。
5. 验证选择文件不包含 test/heldout 证据；生成 heldout 结果后重新核对选择文件内容与哈希未改变。
6. 有意义的测试覆盖标签重映射、过滤后索引/RNG、文件等权指标、选 epoch、选 n_fft、故意篡改预测/汇总/选择时验证器必须拒绝、图片可读性和有界 smoke。
7. 14 个冻结输入 SHA256 前后一致；原 MAT、旧输出不变。
8. 不启动第一/第二阶段级联、不做三分类、不恢复噪声方式实验、不创建定时器、不提交/推送/发布。

完成后写 `completed` 交接；若只有需要 Codex 处理的技术问题，停止执行后写 `needs_codex`，不得自行扩展范围。
