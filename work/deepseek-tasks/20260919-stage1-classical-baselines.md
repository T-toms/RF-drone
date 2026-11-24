# BASE2：第一阶段两个传统基线

## 目标

在不改变现有 DroneRFa 原始数据、窗口索引、六个冻结检查点和 EVAL6 产物的前提下，新增并完成两个 clean 条件的二分类对照：

1. RF0 原始复 IQ 的去直流平均功率阈值基线；
2. RF0 复 IQ 的 PSD 特征 + SVM 基线。

本任务只完成 BASE2。不得开始 `n_fft=512/1024/2048/4096` 的 ResNet 对照；NFFT4 依赖 BASE2 先验收。

## 冻结输入与划分

- 原索引：`dronerfa_baseline/outputs/window_index.csv`，SHA256 `4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac`。
- 新留出索引：`dronerfa_baseline/outputs/heldout_preparation_20260917/heldout_window_index.csv`，已有固定 SHA256 见共享摘要。
- 旧 train/val/test 源文件划分保持不变；新 12 文件只作最终 heldout clean 评价，不得参与特征标准化、阈值、SVM 超参数或模型选择。
- 输入仅 RF0_I/RF0_Q；窗口长度仍为 1,000,000 点；本任务不加任何额外 AWGN，不比较噪声方式。
- 标签仍为 Background=0，Phantom 4 Pro 与 Air 2S 合并为 Drone=1。

## 方法接口

### 1. 能量阈值

- 对每个窗口组成 `x=I+jQ`，去除复均值后计算 `10*log10(mean(abs(x)^2)+eps)`；这里故意使用 RMS 归一化前的原始相对功率。
- 阈值和方向只能由旧 train/val 决定。必须在报告中明确具体选择规则，并证明 test/heldout 未进入选择。
- 使用唯一固定阈值评价旧 test clean 和新 heldout clean；输出逐窗分数、预测、混淆矩阵和指标。

### 2. PSD + SVM

- 对 RF0 复 IQ 去均值并按现有 `rms_normalize` 归一化。
- 使用 1024 点 Hamming、hop 512、双边复频谱；对每个窗口沿时间维平均功率，fftshift 后取稳定 log power，形成固定 1024 维 PSD 特征。允许用 PyTorch/CUDA 分批提取，CPU fallback 必须可用。
- 特征标准化器只拟合旧 train。SVM 使用 class-balanced；核函数及小型超参数候选必须预先写在 manifest，使用旧 val 选择。不得根据 test/heldout 结果改规则。
- 评价旧 test clean 和新 heldout clean；输出逐窗分数/decision function、预测、混淆矩阵和指标。

## 代码与输出

允许新增/修改：

- `dronerfa_baseline/run_classical_baselines.py`
- `dronerfa_baseline/test_stage1_method_comparison.py`
- 如确有必要，可增加一个小型共享模块，但不得重写现有训练/评价逻辑。
- `dronerfa_baseline/requirements.txt` 仅补充实际使用且本机已安装的 `scikit-learn`/`joblib` 依赖。
- 新输出仅限 `dronerfa_baseline/outputs/stage1_method_comparison_20260919/baselines/`。

输出至少包括：

- `energy_threshold/selection.json`
- `energy_threshold/test_window_predictions.csv`
- `energy_threshold/heldout_window_predictions.csv`
- `energy_threshold/metrics.csv`
- `energy_threshold/energy_distributions.png`
- `psd_svm/model.joblib`
- `psd_svm/selection.json`
- `psd_svm/test_window_predictions.csv`
- `psd_svm/heldout_window_predictions.csv`
- `psd_svm/metrics.csv`
- `psd_svm/psd_examples.png`
- `comparison_metrics.csv`
- `experiment_manifest.json`
- `report.md`
- `verification.json`

PSD 数值缓存如保存，应位于本输出目录下并记录 dtype、shape、来源索引和 SHA256；不得生成或保存每个窗口的 STFT PNG。代表性 PSD 图可以保存。

## 指标和报告

- 至少报告 accuracy、balanced accuracy、precision、drone recall、background specificity、F1、TN/FP/FN/TP。
- heldout 主口径继续采用每类别 4 个源文件等权；同时保留 pooled 窗口指标，不得混用。
- 报告必须说明能量阈值可能利用录制功率/增益差异，PSD+SVM 是传统特征基线，均不是新的最终模型。
- 对比当前 ResNet 时只引用已经冻结的 clean 结果，不重新运行或改写 EVAL6。

## 验收条件

1. 依赖导入、CLI `--help` 和限制样本 smoke run 通过。
2. train/val/test/heldout 源文件身份检查通过；heldout 未参与选择。
3. 全量旧 test 和 1820 窗 heldout clean 均有一一对应、有限的预测记录。
4. 特征维度、标准化拟合范围、阈值/SVM 选择规则、随机种子和库版本写入 manifest。
5. 从逐窗 CSV 独立回算指标与 `metrics.csv` 一致；混淆矩阵计数守恒。
6. 原索引、`dronerfa_data.py`、`model.py`、`train_resnet18.py` 和 EVAL6 产物不被修改。
7. 结束前写唯一交接报告，状态用 `completed`、`needs_codex` 或 `needs_user`，并停止执行器等待 Codex 验收。

## 限制

- 不执行 NFFT4、不训练新 ResNet、不修改阈值为当前 ResNet 服务。
- 不下载数据、不写原 MAT 所在目录、不创建 heartbeat 或定时任务。
- 不提交、推送、发布或改变权限。
