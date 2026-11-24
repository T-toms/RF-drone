# BASE2：第一阶段两个传统基线（clean 条件）

生成时间：2026-09-19T04:09:02.335986+00:00

## 1. 任务与范围

本报告只覆盖 BASE2：clean 条件下用 RF0 复 I/Q（1,000,000 点/窗）实现并评价两个传统二分类基线，标签为 Background=0、Drone(Phantom 4 Pro + Air 2S)=1。不训练新 ResNet、不做 AWGN 对照、不做 n_fft 四档比较、不进入第二阶段。

## 2. 冻结输入与划分

- 旧索引 `C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\outputs\window_index.csv`，SHA256 `4e77c7f186c1aae8e04f2e90f2769c0ed399cc94f40b64f99237b65c057b85ac`（train/val/test）。
- 新留出索引 `C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\outputs\heldout_preparation_20260917\heldout_window_index.csv`，SHA256 `6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf`。
- 两套索引的源文件交集 0 个；新 12 个文件只用于最终 heldout clean 评价。

| split | 窗口数 | 背景文件 | Phantom 文件 | Air 2S 文件 |
|---|---|---|---|---|
| train | 920 | 2 | 2 | 2 |
| val | 450 | 1 | 1 | 1 |
| test | 440 | 1 | 1 | 1 |
| heldout | 1820 | 4 | 4 | 4 |

## 3. 选择规则（只用旧 train/val）

### 3.1 能量阈值

- 分数：`10*log10(mean(|x - mean(x)|^2) + eps), x = RF0_I + j*RF0_Q, eps = 1e-12`（原始相对功率，不做 RMS 归一化）。
- 决策：score >= threshold -> Drone (1); score < threshold -> Background (0)。
- 候选：minimum train+val score plus the midpoints between all adjacent unique train+val scores，共 1370 个。
- 选择数据：['train', 'val']，1370 窗（背景 460、无人机 910）。
- 规则：balanced accuracy on train+val windows，平局取 smallest_threshold；最佳 train+val BA=88.68%，并列候选 1 个。
- 最终唯一阈值：**-47.336378 dB**（test/heldout 未进入选择）。

各 split 分数分布（dB）：

| split | 窗口数 | 背景均值 | 无人机均值 | 最小 | 最大 |
|---|---|---|---|---|---|
| train | 920 | -48.959 | -41.189 | -52.283 | -30.656 |
| val | 450 | -48.591 | -41.918 | -52.134 | -32.358 |
| test | 440 | -50.357 | -42.355 | -52.176 | -34.714 |
| heldout | 1820 | -50.824 | -42.451 | -52.171 | -34.408 |

### 3.2 PSD + SVM

- 特征：`rms_normalize` 后的复窗，Hamming n_fft=1024 / hop=512、双边复频谱、沿时间平均功率、`10*log10(P+1e-10)`、fftshift，固定 1024 维。
- 标准化：sklearn.preprocessing.StandardScaler，只拟合旧 train（920 窗）。
- 候选网格（选择前冻结，18 个）：linear / rbf / poly3 / sigmoid，`class_weight=balanced`，`random_state=20260919`。
- 选择规则：balanced accuracy on the old val split，平局取 first candidate in the frozen manifest order；选中 **linear-C0.01**（val BA=99.38%，并列候选 5 个）。
- 决策：SVC.decision_function > 0 -> Drone (1); < 0 -> Background (0); == 0 -> Drone, matching margin > 0 in the ResNet evaluation。

## 4. 结果

主口径为**文件等权**：每个类别 4 个源文件各自的检出/误报率先等权平均，两机型再等权得到无人机召回，BA=0.5×(背景文件等权特异度+无人机等权召回)。pooled 为窗口汇总口径，仅作补充，两个口径不混用。

| 方法 | 评价集 | 文件等权 BA | 文件等权召回 | 文件等权特异度 | pooled accuracy | pooled BA | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|
| energy_threshold | test | 96.96% | 99.64% | 94.29% | 97.95% | 96.98% | 132/8/1/299 |
| energy_threshold | heldout | 98.30% | 100.00% | 96.61% | 98.96% | 98.30% | 541/19/0/1260 |
| psd_svm | test | 98.21% | 97.14% | 99.29% | 97.95% | 98.31% | 139/1/8/292 |
| psd_svm | heldout | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 560/0/0/1260 |

### 4.1 逐源文件比例

| 方法 | 评价集 | 类别 | 源文件 | 窗口数 | 判为 Drone 比例 |
|---|---|---|---|---|---|
| energy_threshold | test | background | T0000_D00_S0011.mat | 140 | 5.71% |
| energy_threshold | test | phantom | T0010_D00_S0011.mat | 140 | 99.29% |
| energy_threshold | test | air2s | T0101_D00_S0011.mat | 160 | 100.00% |
| energy_threshold | heldout | background | T0000_D00_S0100.mat | 140 | 12.86% |
| energy_threshold | heldout | background | T0000_D00_S0101.mat | 140 | 0.71% |
| energy_threshold | heldout | background | T0000_D00_S0110.mat | 140 | 0.00% |
| energy_threshold | heldout | background | T0000_D00_S0111.mat | 140 | 0.00% |
| energy_threshold | heldout | phantom | T0010_D00_S0100.mat | 160 | 100.00% |
| energy_threshold | heldout | phantom | T0010_D00_S0101.mat | 150 | 100.00% |
| energy_threshold | heldout | phantom | T0010_D00_S0110.mat | 150 | 100.00% |
| energy_threshold | heldout | phantom | T0010_D00_S0111.mat | 150 | 100.00% |
| energy_threshold | heldout | air2s | T0101_D00_S0100.mat | 160 | 100.00% |
| energy_threshold | heldout | air2s | T0101_D00_S0101.mat | 180 | 100.00% |
| energy_threshold | heldout | air2s | T0101_D00_S0110.mat | 160 | 100.00% |
| energy_threshold | heldout | air2s | T0101_D00_S0111.mat | 150 | 100.00% |
| psd_svm | test | background | T0000_D00_S0011.mat | 140 | 0.71% |
| psd_svm | test | phantom | T0010_D00_S0011.mat | 140 | 94.29% |
| psd_svm | test | air2s | T0101_D00_S0011.mat | 160 | 100.00% |
| psd_svm | heldout | background | T0000_D00_S0100.mat | 140 | 0.00% |
| psd_svm | heldout | background | T0000_D00_S0101.mat | 140 | 0.00% |
| psd_svm | heldout | background | T0000_D00_S0110.mat | 140 | 0.00% |
| psd_svm | heldout | background | T0000_D00_S0111.mat | 140 | 0.00% |
| psd_svm | heldout | phantom | T0010_D00_S0100.mat | 160 | 100.00% |
| psd_svm | heldout | phantom | T0010_D00_S0101.mat | 150 | 100.00% |
| psd_svm | heldout | phantom | T0010_D00_S0110.mat | 150 | 100.00% |
| psd_svm | heldout | phantom | T0010_D00_S0111.mat | 150 | 100.00% |
| psd_svm | heldout | air2s | T0101_D00_S0100.mat | 160 | 100.00% |
| psd_svm | heldout | air2s | T0101_D00_S0101.mat | 180 | 100.00% |
| psd_svm | heldout | air2s | T0101_D00_S0110.mat | 160 | 100.00% |
| psd_svm | heldout | air2s | T0101_D00_S0111.mat | 150 | 100.00% |

### 4.2 逐源文件分数范围（heldout）

分数范围直接读自两份逐窗 CSV，用来判断"分离"是发生在文件之间还是窗口之间。能量分数是绝对电平，PSD 特征在每窗内做过 RMS 归一化，因此 SVM 的判决不含绝对增益。

| 方法 | 源文件 | 类别 | 窗口数 | 分数最小 | 分数最大 |
|---|---|---|---|---|---|
| energy_threshold | T0000_D00_S0100.mat | background | 140 | -52.1706 | -35.3208 |
| energy_threshold | T0000_D00_S0101.mat | background | 140 | -52.1504 | -46.4207 |
| energy_threshold | T0000_D00_S0110.mat | background | 140 | -52.0826 | -48.7011 |
| energy_threshold | T0000_D00_S0111.mat | background | 140 | -52.0759 | -48.6044 |
| energy_threshold | T0010_D00_S0100.mat | phantom | 160 | -43.3717 | -34.4077 |
| energy_threshold | T0010_D00_S0101.mat | phantom | 150 | -41.7652 | -37.6518 |
| energy_threshold | T0010_D00_S0110.mat | phantom | 150 | -42.8929 | -37.8265 |
| energy_threshold | T0010_D00_S0111.mat | phantom | 150 | -42.8510 | -36.4977 |
| energy_threshold | T0101_D00_S0100.mat | air2s | 160 | -46.6697 | -38.5283 |
| energy_threshold | T0101_D00_S0101.mat | air2s | 180 | -47.1881 | -37.9941 |
| energy_threshold | T0101_D00_S0110.mat | air2s | 160 | -46.6509 | -42.3410 |
| energy_threshold | T0101_D00_S0111.mat | air2s | 150 | -47.1614 | -39.9773 |
| psd_svm | T0000_D00_S0100.mat | background | 140 | -1.7445 | -0.8300 |
| psd_svm | T0000_D00_S0101.mat | background | 140 | -1.9916 | -0.9176 |
| psd_svm | T0000_D00_S0110.mat | background | 140 | -1.8881 | -0.4477 |
| psd_svm | T0000_D00_S0111.mat | background | 140 | -1.7412 | -1.0031 |
| psd_svm | T0010_D00_S0100.mat | phantom | 160 | 0.5048 | 3.4021 |
| psd_svm | T0010_D00_S0101.mat | phantom | 150 | 0.5627 | 3.6431 |
| psd_svm | T0010_D00_S0110.mat | phantom | 150 | 1.2204 | 2.6380 |
| psd_svm | T0010_D00_S0111.mat | phantom | 150 | 1.2365 | 2.8700 |
| psd_svm | T0101_D00_S0100.mat | air2s | 160 | 1.3336 | 3.1722 |
| psd_svm | T0101_D00_S0101.mat | air2s | 180 | 1.2464 | 3.2224 |
| psd_svm | T0101_D00_S0110.mat | air2s | 160 | 1.4115 | 3.1823 |
| psd_svm | T0101_D00_S0111.mat | air2s | 150 | 1.1969 | 3.9929 |

### 4.3 与已冻结 clean ResNet 结果的对比

只引用已验收的 EVAL6 heldout clean 文件等权 BA，未重新运行或改写 EVAL6 产物。

| 模型 | heldout clean 文件等权 BA |
|---|---|
| ResNet18 seed2026_A | 96.41% |
| ResNet18 seed2026_B | 99.28% |
| ResNet18 seed2027_A | 99.02% |
| ResNet18 seed2027_B | 99.07% |
| ResNet18 seed2028_A | 98.79% |
| ResNet18 seed2028_B | 99.40% |
| energy_threshold | 98.30% |
| psd_svm | 100.00% |

## 5. 解释限制

1. **能量阈值可能利用录制功率/增益差异**：分数来自未归一化的原始相对功率，源文件之间、采集会话之间的增益差异会直接进入分数，因此不能把结果解释为"信号能量高于噪声"的物理结论，也不构成设备/距离不变量的证明。
2. **PSD+SVM 是传统特征基线**：1024 维平均 PSD 丢弃了时间结构与相位信息，SVM 只在旧 train 上拟合、旧 val 上选超参数，与 ResNet 的容量和输入形式不可比。
3. 两者都**不是新的最终模型**，只是第一阶段存在性检测的对照；test/heldout 数字没有参与阈值、标准化或 SVM 超参数选择。
4. 源文件划分不等于采集会话独立；heldout 的 12 个文件与旧文件无重叠，但缺少中心频率/增益/采集时间/地点元数据，因此不能宣称跨会话或跨地点泛化。
5. 本任务的 heldout 指标与 EVAL6 的四个噪声档不是同一实验设计，此处只比较 clean 一档。
6. heldout 上 PSD+SVM 的文件等权与 pooled 指标都是 100%，且 1820 个窗口的判决函数全部远离 0（每个源文件的最小/最大判决值见 4.2），不是临界翻转造成的假象；但这一档只有 12 个源文件、3 个类别，不能据此宣称跨会话泛化。能量基线在`T0000_D00_S0100.mat` 上误报 12.86%，而在其余 3 个背景文件上为 0%–0.71%，正是源文件间功率差异的直接体现。

## 6. 校验

`verification.json`：38/38 项通过，失败 0 项。

| 检查 | 结果 |
|---|---|
| old index sha256 matches the frozen value | 通过 |
| heldout index sha256 matches the frozen value | 通过 |
| heldout source files are disjoint from old train/val/test files | 通过 |
| each split has three categories with at least one source file | 通过 |
| canonical indices keep four source files per heldout category | 通过 |
| the 14 frozen inputs (original code, index, checkpoints) are unchanged | 通过 |
| energy_threshold/test: one row per index window, same order | 通过 |
| energy_threshold/test: scores are finite and predictions binary | 通过 |
| energy_threshold/test: confusion counts conserve the window total | 通过 |
| energy_threshold/test: pooled metrics recomputed from the CSV agree | 通过 |
| energy_threshold/test: file-equal metrics recomputed from the CSV agree | 通过 |
| energy_threshold/test: per-file rates are all in [0, 1] | 通过 |
| energy_threshold/heldout: one row per index window, same order | 通过 |
| energy_threshold/heldout: scores are finite and predictions binary | 通过 |
| energy_threshold/heldout: confusion counts conserve the window total | 通过 |
| energy_threshold/heldout: pooled metrics recomputed from the CSV agree | 通过 |
| energy_threshold/heldout: file-equal metrics recomputed from the CSV agree | 通过 |
| energy_threshold/heldout: per-file rates are all in [0, 1] | 通过 |
| psd_svm/test: one row per index window, same order | 通过 |
| psd_svm/test: scores are finite and predictions binary | 通过 |
| psd_svm/test: confusion counts conserve the window total | 通过 |
| psd_svm/test: pooled metrics recomputed from the CSV agree | 通过 |
| psd_svm/test: file-equal metrics recomputed from the CSV agree | 通过 |
| psd_svm/test: per-file rates are all in [0, 1] | 通过 |
| psd_svm/heldout: one row per index window, same order | 通过 |
| psd_svm/heldout: scores are finite and predictions binary | 通过 |
| psd_svm/heldout: confusion counts conserve the window total | 通过 |
| psd_svm/heldout: pooled metrics recomputed from the CSV agree | 通过 |
| psd_svm/heldout: file-equal metrics recomputed from the CSV agree | 通过 |
| psd_svm/heldout: per-file rates are all in [0, 1] | 通过 |
| PSD feature cache matches its recorded dtype/shape/SHA256 | 通过 |
| energy score cache matches its recorded dtype/shape/SHA256 | 通过 |
| standardizer was fitted on the old train split only | 通过 |
| standardizer statistics recomputed from the cached train features agree | 通过 |
| SVM selection only used the old val split and a frozen candidate grid | 通过 |
| energy threshold recomputed from the cache reproduces selection.json | 通过 |
| energy selection touched only train/val scores | 通过 |
| cached PSD features match an independent NumPy/FFT re-derivation | 通过 |

## 7. 产物

- `energy_threshold`: `selection.json`, `test_window_predictions.csv`, `heldout_window_predictions.csv`, `metrics.csv`, `energy_distributions.png`, `stage_manifest.json`
- `psd_svm`: `model.joblib`, `selection.json`, `svm_candidates.csv`, `test_window_predictions.csv`, `heldout_window_predictions.csv`, `metrics.csv`, `psd_examples.png`, `feature_manifest.json`, `feature_reference_check.json`, `model_file_info.json`
- `root`: `comparison_metrics.csv`, `experiment_manifest.json`, `report.md`, `verification.json`
- `cache`: `energy_scores.npy`, `energy_scores_metadata.json`, `energy_powers.npy`, `energy_powers_metadata.json`, `energy_cache.json`, `psd_features.npy`, `psd_features_metadata.json`, `psd_features_split_index.json`

