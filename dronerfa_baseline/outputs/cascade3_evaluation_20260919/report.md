# CASCADE3：第一阶段检测 + 第二阶段机型分类直接串联（三分类）

本报告把已验收的第一阶段 Background/Drone 检测器与已验收的第二阶段 Phantom 4 Pro / Air 2S 分类器**直接串联**，得到 Background / DJI Phantom 4 Pro / DJI Air 2S 三分类结果。三个训练种子各自串联（2026-2026、2027-2027、2028-2028），不做跨种子投票或集成。

冻结协议：RF0 I/Q、1,000,000 点窗、`n_fft=512`、`hop_length=256`、`image_size=256`；每个数据集/条件/窗口只生成一次噪声与一次 `1x256x256` float32 特征张量，同一张张量先给第一阶段，通过后原样交给第二阶段；噪声 RNG 位置取**完整 split 的行号**（背景行同样占位），seed 22026、epoch 0。第一阶段规则 `logit_drone - logit_background > 0` 才执行第二阶段，第二阶段规则 `logit_air2s - logit_phantom > 0` 判 Air 2S，否则 Phantom 4 Pro。不重训、不调门槛、不连接既有预测 CSV，全部从原始 IQ 重算。

## 1. 结论摘要（文件等权主口径）

主口径：先算每个源文件在其真实类上的正确率，类内源文件等权，再三类等权（file-equal accuracy）；表中同时列出 pooled 窗口口径，两者不可混用。`four_condition_mean` 行先在每个 seed 内对四档求均值，再对三个 seed 求均值与样本标准差。

| 数据集 | 条件 | 三 seed 文件等权准确率均值 | 样本标准差 | pooled 准确率均值 | 宏召回均值 | 宏 F1 均值 |
|---|---|---:|---:|---:|---:|---:|
| heldout | clean | 99.75% | 0.23pp | 99.74% | 99.74% | 99.74% |
| heldout | -10 | 99.98% | 0.03pp | 99.98% | 99.98% | 99.98% |
| heldout | -12 | 99.63% | 0.11pp | 99.62% | 99.64% | 99.61% |
| heldout | -15 | 95.14% | 0.73pp | 94.89% | 95.20% | 94.89% |
| heldout | four_condition_mean | 98.62% | 0.27pp | 98.56% | 98.64% | 98.56% |
| old_test | clean | 98.97% | 0.99pp | 99.02% | 98.97% | 99.01% |

## 2. 任务与口径边界

- 本任务是**评价**：不训练模型、不用新 heldout 选择配置或门槛、不做噪声方式实验。
- 第三类“Background”不是新模型：它等于第一阶段判为背景（margin <= 0）的结果，此时 `stage2_executed=false`，第二阶段 logits 与 margin 留空，不做任何伪造。
- 新 heldout 已参与此前的 STFT 点数消融与第二阶段选型背景，因此这里是**描述性留出评价**，不能称“全新未触碰的最终测试集”，也不能称跨会话或跨场景泛化。
- 新 heldout 每类只有 4 个源文件、旧 test 每类只有 1 个源文件；四档条件是人工复 AWGN，不等于现场绝对 SNR。
- 旧 test 只做 clean：四档噪声只在 heldout 上评价，两个数据集的行数与条件数因此不同。

## 3. 数据与规模

| 数据集 | split | 条件 | 每个 seed 级联的预测行数 | 类别窗口 |
|---|---|---|---:|---|
| heldout | heldout | clean, -10, -12, -15 | 1820 | Background 560 / Phantom 610 / Air2S 650 |
| old_test | test | clean | 440 | Background 140 / Phantom 140 / Air2S 160 |

合计预测行数 23160（heldout 21,840 + 旧 test clean 1,320），主键为 (dataset, condition, training_seed, dataset_index_position)。

## 4. 每个 seed/条件的完整指标

| 数据集 | 条件 | seed | 文件等权准确率 | pooled 准确率 | 宏召回 | 宏精确率 | 宏 F1 | Background recall | Phantom recall | Air2S recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heldout | -10 | 2026 | 99.94% | 99.95% | 99.94% | 99.95% | 99.94% | 99.82% | 100.00% | 100.00% |
| heldout | -10 | 2027 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| heldout | -10 | 2028 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| heldout | -12 | 2026 | 99.52% | 99.51% | 99.53% | 99.48% | 99.50% | 99.82% | 100.00% | 98.77% |
| heldout | -12 | 2027 | 99.63% | 99.62% | 99.63% | 99.59% | 99.61% | 99.82% | 100.00% | 99.08% |
| heldout | -12 | 2028 | 99.74% | 99.73% | 99.74% | 99.71% | 99.72% | 100.00% | 100.00% | 99.23% |
| heldout | -15 | 2026 | 94.58% | 94.29% | 94.66% | 94.70% | 94.27% | 100.00% | 99.67% | 84.31% |
| heldout | -15 | 2027 | 94.87% | 94.62% | 94.92% | 94.85% | 94.63% | 99.64% | 98.03% | 87.08% |
| heldout | -15 | 2028 | 95.96% | 95.77% | 96.02% | 95.89% | 95.77% | 99.82% | 98.69% | 89.54% |
| heldout | clean | 2026 | 99.56% | 99.56% | 99.56% | 99.54% | 99.55% | 99.82% | 98.85% | 100.00% |
| heldout | clean | 2027 | 99.68% | 99.67% | 99.67% | 99.70% | 99.68% | 100.00% | 99.02% | 100.00% |
| heldout | clean | 2028 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| old_test | clean | 2026 | 99.29% | 99.32% | 99.29% | 99.30% | 99.29% | 100.00% | 97.86% | 100.00% |
| old_test | clean | 2027 | 97.86% | 97.95% | 97.86% | 98.22% | 97.98% | 100.00% | 93.57% | 100.00% |
| old_test | clean | 2028 | 99.76% | 99.77% | 99.76% | 99.79% | 99.78% | 100.00% | 99.29% | 100.00% |

## 5. 跨 seed 均值与样本标准差（n=3，仅描述性）

| 数据集 | 条件 | 指标 | seed 值 | 均值 | 样本标准差 |
|---|---|---|---|---:|---:|
| heldout | clean | file_equal_accuracy | 0.9955853175,0.9967708333,1 | 99.75% | 0.23pp |
| heldout | clean | file_equal_background_recall | 0.9982142857,1,1 | 99.94% | 0.10pp |
| heldout | clean | file_equal_phantom_recall | 0.9885416667,0.9903125,1 | 99.30% | 0.62pp |
| heldout | clean | file_equal_air2s_recall | 1,1,1 | 100.00% | 0.00pp |
| heldout | clean | pooled_accuracy | 0.9956043956,0.9967032967,1 | 99.74% | 0.23pp |
| heldout | clean | macro_recall | 0.9955796253,0.9967213115,1 | 99.74% | 0.23pp |
| heldout | clean | macro_precision | 0.9953654702,0.9969512195,1 | 99.74% | 0.24pp |
| heldout | clean | macro_f1 | 0.9954519162,0.99682116,1 | 99.74% | 0.23pp |
| heldout | clean | recall_background | 0.9982142857,1,1 | 99.94% | 0.10pp |
| heldout | clean | recall_phantom | 0.9885245902,0.9901639344,1 | 99.29% | 0.62pp |
| heldout | clean | recall_air2s | 1,1,1 | 100.00% | 0.00pp |
| heldout | clean | stage1_background_false_alarm_rate | 0.001785714286,0,0 | 0.06% | 0.10pp |
| heldout | clean | stage1_drone_miss_rate | 0.005555555556,0,0 | 0.19% | 0.32pp |
| heldout | clean | stage2_conditional_accuracy | 1,0.9952380952,1 | 99.84% | 0.27pp |
| heldout | -10 | file_equal_accuracy | 0.9994047619,1,1 | 99.98% | 0.03pp |
| heldout | -10 | file_equal_background_recall | 0.9982142857,1,1 | 99.94% | 0.10pp |
| heldout | -10 | file_equal_phantom_recall | 1,1,1 | 100.00% | 0.00pp |
| heldout | -10 | file_equal_air2s_recall | 1,1,1 | 100.00% | 0.00pp |
| heldout | -10 | pooled_accuracy | 0.9994505495,1,1 | 99.98% | 0.03pp |
| heldout | -10 | macro_recall | 0.9994047619,1,1 | 99.98% | 0.03pp |
| heldout | -10 | macro_precision | 0.9994879672,1,1 | 99.98% | 0.03pp |
| heldout | -10 | macro_f1 | 0.9994459018,1,1 | 99.98% | 0.03pp |
| heldout | -10 | recall_background | 0.9982142857,1,1 | 99.94% | 0.10pp |
| heldout | -10 | recall_phantom | 1,1,1 | 100.00% | 0.00pp |
| heldout | -10 | recall_air2s | 1,1,1 | 100.00% | 0.00pp |
| heldout | -10 | stage1_background_false_alarm_rate | 0.001785714286,0,0 | 0.06% | 0.10pp |
| heldout | -10 | stage1_drone_miss_rate | 0,0,0 | 0.00% | 0.00pp |
| heldout | -10 | stage2_conditional_accuracy | 1,1,1 | 100.00% | 0.00pp |
| heldout | -12 | file_equal_accuracy | 0.9952265212,0.9963029101,0.9973958333 | 99.63% | 0.11pp |
| heldout | -12 | file_equal_background_recall | 0.9982142857,0.9982142857,1 | 99.88% | 0.10pp |
| heldout | -12 | file_equal_phantom_recall | 1,1,1 | 100.00% | 0.00pp |
| heldout | -12 | file_equal_air2s_recall | 0.9874652778,0.9906944444,0.9921875 | 99.01% | 0.24pp |
| heldout | -12 | pooled_accuracy | 0.9950549451,0.9961538462,0.9972527473 | 99.62% | 0.11pp |
| heldout | -12 | macro_recall | 0.9953021978,0.9963278388,0.9974358974 | 99.64% | 0.11pp |
| heldout | -12 | macro_precision | 0.9947784809,0.9959433811,0.9970501475 | 99.59% | 0.11pp |
| heldout | -12 | macro_f1 | 0.99501788,0.9961241241,0.9972315172 | 99.61% | 0.11pp |
| heldout | -12 | recall_background | 0.9982142857,0.9982142857,1 | 99.88% | 0.10pp |
| heldout | -12 | recall_phantom | 1,1,1 | 100.00% | 0.00pp |
| heldout | -12 | recall_air2s | 0.9876923077,0.9907692308,0.9923076923 | 99.03% | 0.24pp |
| heldout | -12 | stage1_background_false_alarm_rate | 0.001785714286,0.001785714286,0 | 0.12% | 0.10pp |
| heldout | -12 | stage1_drone_miss_rate | 0.006349206349,0.004761904762,0.003968253968 | 0.50% | 0.12pp |
| heldout | -12 | stage2_conditional_accuracy | 1,1,1 | 100.00% | 0.00pp |
| heldout | -15 | file_equal_accuracy | 0.9457523148,0.9487400794,0.9596362434 | 95.14% | 0.73pp |
| heldout | -15 | file_equal_background_recall | 1,0.9964285714,0.9982142857 | 99.82% | 0.18pp |
| heldout | -15 | file_equal_phantom_recall | 0.996875,0.9809375,0.9873958333 | 98.84% | 0.80pp |
| heldout | -15 | file_equal_air2s_recall | 0.8403819444,0.8688541667,0.8932986111 | 86.75% | 2.65pp |
| heldout | -15 | pooled_accuracy | 0.9428571429,0.9461538462,0.9576923077 | 94.89% | 0.78pp |
| heldout | -15 | macro_recall | 0.9465994115,0.9491752237,0.9601613823 | 95.20% | 0.72pp |
| heldout | -15 | macro_precision | 0.9469638395,0.9484515499,0.9588560222 | 95.14% | 0.65pp |
| heldout | -15 | macro_f1 | 0.9426545265,0.946273689,0.9577288195 | 94.89% | 0.79pp |
| heldout | -15 | recall_background | 1,0.9964285714,0.9982142857 | 99.82% | 0.18pp |
| heldout | -15 | recall_phantom | 0.9967213115,0.9803278689,0.9868852459 | 98.80% | 0.83pp |
| heldout | -15 | recall_air2s | 0.8430769231,0.8707692308,0.8953846154 | 86.97% | 2.62pp |
| heldout | -15 | stage1_background_false_alarm_rate | 0,0.003571428571,0.001785714286 | 0.18% | 0.18pp |
| heldout | -15 | stage1_drone_miss_rate | 0.07777777778,0.06746031746,0.05476190476 | 6.67% | 1.15pp |
| heldout | -15 | stage2_conditional_accuracy | 0.9948364888,0.9906382979,0.9941225861 | 99.32% | 0.22pp |
| heldout | four_condition_mean | file_equal_accuracy | 0.9839922288,0.9854534557,0.9892580192 | 98.62% | 0.27pp |
| heldout | four_condition_mean | file_equal_background_recall | 0.9986607143,0.9986607143,0.9995535714 | 99.90% | 0.05pp |
| heldout | four_condition_mean | file_equal_phantom_recall | 0.9963541667,0.9928125,0.9968489583 | 99.53% | 0.22pp |
| heldout | four_condition_mean | file_equal_air2s_recall | 0.9569618056,0.9648871528,0.9713715278 | 96.44% | 0.72pp |
| heldout | four_condition_mean | pooled_accuracy | 0.9832417582,0.9847527473,0.9887362637 | 98.56% | 0.28pp |
| heldout | four_condition_mean | macro_recall | 0.9842214991,0.9855560935,0.9893993199 | 98.64% | 0.27pp |
| heldout | four_condition_mean | macro_precision | 0.9841489394,0.9853365376,0.9889765424 | 98.62% | 0.25pp |
| heldout | four_condition_mean | macro_f1 | 0.9831425561,0.9848047433,0.9887400842 | 98.56% | 0.29pp |
| heldout | four_condition_mean | recall_background | 0.9986607143,0.9986607143,0.9995535714 | 99.90% | 0.05pp |
| heldout | four_condition_mean | recall_phantom | 0.9963114754,0.9926229508,0.9967213115 | 99.52% | 0.23pp |
| heldout | four_condition_mean | recall_air2s | 0.9576923077,0.9653846154,0.9719230769 | 96.50% | 0.71pp |
| heldout | four_condition_mean | stage1_background_false_alarm_rate | 0.001339285714,0.001339285714,0.0004464285714 | 0.10% | 0.05pp |
| heldout | four_condition_mean | stage1_drone_miss_rate | 0.02242063492,0.01805555556,0.01468253968 | 1.84% | 0.39pp |
| heldout | four_condition_mean | stage2_conditional_accuracy | 0.9987091222,0.9964690983,0.9985306465 | 99.79% | 0.12pp |
| old_test | clean | file_equal_accuracy | 0.9928571429,0.9785714286,0.9976190476 | 98.97% | 0.99pp |
| old_test | clean | file_equal_background_recall | 1,1,1 | 100.00% | 0.00pp |
| old_test | clean | file_equal_phantom_recall | 0.9785714286,0.9357142857,0.9928571429 | 96.90% | 2.97pp |
| old_test | clean | file_equal_air2s_recall | 1,1,1 | 100.00% | 0.00pp |
| old_test | clean | pooled_accuracy | 0.9931818182,0.9795454545,0.9977272727 | 99.02% | 0.95pp |
| old_test | clean | macro_recall | 0.9928571429,0.9785714286,0.9976190476 | 98.97% | 0.99pp |
| old_test | clean | macro_precision | 0.993006993,0.9822485207,0.9979296066 | 99.11% | 0.80pp |
| old_test | clean | macro_f1 | 0.9928563228,0.9798113483,0.9977668353 | 99.01% | 0.93pp |
| old_test | clean | recall_background | 1,1,1 | 100.00% | 0.00pp |
| old_test | clean | recall_phantom | 0.9785714286,0.9357142857,0.9928571429 | 96.90% | 2.97pp |
| old_test | clean | recall_air2s | 1,1,1 | 100.00% | 0.00pp |
| old_test | clean | stage1_background_false_alarm_rate | 0,0,0 | 0.00% | 0.00pp |
| old_test | clean | stage1_drone_miss_rate | 0.01,0,0 | 0.33% | 0.58pp |
| old_test | clean | stage2_conditional_accuracy | 1,0.97,0.9966666667 | 98.89% | 1.64pp |

## 6. 第一阶段在级联中的行为

| 数据集 | 条件 | seed | 背景误报率 | 无人机漏检率 | 通过第一阶段的真实无人机 | 第二阶段条件准确率 |
|---|---|---:|---:|---:|---:|---:|
| heldout | -10 | 2026 | 0.18% (1 窗) | 0.00% (0 窗) | 1260 | 100.00% |
| heldout | -10 | 2027 | 0.00% (0 窗) | 0.00% (0 窗) | 1260 | 100.00% |
| heldout | -10 | 2028 | 0.00% (0 窗) | 0.00% (0 窗) | 1260 | 100.00% |
| heldout | -12 | 2026 | 0.18% (1 窗) | 0.63% (8 窗) | 1252 | 100.00% |
| heldout | -12 | 2027 | 0.18% (1 窗) | 0.48% (6 窗) | 1254 | 100.00% |
| heldout | -12 | 2028 | 0.00% (0 窗) | 0.40% (5 窗) | 1255 | 100.00% |
| heldout | -15 | 2026 | 0.00% (0 窗) | 7.78% (98 窗) | 1162 | 99.48% |
| heldout | -15 | 2027 | 0.36% (2 窗) | 6.75% (85 窗) | 1175 | 99.06% |
| heldout | -15 | 2028 | 0.18% (1 窗) | 5.48% (69 窗) | 1191 | 99.41% |
| heldout | clean | 2026 | 0.18% (1 窗) | 0.56% (7 窗) | 1253 | 100.00% |
| heldout | clean | 2027 | 0.00% (0 窗) | 0.00% (0 窗) | 1260 | 99.52% |
| heldout | clean | 2028 | 0.00% (0 窗) | 0.00% (0 窗) | 1260 | 100.00% |
| old_test | clean | 2026 | 0.00% (0 窗) | 1.00% (3 窗) | 297 | 100.00% |
| old_test | clean | 2027 | 0.00% (0 窗) | 0.00% (0 窗) | 300 | 97.00% |
| old_test | clean | 2028 | 0.00% (0 窗) | 0.00% (0 窗) | 300 | 99.67% |

“第二阶段条件准确率”只在真实无人机且第一阶段 margin > 0 的窗口上计算，因此它描述的是第二阶段在级联真实输入上的表现，而不是 oracle 二分类结果。

## 7. 最终错误分解

| 数据集 | 条件 | seed | Background->Phantom | Background->Air2S | Phantom->Background | Air2S->Background | Phantom->Air2S | Air2S->Phantom | 总错误 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heldout | -10 | 2026 | 0 | 1 | 0 | 0 | 0 | 0 | 1 |
| heldout | -10 | 2027 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| heldout | -10 | 2028 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| heldout | -12 | 2026 | 0 | 1 | 0 | 8 | 0 | 0 | 9 |
| heldout | -12 | 2027 | 0 | 1 | 0 | 6 | 0 | 0 | 7 |
| heldout | -12 | 2028 | 0 | 0 | 0 | 5 | 0 | 0 | 5 |
| heldout | -15 | 2026 | 0 | 0 | 0 | 98 | 2 | 4 | 104 |
| heldout | -15 | 2027 | 0 | 2 | 1 | 84 | 11 | 0 | 98 |
| heldout | -15 | 2028 | 0 | 1 | 1 | 68 | 7 | 0 | 77 |
| heldout | clean | 2026 | 0 | 1 | 7 | 0 | 0 | 0 | 8 |
| heldout | clean | 2027 | 0 | 0 | 0 | 0 | 6 | 0 | 6 |
| heldout | clean | 2028 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| old_test | clean | 2026 | 0 | 0 | 3 | 0 | 0 | 0 | 3 |
| old_test | clean | 2027 | 0 | 0 | 0 | 0 | 9 | 0 | 9 |
| old_test | clean | 2028 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |

## 8. 逐源文件（支持文件等权主口径）

| 数据集 | 条件 | seed | 源文件 | 真实类 | 窗数 | 正确 | 正确率 | 通过第一阶段 |
|---|---|---:|---|---|---:|---:|---:|---:|
| heldout | -10 | 2026 | T0000_D00_S0100.mat | Background | 140 | 139 | 99.29% | 1 |
| heldout | -10 | 2026 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2026 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2026 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2026 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2026 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2026 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2026 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2026 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2026 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 180 | 100.00% | 180 |
| heldout | -10 | 2026 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2026 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2027 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2027 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2027 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2027 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2027 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2027 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2027 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2027 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2027 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2027 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 180 | 100.00% | 180 |
| heldout | -10 | 2027 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2027 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2028 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2028 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2028 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2028 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -10 | 2028 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2028 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2028 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2028 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -10 | 2028 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2028 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 180 | 100.00% | 180 |
| heldout | -10 | 2028 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | -10 | 2028 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2026 | T0000_D00_S0100.mat | Background | 140 | 139 | 99.29% | 1 |
| heldout | -12 | 2026 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2026 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2026 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2026 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 160 | 100.00% | 160 |
| heldout | -12 | 2026 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2026 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2026 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2026 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 158 | 98.75% | 158 |
| heldout | -12 | 2026 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 179 | 99.44% | 179 |
| heldout | -12 | 2026 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 157 | 98.12% | 157 |
| heldout | -12 | 2026 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 148 | 98.67% | 148 |
| heldout | -12 | 2027 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2027 | T0000_D00_S0101.mat | Background | 140 | 139 | 99.29% | 1 |
| heldout | -12 | 2027 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2027 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2027 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 160 | 100.00% | 160 |
| heldout | -12 | 2027 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2027 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2027 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2027 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 158 | 98.75% | 158 |
| heldout | -12 | 2027 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 179 | 99.44% | 179 |
| heldout | -12 | 2027 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 158 | 98.75% | 158 |
| heldout | -12 | 2027 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 149 | 99.33% | 149 |
| heldout | -12 | 2028 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2028 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2028 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2028 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -12 | 2028 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 160 | 100.00% | 160 |
| heldout | -12 | 2028 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2028 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2028 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -12 | 2028 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 158 | 98.75% | 158 |
| heldout | -12 | 2028 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 180 | 100.00% | 180 |
| heldout | -12 | 2028 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 157 | 98.12% | 157 |
| heldout | -12 | 2028 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 150 | 100.00% | 150 |
| heldout | -15 | 2026 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2026 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2026 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2026 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2026 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 158 | 98.75% | 160 |
| heldout | -15 | 2026 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -15 | 2026 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -15 | 2026 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -15 | 2026 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 136 | 85.00% | 136 |
| heldout | -15 | 2026 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 161 | 89.44% | 163 |
| heldout | -15 | 2026 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 135 | 84.38% | 137 |
| heldout | -15 | 2026 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 116 | 77.33% | 116 |
| heldout | -15 | 2027 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2027 | T0000_D00_S0101.mat | Background | 140 | 138 | 98.57% | 2 |
| heldout | -15 | 2027 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2027 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2027 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 151 | 94.38% | 160 |
| heldout | -15 | 2027 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 148 | 98.67% | 149 |
| heldout | -15 | 2027 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -15 | 2027 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 149 | 99.33% | 150 |
| heldout | -15 | 2027 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 136 | 85.00% | 136 |
| heldout | -15 | 2027 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 165 | 91.67% | 165 |
| heldout | -15 | 2027 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 139 | 86.88% | 139 |
| heldout | -15 | 2027 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 126 | 84.00% | 126 |
| heldout | -15 | 2028 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2028 | T0000_D00_S0101.mat | Background | 140 | 139 | 99.29% | 1 |
| heldout | -15 | 2028 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2028 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | -15 | 2028 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 153 | 95.62% | 160 |
| heldout | -15 | 2028 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 149 | 99.33% | 149 |
| heldout | -15 | 2028 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -15 | 2028 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | -15 | 2028 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 143 | 89.38% | 143 |
| heldout | -15 | 2028 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 170 | 94.44% | 170 |
| heldout | -15 | 2028 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 140 | 87.50% | 140 |
| heldout | -15 | 2028 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 129 | 86.00% | 129 |
| heldout | clean | 2026 | T0000_D00_S0100.mat | Background | 140 | 139 | 99.29% | 1 |
| heldout | clean | 2026 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2026 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2026 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2026 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 158 | 98.75% | 158 |
| heldout | clean | 2026 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | clean | 2026 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 148 | 98.67% | 148 |
| heldout | clean | 2026 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 147 | 98.00% | 147 |
| heldout | clean | 2026 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | clean | 2026 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 180 | 100.00% | 180 |
| heldout | clean | 2026 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | clean | 2026 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 150 | 100.00% | 150 |
| heldout | clean | 2027 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2027 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2027 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2027 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2027 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 157 | 98.12% | 160 |
| heldout | clean | 2027 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 148 | 98.67% | 150 |
| heldout | clean | 2027 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 149 | 99.33% | 150 |
| heldout | clean | 2027 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | clean | 2027 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | clean | 2027 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 180 | 100.00% | 180 |
| heldout | clean | 2027 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | clean | 2027 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 150 | 100.00% | 150 |
| heldout | clean | 2028 | T0000_D00_S0100.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2028 | T0000_D00_S0101.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2028 | T0000_D00_S0110.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2028 | T0000_D00_S0111.mat | Background | 140 | 140 | 100.00% | 0 |
| heldout | clean | 2028 | T0010_D00_S0100.mat | DJI Phantom 4 Pro | 160 | 160 | 100.00% | 160 |
| heldout | clean | 2028 | T0010_D00_S0101.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | clean | 2028 | T0010_D00_S0110.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | clean | 2028 | T0010_D00_S0111.mat | DJI Phantom 4 Pro | 150 | 150 | 100.00% | 150 |
| heldout | clean | 2028 | T0101_D00_S0100.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | clean | 2028 | T0101_D00_S0101.mat | DJI Air 2S | 180 | 180 | 100.00% | 180 |
| heldout | clean | 2028 | T0101_D00_S0110.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| heldout | clean | 2028 | T0101_D00_S0111.mat | DJI Air 2S | 150 | 150 | 100.00% | 150 |
| old_test | clean | 2026 | T0000_D00_S0011.mat | Background | 140 | 140 | 100.00% | 0 |
| old_test | clean | 2026 | T0010_D00_S0011.mat | DJI Phantom 4 Pro | 140 | 137 | 97.86% | 137 |
| old_test | clean | 2026 | T0101_D00_S0011.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| old_test | clean | 2027 | T0000_D00_S0011.mat | Background | 140 | 140 | 100.00% | 0 |
| old_test | clean | 2027 | T0010_D00_S0011.mat | DJI Phantom 4 Pro | 140 | 131 | 93.57% | 140 |
| old_test | clean | 2027 | T0101_D00_S0011.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |
| old_test | clean | 2028 | T0000_D00_S0011.mat | Background | 140 | 140 | 100.00% | 0 |
| old_test | clean | 2028 | T0010_D00_S0011.mat | DJI Phantom 4 Pro | 140 | 139 | 99.29% | 140 |
| old_test | clean | 2028 | T0101_D00_S0011.mat | DJI Air 2S | 160 | 160 | 100.00% | 160 |

## 9. 一致性核对

- 第一阶段 nfft512 与既有同协议预测的一致性：比较 23160 行，最大 |Δlogit_background| = 0.0，最大 |Δlogit_drone| = 0.0，最大 |Δlogit_margin| = 0.0，判定不一致数 = 0。
- 逐窗特征摘要与既有第一阶段 nfft512 摘要文件逐字节一致：True。
- 第二阶段**不**与既有 oracle 二分类 CSV 比较：oracle 结果的 RNG 行位置是“过滤后的无人机行号”，级联按完整 split 行号生成噪声，两者本来就不该相同。
- 特征摘要文件位于 `feature_digests/`，每个窗口 32 字节；特征缓存清单见 `prediction_cache_manifest.json`。
- 冻结输入（配置、两个索引、六个检查点、14 个既有冻结输入）在推理前后各核对一次，见 `frozen_inputs_verification_before.json` 与 `frozen_inputs_verification.json`。
- 独立验证器：passed=True，failures=0（`verification.json`）。

## 10. 图

- `confusion_<dataset>_<condition>.png`：每个 seed 的三分类混淆矩阵（计数与按真实类占比）。
- `recall_by_condition.png`：三类文件等权召回率随噪声条件变化（三 seed 均值 ± 样本标准差）。
- `cascade_flow.png`：串联流程示意（同一张量、seed 配对、两条 margin 规则）。
- 不生成逐窗 PNG。

## 11. 未做与不能声称的内容

- 未训练、未微调、未改阈值、未改前端、未使用 heldout 做任何选择。
- 未做噪声方式对照、未加未知类、未做任意信号上传界面、未恢复任何定时器。
- 不能据此声称跨会话/跨场景/跨设备泛化，也不能把四档人工 AWGN 当作现场绝对 SNR 性能。
- 三个种子并非独立数据源：同一批窗口、同一份噪声被三个种子共享，跨 seed 标准差只反映模型初始化与训练顺序的差异。
