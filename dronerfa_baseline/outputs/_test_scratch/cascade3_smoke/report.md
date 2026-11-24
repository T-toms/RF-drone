# CASCADE3：第一阶段检测 + 第二阶段机型分类直接串联（三分类）

本报告把已验收的第一阶段 Background/Drone 检测器与已验收的第二阶段 Phantom 4 Pro / Air 2S 分类器**直接串联**，得到 Background / DJI Phantom 4 Pro / DJI Air 2S 三分类结果。三个训练种子各自串联（2026-2026、2027-2027、2028-2028），不做跨种子投票或集成。

冻结协议：RF0 I/Q、1,000,000 点窗、`n_fft=512`、`hop_length=256`、`image_size=256`；每个数据集/条件/窗口只生成一次噪声与一次 `1x256x256` float32 特征张量，同一张张量先给第一阶段，通过后原样交给第二阶段；噪声 RNG 位置取**完整 split 的行号**（背景行同样占位），seed 22026、epoch 0。第一阶段规则 `logit_drone - logit_background > 0` 才执行第二阶段，第二阶段规则 `logit_air2s - logit_phantom > 0` 判 Air 2S，否则 Phantom 4 Pro。不重训、不调门槛、不连接既有预测 CSV，全部从原始 IQ 重算。

## 1. 结论摘要（文件等权主口径）

主口径：先算每个源文件在其真实类上的正确率，类内源文件等权，再三类等权（file-equal accuracy）；表中同时列出 pooled 窗口口径，两者不可混用。`four_condition_mean` 行先在每个 seed 内对四档求均值，再对三个 seed 求均值与样本标准差。

| 数据集 | 条件 | 三 seed 文件等权准确率均值 | 样本标准差 | pooled 准确率均值 | 宏召回均值 | 宏 F1 均值 |
|---|---|---:|---:|---:|---:|---:|
| old_test | clean | 100.00% | 0.00pp | 100.00% | 100.00% | 66.67% |

## 2. 任务与口径边界

- 本任务是**评价**：不训练模型、不用新 heldout 选择配置或门槛、不做噪声方式实验。
- 第三类“Background”不是新模型：它等于第一阶段判为背景（margin <= 0）的结果，此时 `stage2_executed=false`，第二阶段 logits 与 margin 留空，不做任何伪造。
- 新 heldout 已参与此前的 STFT 点数消融与第二阶段选型背景，因此这里是**描述性留出评价**，不能称“全新未触碰的最终测试集”，也不能称跨会话或跨场景泛化。
- 新 heldout 每类只有 4 个源文件、旧 test 每类只有 1 个源文件；四档条件是人工复 AWGN，不等于现场绝对 SNR。
- 旧 test 只做 clean：四档噪声只在 heldout 上评价，两个数据集的行数与条件数因此不同。

## 3. 数据与规模

| 数据集 | split | 条件 | 每个 seed 级联的预测行数 | 类别窗口 |
|---|---|---|---:|---|
| old_test | test | clean | 145 | Background 140 / Phantom 5 / Air2S 0 |

合计预测行数 145（heldout 21,840 + 旧 test clean 1,320），主键为 (dataset, condition, training_seed, dataset_index_position)。

## 4. 每个 seed/条件的完整指标

| 数据集 | 条件 | seed | 文件等权准确率 | pooled 准确率 | 宏召回 | 宏精确率 | 宏 F1 | Background recall | Phantom recall | Air2S recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| old_test | clean | 2026 | 100.00% | 100.00% | 100.00% | 100.00% | 66.67% | 100.00% | 100.00% | n/a |

## 5. 跨 seed 均值与样本标准差（n=3，仅描述性）

| 数据集 | 条件 | 指标 | seed 值 | 均值 | 样本标准差 |
|---|---|---|---|---:|---:|
| old_test | clean | file_equal_accuracy | 1 | 100.00% | 0.00pp |
| old_test | clean | file_equal_background_recall | 1 | 100.00% | 0.00pp |
| old_test | clean | file_equal_phantom_recall | 1 | 100.00% | 0.00pp |
| old_test | clean | file_equal_air2s_recall | nan | nan% | 0.00pp |
| old_test | clean | pooled_accuracy | 1 | 100.00% | 0.00pp |
| old_test | clean | macro_recall | 1 | 100.00% | 0.00pp |
| old_test | clean | macro_precision | 1 | 100.00% | 0.00pp |
| old_test | clean | macro_f1 | 0.6666666667 | 66.67% | 0.00pp |
| old_test | clean | recall_background | 1 | 100.00% | 0.00pp |
| old_test | clean | recall_phantom | 1 | 100.00% | 0.00pp |
| old_test | clean | recall_air2s | nan | nan% | 0.00pp |
| old_test | clean | stage1_background_false_alarm_rate | 0 | 0.00% | 0.00pp |
| old_test | clean | stage1_drone_miss_rate | 0 | 0.00% | 0.00pp |
| old_test | clean | stage2_conditional_accuracy | 1 | 100.00% | 0.00pp |

## 6. 第一阶段在级联中的行为

| 数据集 | 条件 | seed | 背景误报率 | 无人机漏检率 | 通过第一阶段的真实无人机 | 第二阶段条件准确率 |
|---|---|---:|---:|---:|---:|---:|
| old_test | clean | 2026 | 0.00% (0 窗) | 0.00% (0 窗) | 5 | 100.00% |

“第二阶段条件准确率”只在真实无人机且第一阶段 margin > 0 的窗口上计算，因此它描述的是第二阶段在级联真实输入上的表现，而不是 oracle 二分类结果。

## 7. 最终错误分解

| 数据集 | 条件 | seed | Background->Phantom | Background->Air2S | Phantom->Background | Air2S->Background | Phantom->Air2S | Air2S->Phantom | 总错误 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old_test | clean | 2026 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 8. 逐源文件（支持文件等权主口径）

| 数据集 | 条件 | seed | 源文件 | 真实类 | 窗数 | 正确 | 正确率 | 通过第一阶段 |
|---|---|---:|---|---|---:|---:|---:|---:|
| old_test | clean | 2026 | T0000_D00_S0011.mat | Background | 140 | 140 | 100.00% | 0 |
| old_test | clean | 2026 | T0010_D00_S0011.mat | DJI Phantom 4 Pro | 5 | 5 | 100.00% | 5 |

## 9. 一致性核对

- 第一阶段 nfft512 与既有同协议预测的一致性：比较 145 行，最大 |Δlogit_background| = 0.0，最大 |Δlogit_drone| = 0.0，最大 |Δlogit_margin| = 0.0，判定不一致数 = 0。
- 逐窗特征摘要与既有第一阶段 nfft512 摘要文件逐字节一致：False。
- 第二阶段**不**与既有 oracle 二分类 CSV 比较：oracle 结果的 RNG 行位置是“过滤后的无人机行号”，级联按完整 split 行号生成噪声，两者本来就不该相同。
- 特征摘要文件位于 `feature_digests/`，每个窗口 32 字节；特征缓存清单见 `prediction_cache_manifest.json`。
- 冻结输入（配置、两个索引、六个检查点、14 个既有冻结输入）在推理前后各核对一次，见 `frozen_inputs_verification_before.json` 与 `frozen_inputs_verification.json`。

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
