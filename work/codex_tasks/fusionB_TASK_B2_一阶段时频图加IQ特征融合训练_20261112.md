# 任务 B2：第一阶段 时频图 + IQ 手工特征 中间层拼接训练（2.4G 基线 L-b，5.8G 基线 H）

下发：Claude（协调者），2026-11-12。执行：Codex。
**不要发送 heartbeat。不要调用 DSH / dsh-cli / mcp__dsh。** 小问题自己决定并写进 REPORT.md。只在以下情况用 orchestration `ask`：§3 任一检查不过；读盘错误、显存/内存/磁盘不足无法绕过；本文与数据明显冲突。完成时发一次 `worker_done`。

## 0. 背景（已定，不需要重新论证）

- 用户 2026-11-12 转达老师意见：第一阶段（有无无人机）也可以把 IQ 特征和时频图一起输入。用户选定：**中间层拼接**（老师原话的“时频图展开 + 特征接在后面”以后单独对比，本任务不做）；2.4G 基线 = 任务 L 的 **L-b**，5.8G 基线 = 任务 H。
- 设计依据：`work/主线_做法B_时频图加IQ手工特征_设计草案_20261110.md` §1、§4；文献 `work/review_20261110/文献_做法B_特征融合.md`。要点：联合训练时强的一路（时频图）常把弱的一路压住，所以必须有“特征全置 0”的对照组 Z 和测试时打乱特征的检验。
- 特征已在 B1 算好：`work/fusionB_20261110/extract.py` 的 `feature()`，缓存 `E:/fusionB_features_20261110/`（19 列，列名 `feature_names.json`；`*_features_meta.csv` 保存了任务 H / bgA 原 meta 的完整行序，含 block_file、start、n_samples）。
- 2026-11-12 清理磁盘时删除了任务 H 时频图缓存（`E:/drone_lab_spec_L20_20261108/`，含 meta 与 checkpoint）、bgA 时频图缓存、任务 L 的缓存与 checkpoint。**IQ 都还在**（`E:/drone_lab_crop_20261107/`、`E:/drone_lab_bgA_20261109/`）。所以本任务要从 IQ 重建时频图，并**重训基线 R**（与 F、Z 同一份缓存、同一份代码，配对比较）。

## 1. 三个组（每个频段，种子 2026/2027/2028，各 40 epoch）

| 组 | 网络 | 输入 |
|---|---|---|
| R | `dronerfa_baseline/model.py` 的 `ResNet18(1,2)`，不改 | 时频图 |
| F | 融合网络（下），ResNet 部分与 R 结构相同 | 时频图 + 10 个特征 |
| Z | 与 F 完全相同的网络 | 时频图 + 特征**全部置 0**（标准化之后置 0，训练、验证、测试都是 0） |

融合网络：复用 `ResNet18` 的 stem、layer1–4、pool（写一个包装类调用这些子模块，**不改 model.py**），得 512 维 → `LayerNorm(512)`；特征 10 维 → `Linear(10,64)` → ReLU → `LayerNorm(64)`；拼接 576 维 → `Dropout(0.3)` → `Linear(576,2)`。初始化用 PyTorch 默认（ResNet 部分照 `_initialize_weights`）。

其余全部照基线：
- 2.4G 照任务 L 的 L-b：数据组成（data 无人机、v3 无人机、v3 背景、bgA、叠加无人机、叠加背景）、类内权重（`work/side_L_wifimix_20261110/train.py` 的 masses）、选模型（val FA_v3 + FA_bgA + Miss，再比 val loss）、阈值 thr_fa1（两种背景各自阈值取大）、val 不含叠加样本。
- 5.8G 照任务 H：`work/train_stage1_L20_20261108/train.py` 的类权重 1/count、选模型（val FA + Miss，再比 val loss）、thr_fa1。
- AdamW lr 1e-3、wd 1e-4、cosine 到 1e-5（T_max 40）、batch 64、AMP 自动选、`seed()`、cudnn.benchmark=True、无增强、不调参。

## 2. 特征

- 用 10 列（按此顺序）：`shape_factor, log_kurt_a, papr999_db, c42n, log_gamma_max, sigma_f, spec_flatness, occ_frac, duty, log1p_n_bursts`。
- 不用：6 个 `ctl_*`（绝对功率、底噪等，是来源阳性对照）；`c40n_abs, c41n_abs, c20n_abs`（B1b：对 3 dB 以下的弱内容极敏感，而标签差异本身很小）。这是协调者的决定，写进 REPORT。
- 计算函数：从 `extract.py` **复制** `config()`、`feature()` 和 `NOTCH` 的读取（不要 import extract.py）。
- 缺失值（NaN）：用该频段训练集（含叠加样本）的中位数填补，逐列计数写进 REPORT。
- 标准化：按频段，用训练集（含叠加样本）每列的均值、标准差做 z-score；之后截断到 [−10, 10]，逐列记录被截断的片数。val、test、所有对照都用训练集的参数。参数存成 json。

## 3. 开工前检查（任一不过就停下 ask）

1. **行序重建**：按 `build_spec_cache.py` 的规则（Task F `index_L20.csv`，keep=True，按 block_file、start 排序）重建 2.4G/5.8G train/val/test meta 和 rejected test meta；bgA 用 `work/bgA_20261109/` 的 L20 索引（keep=True）。重建结果的 sample_id 顺序必须与 B1 缓存的 `*_features_meta.csv` 逐行一致。
2. **特征复现**：每个频段、每个 split、bgA，各随机抽 3 行，从 IQ 用复制的 `feature()` 重算，与 B1 缓存最大绝对差 ≤ 1e-9。
3. **时频图复现**：重建的时频图与任务 H 原缓存无法直接比（已删）。改为：用重建缓存 + R 的训练日志与任务 H / L 原训练日志对比（只描述，不作为停止条件）。另外，随机抽 3 片，`common.spec()` 连算两次，结果完全相同。
4. **叠加可复现**：用 `work/side_L_wifimix_20261110/mix_train_drone.pairs.csv`、`mix_train_bg.pairs.csv` 和任务 H 的 `work/train_stage1_L20_20261108/mix_pairs.csv`，各取前 3 行，按记录的 g 重建 IQ（公式照 `work/side_L_wifimix_20261110/preflight.py` 的 `mixing()`），实际功率比与记录值最大差 ≤ 1e-6 dB。
5. 计数：2.4G 与任务 L 一致（data 无人机 2832、v3 无人机 152、v3 背景 234、bgA train/val/test = 2861/990/1287、叠加无人机 2832、叠加背景 2861）；5.8G 与任务 H 一致。
6. 记录 `work/crop_slice_20261107/`、`work/train_stage1_L20_20261108/`、`work/bgA_20261109/`、`work/side_L_wifimix_20261110/`、`work/fusionB_20261110/`、`work/review_20261109/` 下所有文件和 B1 特征缓存的 sha256，交付前再算一次，必须一致。
7. `nvidia-smi` 查有无其他 python 计算进程；有就 ask。

## 4. 缓存（新目录 `E:/fusionB_B2_20261112/`）

从 IQ 重算时频图（`common.spec()`，float16）和特征（新算的部分），行序与 §3 的 meta 一致：
- 两频段 train/val/test、rejected test；bgA keep 行。特征直接取 B1 缓存，不重算。
- 2.4G 叠加训练样本两组：按 L 的 pairs.csv 用同一个背景、g 重建 IQ，再算时频图和特征。
- 叠加检验：任务 H 的 `mix_pairs.csv`（两频段，三组 × 7 档）；C1：`work/side_L_wifimix_20261110/C1.pairs.csv`。全部在叠加后的 IQ 上**重算**时频图和特征。
- 对照：2.4G 白噪声 80、陷波白噪声 80、013 安静 19（做法照任务 L，`quiet013_metadata.csv`）；5.8G 白噪声 80、陷波白噪声 80（做法照任务 K，`work/side_5p8_controls_20261109/`）。特征要在 IQ 上算；如果 `work/review_20261109/notched_noise_2.4GHz.npy` 存的不是 IQ，就用 `work/review_20261109/ctrl_notched_noise_*.py` 的同一种子重新生成 IQ，并核对时频图与原文件一致。
- 读盘：IQ 按 block_file、start 顺序 mmap 读取。预计新占磁盘 6–10 GB，超过 20 GB 停下 ask。`common.py` 在 import 时会建 H 的 CACHE 空目录，允许，但不往里写文件。

## 5. 训练

- 每个频段先跑 F 的 seed 2026 3 epoch 冒烟。然后正式跑：2.4G R/Z/F 各 3 个种子，再 5.8G R/Z/F 各 3 个种子，共 18 次。checkpoint 放 §4 的目录，断点可续，日志 `run.log`，每 epoch 记录同 L（5.8G 同 H）。
- best.pt 里另存特征标准化参数、特征列名、组名。

## 6. 评估（R / Z / F 并列；每个种子分别报，再给均值±标准差；argmax 与 thr_fa1 两套；虚警给 k/N 和 95% 单侧 Clopper–Pearson 上界）

1. 2.4G：任务 L §6 的全部项目（v3 test 背景按会话、bgA test/val、013、白噪声、陷波白噪声、C1、非 G 漏检、含 G、按型号/状态/带宽、v3 型号召回、同型号、G 单列、被筛掉片段、叠加检验曲线、按型号叠加检验）。5.8G：任务 H 的全部项目 + 白噪声、陷波白噪声。
2. 对照参考：把任务 L 的 L-b、任务 H 的原结果数字并列一栏（取已有 csv），标明“旧 checkpoint，不是本次训练”。
3. **特征依赖度**（只用 F 的模型，test 集和 C1）：
   - (a) 测试时把特征在全部样本之间随机打乱（行置换，rng 20261112，重复 5 次）；
   - (b) 只在同一标签内打乱；
   - (c) 测试时把特征全置 0。
   报告各项指标相对 F 原结果的变化（5 次的均值±标准差）。
4. **只用特征的参照**：B1 的逻辑回归（`work/fusionB_20261110/classifier_metrics.csv`）用的是 13 列，不是本任务的 10 列。用本任务的 10 列、同样的做法（train 上标准化和拟合，val 选 C）重拟一个，报告同样的 test 指标。
5. 图 20（给老师看，中文标注）`图20_时频图加IQ特征_一阶段对比.png`：两频段各一行，每行两栏柱状。左 = 虚警（v3 test 背景、bgA test、C1 合并、013 安静、白噪声、陷波白噪声；5.8G 没有 bgA、C1、013），右 = 召回（data 非 G、v3 无人机、mavic2、mini3pro、G 各项）。每项 R / Z / F 三根柱，误差棒为 3 个种子的标准差。另画特征依赖度图 `fig_feature_dependence.png`。
6. 逐片预测 csv（前缀 R_ / Z_ / F_）。

## 7. 交付

输出目录 `work/fusionB_B2_20261112/`：脚本、日志、表、图、`REPORT.md`（中文，≤ 100 行，直接写数字）：
- 设置、融合网络结构与参数量、与 L / H 的 train.py 的 diff 摘要；10 个特征、填补与截断计数、标准化参数位置；
- §3 检查结果；18 次训练的 best epoch、val 指标、thr_fa1；
- §6 各表、图 20、特征依赖度图、叠加检验图、C1 图；
- 耗时、显存与内存峰值、新占磁盘；异常与自行决定的小事项；
- 末尾“**事实陈述，不下结论**”，关键数字各一行（argmax，R → Z → F，3 种子均值±标准差）：每个频段的 v3 test FA、非 G 漏检、v3 无人机召回；2.4G 另加 bgA test FA、C1 FA、陷波白噪声 FA、mavic2、mini3pro；以及特征依赖度 (a)(c) 下 v3 无人机召回、非 G 漏检的变化。

不改任务 F/H/I/J/L/B1 的任何文件，不重新划分，不删除/移动数据，不装包，不 git 提交，不调用 DSH。完成后发一次 `worker_done`（`--outcome succeeded/failed`，`--report-path work/fusionB_B2_20261112/REPORT.md`）。
