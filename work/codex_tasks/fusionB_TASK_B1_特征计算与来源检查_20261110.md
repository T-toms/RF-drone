# 做法 B 阶段 1（任务 B1）：IQ 手工特征计算、来源可分性检查、只用特征的基线（不训练网络）

下发：Claude（做法 B 协调者 claude-c3），2026-11-10。执行：Codex。
**不要发送 heartbeat。不要调用 DSH / dsh-cli / mcp__dsh。** 小问题按最合理方式处理并写进 REPORT.md，不要停下来问。只有以下情况才用 orchestration `ask`：读盘错误、磁盘或内存不足且绕不过去、本文与数据明显冲突、§3 的实现检查不通过且原因不明。完成时发一次 `worker_done`。**本任务不训练任何神经网络。**

## 0. 背景（已定事项，不需要重新论证）

- 设计文档：`work/主线_做法B_时频图加IQ手工特征_设计草案_20261110.md`（v2）；文献依据：`work/review_20261110/文献_做法B_特征融合.md`。先读设计文档的 §0、§2、§3。
- 做法 B 以后要把"从 IQ 算出的手工特征"拼接到 ResNet18 的时频图嵌入上。本任务先算特征，并回答三个问题：(1) 哪些特征能直接认出"是哪台接收机录的"（捷径），(2) 同一台接收机内部，哪些特征能区分背景和无人机，(3) 加噪后这些区分力还剩多少。另外给出只用特征的分类器基线。
- 数据里接收机和标签强相关：data/ = 接收机 A，几乎全是无人机；v3 = 接收机 B，含全部原始背景。接收机 A 的背景只有任务 I 从镜像频带截出的 bgA（只有 2.4G）。

## 1. 环境与输入（全部只读）

- Python：特征计算用 `C:/Users/Administrator/Desktop/数据集处理/.venv/Scripts/python.exe -I`（numpy + CUDA torch，**没有 scipy/sklearn**）或 `D:/anaconda/python.exe -I`（有 numpy/scipy/sklearn 1.7.2，torch 是 CPU 版，import torch 前设 `KMP_DUPLICATE_LIB_OK=TRUE`）。逻辑回归和 AUC 用 anaconda 的 sklearn。一律 `PYTHONIOENCODING=utf-8`、`encoding='utf-8'`、pathlib 绝对路径，不装包。
- **严禁 `import work/crop_slice_20261107/crop.py`**（导入时会覆盖任务 F 的文件）。可以 import `work/train_stage1_L20_20261108/common.py` 用 `rows()`、`writecsv()`，但不要调它的 `model()`，不要往它的 `CACHE` 写任何东西。
- 片段清单（特征要与这些文件**逐行对齐**，行序不变）：
  - `E:/drone_lab_spec_L20_20261108/{2.4GHz,5.8GHz}_{train,val,test}_spec_meta.csv`（任务 H 的时频图缓存行，均为 keep==True；列含 block_file、start、n_samples、dataset、label、model、state、bw_label、group_id、sample_id、split、cache_index）
  - `E:/drone_lab_spec_bgA_L20_20261109/bgA_meta.csv`（5630 行，**全部计算**以保持与 `bgA_spec.npy` 对齐；分析时只用 keep==True）
  - `E:/drone_lab_spec_L20_20261108/2.4GHz_rejected_test_spec_meta.csv` 和 5.8G 同名文件（若存在；被筛掉的 test 无人机片段，附表用）
- IQ：片段 = `np.load(block_file, mmap_mode='r')[start : start + n_samples]`，complex64，未归一化。块很大（data/ 约 1 GB/块），**按 block_file 分组、按 start 顺序读**，不要随机访问。2.4G：fs = 61.44e6，中心 2430 MHz，n_samples = 1,228,800；5.8G：fs = 100e6，中心 5800 MHz，n_samples = 2,000,000。
- 陷波表 `work/crop_slice_20261107/notch_table.csv`；划分 `work/crop_slice_20261107/group_split.csv`；泛化集 G：data 2.4G `evo lite 6k`、data 2.4G `evo ii pro v3`、v3 5.8G `evo2pro_v3`、v3 5.8G `fpv`。
- v3 的带宽：索引里 v3 的 bw_label 都是 `unknown`。同型号比较时，Mini 5 Pro 的 v3 带宽从原始 JSON 取：`G:/drone/raw/<campaign>/<session_id>/*.json` 的 `declared.bw`（session_id 从 group_id 里取，例如 `v3:20260709-035600_mini5pro` → `20260709-035600_mini5pro`）。取不到或为 null 的记为 unknown，写明。
- 输出：脚本、表、图、REPORT → `work/fusionB_20261110/`；特征缓存 → `E:/fusionB_features_20261110/`（预计 < 50 MB）。E 盘约 144 GB 空闲。

## 2. 特征定义（逐字照做；有歧义按最合理方式处理并写进 REPORT）

对每个片段 z（复数，长 N）：
- 去直流、功率归一化：m = mean(z)；w = z − m；z̃ = w / sqrt(mean|w|²)。a = |z̃|。
- 0.1 ms 帧：2.4G 6144 点，5.8G 10000 点（不足一帧的尾部丢掉）。帧功率 p_k = mean_{帧内} |z̃|²。

**候选特征（13 个，都在 z̃ 上算）：**
1. `shape_factor` = 1 / mean(a)（因为 rms(a) = 1）
2. `log_kurt_a` = ln( E[(a − ā)⁴] / (E[(a − ā)²])² )
3. `papr999_db` = 10·log10( |z̃|² 的 99.9 百分位 )
4. `c42n` = E|z̃|⁴ − |E z̃²|² − 2；存 sign(x)·ln(1+|x|)
5. `c40n_abs` = |E z̃⁴ − 3·(E z̃²)²|；存 ln(1+x)
6. `c41n_abs` = |E[z̃³·conj(z̃)] − 3·E z̃²|；存 ln(1+x)
7. `c20n_abs` = |E z̃²|
8. `log_gamma_max`：c = a/ā − 1；ln( max_k |DFT(c)_k|² / N )（k 从 1 起，不含直流）
9. `sigma_f`：f_n = angle(z̃_{n+1}·conj(z̃_n))·fs/(2π)；只取 a_n > 1 且 a_{n+1} > 1 的 n；std(f_n)/fs。符合条件的样本 < 1000 个时记 NaN 并计数
10. `spec_flatness`：1024 点、Hann、不重叠分帧，功率谱逐帧平均后 fftshift（频率 = 中心 + (k − 512)·fs/1024）；只取通带 2405–2455 / 5755–5845 MHz 的 bin，去掉每个陷波频率 ±0.3 MHz 的 bin；= exp(mean ln P) / mean P
11. `occ_frac`：同一组 bin 的功率转 dB，(dB > 中位数 + 10) 的比例
12. `duty`：thr = (p_k 的 10 百分位)·10^0.6；duty = mean(p_k > thr)
13. `log1p_n_bursts`：上述二值序列中 0→1 的次数 n，存 ln(1+n)

**阳性对照（6 个，用归一化前的 z 计算，只用于检验来源检查本身是否有效，不进模型）：**
14. `ctl_abs_power_db` = 10·log10(mean|z|²)
15. `ctl_floor_db` = 10·log10( 用 z 算的 0.1 ms 帧功率的 10 百分位 )
16. `ctl_dyn_range_db` = 1 ms 帧功率（dB）的 p90 − p10
17. `ctl_dc_rel` = |m| / sqrt(mean|z|²)
18. `ctl_iq_power_ratio_db` = 10·log10( mean(Re w)² / mean(Im w)² )（这里指 mean((Re w)²) / mean((Im w)²)）
19. `ctl_iq_corr` = 相关系数(Re w, Im w)

NaN、Inf 一律保留为 NaN，并在 REPORT 中按特征、按 (dataset, label) 计数；分析时对 NaN 用该特征的 **train** 中位数填补（只在分析时填，缓存保留 NaN）。

存储：每个 meta 文件对应一个 `<name>_features.npy`（行数 × 19，float64，列顺序同上）和一个 `<name>_features_meta.csv`（原 meta 各列 + `row_index`）。另写 `feature_names.json`。

## 3. 实现检查（先做，不过不要往下）

1. **双实现核对**：另写一个最直接、不分块的纯 numpy float64 参考实现，每个频段在 data 无人机、v3 无人机、v3 背景、bgA 中各取 3 个片段，19 个特征逐项比较：相对误差 ≤ 1e-4（或绝对误差 ≤ 1e-6）。
2. **合成信号核对**（写进 REPORT）：
   - 复高斯白噪声：c42n ≈ 0（|值| < 0.01），c20n_abs ≈ 0，spec_flatness 接近 1（通带内 > 0.95），duty ≈ 0（< 0.05）。
   - 恒包络复正弦：shape_factor = 1，c42n = −1（存储值 sign·ln(2) ≈ −0.693）。
   - 白噪声上叠加 10% 占空的强突发（脉冲内功率比底噪高 20 dB）：duty ≈ 0.1（误差 < 0.02）。
3. 行对齐：随机抽 20 行，确认 `<name>_features_meta.csv` 的 sample_id 与原 meta 同行一致。

## 4. 分析

所有 AUC 都报告 `max(AUC, 1−AUC)`，并注明方向（哪一组大）。每个数字都给样本数。

**4.1 来源可分性（检查捷径）**
- **(a) 只用背景，2.4G（主要依据）**：v3 2026-07-10 实验室背景（group_id 含 `20260710`，三个 split 全部）vs bgA（keep==True，三个 split 全部）。逐特征 AUC。另外按 split 分别报告 test 部分的 AUC。附表 (a')：v3 全部背景 vs bgA。
- **(b) 同型号无人机**：2.4G 和 5.8G 的 data `mini5 pro` vs v3 `mini5pro`；2.4G 的 data `evo ii pro v3` vs v3 `evo2pro_v3`。逐特征 AUC；Mini 5 Pro 能按带宽对上的（data bw_label = v3 declared.bw），**按带宽分别报告**，其余合并并注明。
- **多变量**：19 个特征（以及只用 13 个候选特征）各做一次逻辑回归（sklearn，`StandardScaler` 只在 train 上 fit，`C` 在 val 上从 {0.01, 0.1, 1, 10} 中选）。(a) 用 train 拟合、test 评估；(b) 的样本太少，用按 group_id 分组的留一组交叉验证（`LeaveOneGroupOut`），报告 AUC。
- **判据（事先定好，只报告分类结果，不自动删除特征）**：(a) 中 AUC ≥ 0.90 → 标记"接收机指纹"；0.75–0.90 → "可疑"；< 0.75 → "未见来源线索"。5.8G 没有 bgA，标记"未能检验"，只给 (b) 的数字。**阳性对照 14、15 在 (a) 中必须 ≥ 0.90**，否则在 REPORT 首行写明"来源检查无效"，并照常交付其余部分。

**4.2 同一接收机内的标签区分力**
- 接收机 A（2.4G）：data 无人机 vs bgA。接收机 B（两个频段）：v3 无人机 vs v3 背景。逐特征 AUC，按 split 分开（主要看 test）。5.8G 接收机 A 没有背景，写明。

**4.3 加噪后的区分力**
- 只用 test split。每个频段取：最多 400 个 data 无人机（按 model 分层抽样，种子 2026）、全部 v3 无人机、全部 v3 背景、全部 bgA test（2.4G，keep==True）。
- 对每个片段加复高斯白噪声：σ² = mean|z|² / 10^(SNR/10)，SNR ∈ {clean, 0, −5, −10, −15} dB（与报告的定义一致：原片段平均功率 / 噪声功率），噪声种子固定、逐片段不同并记录。加噪后重算 19 个特征。
- 报告 4.2 的两组标签 AUC 随 SNR 的变化（逐特征），画 `fig_B1_auc_vs_snr_<band>.png`。

**4.4 只用特征的分类器（基线）**
- 每个频段：13 个候选特征 → 逻辑回归（同 4.1 的标准化和选 C 方法），训练集样本权重：2.4G 用任务 J 的规则（无人机、v3 背景、bgA 三组：权重 ∝ 1/N_d、0.5/N_v、0.5/N_a，再整体缩放到均值 1）；5.8G 用任务 H 的规则（类别频率倒数，归一化到均值 1）。2.4G 的训练背景包含 bgA 的 train（keep==True），与任务 J 一致。
- test 报告：argmax（p ≥ 0.5）下 v3 背景虚警（按会话）、bgA 虚警（2.4G）、无人机漏检（主表**不含 G**；G 每项单列；另给含 G 的原口径）、data/v3 来源召回、同型号召回。每个虚警数字带 k/N 和 95% 单侧 Clopper–Pearson 上界。并列放上任务 J（2.4G）/ 任务 H（5.8G）时频图模型 3 个种子的同口径数字（从 `work/train_stage1_L20_bgA_20261109/` 和 `work/train_stage1_L20_20261108/` 的现成 csv 里取，不要重推）。
- 再做一次"只用 13 个候选中 (a) 标为'未见来源线索'的特征"的逻辑回归，同样报告。

## 5. 交付

`work/fusionB_20261110/REPORT.md`（中文，≤ 90 行，直接写数字）：
1. §3 实现检查结果（双实现最大误差、合成信号数值、行对齐）。
2. NaN/Inf 计数表。
3. 4.1 表：每行一个特征，列 = (a) AUC / 方向 / 标记，(a') AUC，(b) 各同型号组 AUC；再加多变量 AUC。
4. 4.2 表、4.3 图和表（每个 SNR 一列）。
5. 4.4 表（只用特征的分类器 vs 时频图模型）。
6. 一张给老师看的图 `图15_手工特征的来源与标签区分力.png`（中文标注）：左 = 2.4G 各特征在 (a) 中的来源 AUC 柱状图（阳性对照用不同颜色，画 0.75 和 0.9 两条虚线）；中 = 同一特征在 4.2 接收机 A 内部（data 无人机 vs bgA）的标签 AUC；右 = 标签 AUC 随 SNR 的变化（选 4 个特征）。
7. 耗时、读盘量、异常与自行决定的小事项。
8. 末尾"**事实陈述，不下结论**"：关键数字各一行。

保护：开工前与交付前对 `work/crop_slice_20261107/`、`work/train_stage1_L20_20261108/`、`work/bgA_20261109/`、`work/train_stage1_L20_bgA_20261109/`、`work/side_5p8_controls_20261109/` 算 sha256，必须一致。不训练神经网络、不改上述目录、不删除/移动数据、不装包、不 git 提交、不调用 DSH。完成后发一次 `worker_done`（`--outcome succeeded/failed`，`--report-path` 指向 REPORT.md）。
