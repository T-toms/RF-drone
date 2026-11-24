# STAGE1-12DRONES-A 验收证据：104 MAT / 13 类一阶段（Background vs 12 机型）全量训练

生成时间：2026-09-26（本地）
阶段：STAGE1-12DRONES-A（仅第一阶段；不含第二阶段，不含任何测试角色评价）
数据权威：冻结划分 `work/meta_split2/split_candidate_104_20260926.csv`
（SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`）
G 正式清单 `G:\DroneRFa数据集\dataset_manifest.csv`
（SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`，本轮未改动）

---

## 1. 一句话结论

按冻结划分在全部 **104 个 MAT / 13 类** 上建立了第一阶段数据入口（15,790 个 1M 窗），
并使用**全部 9,590 个 train 窗 + 全部 2,360 个 val 窗、不抽样、5 epoch** 完成了
Background(T0000) vs Drone(其余 12 个 T 码) 的一阶段训练；最佳 epoch=3 的 checkpoint
可重新加载，独立核验 34/34 项、负向对照 8/8 项、回归测试 39 项全部通过，
训练/验证读取审计证明 26 个测试角色 MAT 的 RF0 从未被打开。

本阶段**不报告** 12 机型识别准确率：那属于尚未开始的第二阶段与串联评价。
下文的 val 指标只是**开发选型指标**。

---

## 2. 交付物与哈希

### 2.1 数据入口（`dronerfa_baseline/outputs/stage1_12drones_20260926/`）

| 产物 | 说明 | SHA256 |
| --- | --- | --- |
| `window_index_stage1_12drones.csv` | 15,790 行窗口索引，覆盖 104 MAT | `5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6` |
| `dataset_summary.json` | 角色/类别/窗口/样本计数 | `a7431ee61e559eb9542d58709daa9ac17dd1efe7ec7e1657378c83db389e0a8d` |
| `source_identity.json` | 104 个源 MAT 的大小、冻结 SHA256、HDF5 形状/分块/压缩 | `0d6890dd8b1cf614c80a6f0a0f260b235c76431b4c029bd4fcfb058302b68a7f` |
| `read_probe.json` | train/val 有界真实 RF0 读取探针 + 测试角色"未读取"清单 | `fe26d513d4c27aeb7e72b6a7170f4c2418b34a2580f8eb4272035fd231c11f33` |
| `index_verification.json` | 索引自检 27 项 | `81df5eba98a84b33edf049b898d111b8f2efeaf6d025dfe609f8a7725e69f450` |
| `index_negative_controls.json` | 索引负向对照 6/6 | `dcbcb7e174022ec382f80347ea74c91060c65c3859dedef2038cd1cb321697dd` |
| `data_scheme_stage1_12drones.md` | 数据方案说明 | — |
| `index_build_result.json` | 索引构建结果与哈希记录 | — |

### 2.2 训练产物（`.../stage1_12drones_20260926/seed_2026/`）

| 产物 | 说明 | SHA256 |
| --- | --- | --- |
| `best_model.pt` | 最佳 epoch=3 checkpoint（44,768,907 B） | `37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b` |
| `run_config.json` | 完整协议、输入身份、逐 epoch 历史、只读审计摘要 | `55bebe3e05eed9287c32944db26a98ec5b947870884a445c186069f5ac6b9b7e` |
| `read_audit.json` | 打开文件与逐文件窗口读取计数 | `be4b5ce29693bfb7a64cc9cd301453d3f8493ff4550ba6d978dc9b398285a7e7` |
| `selection.json` | 选型规则与逐 epoch 均值 | `9766f45e3ea4848b29ce7face1d6b91a37b409746fef7dfbfb3064c331efbad8` |
| `training_history.csv` | 5 epoch 训练/验证汇总 | `76cd68d1c7853741c9f3d7e05b6cbf0ea098ca3841a8ef5d10609b8a85ef73dc` |
| `validation_metrics.csv` | 5 epoch × 4 条件 = 20 行验证指标 | `1546097cabbbd5eb3a7750d65229a3a1da969b630ba024d80a6b052bb0c5fec3d` |
| `validation_scores_epoch_0{1..5}.csv` | 每 epoch 每窗口 margin/预测（各 9,440 行） | 见目录 |
| `checkpoint_reload_check.json` | checkpoint 重载 28 项比较，`max_float_delta=0.0` | — |
| `validation_cache_manifest.json` | 4 个条件的验证特征缓存身份（各 2,360 窗） | — |
| `stage1_train.log` | 训练日志（含逐批进度与逐 epoch 指标） | — |
| `progress.json` / `environment.json` / `input_identity.json` / `pre_run_config.json` / `summary.json` | 进度、环境、输入身份、汇总 | — |

### 2.3 独立核验（`.../stage1_12drones_20260926/verification.json`）

SHA256 `607c4023ee17b68777170d4a4c7355c5f7a28ef49c227ad57df0911a7e16fe0e`，
结论 `passed=true`，34/34 检查通过、负向对照 8/8 命中。

### 2.4 代码（均在 `dronerfa_baseline/`，均为新增，未修改既有脚本）

| 文件 | 作用 |
| --- | --- |
| `build_stage1_12drones_index.py` | 104 MAT 窗口索引 + 源身份 + 有界读取探针 + 自检/负向对照 |
| `run_stage1_12drones_training.py` | 第一阶段训练驱动（含只读审计、选型、重载自检） |
| `verify_stage1_12drones_training.py` | 独立核验（不导入训练驱动，自带 STFT/噪声实现） |
| `test_stage1_12drones.py` | 39 项回归测试 |
| `benchmark_stage1_reads.py` | 只读磁盘读带宽诊断（有界） |

未改动：`run_cascade3_newsplit_training.py`、`build_newsplit_index.py`、
`dronerfa_data.py`、`model.py`、`train_resnet18.py`、三类索引与 checkpoint、
`work/meta_split2/*`、G 盘任何文件。

---

## 3. 数据入口：标签、角色与窗口

### 3.1 标签空间

- `T0000` = Background = `label` 0（4 train / 2 val / 0 dev test / 2 exposed test）
- 其余 12 个 T 码（T0001、T0010、T0011、T0100、T0101、T0110、T0111、T1000、T1001、
  T1010、T1011、T1100）= Drone = `label` 1（各 5 train / 1 val / 2 dev test / 0 exposed test）
- 冻结字段 `class_id`、`class_index`、`prior_use_role`、`archive_segment_role`、
  `exposed_to_prior_selection` **原样保留**用于溯源，但不参与 Stage1 损失或选型。
  角色 100% 取自冻结 CSV 的 `role` 列，绝不从 S 码或旧 `source_table_104.csv` 推断。

### 3.2 窗口与角色统计

| role | 文件数 | 窗口数 | label 0 | label 1 | 本阶段用途 |
| --- | --- | --- | --- | --- | --- |
| train | 62 | 9,590 | 600 | 8,990 | 模型拟合（全部窗口，每 epoch 全用） |
| val | 16 | 2,360 | 280 | 2,080 | 选型（clean/-10/-12/-15 dB 四条件） |
| development_test | 20 | 2,950 | 0 | 2,950 | 不读取，仅元数据登记 |
| development_test_exposed | 6 | 890 | 280 | 610 | 不读取，仅元数据登记 |
| 合计 | 104 | 15,790 | 1,160 | 14,630 | — |

窗口规则：RF0 的 `RF0_I + 1j*RF0_Q`，每窗 1,000,000 采样（10.0 ms @ 100 MS/s），
非重叠、1M 对齐，尾部不足一窗丢弃。104 个源文件的长度均为 1M 的整数倍
（110M～210M 采样，合计 15,790,000,000 采样），因此**实际丢弃 0 个采样**；
"丢弃尾部"是规则的一部分，本数据集恰好没有余数。

### 3.3 有界真实读取探针（未全量重哈希）

- 冻结 CSV 已含经 Codex 验收的 MAT SHA256，本轮**未**为文档重算约 100 GiB 哈希；
  索引与行内哈希全部来自冻结清单，并逐一核对磁盘大小（104/104 相符）。
- 真实 RF0 读取：78 个 train/val 文件 × 3 个窗（首/中/末）× 2 通道 × 3 个 4,096 采样切片，
  共 234 个探测窗、5,750,784 采样；另对每个可读角色取 1 个整窗（1,000,000 采样）逐字节哈希，
  供独立核验复算。
- `read_probe.json` 中 `test_roles_read_count = 0`；测试角色文件只作为文件名清单登记。

---

## 4. 训练协议（全部冻结，未做任何削减）

| 项目 | 取值 |
| --- | --- |
| 输入 | RF0 I/Q → 每窗去均值 + RMS 归一化 |
| 时频前端 | Hamming STFT `n_fft=512`、`hop=256`、256×256 log-power 图（`model.SpectrogramFrontEnd` 原样） |
| 模型 | `ResNet18(input_channels=1, num_classes=2)`，类别映射 {0:Background, 1:Drone} |
| 训练噪声 | `A_original` = `clean,10,5,0,-5,-10,-12` dB，每窗 `default_rng(seed + epoch*len + index)` 抽取 |
| 优化 | Adam lr=1e-3、weight decay=0、batch=4、类别加权交叉熵、CUDA AMP float16 |
| 轮数/种子 | 5 epoch、seed 2026（`cudnn.deterministic=True`、`benchmark=False`） |
| 训练用量 | **全部** 9,590 train 窗/epoch（62 文件）；无抽样 |
| 验证 | **全部** 2,360 val 窗（16 文件），clean/-10/-12/-15 dB 四条件固定噪声（seed 12026）；无抽样 |
| 选型 | 四条件 pooled balanced accuracy 均值最大；完全同分取最早 epoch |
| 判定 | Drone iff `float32(logit_drone - logit_background) > 0` |

硬件/环境：RTX 4060 Laptop（8 GB）；torch 2.12.1+cu130；GPU 峰值占用约 1.7–2.2 GiB。
实际耗时：验证缓存约 31.5 min（4 条件），5 个 epoch 训练+验证约 3.2 h，重载自检约 1 min。

---

## 5. 结果（仅开发选型指标）

### 5.1 逐 epoch 验证（val，16 文件 / 2,360 窗）

| epoch | train loss | train BA | mean pooled BA（选型键） | mean file-equal BA | clean BA | -10 dB | -12 dB | -15 dB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3596 | 0.6754 | 0.893879 | 0.887291 | 0.942960 | 0.915076 | 0.900515 | 0.816964 |
| 2 | 0.2111 | 0.8153 | 0.780108 | 0.779139 | 0.990076 | 0.760989 | 0.734650 | 0.634718 |
| **3** | **0.1534** | **0.8704** | **0.900137** | **0.899000** | **0.986435** | **0.870021** | **0.881662** | **0.862431** |
| 4 | 0.1235 | 0.9070 | 0.769961 | 0.769750 | 0.951648 | 0.733070 | 0.714251 | 0.680872 |
| 5 | 0.1165 | 0.9112 | 0.744342 | 0.744143 | 0.845913 | 0.736470 | 0.711195 | 0.683791 |

选中 **epoch 3**（`selection.json`，规则与 tie-break 与冻结口径一致）。

### 5.2 最佳 epoch 的验证混淆矩阵（Drone 为正类）

| 条件 | tp | tn | fp | fn | pooled BA | file-equal BA |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 2031 | 279 | 1 | 49 | 0.986435 | 0.986588 |
| -10 dB | 1985 | 220 | 60 | 95 | 0.870021 | 0.868459 |
| -12 dB | 1922 | 235 | 45 | 158 | 0.881662 | 0.879773 |
| -15 dB | 1686 | 256 | 24 | 394 | 0.862431 | 0.861180 |

### 5.3 读法（重要）

- 这四条只是**开发选型指标**：val 文件在 META-SPLIT2 中按源文件划分，但与本轮 train
  文件同处一个文件时间簇，只可称"文件级留出"，不是跨采集会话或跨环境泛化。
- epoch 2/4/5 显示 clean BA 上升而低 SNR 大幅退化（背景虚警上升），说明
  背景类只有 4 个 train 文件、600 个窗，"高信噪比更准"与"低信噪比更稳"并不单调同步；
  选型键按协议取四条件均值，因此选中 epoch 3。
- 上述数字**不得**当作 12 机型识别准确率，也**不得**当作现场准确率。

---

## 6. 验收检查（实际执行）

### 6.1 索引自检与负向对照（构建时）

`index_verification.json`：27/27 通过。要点：104 文件全集、13 类各 8 文件、零跨角色、
标签 0 仅 T0000、窗口 1M 对齐、行窗口数 = `floor(samples/1e6)`、
`(filename, window_index)` 唯一、全部路径存在且在 G 数据根内、无 legacy F:/junction 路径、
磁盘大小与冻结清单一致、行内 SHA256 等于冻结哈希、测试角色与 train/val 文件集互斥、
探针未触碰测试角色、`T1010/T1011/T1100` 未伪造距离码、其余 9 个机型均为 `D00`。

`index_negative_controls.json`：6/6 命中（角色改写、标签翻转、窗口错位 1 采样、
哈希替换、文件缺失、测试文件混入探针）。

### 6.2 独立核验（训练后，`verification.json`）

`verify_stage1_12drones_training.py` 不导入训练驱动与索引构建器，自带 RMS/噪声/STFT 实现，
在 CPU 上复算。结果 **34/34 通过**：

| 组 | 检查 | 结果 |
| --- | --- | --- |
| A | 冻结划分哈希、G 清单哈希、索引哈希、104 文件/13 类、索引行数、文件全集 | 7/7 |
| B | 窗继承源角色、标签空间、窗口几何、文件集互斥、角色构成 62/16/20/6、summary 一致、构建期核验 | 7/7 |
| C | 打开文件集 = 精确 78 个 train+val 文件；测试角色 0 次打开；自报与复算一致；逐文件读取预算 = train×5 + val×4；四条件齐全 | 5/5 |
| D | checkpoint 的标签映射/判定规则、索引与划分哈希 | 2/2 |
| E | 缓存清单覆盖、score 文件覆盖、**全部指标从逐窗决策独立复算（max_delta = 0.0）**、重放窗口代表性、margin 复现、逐窗决策复现（0 不一致）、选型规则、无更优 epoch、协议声明、全量使用、重载自检、checkpoint 哈希 | 12/12 |
| F | 负向对照 | 8/8 |

关键量化证据：

- 读取审计：`opened_files` 78 个，与冻结 train(62)+val(16) 集合**完全相等**；
  `test_files_touched = []`；`windows_read = 57,390 = 9,590×5 + 2,360×4`，逐文件预算零违例。
- 独立重放：480 个 val 窗（16 文件全覆盖、label 0/1 均有）× 4 条件 = 1,920 次推理，
  与训练记录的逐窗决策**完全一致**（0 不一致），margin 最大差 `6.26e-3`（float32 设备差，
  容差 0.05）。
- checkpoint：重载 28 项比较全等，`state_dict` 哈希两侧相同，`max_float_delta = 0.0`。

### 6.3 负向对照（8/8 命中）

| 对照 | 命中方式 |
| --- | --- |
| 某 train 文件多计 1 个 val 窗读取 | 读取预算 C4 失败 |
| 某 development_test 文件混入 `opened_files` | 打开文件集 C1/C2 失败 |
| 记录的逐窗标签翻转 | 指标复算 E3 抛错/不一致 |
| 用随机初始化权重替换 checkpoint | 重放决策与记录不一致（E6） |
| 篡改 epoch 历史 | 选型 argmax 与记录不符（E7） |
| 同分却选较晚 epoch | tie-break 断言失败（E7） |
| 无人机窗被标成背景 | 标签契约 B2 失败 |
| 审计中删掉验证缓存读取 | 读取预算 C4 失败 |

### 6.4 回归测试

`python -m unittest test_stage1_12drones`：**39 项全部通过**（约 19 s）。
覆盖：冻结权威与角色期望、`T1010/T1011/T1100` 无距离字段解析、索引 schema 与几何、
summary 一致、探针不触测试角色、`check_index_identity`、索引负向对照复现、
指标手算核对、选型与 tie-break、重放窗口选取、噪声/前端复刻（与
`dronerfa_data`、`model.SpectrogramFrontEnd` 逐位一致）、只读审计、
核验脚本不导入训练驱动的源码断言、以及已完成运行产物的只读核对。

### 6.5 只读磁盘诊断（有界，辅助）

`benchmark_stage1_reads.py 40`：train 62 文件 124 次整窗读取均值 0.159 s/窗、
val 16 文件 32 次均值 0.113 s/窗，单次整窗读 16 MB。
说明本盘随机整窗读带宽是训练吞吐的主要限制（GPU 利用率长期 <30%），
不构成正确性风险，但下一阶段排期需按 ~4–5 窗/s 估算。

---

## 7. 限制与不得声称的结论

1. **未做第二阶段、未做串联评价、未对任何测试角色推理**。本阶段没有 12 机型识别结果。
2. val 指标是开发选型指标；13 类整体只是开发评估；新 10 类的 20 个 `development_test`
   文件同属一个文件时间簇，旧 3 类的 6 个 `development_test_exposed` 文件此前已暴露。
   因此**不能**声称 13 类独立最终测试、跨采集会话或跨环境泛化。
3. 二分类目标本身混合了两类非机型因素：9 个机型名带 `D00`（论文表 3：20～40 m），
   `T1010/T1011/T1100` 无距离字段，且 12 个机型分属两个文件时间簇/采集夜；
   Stage1 只学"背景 vs 无人机"，上述因素不会被本阶段结果区分。
4. 背景类只有 4 个 train 文件（600 窗），无人机侧 8,990 窗；类别加权与 file-equal 指标
   只是缓解，不能替代独立背景来源。epoch 2/4/5 的低 SNR 虚警上升即为此结构的体现。
5. 本节所有数字都来自**冻结划分内的同批文件**；G 盘清单、冻结 CSV、旧三类产物均未被修改。

---

## 8. 复现命令

```powershell
# 0) 环境：项目 venv 的 site-packages + 基础解释器（本会话沙箱下 venv 启动器被拒）
$env:PYTHONPATH = "C:\Users\Administrator\Desktop\数据集处理\.venv\Lib\site-packages"
$env:PYTHONIOENCODING = "utf-8"
cd C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline

# 1) 索引（已构建；重建会因已存在而拒绝覆盖）
D:\anaconda\python.exe build_stage1_12drones_index.py
D:\anaconda\python.exe build_stage1_12drones_index.py --verify-only

# 2) 训练（完整协议；已存在 checkpoint 时拒绝覆盖，除非 --force）
D:\anaconda\python.exe run_stage1_12drones_training.py --mode full --device cuda
D:\anaconda\python.exe run_stage1_12drones_training.py --status

# 3) 独立核验与测试
D:\anaconda\python.exe verify_stage1_12drones_training.py
D:\anaconda\python.exe -m unittest test_stage1_12drones
```

已知环境注意：本会话文件沙箱对"管道/重定向捕获子进程输出"的调用会返回
`Access is denied`，需直接运行（输出落到 `stage1_train.log`）。
