# 任务 W：第一阶段窗长比较（L = 10 / 30 / 40 / 50t ms，时频图输入）——草稿，待用户确认后下发

下发：Claude（主线协调者），2026-11-10。执行：Codex。
**不要发送 heartbeat。不要调用 DSH / dsh-cli / mcp__dsh。** 小问题自己决定，写进 REPORT.md。只在以下情况用 orchestration `ask`：§5 任一复现检查不过；读盘错误或内存/磁盘超限；本文与数据明显冲突。完成时发一次 `worker_done`。

> 【2026-11-11 用户已确认开跑（“开始训练吧，还是 codex 帮你”），按本稿 ①② 执行】① 时频图固定 256×256；② 比较以按型号的无人机召回为主，虚警逐 L 列片数与上界，长窗长注明不可比。主线对话当时不在线，由支线对话（Orca run `run_253f4d8cfc53`）代为下发和验收，结论交回主线。

## 0. 背景与已定事项（不需要重新论证）

- 报告第 2 步：比较不同窗长。数据已按 L = 10–50 ms 切好（任务 F）。
- **唯一的自变量是 L。** 其余设置逐项沿用当前基线（协作约定 §5）：
  - 2.4G = 任务 L 的 L-b 数据组成 + 任务 M 的 `spec_fill` 时频图（`work/side_L_wifimix_20261110/`、`work/side_M_notchfill_20261110/`）。
  - 5.8G = 任务 H（`work/train_stage1_L20_20261108/`），`common.spec`，无 bgA、无叠加。
- **L = 20 不重训**，直接引用任务 M（2.4G）和任务 H（5.8G）的结果；本任务只用新代码对它们的 checkpoint 复推做核对（§5）。
- L50 用户 2026-11-10 定为 **L50t**：不重叠，每块再补一个与块尾对齐的窗口。索引由支线任务 N2 生成：`work/side_N2_L50tail_index_20261110/index_L50t.csv`（含 `on_original_grid`、`orig_sample_id`）。L10/30/40 用任务 F 的 `index_L{L}.csv`，不重叠。半重叠索引（任务 N 的 L30h/40h/50h）**不用**。
- N2 已交付（支线复核、主线抽查通过）：4563 行 = 原窗口 4092（与任务 F `index_L50.csv` 逐行一致）+ 块尾窗口 471（每块最多 1 个）。v3 块的两片只重叠 1 ms；data 块的尾窗口与前一片重叠 1–47 ms（中位 1 ms）。
- 已知事实（报告里要写明）：v3 的块截频后只有 99.6 ms 和 49.6 ms 两种；49.6 ms 的块在 L50t 一片也切不出，2.4G train 背景组 `v3:noise_adrv_2450`（一组无 WiFi、有零星突发的接收机 B 背景，**不是纯噪声**）在 L50t 整组缺失；`075302_ambient@2440` 从 14 片降到 4 片，5.8G 的 3 个 ambient 组从 18 片降到 8 片。v3 尾窗口被筛掉 8.9%（原窗口 2.7%），33 个全是 v3 无人机，主要是 lack_feature，规则照旧不处理，报告里说明 L50t 的 v3 无人机 keep 比例偏低。
- L50t 的 keep 不重复时长已不少于 L20（2.4G v3 背景 train 5.15 s 对 4.68 s，v3 无人机 3.81 对 3.04，data 无人机 59.3 对 56.6；5.8G v3 背景 4.65 对 3.88）。

## 1. 数据（每个 L、每个频段）

- 片段：索引中 `band` 匹配、`keep==True` 的行；split 用索引自带的 split（来自 `group_split.csv`，不重新划分）。
- **bgA（只 2.4G）**：
  - L10/30/40：直接用 `work/bgA_20261109/index_bgA_L{L}.csv` 中 `keep==True` 的行（已与 `index_L{L}.csv` 的 2.4G data 行一一镜像，keep 按块级 6 个坏块定）。
  - L50t：按同一规则由 `index_L50t.csv` 的 2.4G data 行镜像生成：`block_file` = `E:/drone_lab_bgA_20261109/2.4GHz/<data 块 stem>__bgA_s0.npy`，start/n_samples/group_id/split 与 data 行相同，`dataset='dataA_mirror'`、`label='background'`、5 个 flags 为 False，keep = 块 stem 不在 `index_bgA_L50.csv` 里 keep=False 的块集合中，`paired_sample_id` = data 行 sample_id，并保留 `on_original_grid`。检查：`on_original_grid==True` 的行与 `index_bgA_L50.csv` 逐行一致（block_file/start/n_samples/split/keep）。
  - 生成脚本 `full_driver.py` 只读参考，**不要 import**、不要运行 `work/bgA_20261109/` 下任何脚本。
- **训练叠加样本（只 2.4G，规则照搬任务 L 的 `generate.py`）**：`rng = np.random.default_rng(20261110)`；先遍历 train 集 data 无人机行（按索引行序），再遍历 train 集 bgA 行（按索引行序），每行依次抽 `X = rng.uniform(-5, 15)`、`j = rng.integers(0, N_bg)`，N_bg = 该 L 的 train v3 背景片数；功率比按整片平均功率，`z = b + g·d`（叠加函数从 `work/side_L_wifimix_20261110/preflight.py` 的 `mixing` 复制）。配对写成 `mix_train_{drone,bg}_L{L}.pairs.csv`。
- **测试叠加（两个频段，规则照搬 `work/train_stage1_L20_20261108/mix_test.py`）**：三组 data_drone / v3_drone / v3_background，XS = [−10, −5, 0, 5, 10, 15, 20] dB，背景取同 L 的 test v3 背景，配对规则 `j = i % N`（v3_background 组 `j = (i+1) % N`）。
- **白噪声对照（只 2.4G）**：80 片、长度 = L 对应样本数，`rng = np.random.default_rng(20261109)`，复数标准高斯。

## 2. 时频图

- 2.4G：从 `work/side_M_notchfill_20261110/fill.py` **复制** `spec_new`（method='M'）及其常量到本任务目录（不要 import pilot 或 side_M 的模块）。5.8G：从 `work/train_stage1_L20_20261108/common.py` 复制 `spec`。
- 参数不变：Hann 1024、hop 512、center=False、双边 FFT、取通带频点、（2.4G 修补陷波凹坑）、功率域 `adaptive_avg_pool2d` 到 **256×256**、加中位功率×1e−3 转 dB、逐片标准化、float16。
- 记录每个 L 的 STFT 帧数和"每列代表的时间"（帧数 × 512 / fs / 256 ≈ L/256），写进报告。
- 叠加、加噪一律先在 IQ 上做，再过时频图函数。IQ 读取按 `block_file`、`start` 排序，mmap 只读，按原行索引写回。

## 3. 训练

- 2.4G：`work/side_M_notchfill_20261110/train.py`（即任务 L 的 L-b 训练脚本）**逐字复制**，只改 L 相关的路径与片数断言。五个子类（原无人机 / v3 背景 / bgA / 叠加无人机 / 叠加背景）的类内权重 无人机:叠加无人机 = 1:1，v3 背景:bgA:叠加背景 = 2:1:1，类总权重各 0.5；选模型 = val 的 FA_v3 + FA_bgA + Miss 最小（并列看 val loss）；thr_fa1 取两种背景阈值中较大的。
- 5.8G：`work/train_stage1_L20_20261108/train.py` 逐字复制，只改 L 相关路径。类别频率倒数加权，选模型 = val 的 (FA + Miss, loss) 字典序最小。
- 共同：`ResNet18(1,2)`（`dronerfa_baseline/model.py`，不改）、AdamW lr 1e−3 / wd 1e−4、cosine T_max 40 / eta_min 1e−5、40 epoch、batch 64、bf16 AMP、种子 2026/2027/2028、无其他增强。每个 (L, 频段) 先跑 3 epoch 冒烟（种子 2026），正式训练同种子从头。
- 交付 `train_vs_M.diff`、`train_vs_H.diff`，diff 里只能出现路径/L/断言的改动。
- 顺序：L10 → L30 → L40 → L50t；每个 L 内先 2.4G 后 5.8G。

## 4. 评估（每个 L × 频段 × 种子；argmax 与 thr_fa1 两种判决都报）

1. 主表（口径照 `work/主线说明_20261109.md` §2）：不含泛化集 G 的无人机漏检；按 (dataset, model, state, bw_label) 的漏检；**按文件（block_file）的漏检**；G 每项单列；"该状态未训练"的项单列；混淆矩阵。
2. 虚警：v3 test 背景（总体与按会话）、bgA test（2.4G）、白噪声（2.4G），都给 k/N 和 95% 单侧 Clopper–Pearson 上界。L50t 的口径：v3 背景两片只重叠 1 ms，**全部片段计入 CP**；bgA 是 data 块的镜像，尾窗口与前一片最多重叠 47 ms，**CP 只用 `on_original_grid==True` 的子集**，全集数字另列并注明片段不独立。
3. 测试叠加：三组 × 7 档的检出率（v3_background 组为虚警率）。
4. 同型号：5.8G Mini 5 Pro 的 data/ 与 v3；2.4G EVO II Pro V3 的 data/ 与 v3（**data/ 那一项属 G**，单列；v3 那一项计入 nonG。【2026-11-11 支线更正：原文写成"后者属 G"，与主线说明 §2 相反，已按 §2 回复 worker】）。§5.4 的"§6 计数表"不存在，改用任务 F `counts.csv`、N2 `counts.csv` 和 bgA 计数文件对照（已回复 worker）。
5. 3 个种子的均值、标准差、最差值。

## 5. 复现检查（任一不过就停下 ask；全部在训练前完成）

1. **时频图**：用本任务的函数在 L=20 重算任务 M 的 2.4G 缓存（`E:/side_M_notchfill_20261110/H/2.4GHz_test_spec.npy`）和任务 H 的 5.8G 缓存（`E:/drone_lab_spec_L20_20261108/5.8GHz_test_spec.npy`）中各 ≥ 30 片（含 v3 与 data），float32 比较最大差 ≤ 0.02。
2. **训练配对**：在 L=20 上按 §1 规则重新生成的配对，与 `work/side_L_wifimix_20261110/mix_train_{drone,bg}.pairs.csv` 的 sample_id、background_sample_id 完全一致，X、g 相对差 ≤ 1e−9。
3. **评估代码**：用本任务的评估代码对任务 M（2.4G，3 个种子 `best.pt`）和任务 H（5.8G）的 checkpoint 在 L=20 上复推，主表数字（nonG_miss、v3_test_FA、bgA_test_FA，argmax 与 thr_fa1）与 `work/side_M_notchfill_20261110/metrics_paired.csv`、`work/train_stage1_L20_20261108/` 的结果一致（片数完全一致；逐片 p_drone 最大差 ≤ 1e−3）。
4. **索引与划分**：每个 L 的片数与 §6 计数表一致；同一 group 不跨 split；bgA L50t 原网格行与 `index_bgA_L50.csv` 一致。
5. 记录 `work/crop_slice_20261107/`、`work/train_stage1_L20_20261108/`、`work/bgA_20261109/`、`work/side_L_wifimix_20261110/`、`work/side_M_notchfill_20261110/`、`work/side_N2_L50tail_index_20261110/` 下所有文件的 sha256，交付前再算一次，必须一致。

## 6. 资源与目录

- 代码与小文件：`work/main_W_window_20261110/`；缓存与 checkpoint：`E:/main_W_window_20261110/`。不写其他任何目录。
- 预计：GPU 约 3–3.5 小时（2.4G 训练约 2 h，L10 占一半；5.8G 约 0.4 h；生成时频图约 0.7 h）；新占磁盘约 9 GB，**超过 20 GB 立即停下 ask**。
- 内存 16 GB：训练缓存按 float16 常驻；单进程工作集超过 8 GB 时改为 mmap 按批读。不要和其他大内存任务同时跑。
- 开工前检查 `nvidia-smi` 无其他 Python 计算进程，否则 ask。
- Python：`C:/Users/Administrator/Desktop/数据集处理/.venv/Scripts/python.exe -I`，`PYTHONIOENCODING=utf-8`，关闭 pyc 写入。不装包、不 git 提交、不改划分、不删除任何文件。

## 7. 交付：`work/main_W_window_20261110/REPORT.md`（中文，≤ 120 行，直接写数字）

1. 每个 L 的片数表：band × (dataset, label) × split；keep 覆盖的不重复时长（秒）；v3 test 背景片数和 0 错时的 95% 上界。
2. 每个 L 的时间分辨率（每列多少 ms），以及 0.4 ms 突发约占几列。
3. 主对比表（L = 10 / 20 / 30 / 40 / 50t，L20 来自任务 M/H）：nonG 漏检、按型号漏检、v3 无人机召回、v3 test 虚警（k/N 与 U95）、bgA 虚警、白噪声虚警；每格 3 种子均值 ± 标准差和最差值。
4. 图：按型号的漏检随 L 变化（每个种子一个点，连均值）；同一段无人机信号在各 L 下的时频图（2.4G、5.8G 各一组）；测试叠加的检出率随 X 的曲线（每个 L 一条）。
5. §5 复现检查结果；L50t 的覆盖说明和缺失组。
6. **只陈述事实，不下结论。** 差异小于种子间波动的，标明"不能区分"。
