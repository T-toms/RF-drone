# 任务 M：2.4G 时频图修补陷波凹坑后重训（在任务 L 的 L-b 基线上只改这一件事）

下发：Claude（支线协调者），2026-11-10。执行：Codex。
**不要发送 heartbeat。不要调用 DSH / dsh-cli / mcp__dsh。** 小问题自己决定并写进 REPORT.md。只在以下情况用 orchestration `ask`：§3 任一检查不过；读盘错误、显存/内存/磁盘不足无法绕过；本文与数据明显冲突。完成时发一次 `worker_done`。

## 0. 背景（已查清，不需要重新论证）

- 任务 F 对所有 2.4G 样本在 2426.88、2440、2450 MHz 做了陷波，IQ 里留下了三道窄而深的凹坑。在纯噪声片段里，逐片标准化后这三行比邻行低 2–7 个标准单位，模型会把它当成线索：第一轮模型对陷波白噪声 100% 判无人机，任务 L 的 L-b 种子 2026 在任务 K 里对陷波白噪声也是 99.5%。
- 用户决定：2.4G 基线改为任务 L 的 **L-b**（data 无人机与 bgA 都叠加 train v3 背景）；陷波凹坑必须修。
- 协调者已做试算：`work/review_20261109/notch_hole_profile.py`（量凹坑宽度）和 `work/review_20261109/notch_fill_pilot.py`（修补方法与对比图 `notch_fill_pilot.png`、`notch_fill_pilot_zoom.png`）。
- 修补方法（镜像复制，记为 M）：在 STFT 原始频点上（池化前，60 kHz/点），把三处受影响频点逐帧换成坑外对称位置的频点。坑左半边复制左侧外面的点，右半边复制右侧外面的点。之后照原流程池化、取 dB、逐片标准化。
- 纯噪声上，修补后陷波行与邻行的均值差从 −2.0/−5.2/−6.9 降到 −0.06/−0.12/−0.07，行内标准差比 0.92–1.07。未陷波白噪声在原流程下是 −0.02 到 −0.09、0.93–1.05，两者一致。
- **IQ 不改**，只改时频图函数。叠加、加噪、对照都在 IQ 上做，然后统一过新函数，所以所有输入一致。

## 1. 唯一的改动

| 项 | L-b（基线） | 本任务 M |
|---|---|---|
| 时频图函数 | `common.spec(iq,'2.4GHz')` | `spec_fill(iq)`：与 `common.spec` 完全相同，只在取通带频点之后、池化之前，对受影响频点做镜像复制 |
| 其他 | — | 与 L-b 完全相同：数据组成、叠加配对（沿用 L 的 `mix_train_*.pairs.csv`，不重新抽样）、权重、选模型、阈值、超参、种子 2026/2027/2028 |

`spec_fill` 的实现：从 `work/review_20261109/notch_fill_pilot.py` **复制** `HOLES`、`keep`、`hole_idx`、`fill_bins`（method='M'）、`spec_new` 到本任务目录（不要 import 该文件）。受影响频点固定为：
- 2426.88：2426.76–2427.00 MHz，4 个频点；
- 2440：2439.84–2440.20 MHz，6 个频点；
- 2450：2449.80–2450.16 MHz，6 个频点。

REPORT 中列出这 16 个频点的频率。

## 2. 输入（全部只读）

- Python：`C:/Users/Administrator/Desktop/数据集处理/.venv/Scripts/python.exe -I`，`PYTHONIOENCODING=utf-8`，pathlib 绝对路径，不装包。
- 任务 L 代码 `work/side_L_wifimix_20261110/`（`support.py`、`train.py`、`generate.py`、`preflight.py`、`evaluate.py`、`controls.py`、`finish.py` 等可以复制改写）。L 的叠加配对 `mix_train_drone.pairs.csv`、`mix_train_bg.pairs.csv`；C1 配对 `C1.pairs.csv`；L-b checkpoint 在 `E:/side_L_wifimix_20261110/`；L 的逐片预测与 `metrics_paired.csv`。
- 任务 H 的 meta 与 IQ：`E:/drone_lab_spec_L20_20261108/2.4GHz_{train,val,test}_spec_meta.csv`（含 block_file/start/n_samples）、`2.4GHz_rejected_test_spec_meta.csv`（若有）；三组 mix 测试的 `.pairs.csv`。
- bgA meta：`E:/drone_lab_spec_bgA_L20_20261109/bgA_meta.csv`（只用 keep==True）。
- 对照输入：`work/review_20261109/notched_noise_2.4GHz.npy`（陷波白噪声 80 片）；纯白噪声按 J/L 的同一种子生成；013 安静片段按 L 的做法。
- `common.py` 可以 import 用 `spec()`、`rows()`、`writecsv()`、`seed()`、`image()`。**不要用 `model()`，不要往任务 H 的 `CACHE`、任务 J 的 `DEST`、任务 L 的 `DEST` 写任何东西。严禁 import `work/crop_slice_20261107/crop.py`。**

## 3. 开工前检查（任一不过就停下 ask）

1. **函数一致性**：随机抽 H 的 test 3 片、bgA 3 片。`spec_fill` 去掉修补（HOLES 为空）后与 `common.spec` 逐像素完全相同；带修补时，除了受影响频点所在的行（每处 2–3 行），其余行与 `common.spec` 的差别只来自逐片标准化的整体平移和缩放。具体检验：对非陷波行做线性回归 new = a·old + b，残差最大绝对值 ≤ 0.01。
2. **叠加可复现**：按 L 的 `mix_train_drone.pairs.csv` 前 3 行和 `mix_train_bg.pairs.csv` 前 3 行，用 `common.spec` 重算，与 L 缓存 `mix_train_*.npy` 对应元素最大差 ≤ 0.01。
3. **修补效果**：在 bgA test 40 片、陷波白噪声 40 片上复现 `notch_fill_pilot.py` 的行统计（陷波行均值差、标准差比）。要求每处 |均值差| ≤ 0.2，标准差比在 0.85–1.15 之间。
4. 计数与 L 相同：data 无人机 2832、v3 无人机 152、v3 背景 234、bgA train/val/test = 2861/990/1287、叠加无人机 2832、叠加背景 2861。
5. 记录 `work/crop_slice_20261107/`、`work/train_stage1_L20_20261108/`、`work/bgA_20261109/`、`work/train_stage1_L20_bgA_20261109/`、`work/side_L_wifimix_20261110/`、`work/review_20261109/` 下所有文件的 sha256，交付前再算一次，必须一致。
6. `nvidia-smi` 查有无其他 python 计算进程；有就 ask。

## 4. 重算时频图缓存（新目录 `E:/side_M_notchfill_20261110/`）

全部从 IQ 用 `spec_fill` 重算，行序与原 meta 一致，float16：
- H 的 2.4G train/val/test，以及被 keep 筛掉的 test 片段；
- bgA 全部 keep==True 的行（按 bgA_meta 行序，可只存 keep 行并另存索引）；
- 叠加训练样本两组：按 L 的 pairs.csv 用同一个 X、背景、g 重建 IQ（公式见任务 L 任务书 §4），再算 `spec_fill`；
- 叠加检验三组（H mix 的 pairs.csv）× 7 档；C1（L 的 C1.pairs.csv）× 7 档；
- 对照：纯白噪声 80、陷波白噪声 80、013 安静 19。

读盘：IQ 按 block_file、start 顺序 mmap 读取。v3 背景 IQ 可以预读。预计新占磁盘约 4 GB，超过 20 GB 就停下 ask。

## 5. 训练

- 设置与 L-b 完全相同，只换输入缓存。先跑 seed 2026 的 3 epoch 冒烟，再跑 3 个种子 × 40 epoch。checkpoint 放 §4 的新目录，断点可续，日志 `run.log`，每 epoch 记录同 L。
- **不重训 L-b**，L-b 的数字直接取任务 L 的 csv。

## 6. 评估（L-b 与 M 并列；3 个种子分别报告，再给均值±标准差；argmax 与 thr_fa1 两套；虚警给 k/N 和 95% 单侧 Clopper–Pearson 上界）

1. 任务 L §6 的全部项目照做（v3 test 背景按会话、bgA test/val、013、白噪声、陷波白噪声、C1、非 G 漏检、含 G、按型号/状态/带宽、v3 型号召回、同型号、G 单列、被筛掉片段、叠加检验曲线、按型号叠加检验）。M 用修补后的缓存，L-b 用原缓存（即 L 已有结果）。
2. **交叉对照（新增，只推理）**：
   - 用 L-b 的 3 个模型推理修补后的 test 缓存、bgA test、陷波白噪声。看 L-b 模型是否依赖凹坑。
   - 用 M 的 3 个模型推理原流程（未修补）的陷波白噪声、bgA test。看 M 模型在旧输入上会怎样。
3. **陷波行的模型关注度**（只描述）：对 bgA test 前 200 片，把修补后图像中陷波所在行再人工压低回原流程的凹坑深度（行均值减去协调者测得的差值 2.0/5.2/6.9，只改这些行），看 M 模型判无人机比例的变化。
4. 图 15（给老师看，中文标注）`图15_陷波修补前后_2.4GHz.png`：
   - 上排是同一片 bgA 纯噪声、同一片陷波白噪声，修补前后时频图对比，放大 2420–2455 MHz；
   - 下排是两栏柱状：左 = 虚警（v3 test 背景、bgA test、C1 合并、013 安静、白噪声、陷波白噪声），右 = 召回（data 非 G、v3 无人机、mavic2、mini3pro、G 各项），每项 L-b / M 两根柱，误差棒为 3 个种子的标准差。
5. 逐片预测 csv 同 L 的命名规则（前缀 Lb_ / M_）。

## 7. 交付

输出目录 `work/side_M_notchfill_20261110/`：脚本、日志、表、图、`REPORT.md`（中文，≤ 90 行，直接写数字）：
- 设置 + 与 L 的 `train.py`、`support.py` 的 diff 摘要（应只涉及时频图函数与路径）+ 16 个受影响频点；
- §3 检查结果；3 次训练的 best epoch、val 指标、thr_fa1；
- §6 各表、图 15、叠加检验图、C1 图；
- 耗时、显存与内存峰值、新占磁盘；异常与自行决定的小事项；
- 末尾“**事实陈述，不下结论**”，关键数字各一行（argmax，L-b → M）：陷波白噪声 FA、白噪声 FA、bgA test FA、v3 test FA、C1 FA、v3 无人机召回、mavic2、mini3pro、非 G 漏检；以及交叉对照中“L-b 模型在修补输入上”和“M 模型在未修补陷波白噪声上”的 FA。

不训练 5.8G、不改任务 F/H/I/J/L 的任何文件、不重新划分、不删除/移动数据、不装包、不 git 提交、不调用 DSH。完成后发一次 `worker_done`（`--outcome succeeded/failed`，`--report-path work/side_M_notchfill_20261110/REPORT.md`）。
