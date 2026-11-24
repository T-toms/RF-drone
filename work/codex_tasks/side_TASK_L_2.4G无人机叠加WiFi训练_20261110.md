# 任务 L：2.4G 一阶段训练加入“无人机 + 环境 WiFi”叠加样本（两组对照 L-a / L-b）

下发：Claude（支线协调者），2026-11-10。执行：Codex。
**不要发送 heartbeat。不要调用 DSH / dsh-cli / mcp__dsh。** 小问题按最合理方式处理并写进 REPORT.md，不要停下来问。只在以下情况用 orchestration `ask`：§3 任一检查不过；§4 试算图完成（**必须** ask，等回复再继续）；读盘错误、显存/内存/磁盘不足无法绕过；本文与数据明显冲突。完成时发一次 `worker_done`。

## 0. 背景与目的（已查清，不需要重新论证）

- 任务 H/J 的 2.4G 模型在城区 v3 无人机上召回只有约 58–62%（mavic2、mini3pro 漏检 60–100%）。叠加检验（`work/train_stage1_L20_20261108/mix_test.py`）中，同一批无人机叠到 v3 背景上检出率明显下降，叠白噪声反而升高。解释：训练集里的 data/ 无人机（接收机 A、实验室）全都没有环境 WiFi，模型学到“有 WiFi 偏背景”。
- 文献做法（Glüge 等 2024，IEEE J-RFID《Robust Low-Cost Drone Detection and Classification Using CNNs in Low SNR Environments》II-B/II-C 节，协调者读过全文）：暗室录的无人机 IQ 与另录的 WiFi/蓝牙背景 IQ 按功率比离线叠加，作为训练样本；**背景类也用同一公式做“背景+背景”叠加**，使“被叠加过”这件事本身不区分类别。
- 本任务分两组，各 3 个种子：
  - **L-a**：在任务 J 的训练集上，只加入“train 集 data/ 无人机 + train 集 v3 背景”的叠加样本（标无人机）。与 J 只差这一件事。
  - **L-b**：在 L-a 上，再加入“train 集 bgA + train 集 v3 背景”的叠加样本（标背景），即文献的对称做法。与 L-a 只差这一件事。
- 担心的新捷径：叠加样本 = 接收机 A 底噪 + 接收机 B 的 WiFi，若只在无人机类出现，模型可能学成“A 底噪 + WiFi = 无人机”。§6 的对照 C1 专门检验这一点。

## 1. 唯一的改动

| 项 | 任务 J | L-a | L-b |
|---|---|---|---|
| 2.4G train 无人机 | data 2832 + v3 152 = 2984 | 同 J + **叠加无人机 2832**（每个 data/ train 无人机片段 1 份） | 同 L-a |
| 2.4G train 背景 | v3 234 + bgA 2861 | 同 J | 同 J + **叠加背景 2861**（每个 bgA train keep 片段 1 份） |
| 损失权重（类内） | 无人机类内均匀；背景类内 v3:bgA = 1:1 | 无人机类内 原始:叠加 = 1:1；背景类内同 J | 无人机类同 L-a；背景类内 v3 : bgA : 叠加背景 = 2 : 1 : 1 |
| val、选模型、thr_fa1 | J 的规则 | **不变**（val 不加叠加样本） | **不变** |

权重（逐样本，写进 REPORT 的实际数值）：两类总权重相等（各 0.5）。各子类权重按上表比例分配，再除以该子类的片数，最后整体缩放到 train 集上 w 的均值 = 1。batch 损失 = Σ(w_i·CE_i) / Σ(w_i)。val loss 沿用 J 的权重（以 train 的 N_d、N_v、N_a 计算，与 J 完全一致）。
理由：若按片数加权，L-b 中 v3 背景（唯一的原样 WiFi 背景）在背景类里的占比会从 50% 掉到约 4%，等于同时改了第二件事。

**其他全部与任务 J 相同**：输入（已有缓存，不重算）、`ResNet18(1,2)`（不改）、AdamW lr 1e-3、wd 1e-4、cosine 到 1e-5（T_max 40）、40 epoch、batch 64、AMP 自动选、种子 2026/2027/2028、`seed()`、cudnn.benchmark=True、不做其他增强、不调参。训练脚本以 `work/train_stage1_L20_bgA_20261109/train.py` 为底本复制修改，REPORT 中列出 diff（只应涉及数据拼接、权重、路径）。

## 2. 输入（全部只读）

- Python：`C:/Users/Administrator/Desktop/数据集处理/.venv/Scripts/python.exe -I`（CUDA torch，无 scipy，本任务不需要）。`PYTHONIOENCODING=utf-8`，文本 `encoding='utf-8'`，pathlib 绝对路径，不装包。
- 任务 H 时频图缓存与 meta：`E:/drone_lab_spec_L20_20261108/2.4GHz_{train,val,test}_spec.npy` + `_spec_meta.csv`（含 block_file、start、n_samples）。叠加检验缓存 `2.4GHz_mix_{data_drone,v3_drone,v3_background}.npy` 及同名 `.pairs.csv`。
- bgA：时频图 `E:/drone_lab_spec_bgA_L20_20261109/bgA_spec.npy` + `bgA_meta.csv`（只用 keep==True；IQ 在 `block_file` 列，位于 `E:/drone_lab_bgA_20261109/`）。
- 任务 J：代码 `work/train_stage1_L20_bgA_20261109/`（`support.py`、`train.py`、`evaluate.py`、`run_eval.py` 可复制改写）；checkpoint `E:/drone_lab_stage1_L20_bgA_20261109/2.4GHz_seed<s>_best.pt`；逐片预测 csv 在 J 目录。
- `work/train_stage1_L20_20261108/common.py`：可 import 用 `spec()`、`predict()`、`rows()`、`writecsv()`、`seed()`、`image()`。**不要用 `model()`，不要往它的 `CACHE` 写任何东西。** J 的 `support.py` 同理，不要往它的 `DEST` 写。
- **严禁 import `work/crop_slice_20261107/crop.py`**（导入时会覆盖任务 F 的产出）。
- 划分：`work/crop_slice_20261107/group_split.csv`。

## 3. 开工前检查（任一不过就停下 ask）

1. 计数：2.4G train 中 data 无人机 2832、v3 无人机 152、v3 背景 234；bgA keep==True 的 train/val/test = 2861/990/1287。
2. 所有 train v3 背景、data 无人机、bgA 的 group_id 在 `group_split.csv` 中都是 train（0 不一致）。
3. 重算一致：data 无人机 train、v3 背景 train、bgA train 各抽 3 行，从 IQ 用 `spec()` 重算，与缓存最大绝对差 ≤ 0.01。
4. 叠加公式一致：用 H 的 `2.4GHz_mix_data_drone.pairs.csv` 前 3 行（sample_id、background_sample_id、g），按 §4 的公式重算，与 `2.4GHz_mix_data_drone.npy` 对应元素最大绝对差 ≤ 0.01。证明本任务的叠加与已有叠加检验是同一种操作。
5. J seed 2026 checkpoint 对 2.4G test 推理，与 J 的 `test_predictions_2.4GHz_seed2026.csv` 同 sample_id 的 p_drone 最大差 < 1e-3。
6. 记录 `work/crop_slice_20261107/`、`work/train_stage1_L20_20261108/`、`work/bgA_20261109/`、`work/train_stage1_L20_bgA_20261109/` 下所有文件的 sha256，交付前再算一次，必须一致。
7. GPU：`nvidia-smi` 查有无其他计算进程。若有别的训练或大批量推理在跑（例如做法 B 的任务），**不要开始 §4 的全量生成和 §5 的训练**，ask 协调者。

## 4. 叠加样本生成

公式与 `mix_test.py` 相同：设源片段 d（data 无人机或 bgA）、背景片段 b（train v3 背景），P_d = mean(|d|²)，P_b = mean(|b|²)（float64），功率比 X dB：
g = sqrt(P_b / P_d · 10^(X/10))，y = b + g·d，输入 = `spec(y, '2.4GHz')`。

- 随机数：`np.random.default_rng(20261110)`。先为叠加无人机（按 data 无人机 train 行序）依次抽 X ~ U[−5, 15]（连续均匀）和背景下标 j ~ 均匀整数 [0, 234)；再用同一个 rng 继续为叠加背景（按 bgA train keep 行序）抽。两组都只用 **train** 的 v3 背景（234 片，不筛选）。
- 两组 L-a/L-b 与 3 个种子**共用同一份叠加样本**（离线生成一次）。
- 每行记录 sample_id、background_sample_id、background group_id、X、g、P_d、P_b、实际功率比 → `mix_train_drone.pairs.csv`、`mix_train_bg.pairs.csv`。
- 读盘：v3 train 背景 234 片 IQ 预读入内存（约 2.3 GB，complex64）；源片段按 block_file、start 顺序 mmap 读取。输出 float16 memmap，写到缓存目录。

**试算（必须先做，完成后 ask，等协调者回复再全量生成）：**
- `fig_L_pilot_2.4GHz.png`：train 中每个 data 无人机型号（共 11 个）各取 1 片（rng 20261110 选）。每行依次是：原片、所用背景、X = −5/0/5/15 dB 的叠加结果。背景统一取 group `v3:20260709-020956_ambient` 的第 1 片。另加 2 行“bgA + 同一背景”。用 `common.image()` 画，中文标题。
- 同时给出每个型号原片的突发占空比（时频图上超过该片中位数 +3 个标准单位的像素比例）供参考。
- ask 正文写图的路径，等回复。协调者可能据图要求收窄 X 范围，按回复执行。

## 5. 训练

- 两组 × 种子 2026/2027/2028，共 6 次。先跑 L-a seed 2026 的 3 epoch 冒烟，确认速度、显存、内存（float16 常驻，不要整体复制成 float32），再全量。顺序：L-a 三个种子 → L-b 三个种子。
- 每 epoch 记录：train_loss、val_loss、val FA_v3、val FA_bgA、val Miss、lr、秒数 → `train_log_{La,Lb}_seed<s>.csv`；画 `fig5_curves_2.4GHz.png`（6 次训练）。
- checkpoint 与 done 标记放 `E:/side_L_wifimix_20261110/`（新目录）。best 中保存 thr_fa1、两种背景各自的阈值和片数、权重数值。后台运行、日志 `run.log`、断点可续（照 J）。

## 6. 评估

J、L-a、L-b 并列（J 的数字从 J 已有 csv 取；需要时用 J checkpoint 重推）。3 个种子分别报告，再给均值±标准差（ddof=1）。argmax 与 thr_fa1 两套。每个虚警数字带 k/N 和 95% 单侧 Clopper–Pearson 上界。

1. **J 的全部评估项照做**（任务 J 任务书 §5 的 1–4、7：v3 test 背景按会话、bgA test/val、013 安静、白噪声、陷波白噪声、test 无人机漏检（主表不含 G，G 单列，含 G 原口径）、按 (dataset, model, state, bw_label) 分表、被筛掉的 test 片段、按来源与同型号召回）。另把 v3 无人机按型号单列召回（重点 mavic2、mini3pro）。
2. **叠加检验**：H 的三组 mix 缓存（全是 test 无人机 + test 背景，训练没见过），每个 X 的检出率/虚警。画 `fig3_mix_test_2.4GHz.png`：J 虚线，L-a、L-b 实线。注明训练用的 X 范围（−5 到 15 dB），−10 和 +20 dB 在范围外。
3. **对照 C1（新捷径检验，必须做）**：bgA **test** 片段（rng 20261111 不放回抽 200 片）叠加 v3 **test** 背景（第 i 片配 i % 80），X = −10/−5/0/5/10/15/20 dB，公式同 §4，标签为背景。报告每个 X 判无人机的比例。缓存到本任务缓存目录。
4. **对照 C2**：v3 test 背景原样（不叠）FA 已在第 1 项；另报告 H 的 `v3_background` mix 组（背景+背景）各 X 的 FA。
5. **v3 背景的 WiFi 粗指标**（只描述，不用于筛选）：对每个 v3 背景 group（train/val/test），取缓存时频图每行沿时间的中位数，得频率轮廓；ch1 = 2408–2416 MHz 均值、ch6 = 2433–2441 MHz 均值、参考 = 2451–2454 MHz 均值，报告 group 内各片 (ch1−参考)、(ch6−参考) 的中位数。行 i 的频率 = 2405 + (i+0.5)·50/256 MHz。写明这是粗指标，参考带靠近陷波和滤波器边缘。
6. **图 14**（给老师看，中文标注）`图14_叠加WiFi训练前后_2.4GHz.png`：两栏柱状。左栏虚警：v3 test 背景、bgA test、C1（7 个 X 合并）、013 安静、陷波白噪声。右栏召回：data 无人机（非 G）、v3 无人机、v3 mavic2、v3 mini3pro、同型号 Mini 5 Pro（data / v3）、EVO II Pro V3（data / v3）、G 各项。每项 J / L-a / L-b 三根柱，误差棒为 3 个种子的标准差。另画 C1 的 X 曲线图 `fig_C1_2.4GHz.png`。
7. 逐片预测：`test_predictions_2.4GHz_{La,Lb}_seed<s>.csv`（sample_id, dataset, model, state, bw_label, group_id, label, p_drone, pred_argmax, pred_thr），bgA test、C1 各一份。

## 7. 交付

输出目录 `work/side_L_wifimix_20261110/`：脚本、日志、表、图、`REPORT.md`（中文，≤ 100 行，直接写数字）：
- 设置（一段）+ 与 J `train.py` 的 diff 摘要 + 两组权重的实际数值；
- §3 检查结果；试算图与协调者回复的处理；叠加样本 X 与功率比的实际分布（直方图或分位数）；
- 6 次训练的 best epoch、val FA_v3 / FA_bgA / Miss、thr_fa1；
- §6 各表（J / L-a / L-b 并列）、图 14、mix 图、C1 图、WiFi 粗指标表；
- 耗时、显存与内存峰值、新占磁盘；异常与自行决定的小事项；
- 末尾“**事实陈述，不下结论**”，关键数字各一行（argmax，J → L-a → L-b）：v3 无人机召回、mavic2 召回、mini3pro 召回、非 G 漏检、v3 test FA、bgA test FA、C1 FA（合并）、陷波白噪声 FA、mix 检验 0 dB 处 data 无人机检出率。

不训练 5.8G、不改任务 F/H/I/J 的任何文件、不重新划分、不重算已有缓存、不删除/移动数据、不装包、不 git 提交、不调用 DSH。完成后发一次 `worker_done`（`--outcome succeeded/failed`，`--report-path` 指向 `work/side_L_wifimix_20261110/REPORT.md`）。
