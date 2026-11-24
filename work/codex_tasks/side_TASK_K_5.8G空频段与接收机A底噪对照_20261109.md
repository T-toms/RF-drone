# 支线任务 K：5.8G 的"空频段 + 陷波"对照与接收机 A 底噪代理（只推理，不训练）

下发：Claude（支线协调者 claude-c3），2026-11-09。执行：Codex。
**不要发送 heartbeat。不要调用 DSH / dsh-cli / mcp__dsh。** 小问题按最合理方式处理并写进 REPORT.md，不要停下来问；只有读盘错误、显存/内存/磁盘不足无法绕过、或本文与数据明显冲突时才用 orchestration `ask`。完成时发一次 `worker_done`。**本任务不训练任何模型。**

## 0. 目的与背景（已查清，不需要重新论证）

- 2.4G 第一轮模型（任务 H）在接收机 A 背景上虚警 100%。对照显示原因是训练背景里从来没有"空频段"：合成白噪声经过同样的 FIR + 陷波，3 个种子也 100% 判无人机（`work/review_20261109/ctrl_notched_noise_eval.py`、`ctrl_fill_notch_rows.py`）。任务 J 把接收机 A 镜像背景加入训练后，2.4G 虚警降到 0%。
- **5.8G 没有接收机 A 背景**：data/ 的 5.8G 录制在时间上最长空隙 11 ms，频率上没有空闲频带（无人机跳频覆盖整个采样带宽）。所以 5.8G 模型在部署接收机上的虚警目前测不了，5.8G 可能有同样的 100% 误报问题。
- 本任务回答：**(1) 5.8G 模型对"空频段 + 陷波"是否也判无人机；(2) 用接收机 A 自己的 5.8G 底噪形状合成的噪声，模型怎么判。** (2) 是代理，必须先在 2.4G 上证明代理可信（§3），才能用于 5.8G。

## 1. 环境与输入（全部只读）

- scipy 步骤（FIR、陷波）：`D:/anaconda/python.exe -I`，先设 `KMP_DUPLICATE_LIB_OK=TRUE`。torch/GPU 步骤：`C:/Users/Administrator/Desktop/数据集处理/.venv/Scripts/python.exe -I`（没有 scipy）。一律 `PYTHONIOENCODING=utf-8`、`encoding='utf-8'`、pathlib 绝对路径，不装包。
- **严禁 `import work/crop_slice_20261107/crop.py`**（导入时会覆盖任务 F 的文件）。需要的处理照抄：终端 FIR 从 `work/crop_slice_20261107/filters.npz` 读（键 `'2.4GHz'` 143 taps、`'5.8GHz'` 193 taps），`scipy.signal.fftconvolve(x, h, mode='same')`；陷波从 `work/crop_slice_20261107/notch_table.csv` 读对应频段的行，逐个调用 `work/prep_lines_20261104/lab_preproc.py` 的 `notch(z, FS, (freq_mhz − FC)·1e6, half_width_khz·500.)`（FS 2.4G 61.44e6 / 5.8G 100e6；FC 2430 / 5800）。参考实现：`work/review_20261109/ctrl_notched_noise_gen.py`（2.4G 版，两端各补 20000 点再截掉）。
- 时频图：`work/train_stage1_L20_20261108/common.py` 的 `spec(iq, band)`（可以 import；不要调用它的 `model()`，不要往它的 `CACHE` 写东西）。片段长度 2.4G 1,228,800 点，5.8G 2,000,000 点。
- 模型（`dronerfa_baseline/model.py` 的 `ResNet18(1,2)`，checkpoint 里有 `thr_fa1`）：
  - 任务 H（第一轮）：`E:/drone_lab_spec_L20_20261108/{2.4GHz,5.8GHz}_seed{2026,2027,2028}_best.pt`
  - 任务 J（2.4G 第二轮，加了接收机 A 背景）：`E:/drone_lab_stage1_L20_bgA_20261109/2.4GHz_seed{2026,2027,2028}_best.pt`
- 任务 F 的 1 ms cell：`work/crop_slice_20261107/cells/<block_stem>/mean.npy`，dB，形状 (cells, bins)：2.4G 512 bin（120 kHz），5.8G 1024 bin（97.65625 kHz），频率 = FC + (k − bins/2)·bin 宽（**先用一个已知信号核对频率方向与偏移，结果写进 REPORT**）。块清单 `work/crop_slice_20261107/blocks.csv`（dataset、band、label、group_id、block_file）；划分 `group_split.csv`。
- 已有缓存（直接读，不重算）：任务 H `E:/drone_lab_spec_L20_20261108/{band}_{train,val,test}_spec.npy` + `_spec_meta.csv`；接收机 A 镜像背景 `E:/drone_lab_spec_bgA_L20_20261109/bgA_spec.npy` + `bgA_meta.csv`（**只用 keep==True**）。
- 输出：`work/side_5p8_controls_20261109/`（脚本、REPORT、图、表）；合成 IQ 若要落盘放 `E:/side_5p8_controls_20261109/`（预计 < 10 GB）。

## 2. 底噪形状的估计（两台接收机 × 两个频段）

对每台接收机、每个频段，从任务 F 的 cell 估计"无信号时的平均功率谱形状" S(f)：
1. 选块：接收机 A = `dataset=='data'` 的块（drone）；接收机 B = `dataset=='v3'`、`label=='background'`、group_id 含 `20260710`（07-10 实验室，几乎没有 WiFi）的块，**只用 train split 的组**。
2. 每块：每个 bin 取全部 cell 的第 10 百分位（dB）；同时记录每个 bin 的"占用率" = cell > 该块噪声底 + 3 dB 的比例（噪声底 = 50 个 1 MHz 子带各自中位数中最低的那个子带的中位数；与任务 I 相同的口径）。
3. 跨块合成：每个 bin 只用占用率 < 50% 的块，取这些块 p10 的中位数；参与块数 < 5 的 bin 记为缺失，用两侧最近的有效 bin 线性插值。陷波频点 ±0.3 MHz 内的 bin 一律当缺失插值（陷波在 §2.5 单独做）。
4. 报告每个 bin 的参与块数分布、插值的 bin 数与位置；画 S(f)（两台接收机、两个频段，四条曲线）。
5. 合成噪声：复高斯白噪声（每个片段独立，种子固定并记录），在频域按 sqrt(10^(S(f)/10)) 整形（S 插值到该片段 FFT 网格上；整段一次 FFT 即可，循环边界对噪声无影响），然后**再**按 §1 依次做终端 FIR 和陷波（与真实样本的处理链一致）。每类合成 200 个片段。
6. 核对：对合成片段用同样的 1 ms × bins 网格算 cell，按同样的 p10 方法重估 S'(f)，与 S(f) 比较（去掉陷波 ±0.3 MHz），最大绝对差应 ≤ 0.3 dB；不满足就先修，不要往下做。

## 3. 代理可信性检验（在 2.4G 上做，事先定好判据）

在 2.4G 上，代理"接收机 A 底噪形状噪声"（下称 proxyA）应当表现得像真实的接收机 A 镜像背景（bgA test）：
- 对任务 H 和任务 J 各 3 个种子，分别算 proxyA 200 片和 bgA test（keep==True 全部 1287 片）的"判为无人机"比例（argmax 与 thr_fa1）。
- **判据：**每个种子、每个模型、每种判决，|proxyA − bgA| ≤ 10 个百分点，算代理可信。另外报告两者模型输入的统计量：时间平均频率剖面（256 行）之间的相关系数与最大差（标准化单位），以及"像素 > +3"的占比。
- 第二个检验：proxyB（接收机 B 07-10 实验室底噪形状）在 2.4G 上与真实 v3 07-10 test 背景对比，同样判据。
- **不通过时：**照实报告，§4 中所有 proxy 的 5.8G 数字标"代理未通过 2.4G 检验，仅供参考"，其余照做，不要 ask。

## 4. 5.8G 对照（任务 H 的 5.8G 模型，3 个种子，argmax 与 thr_fa1）

每项 200 片（真实数据按实际片数），报告"判为无人机"的比例、k/N、95% 单侧 Clopper–Pearson 上界：
- (a) 纯复高斯白噪声（不做 FIR、不陷波）。
- (b) 白噪声 + 5.8G FIR + 陷波。
- (c) proxyA（5.8G 接收机 A 底噪形状 + FIR + 陷波）。
- (d) proxyB（5.8G 接收机 B 07-10 实验室底噪形状 + FIR + 陷波）。
- (e) 真实对照：v3 5.8G test 背景（按会话）；data 5.8G test 无人机召回（不含泛化集）。
- (f) **抹平陷波行**：对 (b)(c) 以及 (e) 的两组，把时频图上陷波频点 ±0.45 MHz 的行用相邻行线性插值替换、再逐样本标准化，重新推理（方法同 `work/review_20261109/ctrl_fill_notch_rows.py`，5.8G 陷波频点 5800、5795.5009375 MHz）。用于判断误报是否由陷波凹坑造成。
- 同样的 (a)–(d)、(f) 也对 2.4G 的任务 H 与任务 J 模型做一遍，放同一张表，便于对照。

## 5. 交付

`work/side_5p8_controls_20261109/REPORT.md`（中文，≤ 80 行，直接写数字）：
1. §1 频率方向核对；§2 四条 S(f) 的参与块数、插值情况、§2.6 重估误差。
2. §3 代理检验表（逐种子、逐模型、逐判决）与是否通过。
3. §4 总表：行 = 输入 (a)–(f)，列 = 频段 × 模型（2.4G-H、2.4G-J、5.8G-H）× 种子 × 判决。
4. 图 13（给老师看，中文标注）`图13_空频段与底噪代理对照.png`：左 = 四条底噪形状 S(f)（两个频段分两个子图，标出陷波位置）；右 = 各输入的"判为无人机"比例柱状（3 个种子均值，误差棒为标准差，2.4G-H / 2.4G-J / 5.8G-H 三组）。
5. 耗时、显存、异常与自行决定的小事项。
6. 末尾"**事实陈述，不下结论**"：关键数字各一行。

不训练、不改任务 F/H/I/J 的任何文件、不改共同目录、不删除/移动数据、不装包、不 git 提交、不调用 DSH。开工前与交付前对 `work/crop_slice_20261107/`、`work/train_stage1_L20_20261108/`、`work/bgA_20261109/`、`work/train_stage1_L20_bgA_20261109/` 算 sha256，必须一致。完成后发一次 `worker_done`（`--outcome succeeded/failed`，`--report-path` 指向 REPORT.md）。
