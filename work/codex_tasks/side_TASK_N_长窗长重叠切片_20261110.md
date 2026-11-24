# 任务 N：L = 30/40/50 ms 改为半重叠切片，重新生成索引（只做索引，不训练）

下发：Claude（支线协调者），2026-11-10。执行：Codex。
**不要发送 heartbeat。不要调用 DSH / dsh-cli / mcp__dsh。** 小问题自己决定，写进 REPORT.md。只在以下情况用 orchestration `ask`：§3 任一检查不过；读盘错误；本文与数据明显冲突。完成时发一次 `worker_done`。

## 0. 背景（已查清，不需要重新论证）

- 任务 F（`work/crop_slice_20261107/finalize.py`）在 1 ms 网格上不重叠切片，步长 = L。v3 的块截频后只有 99.6 ms 和 49.6 ms 两种长度，即 99 或 49 个 1 ms 单元。
- 后果：L = 50 时，99 单元的块只切出 1 片，丢掉 49 ms；49 单元的块一片都切不出。2.4G 的 v3 train 背景总时长从 L20 的 4.68 s 降到 L50 的 2.6 s，train 背景组 `v3:noise_adrv_2450` 整组消失。
- 用户 2026-11-10 决定：长窗长重新切片，允许重叠；同源片段不能跨 train/val/test。
- 本任务只生成新的索引 csv，**不改任务 F 的任何文件**。L = 10、20 不变，继续用任务 F 的 `index_L10.csv`、`index_L20.csv`。

## 1. 切片规则（唯一的改动）

| L | 任务 F | 本任务 |
|---|---|---|
| 10、20 | 步长 L | 不变，不生成 |
| 30 / 40 / 50 | 步长 L | **步长 L/2 = 15 / 20 / 25 ms**，所有来源、所有频段一律如此（data 无人机、v3 无人机、v3 背景） |

- 起点 j 仍在 1 ms 网格上：j = 0, S, 2S, …，满足 j + L ≤ 块单元数。不跨块，不补零，不做尾部对齐。
- 新窗口 = 任务 F 原窗口（j 是 L 的整数倍）+ 错开半个窗长的窗口。每行加两列：
  - `on_original_grid`：j % L == 0 时为 True；
  - `orig_sample_id`：为 True 的行填任务 F 对应行的 sample_id，否则留空。
- `sample_id` 用新前缀 `L{L}h_{序号:08d}`（h = half overlap），和任务 F 的 id 不会冲突。
- split 照任务 F 的 `group_split.csv` 按 group_id 取，不重新划分。

## 2. 筛选（老师规则 v3b），必须保证原窗口的结果不变

- 筛选逻辑从 `work/crop_slice_20261107/finalize.py` 的 `screening()`、`intervals()` **复制**到本任务目录。**严禁 import `crop.py` 或 `finalize.py`**：它们导入时会执行 `design()`，覆盖任务 F 的文件。
- 需要的常量：
  - FS = {2.4GHz: 61440000, 5.8GHz: 100000000}，FC = {2430, 5800}，WIDTH = {25, 45}；
  - NOTCHES 从 `work/crop_slice_20261107/notch_table.csv` 读，`freq_mhz`、`half_width_khz` 转成 float。
- 1 ms 单元直接读任务 F 的缓存 `work/crop_slice_20261107/cells/<块文件名 stem>/{mean,max}.npy`，用 mmap 只读，不重算，不写入。
- 背景组照任务 F 的做法：不筛选，flags 全 False，keep = True。
- 无人机组：**所有参考统计量只用任务 F 的原窗口（步长 L）计算**，然后用同一套参考量对全部新窗口逐个打 flags。参考统计量包括：
  - `chars`、`primary`（任务 F 用的是 L=50 的原窗口）；
  - 每个 L 的 `dense`、`med`、`meds`、`outlim`、`tested`。
  这样原窗口的 flags 必然与任务 F 完全相同，新窗口用的也是同一把尺子。REPORT 中写明这一点。
- 每行列与任务 F 的 index 相同（block_file … keep），另加 §1 的两列。

## 3. 检查（任一不过就停下 ask）

1. **原窗口复现**：本任务中 `on_original_grid==True` 的行，与任务 F 的 `index_L{L}.csv` 逐行对应（block_file、start、n_samples）。行数相同，5 个 flags 与 keep **完全一致**，三个 L、两个频段都要查。
2. **不泄露**：每行 split 等于其 group_id 在 `group_split.csv` 中的 split。同一 block_file 的所有行 split 相同。任一 group 都不出现在两个 split 中。
3. **边界**：0 ≤ start，start + n_samples ≤ 块长度（样本数）；start 是 1 ms 样本数的整数倍；n_samples = L × 1 ms 样本数。
4. **数量自洽**：每个块的窗口数 = floor((单元数 − L)/S) + 1（单元数 < L 时为 0）。
5. 记录 `work/crop_slice_20261107/`、`work/train_stage1_L20_20261108/`、`work/bgA_20261109/`、`work/side_L_wifimix_20261110/`、`work/side_M_notchfill_20261110/` 下所有文件的 sha256，交付前再算一次，必须一致。

## 4. 交付

输出目录 `work/side_N_overlap_index_20261110/`，内容：
- `index_L30h.csv`、`index_L40h.csv`、`index_L50h.csv`；
- 脚本、`verification.json`、`REPORT.md`（中文，≤ 60 行，直接写数字）。

REPORT 要包含：

1. 切片与筛选规则，各一段。
2. §3 检查结果。
3. 计数表：L × band × (dataset, label) × split，列出“任务 F 片数 / 本任务片数 / 本任务 keep 片数 / keep 覆盖的不重复时长（秒）”。不重复时长按每块的窗口并集计算，不是片数 × L。L10、L20 也列出任务 F 的数字作参照。
4. 2.4G 与 5.8G 的 v3 背景按 group 列出 L20 与 L50h 的片数，说明还有哪些组整组切不出。
5. 新窗口中被筛掉（keep=False）的比例，与原窗口对比，按 band × state 列。
6. 一段说明：重叠片段彼此不独立。评估时，虚警的 Clopper–Pearson 上界应该只用 `on_original_grid==True` 的子集算，或者另外注明。

不训练、不生成时频图、不改任务 F 及其他目录的任何文件、不重新划分、不装包、不 git 提交、不调用 DSH。Python 用 `C:/Users/Administrator/Desktop/数据集处理/.venv/Scripts/python.exe -I`（本任务只需 numpy），`PYTHONIOENCODING=utf-8`。完成后发一次 `worker_done`（`--report-path work/side_N_overlap_index_20261110/REPORT.md`）。
