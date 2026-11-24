# CASCADE3-NEWSPLIT-A 数据方案（三类新冻结划分窗口索引）

生成时间（UTC）：2026-09-26T03:49:22.588298+00:00

## 权威输入身份

- 冻结决定：`work/meta_split2/split_candidate_104_20260926.csv`，SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`（由 `CODEX-FREEZE-20260926.json` 冻结）。
- G 盘正式清单：`G:\DroneRFa数据集\dataset_manifest.csv`，SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`。
- 数据根：`G:\DroneRFa数据集\original`；24 个 MAT 的名称、大小与 SHA256 均与冻结表逐项一致。
- 角色只取自冻结 CSV；`work/meta_split1/source_table_104.csv` 的同名列在本流程中未被读取。

## 窗口规则

- 通道：RF0，I/Q 各一列实数，100 MS/s 复数基带（与既有 reader 一致）。
- 窗口：1,000,000 采样（10.0 ms），非重叠，步长 = 窗口长度，1M 对齐。
- 每个源文件取全部完整 1M 窗：`windows = floor(source_samples / 1,000,000)`，尾段不足 1M 丢弃（本批 24 个文件尾段均为 0）。
- 10M 采集块是 1M 的整数倍，因此窗口不会跨越 10M 块边界。
- `window_index` 是文件内 0 起的全局窗序号；`block_index`/`window_in_block` 是同一窗口在 10,000,000 采样块内的等价定位。

## 标签空间

- `class_index`（最终三类）：0 = Background（T0000），1 = DJI Phantom 4 Pro（T0010），2 = DJI Air 2S（T0101）。
- `stage1_label`（阶段 1 二元）：0 = Background，1 = Drone（与旧索引 `label` 语义一致，`label` 列等于 `stage1_label`）。
- `stage2_label`（阶段 2）：仅两个无人机类有值，0 = Phantom 4 Pro，1 = Air 2S；Background 行 `stage2_eligible=0` 且 `stage2_label` 为空。

## 规模（窗口数 / 源文件数）

| 角色 | 文件 | 窗口 | Background | Phantom 4 Pro | Air 2S | 阶段1 背景 | 阶段1 无人机 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 12 | 1810 | 600 | 590 | 620 | 600 | 1210 |
| val | 6 | 930 | 280 | 310 | 340 | 280 | 650 |
| development_test_exposed | 6 | 890 | 280 | 300 | 310 | 280 | 610 |
| 合计 | 24 | 3630 | 1160 | 1200 | 1270 | 1160 | 2470 |

阶段 2 视图（仅无人机类）：训练 1210，验证 650，测试 610，合计 2470 窗。

## 与旧索引的规模对比（仅供 Codex 决定是否抽样）

| 项 | 旧 12 文件索引 | 本索引 | 比值 |
| --- | --- | --- | --- |
| 训练窗 | 920 | 1810 | 1.97x |
| 验证窗 | 450 | 930 | 2.07x |
| 测试窗 | 440 | 890 | 2.02x |

本流程没有抽样、没有改窗口规则；是否抽样或缩短训练由 Codex 决定。

## 使用约束

- 一个源 MAT 只出现在一个角色中；窗口继承源文件角色，未来任何派生划分都必须继续继承。
- `development_test_exposed` 的 6 个 MAT 此前已用于旧实验（`exposed_to_prior_selection=True`），只能作已暴露开发对照，不得称为全新 final test 或跨会话泛化。
- 测试文件不得用于训练、选 epoch、选阈值或选超参数；训练视图只用 `source_role == "train"`。
- 本流程未训练模型、未改动 G 盘数据、未改动任何历史实验产物。

## 产物

- `window_index_newsplit3.csv`（SHA256 `deed870f9693c5ebe93fb80a9647b452e90680dd5b1f21372e108ded4bf4d231`）：全部 3630 窗。
- `stage2_view_newsplit3.csv`（SHA256 `76001e671cf3ebe492b191af87a85b5368caed85e03c84c4c249ccc5081642de`）：无人机窗投影，含稳定 `stage2_position`。
- `newsplit3_dataset_summary.json`：机器可读计数与逐文件明细。
- `newsplit3_source_identity.json`：24 个 MAT 的大小、SHA256、HDF5 形状。
- `newsplit3_verification.json`：独立核验与反向对照结果（由 `verify_newsplit_index.py` 生成）。

## 复现命令

```
python dronerfa_baseline/build_newsplit_index.py \
  --freeze-json work/meta_split2/CODEX-FREEZE-20260926.json \
  --data-root 'G:\DroneRFa数据集\original'
python dronerfa_baseline/verify_newsplit_index.py --sha256-mode recompute
python -m unittest dronerfa_baseline.test_newsplit_index -v
```
