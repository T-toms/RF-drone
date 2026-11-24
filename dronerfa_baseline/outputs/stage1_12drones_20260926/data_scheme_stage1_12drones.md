# STAGE1-12DRONES-A 数据入口方案（104 MAT / 13 类）

生成时间（UTC）：2026-09-26T08:41:43.625046+00:00

## 冻结输入

- 划分权威：`work/meta_split2/CODEX-FREEZE-20260926.json` → `work/meta_split2/split_candidate_104_20260926.csv` (SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`)
- G 正式清单：`G:\DroneRFa数据集\dataset_manifest.csv` (SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`)
- 数据根：`G:\DroneRFa数据集\original`；104 个 MAT 全部按冻结 `role` 索引，源文件不跨角色。

## Stage1 标签空间

- `T0000` = Background = 0；其余 12 个 T 码 = Drone = 1。
- `class_id` 等冻结字段原样保留，仅用于溯源，不参与 Stage1 损失。

## 窗口规则

- RF0 I/Q, non-overlapping 1,000,000-sample windows, stride 1,000,000, tail < 1,000,000 samples dropped；每窗 1000000 采样 @ 100000000 Hz（10.0 ms）。

## 角色与窗口统计

| role | 文件数 | 窗口数 | label 0 | label 1 | 本阶段用途 |
| --- | --- | --- | --- | --- | --- |
| train | 62 | 9590 | 600 | 8990 | 模型拟合（全部窗口） |
| val | 16 | 2360 | 280 | 2080 | 选型（clean/-10/-12/-15 dB 四条件） |
| development_test | 20 | 2950 | 0 | 2950 | 不读取，仅元数据登记 |
| development_test_exposed | 6 | 890 | 280 | 610 | 不读取，仅元数据登记 |

## 源身份与核验

- 索引 SHA256：`5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6`
- 汇总 JSON SHA256：`a7431ee61e559eb9542d58709daa9ac17dd1efe7ec7e1657378c83db389e0a8d`
- 源身份 JSON SHA256：`0d6890dd8b1cf614c80a6f0a0f260b235c76431b4c029bd4fcfb058302b68a7f`
- 读取探针 JSON SHA256：`fe26d513d4c27aeb7e72b6a7170f4c2418b34a2580f8eb4272035fd231c11f33`

## 限制

- 冻结划分是**开发用途**划分：新 10 类的 20 个测试文件与此前未参与选型但同属一个文件时间簇，旧 3 类的 6 个测试文件此前已暴露；不得称 13 类独立最终测试或跨采集会话。
- 9 个机型文件名含 `D00`（论文表 3：20～40 m）；`T1010/T1011/T1100` 文件名无距离字段，距离不可确认，本索引不推定其距离。
- Stage1 只输出 Background vs Drone，本阶段不对任何测试角色做评价。
