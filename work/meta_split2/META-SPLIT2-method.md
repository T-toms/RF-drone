# META-SPLIT2：13 类开发用源文件级划分候选（方法与边界）

状态：**候选，未冻结**。本文档描述 `work/meta_split2/` 生成的 104 行源文件级划分工件，
交 Codex 验收后由 Codex 决定是否正式冻结。划分规则由 Codex 冻结，本目录只做实现与核验。

生成日期：2026-09-26。项目：`C:\Users\Administrator\Desktop\数据集处理`。

---

## 1. 范围与不做的事

本轮**只**产生候选划分与核验证据：

- **不**修改 `G:\DroneRFa数据集\dataset_manifest.csv`（起草前后 SHA256 均为
  `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`）。
- **不**训练、**不**提取 1M 窗、**不**下载、**不**改任何 MAT、**不**动 `work/meta_split1/`、
  `work/meta_header1/`、`work/sel12g/`、`dronerfa_baseline/`。
- **不**重算 107 GB 全量哈希：`mat_sha256` 一律引用 SELECT12-G3 的记录；本轮只对 104 个
  文件做只读 `stat`（存在性 + 字节数），不做载荷读取。
- **不**根据任何模型准确率改动本规则。

## 2. 固定输入与身份核对

| 输入 | 路径 | 身份 |
|---|---|---|
| G 正式清单 | `G:\DroneRFa数据集\dataset_manifest.csv` | SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`（104 行，与任务冻结值一致） |
| 文件头时间 | `work/meta_split1/source_table_104.csv` | 104 行；本轮只取 `mat_created_local` / `acquisition_campaign_key` / `acquisition_night_key` |
| MAT 哈希 | `work/sel12g/sel12g-mat-sha256-20260925.json` | 104 条；`count=104`、`bytes=107,450,183,727` |
| 既往使用佐证 | `work/meta_split1/prior_use_scan.json` | 只作为「新十类未见于模型产物」的佐证，不参与划分规则 |

三者文件名集合完全一致（104/104/104），每个文件的 `size_bytes`、`class_id`、`origin`、
`G:` 绝对路径逐条一致；104 个文件在盘上全部存在且字节数与清单相符。

时间字段可信性检查（通过）：`mat_created_local` 全部可解析为 ISO 本地时间；由其重算的
`mat_created_day`、`acquisition_night_key`（减 6 小时）、`acquisition_campaign_key` 与
source table 记录逐条相同。**这些时间仍是 MATLAB `Created on` 文件头时间，只是文件时间代理，
不是已证明的无线电采集时间，也不是会话 ID**；本轮不据此主张任何跨会话属性。

一个术语冲突（已在候选表中分离记录，不是阻塞项）：`source_table_104.csv` 的
`prior_use_role` 列**是按文件名 S 码推出的归档分段语义**（S0000/S0001 是 DroneRFa 上游
自带的 train 命名，S0010/S0011/S01xx 同理），对旧 24 个文件恰好等同于本项目既往用途，
但对新 80 个文件**不是**既往用途证据。候选表因此把两者拆成两列：

- `prior_use_role`：本项目真实的既往用途。新十类 80 个文件一律
  `not_previously_used_in_project`；旧 24 个文件沿用其既有角色。
- `archive_segment_role` / `archive_segment_evidence`：保留上游 S 码语义原文，避免静默丢弃输入信息。

## 3. 冻结规则（实现即此，未作任何改动）

**新 10 类**（`origin=selected_80`，每类 8 文件）：按 MAT 文件头时间升序、同时间按文件名升序，
第 1–5 个 `train`、第 6 个 `val`、第 7–8 个 `development_test`。

**旧 3 类** T0000/T0010/T0101：原 `development_original` 的 4 个文件全进 `train`；
原 `evaluated_holdout` 的 4 个按时间升序，前 2 个 `val`、后 2 个 `development_test_exposed`。

补充约束：每个源 MAT 恰好出现一次；分组单元是**源文件**（`minimum_group_key = file:<name>`），
将来由该文件生成的任何 1M 窗必须与其源文件同角色，**禁止同一文件跨 split**；
`development_test` 不得改名为 `final_test`。

规则结果（21 项机器检查全部通过）：

- 总计 **62 train / 16 val / 26 test**（新十类 test 20 + 旧三类 exposed test 6），104 行无重复、无跨 split。
- 13 类每类均为 8 个文件：新十类 5/1/2，旧三类 4/2/2。
- 每类时间排序后角色块严格为「train 段 → val → test」，且每类只落在一个 `campaign_proxy` 内。
- 类内不存在同秒文件头时间，规则里的同时间按文件名升序**从未被触发**（仅为防御性实现）。

### 每类结果

| class_id | class_index | 类别 | origin | campaign_proxy | night_proxy | train | val | test | test 角色 |
|---|---|---|---|---|---|---|---|---|---|
| T0000 | 0 | Background | baseline_24 | C20221127_T0000 | N20221127 | 4 | 2 | 2 | development_test_exposed |
| T0001 | 1 | DJI Phantom 3 | selected_80 | C20221127_T0001 | N20221127 | 5 | 1 | 2 | development_test |
| T0010 | 10 | DJI Phantom 4 Pro | baseline_24 | C20221127_T0010 | N20221127 | 4 | 2 | 2 | development_test_exposed |
| T0011 | 11 | DJI Matrice 200 | selected_80 | C20221127_T0011 | N20221127 | 5 | 1 | 2 | development_test |
| T0100 | 100 | DJI Matrice 100 | selected_80 | C20221127_T0100 | N20221127 | 5 | 1 | 2 | development_test |
| T0101 | 101 | DJI Air 2S | baseline_24 | C20221128_T0101 | N20221127 | 4 | 2 | 2 | development_test_exposed |
| T0110 | 110 | DJI Mini 3 Pro | selected_80 | C20221128_T0110 | N20221127 | 5 | 1 | 2 | development_test |
| T0111 | 111 | DJI Inspire 2 | selected_80 | C20221128_T0111 | N20221127 | 5 | 1 | 2 | development_test |
| T1000 | 1000 | DJI Mavic Pro | selected_80 | C20221128_T1000 | N20221127 | 5 | 1 | 2 | development_test |
| T1001 | 1001 | DJI Mini 2 | selected_80 | C20221128_T1001 | N20221127 | 5 | 1 | 2 | development_test |
| T1010 | 1010 | DJI Mavic 3 | selected_80 | C20230227_T1010 | N20230227 | 5 | 1 | 2 | development_test |
| T1011 | 1011 | DJI Matrice 300 | selected_80 | C20230227_T1011 | N20230227 | 5 | 1 | 2 | development_test |
| T1100 | 1100 | DJI Phantom 4 Pro RTK | selected_80 | C20230227_T1100 | N20230227 | 5 | 1 | 2 | development_test |

新十类的 val 一律是各类的第 6 个文件（T0001/T0011/T0100/T0110/T0111/T1000/T1001 为
`*_D00_S0101.mat`，T1010/T1011/T1100 为 `*_S0101.mat`），test 一律是第 7、8 个
（`*_S0110.mat`、`*_S0111.mat`）。这是**类内时间阻断的文件级留出，仍属同一个文件时间簇**，
不是独立采集会话。

## 4. 候选表字段

`split_candidate_104_20260926.csv`（104 行 × 24 列，UTF-8 BOM + CRLF，
SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`）：

| 列 | 含义 |
|---|---|
| `file_name` | 源 MAT 文件名 |
| `class_id` / `class_name` / `class_index` | 类别标识、名称、清单序号 |
| `role` | 本划分角色：`train` / `val` / `development_test` / `development_test_exposed` |
| `role_rank_in_class` | 类内按文件时间的位次 1–8 |
| `role_purpose` | 该角色的用途说明 |
| `metrics_scope` | 指标归属（见第 6 节） |
| `prior_use_role` / `prior_use_evidence` | 本项目既往用途及其证据 |
| `archive_segment_role` / `archive_segment_evidence` / `archive_s_code` | 上游 S 码语义原文（保真保留） |
| `former_partition` | G 清单值；新 80 个文件在清单中为空，按 `source_table_104.csv` 的口径归一为 `NR` |
| `origin` | `baseline_24`（旧 24）/ `selected_80`（新 80） |
| `exposed_to_prior_selection` | 是否已暴露于本项目既往选型 |
| `file_time_proxy` | MAT `Created on` 文件头时间（文件时间代理） |
| `campaign_proxy` / `night_proxy` | 类内文件时间簇键 / 夜键（减 6 小时，人为夜界） |
| `mat_sha256` | SELECT12-G3 记录的 MAT SHA256（本轮未重算） |
| `g_path` | G 盘绝对路径 |
| `size_bytes` | 清单字节数 |
| `minimum_group_key` | 最小分组单元 = `file:<name>` |
| `window_rule` | 未来窗口必须继承源文件角色，禁止同文件跨 split |

`split_candidate_104_20260926.json` 额外记录输入身份哈希、规则原文、逐类明细、列语义、
指标边界与禁用表述。CSV 逐字节可复现（连跑两次字节相同，SHA256 恒为
`d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`）；JSON 除
`generated_at` 外语义可复现（语义 SHA256
`6d6f3e1a94c48343e1b33626053d52275c95e4aa4e6d20a33b0d780ad5b5d86d`）。

## 5. 核验

`verify_split_candidate.py` 不信任 builder 的输出：它从三个固定输入**独立重推**整张划分表，
再与候选逐字段比对。21 项检查全部通过（`verification_meta_split2.json`）：

- `C00` 三个输入覆盖同一文件集合；`C01` G 清单身份哈希；`C02` 104 行 / 13 类 / 每类 8；
  `C03` 类–origin 组成；`C04` 三输入逐文件一致；`C05` 盘上 104 文件存在且字节数相符。
- `C06` 角色计数 62/16/20/6；`C07` 每类角色模式；`C08` 每文件恰好一个角色；
  `C09` 类内时间阻断且不跨 campaign_proxy；`C10` 类内无同秒时间。
- `C11` 旧三类 train 只来自 `development_original` 且共 12 个；`C12` 旧三类 val/test 共 12 个全部历史已暴露；
  `C13` 新十类 80 个既未 `exposed`、也在 `prior_use_scan.json` 的 data/model 产物命中中为 0。
- `C14` 命名门（禁止 `final_test` / `fresh` / `cross-session` 等）；`C14b` 既往用途证据语义；
  `C15` 候选存在且必需列齐全；`C16` 候选与独立重推逐字段一致（13 列）；
  `C17` 104 行 / 104 唯一文件名 / 104 唯一哈希；`C18` G 清单未被改动；`C19` CSV 与 JSON 一致。

`negative_control.py` 证明核验不是空转：13 个用例（含 1 个基线）分别篡改**副本**，
断言核验器必须以预期检查项拒绝，实际 13/13 符合预期（`negative_control_20260926.json`）。
用例覆盖：test→train、角色改名 `final_test`、SHA256 篡改、路径改到 `F:`、
同计数下 val↔test 互换（证明不只是数数）、重复行、暴露标志篡改，
以及四类输入副本变异（清单 `former_partition` 破坏 → 触发 8 项含 `C01/C07/C11/C16`；
source 表缺行 → `C00/C09/C16`；清单缺行 → 触发 8 项含 `C01/C00/C02/C16`；
`prior_use_scan` 改判为已用 → `C13`；source 表造出同秒时间 → `C10`）。
`C03/C14/C14b` 是对重推逻辑本身的守卫（类–origin 组成、角色命名、证据语义），
只有在冻结规则或被改坏时才会失败，由干净基线用例保证，不设篡改用例。
全部变异只写在 `work/meta_split2/_selftest_tmp/`，运行后已删除。

复算方式：

```powershell
python work\meta_split2\build_split_candidate.py      # 生成候选
python work\meta_split2\verify_split_candidate.py     # 21 项核验
python work\meta_split2\negative_control.py           # 13 项反向对照
```

## 6. 将来训练/验证/报告的指标边界

- **13 类整体结果只能叫开发评估**（development evaluation）。不得写成现场通用准确率、
  跨会话、跨接收机、开放集或部署鲁棒性。
- **新十类的 20 个 file-level test 可以单列为主指标**（`metrics_scope=new10_file_level_test_primary`），
  但仍必须同时声明它是**同一文件时间簇内的时间阻断文件留出**，不是全新独立采集会话；
  20 个测试文件来自 10 个类、每类仅 2 个源文件，样本量与类别覆盖有限。
- **旧三类的 6 个测试文件只作已暴露对照**（`metrics_scope=legacy_exposed_control`），
  不参与「未用于此前选型」的主指标；它们此前已进入 EVAL6/BASE2/NFFT4/CASCADE3 等评价。
  原 `evaluated_holdout` **不得**改写成 fresh final test。
- 旧三类 train 段包含其历史上的 validation/test 文件（S0010/S0011）：旧三类的既往划分已无法
  提供干净留出，规则刻意把 12 个旧文件全部并入 train，只留 12 个已暴露的旧 holdout 作 val/test。
  报告需明说这一点。
- 新十类 80 个文件虽未参与本项目既往选型，但 `campaign_proxy` 显示它们与训练文件同簇
  （10 类同批次；T1010/T1011/T1100 位于另一个夜 `N20230227`，同样类内同簇）。
  因此新十类 test 只能支持「同批次文件级留出」的结论。
- 禁用表述：fresh final test、独立采集会话、cross-session、open-set 泛化、现场/部署准确率。
- 独立最终测试仍需可核实的新采集或外部来源，本轮 104 文件无法提供。

## 7. 已知限制

1. 时间只是 MATLAB 文件头时间代理；同批次判断为「时间上相邻，疑似同批」，不是采集会话证据。
2. `campaign_proxy` 按「类 + 文件日期」定义，类内天然只有一个簇；它**不能**用来证明会话独立性。
3. 新十类 test 每类只有 2 个源文件，文件级留出的方差会很大；跨 seed 报告需谨慎。
4. `mat_sha256` 未在本轮重算，沿用 SELECT12-G3（2026-09-25）的记录；本轮只核对清单字节数与盘上字节数一致。
5. 本候选**尚未冻结**：在 Codex 验收并正式冻结前，不得据此启动训练、窗口提取或正式结果报告。
