# DATA-MERGE1：统一 F 盘 DroneRFa 原始数据目录

## 目标

把 `F:\DroneRFa数据集\original` 与 `F:\DroneRFa数据集\heldout_20260917` 顶层的 MAT 原始数据合并到一个目录：

`F:\DroneRFa数据集\original`

保留原先 development/heldout 的实验角色信息，但改为由数据清单记录，不再依赖两个物理数据目录。

## 已核实事实

- 基准目录：`F:\DroneRFa数据集`
- 源目录：`F:\DroneRFa数据集\heldout_20260917`
- 目标目录：`F:\DroneRFa数据集\original`
- 目标目录当前有 12 个顶层 `.mat`，合计 12,340,116,227 字节。
- 源目录当前有 12 个顶层 `.mat`，合计 12,382,967,102 字节。
- 两组 MAT 文件名冲突数为 0。
- 源目录除 12 个 MAT 外，只有 `downloads` 子目录；它是下载日志、断点和记录，不是数据集样本。
- 用户明确要求合并这两个 F 盘数据目录。

## 必须执行

1. 开始前用 `[IO.Path]::GetFullPath()` 再次解析上述 base/source/destination，确认 source 和 destination 都严格位于 `F:\DroneRFa数据集` 下；若不满足立即停止。
2. 再次确认源目录恰好 12 个顶层 MAT、目标目录恰好 12 个顶层 MAT、同名冲突为 0；任何条件不满足都停止，不覆盖文件。
3. 逐个使用字面量路径移动源目录顶层的 12 个 MAT 到 `F:\DroneRFa数据集\original`。禁止使用覆盖参数，禁止删除任何 MAT。
4. MAT 移动完成后，把只剩 `downloads` 的 `F:\DroneRFa数据集\heldout_20260917` 重命名为 `F:\DroneRFa数据集\download_records_20260917`。若目标记录目录已存在则停止并报告，不合并、不覆盖。
5. 在 `F:\DroneRFa数据集\dataset_manifest.csv` 生成24行文件清单，至少包含：`file_name,class_id,class_name,former_partition,unified_path,size_bytes`。原 `original` 的12个文件标记 `development_original`；本次移动的12个文件标记 `evaluated_holdout`。类别映射：T0000=Background，T0010=DJI Phantom 4 Pro，T0101=DJI Air 2S。
6. 创建项目内 `work/dataset-unification-20260922.json`，记录迁移前后目录、精确文件数与总字节数、12个移动文件名、无冲突检查、记录目录新路径、manifest 路径和最终检查结果。
7. 验收最终状态：
   - `F:\DroneRFa数据集\original` 顶层恰好24个 MAT；
   - 总字节数恰好 24,723,083,329；
   - 原 `heldout_20260917` 路径不存在；
   - `download_records_20260917` 存在且仍包含原 `downloads`；
   - `dataset_manifest.csv` 恰好24条数据行、24个唯一文件名、3个类别各8个文件；
   - 目标目录中的每个文件大小与迁移前记录一致。

## 路径兼容与科学边界

- 不修改已有冻结实验配置、历史输出、模型权重或历史 CSV；它们可能仍记录旧绝对路径，作为历史证据保留。
- 在迁移证据中明确记录旧路径到新路径的映射。以后重跑旧实验时，应通过 manifest/remap 解析新路径。
- `evaluated_holdout` 已经用于既有评估，不能改称“未触碰最终测试集”。
- 不移动或修改 `F:\DroneRFa数据集\analysis_outputs`。
- 不下载数据，不训练模型，不清理下载日志，不删除目录树。

## 允许修改

- `F:\DroneRFa数据集\original`（仅接收这12个 MAT）
- `F:\DroneRFa数据集\heldout_20260917`（仅移动顶层12个 MAT并在空出后重命名）
- `F:\DroneRFa数据集\download_records_20260917`
- `F:\DroneRFa数据集\dataset_manifest.csv`
- `work/dataset-unification-20260922.json`
- 本次交接文件

## 交接

完成后写 `completed`；如果遇到权限、重名、数量或字节数不符，写 `needs_codex` 并停止，不能自行扩大范围或覆盖文件。
