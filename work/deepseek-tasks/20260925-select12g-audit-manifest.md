# SELECT12-G3：104 文件审计与 G 清单提案

Codex 已验收下载阶段：冻结目标 104/104 个 MAT 在 `G:\DroneRFa数据集\original`，文件名和官方预期字节数逐一匹配；下载状态中 42/42 个新下载项 completed，最后 16 文件续传 failures=0，当前无下载锁。旧 `G:\DroneRFa数据集\dataset_manifest.csv` 仅 24 行，路径仍写旧 F 盘。用户要求判断后，Codex 决定：新增 80 个文件登记为 `split_status=unassigned`；原 24 个保留历史用途；在审计和清单验收前保留 `.restart` 裸块。此任务不冻结 train/val/test，不训练，不删除裸块。

工作范围：

1. 修正 `work/sel12g/audit_g_mats.py` 中 `np.asarray(probe).reshape(-1)` 会将整条超长 HDF5 通道加载入内存的问题，仅读取约 1000 个采样作有限值探针；兼容现有 (1,N) 和一维通道，拒绝不支持形状。运行 104 文件的只读结构审计，核查清单名/大小、HDF5 四通道、长度一致、浮点类型及有限值探针。审计输出写项目 `work/sel12g/`，不改 MAT。
2. 对 104 个目标 MAT 独立重算 SHA256，输出逐文件大小、哈希、路径与状态；可复用 `work/sel12g/hash_g_mats.py`，记录执行时间与完整性。若 DSH 沙箱拒绝执行项目 Python，不请求宽泛 danger-full-access 来做只读扫描；保留实现并报告 needs_codex，让 Codex 用受控命令执行。
3. 在项目 `work/sel12g/` 生成**候选** 104 行 CSV 与差异报告，不直接覆盖 G 的正式 `dataset_manifest.csv`，由 Codex 验收后发布。保留旧清单前 7 列语义（file_name、class_id、class_name、former_partition、unified_path、size_bytes、class_index），追加 `split_status`、`evaluation_role`、`origin`。所有 unified_path 指向实际 G 原始文件；旧 24 行保持原 class/size/former_partition，并将旧 development_original 12 标为 `legacy_development`、旧 evaluated_holdout 12 标为 `previously_evaluated`。新增 80 行 former_partition 为空、split_status=`unassigned`、origin=`selected_80`；evaluation_role 不得写 `final_test`、`heldout` 或暗示未参与选型的结论。按冻结目标表核对恰好 13 类各 8 文件、无重复文件名、无 F 路径、文件大小和哈希证据齐全。
4. 输出清单变更报告：旧 24 行如何保留，新增 80 行详情，任何异常。写 completed/needs_codex/needs_user 交接文件。

科学边界：旧 24 文件含 12 个开发文件和 12 个已评估留出文件，不能重新标为全新 final test。META-SPLIT 是后续独立任务，需先核查采集会话、重复及来源分组；至少源文件不能跨 split。当前 104 文件不足以宣称 13 类都有完全未参与选型的 final test。成功 `.restart` 裸块约 39.7 GiB 暂存，待本任务及正式 G 清单验收后再单独审查清理范围。
