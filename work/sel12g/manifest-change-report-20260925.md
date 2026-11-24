# SELECT12-G3 G 盘候选清单变更报告

生成时间（UTC）：2026-09-25T09:36:15.076298+00:00
候选 CSV：`C:\Users\Administrator\Desktop\数据集处理\work\sel12g\dataset_manifest_candidate_104_20260925.csv`（104 行，107450183727 字节）

本文件是候选方案，**没有**改写 `G:\DroneRFa数据集\dataset_manifest.csv`。

## 旧 24 行如何保留

- 七个旧列 `file_name,class_id,class_name,former_partition,unified_path,size_bytes,class_index` 逐行保留原 class/大小/former_partition 语义（已核对与冻结目标表一致）。
- `unified_path` 由旧 F 盘路径改写为实际 G 盘文件路径；旧路径原值保留在差异报告的 `legacy_unified_path` 字段。
- 旧 `development_original` 12 行 → `evaluation_role=legacy_development`；旧 `evaluated_holdout` 12 行 → `evaluation_role=previously_evaluated`；两者 `split_status=preserve_former_partition`。
- 这 12 个已评估留出文件已经参与过模型/n_fft 选型，**不能**重新标成全新 final test。

## 新增 80 行

- `former_partition` 为空、`split_status=unassigned`、`evaluation_role=unassigned`、`origin=selected_80`。
- 覆盖 10 个新机型 × 8 个 S 段；加上旧 3 类，共 13 类，每类 8 文件。
- 每行都带实到字节数（与冻结归档索引一致）和独立 SHA256 证据。

## 每类文件数

| class_id | class_name | 文件数 | 旧 24 | 新增 80 | 字节 |
| --- | --- | --- | --- | --- | --- |
| T0000 | Background | 8 | 8 | 0 | 7534708491 |
| T0001 | DJI Phantom 3 | 8 | 0 | 8 | 7827156925 |
| T0010 | DJI Phantom 4 Pro | 8 | 8 | 0 | 8524116981 |
| T0011 | DJI Matrice 200 | 8 | 0 | 8 | 8347792345 |
| T0100 | DJI Matrice 100 | 8 | 0 | 8 | 8111215401 |
| T0101 | DJI Air 2S | 8 | 8 | 0 | 8664257857 |
| T0110 | DJI Mini 3 Pro | 8 | 0 | 8 | 8330502610 |
| T0111 | DJI Inspire 2 | 8 | 0 | 8 | 8162201043 |
| T1000 | DJI Mavic Pro | 8 | 0 | 8 | 8144423680 |
| T1001 | DJI Mini 2 | 8 | 0 | 8 | 9203027965 |
| T1010 | DJI Mavic 3 | 8 | 0 | 8 | 7062910976 |
| T1011 | DJI Matrice 300 | 8 | 0 | 8 | 8793935143 |
| T1100 | DJI Phantom 4 Pro RTK | 8 | 0 | 8 | 8743934310 |

## 异常

- 无。

## 内容重复组（SHA256 相同，供后续 META-SPLIT 防泄漏核查）

- 未发现 104 个文件之间的 SHA256 重复。

## 检查项

- [PASS] audit_covers_all_targets: `{"audit_counts": {"mat_on_g": 104, "valid": 104, "invalid": 0, "unknown": 0, "target_total": 104, "target_present_on_g": 104, "target_missing_on_g": 0}, "targets": 104}`
- [PASS] audit_all_valid: `{"invalid": 0, "problems": []}`
- [PASS] hash_report_covers_all_targets: `{"hash_count": 104, "targets": 104}`
- [PASS] legacy_manifest_row_count: `{"rows": 24}`
- [PASS] legacy_file_names_unique: `{"unique": 24}`
- [PASS] legacy_rows_match_frozen_targets: `[]`
- [PASS] legacy_partition_roles_known: `{"development_original": 12, "evaluated_holdout": 12}`
- [PASS] candidate_row_count: `{"rows": 104}`
- [PASS] new_row_count: `{"new_rows": 80}`
- [PASS] file_names_unique: `{"duplicates": []}`
- [PASS] classes_and_files_per_class: `{"classes": 13, "counts": {"T0000": 8, "T0001": 8, "T0010": 8, "T0011": 8, "T0100": 8, "T0101": 8, "T0110": 8, "T0111": 8, "T1000": 8, "T1001": 8, "T1010": 8, "T1011": 8, "T1100": 8}}`
- [PASS] all_paths_point_to_g_original: `{"bad_paths": []}`
- [PASS] no_forbidden_evaluation_role: `{"forbidden_roles": []}`
- [PASS] new_rows_are_unassigned: `{"new_rows": 80}`
- [PASS] new_targets_have_no_former_partition: `[]`
- [PASS] hash_evidence_complete: `{"missing": []}`
- [PASS] no_outstanding_issues: `[]`
- [PASS] download_state_matches_frozen_totals: `{"path": "G:\\DroneRFa数据集\\download_records_12models_20260922\\select12g-state.json", "state_entries": 42, "state_entries_completed": 42, "new_targets": 80, "new_targets_covered_by_this_state_file": 42, "covered_targets_all_completed": true, "new_targets_landed_before_this_state_round": 38, "frozen_totals_already_landed_on_g": 38, "frozen_totals_still_missing_on_g": 42, "historical_errors": 2}`
