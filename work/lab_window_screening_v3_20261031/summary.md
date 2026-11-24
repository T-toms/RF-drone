# 任务 D：窗口筛选 v3
缓存筛选；仅画廊读取 MAT；未训练、未划分、未修改原始数据。
epoch 跨界窗归 epoch_start；无持续特征时按 50 ms 出现率回退；灰色仅表示无效 cell。
10ms/recording：保留 91.58% / mean_coverage 85.31% / full_coverage_rate 68.00%；2.4GHz:保留 92.79% / mean_coverage 83.95% / full_coverage_rate 65.56%；5.8GHz:保留 89.78% / mean_coverage 87.33% / full_coverage_rate 71.62%；flight:保留 93.18% / mean_coverage 85.21% / full_coverage_rate 67.51%；standby:保留 90.12% / mean_coverage 85.41% / full_coverage_rate 68.45%
10ms/epoch：保留 93.20% / mean_coverage 82.58% / full_coverage_rate 61.24%；2.4GHz:保留 94.67% / mean_coverage 81.65% / full_coverage_rate 59.02%；5.8GHz:保留 91.02% / mean_coverage 83.96% / full_coverage_rate 64.54%；flight:保留 94.79% / mean_coverage 82.16% / full_coverage_rate 61.56%；standby:保留 91.74% / mean_coverage 82.96% / full_coverage_rate 60.94%
15ms/recording：保留 91.04% / mean_coverage 90.68% / full_coverage_rate 78.70%；2.4GHz:保留 93.10% / mean_coverage 89.14% / full_coverage_rate 75.35%；5.8GHz:保留 87.98% / mean_coverage 92.98% / full_coverage_rate 83.68%；flight:保留 92.19% / mean_coverage 90.17% / full_coverage_rate 77.25%；standby:保留 89.99% / mean_coverage 91.16% / full_coverage_rate 80.03%
15ms/epoch：保留 94.00% / mean_coverage 89.22% / full_coverage_rate 74.47%；2.4GHz:保留 96.09% / mean_coverage 87.77% / full_coverage_rate 71.09%；5.8GHz:保留 90.89% / mean_coverage 91.38% / full_coverage_rate 79.48%；flight:保留 93.95% / mean_coverage 88.39% / full_coverage_rate 73.46%；standby:保留 94.03% / mean_coverage 89.98% / full_coverage_rate 75.39%
20ms/recording：保留 92.05% / mean_coverage 94.15% / full_coverage_rate 85.94%；2.4GHz:保留 93.25% / mean_coverage 92.87% / full_coverage_rate 83.37%；5.8GHz:保留 90.26% / mean_coverage 96.06% / full_coverage_rate 89.76%；flight:保留 93.25% / mean_coverage 93.31% / full_coverage_rate 83.67%；standby:保留 90.95% / mean_coverage 94.91% / full_coverage_rate 88.01%
20ms/epoch：保留 94.47% / mean_coverage 93.41% / full_coverage_rate 83.61%；2.4GHz:保留 95.99% / mean_coverage 92.02% / full_coverage_rate 80.73%；5.8GHz:保留 92.20% / mean_coverage 95.50% / full_coverage_rate 87.90%；flight:保留 94.48% / mean_coverage 92.31% / full_coverage_rate 81.32%；standby:保留 94.46% / mean_coverage 94.42% / full_coverage_rate 85.71%
50ms/recording：保留 92.86% / mean_coverage 99.47% / full_coverage_rate 98.55%；2.4GHz:保留 93.98% / mean_coverage 99.41% / full_coverage_rate 98.46%；5.8GHz:保留 91.18% / mean_coverage 99.56% / full_coverage_rate 98.68%；flight:保留 93.38% / mean_coverage 99.45% / full_coverage_rate 98.45%；standby:保留 92.39% / mean_coverage 99.48% / full_coverage_rate 98.64%
50ms/epoch：保留 95.13% / mean_coverage 99.34% / full_coverage_rate 98.23%；2.4GHz:保留 96.57% / mean_coverage 99.29% / full_coverage_rate 98.11%；5.8GHz:保留 92.96% / mean_coverage 99.41% / full_coverage_rate 98.42%；flight:保留 94.81% / mean_coverage 99.39% / full_coverage_rate 98.40%；standby:保留 95.41% / mean_coverage 99.28% / full_coverage_rate 98.08%
触发占比 10ms/recording：lack_feature=0.73%；primary_weak=3.41%；power_shift=4.37%；floor_raised=0.00%；interference=1.16%
触发占比 10ms/epoch：lack_feature=1.05%；primary_weak=2.55%；power_shift=2.87%；floor_raised=0.00%；interference=1.11%
最低 10ms：005_5.8GHz_10M-signal__2026-09-20-20-14-22 46.00%；主因 primary_weak
最低 10ms：004_5.8GHz_10M-signal__2026-09-20-19-59-29 47.64%；主因 primary_weak
最低 10ms：011_2.4GHz_10M-signal__2026-09-20-18-54-38 50.50%；主因 primary_weak
最低 10ms：F007_2.4GHz_10M-signal__2026-09-20-20-29-54 58.00%；主因 power_shift
最低 10ms：F005_2.4GHz_40M-signal__2026-09-20-20-09-11 59.50%；主因 power_shift
触发占比 20ms/recording：lack_feature=1.78%；primary_weak=2.44%；power_shift=4.47%；floor_raised=0.00%；interference=0.70%
触发占比 20ms/epoch：lack_feature=1.79%；primary_weak=1.24%；power_shift=2.62%；floor_raised=0.00%；interference=0.68%
最低 20ms：F008_5.8GHz_20M-signal__2026-09-20-20-47-06 30.00%；主因 primary_weak
最低 20ms：011_2.4GHz_10M-signal__2026-09-20-18-54-38 51.00%；主因 power_shift
最低 20ms：005_5.8GHz_10M-signal__2026-09-20-20-14-22 54.00%；主因 primary_weak
最低 20ms：F007_2.4GHz_10M-signal__2026-09-20-20-29-54 54.00%；主因 power_shift
最低 20ms：004_5.8GHz_10M-signal__2026-09-20-19-59-29 57.89%；主因 power_shift
rec_keep recording：{"10": 69, "15": 7, "none": 13, "20": 3, "50": 4}
rec_cover recording：{"10": 35, "20": 13, "50": 33, "15": 11, "none": 4}
recommended recording：{"10": 27, "20": 13, "50": 31, "15": 9, "none": 16}
rec_keep epoch：{"10": 74, "20": 3, "15": 5, "none": 9, "50": 5}
rec_cover epoch：{"10": 20, "20": 17, "50": 44, "15": 13, "none": 2}
recommended epoch：{"10": 18, "20": 13, "50": 42, "none": 11, "15": 12}
no_persistent 清单：见 no_persistent.csv（3 个）
对比 C 10ms/recording：49.23% → 91.58%（+42.35 百分点）
对比 C 10ms/epoch：45.90% → 93.20%（+47.30 百分点）
对比 C 15ms/recording：59.88% → 91.04%（+31.16 百分点）
对比 C 15ms/epoch：59.08% → 94.00%（+34.92 百分点）
对比 C 20ms/recording：65.84% → 92.05%（+26.21 百分点）
对比 C 20ms/epoch：68.47% → 94.47%（+26.00 百分点）
对比 C 50ms/recording：76.00% → 92.86%（+16.86 百分点）
对比 C 50ms/epoch：81.19% → 95.13%（+13.94 百分点）
环境区间（n_recordings为区间内bin最低覆盖数，all_bins_recordings为整段共同覆盖数）：{"2.4GHz": [{"lo_MHz": 2401.92, "hi_MHz": 2402.04, "n_recordings": 46, "all_bins_recordings": 46, "min_bin_recordings": 46, "total_recordings": 57}, {"lo_MHz": 2425.44, "hi_MHz": 2426.64, "n_recordings": 46, "all_bins_recordings": 44, "min_bin_recordings": 46, "total_recordings": 57}, {"lo_MHz": 2426.88, "hi_MHz": 2427.0, "n_recordings": 57, "all_bins_recordings": 57, "min_bin_recordings": 57, "total_recordings": 57}], "5.8GHz": [{"lo_MHz": 5777.08, "hi_MHz": 5786.5599999999995, "n_recordings": 32, "all_bins_recordings": 30, "min_bin_recordings": 32, "total_recordings": 39}, {"lo_MHz": 5794.72, "hi_MHz": 5798.5599999999995, "n_recordings": 32, "all_bins_recordings": 31, "min_bin_recordings": 32, "total_recordings": 39}, {"lo_MHz": 5798.92, "hi_MHz": 5799.88, "n_recordings": 32, "all_bins_recordings": 32, "min_bin_recordings": 32, "total_recordings": 39}, {"lo_MHz": 5800.24, "hi_MHz": 5800.84, "n_recordings": 32, "all_bins_recordings": 32, "min_bin_recordings": 32, "total_recordings": 39}, {"lo_MHz": 5815.24, "hi_MHz": 5820.04, "n_recordings": 32, "all_bins_recordings": 30, "min_bin_recordings": 32, "total_recordings": 39}]}
协调者修订：环境频点为同频段≥80%录制 occ_any≥0.05 的掩膜内bin；子带≥50% bin为环境才整体排除，其余剔除环境bin；非稠密持续特征见 non_dense_persistent.csv。
无持续特征时 coverage=1；主子带从非环境子带按50ms出现率回退，无非环境子带时主子带为none且主信号/功率标记为假。
间隙为合并主子带的有效 1ms 序列；所有 L/ref 共用 recording 主子带间隙，逐窗参照主子带完整记录在索引。
验收：verification.json 八项全部 pass；44996 窗与 B 索引一致，012 主带出现率100%，随机5窗独立重算一致，104张PNG均>30KB，6629个原始文件路径/大小/mtime逐项一致（完整元数据基线在画廊读取前记录）。
