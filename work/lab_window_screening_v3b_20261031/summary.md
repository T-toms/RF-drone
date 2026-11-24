# 任务 D：窗口筛选 v3b
缓存筛选；仅画廊读取 MAT；未训练、未划分、未修改原始数据。
epoch 跨界窗归 epoch_start；无持续特征时按 50 ms 出现率回退；灰色仅表示无效 cell。
10ms/recording：保留 90.12% / mean_coverage 82.61% / full_coverage_rate 60.79%；2.4GHz:保留 92.46% / mean_coverage 79.90% / full_coverage_rate 57.66%；5.8GHz:保留 86.63% / mean_coverage 86.64% / full_coverage_rate 65.44%；flight:保留 93.18% / mean_coverage 82.99% / full_coverage_rate 61.63%；standby:保留 87.31% / mean_coverage 82.27% / full_coverage_rate 60.02%
10ms/epoch：保留 93.42% / mean_coverage 81.00% / full_coverage_rate 54.65%；2.4GHz:保留 94.31% / mean_coverage 78.75% / full_coverage_rate 52.77%；5.8GHz:保留 92.10% / mean_coverage 84.33% / full_coverage_rate 57.46%；flight:保留 94.98% / mean_coverage 81.51% / full_coverage_rate 56.83%；standby:保留 92.00% / mean_coverage 80.52% / full_coverage_rate 52.66%
15ms/recording：保留 90.89% / mean_coverage 88.89% / full_coverage_rate 73.68%；2.4GHz:保留 92.94% / mean_coverage 86.40% / full_coverage_rate 69.10%；5.8GHz:保留 87.84% / mean_coverage 92.59% / full_coverage_rate 80.50%；flight:保留 92.90% / mean_coverage 88.59% / full_coverage_rate 72.99%；standby:保留 89.05% / mean_coverage 89.17% / full_coverage_rate 74.32%
15ms/epoch：保留 94.51% / mean_coverage 88.01% / full_coverage_rate 69.76%；2.4GHz:保留 95.92% / mean_coverage 85.78% / full_coverage_rate 65.94%；5.8GHz:保留 92.42% / mean_coverage 91.32% / full_coverage_rate 75.43%；flight:保留 95.36% / mean_coverage 87.79% / full_coverage_rate 69.79%；standby:保留 93.73% / mean_coverage 88.21% / full_coverage_rate 69.73%
20ms/recording：保留 90.92% / mean_coverage 92.93% / full_coverage_rate 82.73%；2.4GHz:保留 92.60% / mean_coverage 91.12% / full_coverage_rate 79.06%；5.8GHz:保留 88.43% / mean_coverage 95.63% / full_coverage_rate 88.19%；flight:保留 94.08% / mean_coverage 92.14% / full_coverage_rate 80.57%；standby:保留 88.03% / mean_coverage 93.66% / full_coverage_rate 84.71%
20ms/epoch：保留 94.87% / mean_coverage 92.59% / full_coverage_rate 80.52%；2.4GHz:保留 95.57% / mean_coverage 90.86% / full_coverage_rate 77.10%；5.8GHz:保留 93.82% / mean_coverage 95.17% / full_coverage_rate 85.62%；flight:保留 95.07% / mean_coverage 91.82% / full_coverage_rate 78.66%；standby:保留 94.68% / mean_coverage 93.30% / full_coverage_rate 82.23%
50ms/recording：保留 92.12% / mean_coverage 99.35% / full_coverage_rate 98.29%；2.4GHz:保留 93.72% / mean_coverage 99.21% / full_coverage_rate 98.07%；5.8GHz:保留 89.74% / mean_coverage 99.56% / full_coverage_rate 98.62%；flight:保留 94.81% / mean_coverage 99.32% / full_coverage_rate 98.18%；standby:保留 89.67% / mean_coverage 99.38% / full_coverage_rate 98.39%
50ms/epoch：保留 96.02% / mean_coverage 99.39% / full_coverage_rate 98.21%；2.4GHz:保留 96.49% / mean_coverage 99.28% / full_coverage_rate 98.07%；5.8GHz:保留 95.33% / mean_coverage 99.55% / full_coverage_rate 98.42%；flight:保留 96.19% / mean_coverage 99.48% / full_coverage_rate 98.40%；standby:保留 95.87% / mean_coverage 99.31% / full_coverage_rate 98.03%
触发占比 10ms/recording：lack_feature=0.89%；primary_weak=1.95%；power_shift=6.92%；floor_raised=0.00%；interference=1.20%
触发占比 10ms/epoch：lack_feature=1.29%；primary_weak=1.49%；power_shift=3.23%；floor_raised=0.00%；interference=1.15%
最低 10ms：005_5.8GHz_10M-signal__2026-09-20-20-14-22 15.50%；主因 power_shift
最低 10ms：005_5.8GHz_40M-signal__2026-09-20-20-12-48 38.50%；主因 power_shift
最低 10ms：011_2.4GHz_10M-signal__2026-09-20-18-54-38 50.50%；主因 primary_weak
最低 10ms：003_5.8GHz_20M-signal__2026-09-20-20-17-17 57.89%；主因 power_shift
最低 10ms：F007_2.4GHz_10M-signal__2026-09-20-20-29-54 58.00%；主因 power_shift
触发占比 20ms/recording：lack_feature=2.12%；primary_weak=0.97%；power_shift=6.34%；floor_raised=0.00%；interference=0.70%
触发占比 20ms/epoch：lack_feature=2.11%；primary_weak=0.49%；power_shift=2.35%；floor_raised=0.00%；interference=0.68%
最低 20ms：005_5.8GHz_10M-signal__2026-09-20-20-14-22 13.00%；主因 power_shift
最低 20ms：005_5.8GHz_40M-signal__2026-09-20-20-12-48 47.00%；主因 power_shift
最低 20ms：011_2.4GHz_10M-signal__2026-09-20-18-54-38 51.00%；主因 power_shift
最低 20ms：F007_2.4GHz_10M-signal__2026-09-20-20-29-54 54.00%；主因 power_shift
最低 20ms：F005_2.4GHz_40M-signal__2026-09-20-20-09-11 58.00%；主因 power_shift
rec_keep recording：{"10": 68, "15": 4, "50": 5, "20": 5, "none": 14}
rec_cover recording：{"50": 45, "20": 12, "15": 11, "none": 4, "10": 24}
recommended recording：{"50": 42, "20": 9, "15": 9, "none": 16, "10": 20}
rec_keep epoch：{"10": 72, "20": 6, "15": 5, "50": 4, "none": 9}
rec_cover epoch：{"50": 57, "20": 15, "15": 8, "10": 14, "none": 2}
recommended epoch：{"50": 52, "20": 14, "none": 11, "15": 9, "10": 10}
no_persistent 清单：无
对比 C 10ms/recording：49.23% → 90.12%（+40.88 百分点）
对比 C 10ms/epoch：45.90% → 93.42%（+47.53 百分点）
对比 C 15ms/recording：59.88% → 90.89%（+31.01 百分点）
对比 C 15ms/epoch：59.08% → 94.51%（+35.43 百分点）
对比 C 20ms/recording：65.84% → 90.92%（+25.08 百分点）
对比 C 20ms/epoch：68.47% → 94.87%（+26.40 百分点）
对比 C 50ms/recording：76.00% → 92.12%（+16.12 百分点）
对比 C 50ms/epoch：81.19% → 96.02%（+14.83 百分点）
环境区间（n_recordings为区间内bin最低覆盖数，all_bins_recordings为整段共同覆盖数）：{"2.4GHz": [{"lo_MHz": 2426.88, "hi_MHz": 2427.0, "n_recordings": 57, "all_bins_recordings": 57, "min_bin_recordings": 57, "total_recordings": 57}], "5.8GHz": [{"lo_MHz": 5799.76, "hi_MHz": 5799.88, "n_recordings": 38, "all_bins_recordings": 38, "min_bin_recordings": 38, "total_recordings": 39}]}
协调者修订：环境频点为同频段≥95%录制 occ_any≥0.05 的掩膜内bin；子带≥50% bin为环境才整体排除，其余剔除环境bin；非稠密持续特征见 non_dense_persistent.csv。
无持续特征时 coverage=1；主子带从非环境子带按50ms出现率回退，无非环境子带时主子带为none且主信号/功率标记为假。
间隙为合并主子带的有效 1ms 序列；所有 L/ref 共用 recording 主子带间隙，逐窗参照主子带完整记录在索引。
v3 -> v3b 10ms/recording: retention 91.58% -> 90.12%; full_coverage_rate 68.00% -> 60.79%.
v3 -> v3b 15ms/recording: retention 91.04% -> 90.89%; full_coverage_rate 78.70% -> 73.68%.
v3 -> v3b 20ms/recording: retention 92.05% -> 90.92%; full_coverage_rate 85.94% -> 82.73%.
v3 -> v3b 50ms/recording: retention 92.86% -> 92.12%; full_coverage_rate 98.55% -> 98.29%.
v3 -> v3b recommended: recording {"10": 27, "20": 13, "50": 31, "15": 9, "none": 16} -> {"50": 42, "20": 9, "15": 9, "none": 16, "10": 20}; epoch {"10": 18, "20": 13, "50": 42, "none": 11, "15": 12} -> {"50": 52, "20": 14, "none": 11, "15": 9, "10": 10}
D2 environment: 2.4GHz only 2426.88-2427.00 MHz (57/57); 5.8GHz 5799.76-5799.88 MHz (38/39); threshold unchanged at 0.95.
