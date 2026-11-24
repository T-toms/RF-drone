# 任务 C：窗口筛选 v2
缓存筛选；仅画廊读取 MAT；未训练、未划分、未修改原始数据。
epoch 跨界窗归 epoch_start；无持续特征时按 50 ms 出现率回退；灰色仅表示无效 cell。
10ms/recording：保留 49.23% / 覆盖 63.22%；2.4GHz:保留 47.31% / 覆盖 61.85%；5.8GHz:保留 52.09% / 覆盖 65.27%；flight:保留 51.57% / 覆盖 62.81%；standby:保留 47.09% / 覆盖 63.60%
10ms/epoch：保留 45.90% / 覆盖 57.09%；2.4GHz:保留 45.39% / 覆盖 57.17%；5.8GHz:保留 46.66% / 覆盖 56.98%；flight:保留 48.79% / 覆盖 57.83%；standby:保留 43.24% / 覆盖 56.41%
15ms/recording：保留 59.88% / 覆盖 75.16%；2.4GHz:保留 58.26% / 覆盖 71.86%；5.8GHz:保留 62.29% / 覆盖 80.07%；flight:保留 61.66% / 覆盖 73.22%；standby:保留 58.25% / 覆盖 76.94%
15ms/epoch：保留 59.08% / 覆盖 71.18%；2.4GHz:保留 57.44% / 覆盖 68.83%；5.8GHz:保留 61.51% / 覆盖 74.67%；flight:保留 61.27% / 覆盖 70.23%；standby:保留 57.07% / 覆盖 72.05%
20ms/recording：保留 65.84% / 覆盖 83.53%；2.4GHz:保留 64.33% / 覆盖 80.73%；5.8GHz:保留 68.08% / 覆盖 87.69%；flight:保留 68.05% / 覆盖 80.48%；standby:保留 63.81% / 覆盖 86.32%
20ms/epoch：保留 68.47% / 覆盖 81.32%；2.4GHz:保留 67.60% / 覆盖 78.92%；5.8GHz:保留 69.76% / 覆盖 84.89%；flight:保留 69.50% / 覆盖 78.88%；standby:保留 67.52% / 覆盖 83.56%
50ms/recording：保留 76.00% / 覆盖 98.23%；2.4GHz:保留 77.20% / 覆盖 98.07%；5.8GHz:保留 74.21% / 覆盖 98.49%；flight:保留 78.31% / 覆盖 98.01%；standby:保留 73.89% / 覆盖 98.44%
50ms/epoch：保留 81.19% / 覆盖 98.21%；2.4GHz:保留 84.27% / 覆盖 98.15%；5.8GHz:保留 76.58% / 覆盖 98.29%；flight:保留 83.72% / 覆盖 98.40%；standby:保留 78.88% / 覆盖 98.03%
触发占比 10ms/recording：lack_feature=36.78%；primary_weak=1.95%；power_shift=20.32%；floor_raised=0.00%；interference=1.20%
触发占比 10ms/epoch：lack_feature=42.91%；primary_weak=1.49%；power_shift=17.40%；floor_raised=0.00%；interference=1.15%
最低 10ms：012_2.4GHz_10M-signal__2026-09-20-18-58-55 0.00%；主因 lack_feature
最低 10ms：F012_2.4GHz_10M-signal__2026-09-20-18-59-30 0.00%；主因 lack_feature
最低 10ms：011_2.4GHz_10M-signal__2026-09-20-18-54-38 1.00%；主因 lack_feature
最低 10ms：F003_2.4GHz_10M-signal__2026-09-20-19-48-20 1.50%；主因 lack_feature
最低 10ms：F007_5.8GHz_10M-signal__2026-09-20-20-32-08 2.06%；主因 lack_feature
触发占比 20ms/recording：lack_feature=16.47%；primary_weak=0.97%；power_shift=21.18%；floor_raised=0.00%；interference=0.70%
触发占比 20ms/epoch：lack_feature=18.68%；primary_weak=0.49%；power_shift=15.93%；floor_raised=0.00%；interference=0.68%
最低 20ms：011_2.4GHz_10M-signal__2026-09-20-18-54-38 6.00%；主因 lack_feature
最低 20ms：012_2.4GHz_10M-signal__2026-09-20-18-58-55 7.00%；主因 lack_feature
最低 20ms：F012_2.4GHz_10M-signal__2026-09-20-18-59-30 9.00%；主因 lack_feature
最低 20ms：F007_5.8GHz_10M-signal__2026-09-20-20-32-08 21.65%；主因 lack_feature
最低 20ms：F013_2.4GHz_10M-signal__2026-09-20-19-03-17 22.00%；主因 power_shift
推荐 recording：{"none": 58, "50": 17, "15": 2, "10": 16, "20": 3}
推荐 epoch：{"50": 25, "none": 53, "20": 5, "10": 9, "15": 4}
no_persistent 清单：无
对比 B 10ms/recording：80.19% → 49.23%（-30.96 百分点）
对比 B 10ms/epoch：77.51% → 45.90%（-31.61 百分点）
对比 B 15ms/recording：77.22% → 59.88%（-17.34 百分点）
对比 B 15ms/epoch：75.35% → 59.08%（-16.27 百分点）
对比 B 20ms/recording：80.02% → 65.84%（-14.18 百分点）
对比 B 20ms/epoch：79.24% → 68.47%（-10.77 百分点）
对比 B 50ms/recording：78.42% → 76.00%（-2.42 百分点）
对比 B 50ms/epoch：81.14% → 81.19%（+0.05 百分点）
间隙为合并主子带的有效 1ms 序列；所有 L/ref 共用 recording 主子带间隙，逐窗参照主子带完整记录在索引。

验证第 5 项 withdrawn（coordinator expectation withdrawn）：指定窗 dF=-0.12343788146972656 dB；协调者确认画廊独立色标造成误判并撤回预期，3 dB 阈值未变，其余验证全部 pass。
