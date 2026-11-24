# 固定频段截取、切片、筛选及同源划分结果

完成 489 块、173 个同源组；IQ 输出 E:/drone_lab_crop_20261107，complex64，无归一化；未训练、未装包、未提交。 实际 IQ 占用 141.598 GB；E 盘剩余 87.97 GB；总耗时约 2.51 小时（含冒烟、验收）。
输入路径实际为正斜杠，已兼容反斜杠；data model 由数字 code 映射，v3 背景 model=none；v3 fc 使用文件名。
纳入块数：data/2.4GHz/drone=61；data/5.8GHz/drone=39；v3/2.4GHz/background=105；v3/2.4GHz/drone=94；v3/5.8GHz/background=80；v3/5.8GHz/drone=110
纳入原始 MAT 文件数：data=3821；v3=760；逐块纳入清单与源路径见 blocks.csv。
排除 v3 文件 898 个（源表排除740、完整频段不覆盖158）；主要中心 MHz/文件数：2367.06/10；2500/80；2514.59/10；2589.19/10；2600/28；2663.78/10；5760/10；5787.5/10；5810/10；5881.56/10；其他扫频/会话710个；逐文件/频率明细见 excluded_files.csv、excluded_frequencies.csv。
data 只用 include=True 的 100 个有效 run（96 次录制）；原始无效/重复文件不纳入；008/F008 共8次录制目录省略带宽 M 字符，沿用任务 A 解析为 10M/20M/40M，逐录制见 metadata_corrections.csv。
连续性 07-10_lab：pass，两段区间 [2000,5000000)、[5000000,10000000)，最低首尾相关 0.99996749。
连续性 07-11_home：pass，两段区间 [2000,5000000)、[5000000,10000000)，最低首尾相关 0.99995045。
连续性 075302：pass，两段区间 [4000,5000000)、[5000000,10000000)，最低首尾相关 0.99999940。
已验证 250M 类按同 NN 拼接；扫频复核相接后同 NN 拼接；noise_adrv、缺配对文件单独成块；不同 NN 不拼接。
终端 FIR 2.4GHz：143 taps，通带波动 0.000523 dB，阻带衰减 89.91 dB；两端各裁 12288 点（0.2 ms）。
终端 FIR 5.8GHz：193 taps，通带波动 0.000587 dB，阻带衰减 89.96 dB；两端各裁 20000 点（0.2 ms）。
重采样使用显式 100 dB Kaiser FIR；所有来源共用对应频段的终端 FIR（filters.npz）；分段 50 ms、双侧 2 ms 重叠，避免内存溢出。
统一陷波（频率 MHz / 半宽 kHz）：2.4GHz 2440.00000000/100；2.4GHz 2450.00000000/100；2.4GHz 2426.88000000/60；5.8GHz 5800.00000000/100；5.8GHz 5795.50093750/60；完整来源见 notch_table.csv。 lab_preproc.notch 的 width 参数内部翻倍，因此调用传 half_width/2，确保最终半宽与陷波表一致。
1 ms cell：2.4G NFFT=512（120 kHz/bin），5.8G NFFT=1024（97.65625 kHz/bin），Hann、不重叠、不折叠；5.8G 每 cell 用 97 帧，末 672 点不入谱。
筛选按新 IQ 计算 recording 参考底噪、占用≥5% 子带，间隔≤2有效 bin 合并；固定陷波区作为环境 bin；v3b 五项阈值不变；背景全保留，仅标记不删除。

|L ms|总片段|keep 后|保留率|
|---|---:|---:|---:|
|10|22370|18723|83.70%|
|20|10955|9551|87.18%|
|30|7418|6574|88.62%|
|40|5435|4924|90.60%|
|50|4092|3728|91.10%|

|L（各列总数/keep）|split|data 2.4 无人机|data 5.8 无人机|v3 2.4 无人机|v3 2.4 背景|v3 5.8 无人机|v3 5.8 背景|
|---|---|---|---|---|---|---|---|
|10|train|6543/5364|4489/3451|360/330|520/520|450/407|435/435|
|10|val|2189/1670|1571/1498|216/201|180/180|225/206|135/135|
|10|test|2587/2094|1580/1410|270/228|180/180|305/279|135/135|
|20|train|3254/2832|2236/1806|160/152|234/234|200/165|194/194|
|20|val|1089/936|783/732|96/87|80/80|100/90|60/60|
|20|test|1287/1120|786/688|120/105|80/80|136/130|60/60|
|30|train|2169/1909|1489/1227|120/113|169/169|150/138|144/144|
|30|val|726/638|521/502|72/69|60/60|75/68|45/45|
|30|test|858/737|524/479|90/79|60/60|101/92|45/45|
|40|train|1611/1428|1109/944|80/75|117/117|100/81|97/97|
|40|val|539/501|389/382|48/47|40/40|50/43|30/30|
|40|test|637/581|390/373|60/52|40/40|68/63|30/30|
|50|train|1279/1152|886/748|40/38|52/52|50/48|47/47|
|50|val|429/396|310/306|24/23|20/20|25/25|15/15|
|50|test|507/470|310/295|30/26|20/20|33/32|15/15|

秒数与 keep 秒数见 counts.csv；逐 model × split 数量见 model_counts.csv；L × band × state 保留率见 retention.csv。
10ms 保留率：2.4GHz/standby=77.48%；5.8GHz/standby=77.25%；2.4GHz/flight=84.43%；5.8GHz/flight=88.93%；2.4GHz/unknown=94.96%；5.8GHz/unknown=94.78%
20ms 保留率：2.4GHz/standby=86.27%；5.8GHz/standby=79.91%；2.4GHz/flight=87.48%；5.8GHz/flight=89.43%；2.4GHz/unknown=95.84%；5.8GHz/unknown=93.20%
30ms 保留率：2.4GHz/standby=87.08%；5.8GHz/standby=83.82%；2.4GHz/flight=88.00%；5.8GHz/flight=90.29%；2.4GHz/unknown=96.32%；5.8GHz/unknown=95.00%
40ms 保留率：2.4GHz/standby=89.86%；5.8GHz/standby=87.51%；2.4GHz/flight=90.31%；5.8GHz/flight=92.35%；2.4GHz/unknown=96.36%；5.8GHz/unknown=91.73%
50ms 保留率：2.4GHz/standby=91.63%；5.8GHz/standby=87.64%；2.4GHz/flight=90.48%；5.8GHz/flight=91.43%；2.4GHz/unknown=96.24%；5.8GHz/unknown=98.38%
划分种子2026，同一 group 的所有 L 共用一个 split；优先保证 model 覆盖，按时长贪心。组数不足层/model：('data', '2.4GHz', 'drone')/atom se=2组；('data', '2.4GHz', 'drone')/avata=2组；('data', '2.4GHz', 'drone')/evo ii pro v3=1组；('data', '2.4GHz', 'drone')/evo lite 6k=1组；('data', '2.4GHz', 'drone')/mini4 pro=2组；('v3', '2.4GHz', 'drone')/evo2pro_v3=2组；('v3', '5.8GHz', 'drone')/evo2pro_v3=1组；('v3', '5.8GHz', 'drone')/fpv=1组；详情 group_split.csv。

|来源/频段/标签|train 组数/秒/占比|val 组数/秒/占比|test 组数/秒/占比|
|---|---|---|---|
|data/2.4GHz/drone|33/65.785/57.8%|11/21.996/19.3%|13/25.995/22.8%|
|data/5.8GHz/drone|23/45.021/58.7%|8/15.756/20.6%|8/15.871/20.7%|
|v3/2.4GHz/background|12/5.823/59.4%|4/1.992/20.3%|4/1.992/20.3%|
|v3/2.4GHz/drone|8/3.983/42.6%|5/2.390/25.5%|6/2.987/31.9%|
|v3/5.8GHz/background|10/4.829/61.8%|3/1.494/19.1%|3/1.494/19.1%|
|v3/5.8GHz/drone|10/4.979/45.9%|5/2.490/22.9%|7/3.385/31.2%|

底噪形状 2.4GHz：最大差 12.261 dB @ 2438.977500 MHz（data_drone_2400.0_2.4GHz / v3_background_2450.0_2.4GHz）；未自行修正，全部两两差值见 verification.json，图见 floor_shape_2.4GHz.png。
底噪形状 5.8GHz：最大差 2.551 dB @ 5830.004883 MHz（data_drone_5800.0_5.8GHz / v3_background_5800.0_5.8GHz）；未自行修正，全部两两差值见 verification.json，图见 floor_shape_5.8GHz.png。
验证状态：v3_contiguity=pass；filter_chain=pass；recompute=pass；notch=pass；floor_shape=fail；leakage_and_boundaries=pass；counts=pass；source_unchanged=pass；csv_leakage=pass；csv_boundaries=pass；screening_flags=pass
重算最大相对误差 7.41e-08；陷波残留最大 -21.886 dB，外侧功率变化最大 0.002112 dB。
异常：底噪形状差异超过1 dB，包含低功率 cell 仍保留的真实信号影响，不将其强行平坦化；其他失败项如有，以 verification.json 数值为准。

**已知残余风险**：data/ 的接收机只有无人机样本、没有背景样本，此来源与标签相关性无法由本次预处理消除。
