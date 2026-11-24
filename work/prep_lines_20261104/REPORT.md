# 任务 E：截频带前谱线与同源核查
已完成 189/189 个录制/会话；恢复原有 127 个断点，仅新增 62 个，原断点 SHA256 全部不变=True。
未训练、未生成样本、未保存切片、未划分；G:/drone 只读；未调用 DSH、未发送 heartbeat。
## 谱线测量与结论
65536 点 Hann、50% 重叠；局部±1MHz中位数突出≥10dB、−10dB宽≤50kHz、组内±20kHz合并；共2700条合并候选。
均匀抽5个有效文件，每会话≥200ms（v3两文件约100ms不足）；10ms整块统计出现率、漂移与幅度波动。
各组完整频率/中心频点/出现率见 lines_inventory.csv；group_summary.csv 给出分组清单，lines_by_recording.csv 给出逐录制重叠。
逐录制候选中155/3157条出现率≥0.9；稳定仅表示单音候选，无法由IQ确定硬件本振归因。
## 2026-07-11_home 的100–185条细线专项核查
日期前缀2026-07-11_home对应原始会话20260710-11xxxx，不能按文件名前缀日期误选断点。
长时谱局部峰判据不足以证明本振泄露：局部中位数会被宽带包络凹口拉低；突发谱峰也可满足50kHz宽度。
20260710-114317_ambient：165候选，稳定1，10ms出现率中位数0.0476，宽度中位数27.47kHz。
20260710-114353_ambient：119候选，稳定0，10ms出现率中位数0.0476，宽度中位数28.99kHz。
20260710-114525_ambient：112候选，稳定1，10ms出现率中位数0.0476，宽度中位数24.41kHz。
20260710-114636_ambient：111候选，稳定0，10ms出现率中位数0.0476，宽度中位数24.41kHz。
20260710-114713_ambient：175候选，稳定2，10ms出现率中位数0.0476，宽度中位数28.99kHz。
20260710-114757_ambient：185候选，稳定1，10ms出现率中位数0.0476，宽度中位数27.47kHz。
这些会话的百余峰主体不符合稳定本振/连续单音，属于瞬态窄峰误归为泄露候选；与突发宽带/调制信号一致，不能仅凭PSD证明具体为WiFi。
修正：保留原始候选测量不删数；lines_classified.csv增加出现率≥0.9稳定性门槛，自动消除还要求至少3个验证块全部达标；其余crop_plan标review_required。
低出现率也可能是间歇单音，故保留unconfirmed而非断言所有峰均为WiFi；不应对背景百余候选批量陷波。
## 消除方法及验证
lab_preproc.py：notch为复下变频后6阶Butter高通、双向零相位，半宽max(2×线宽,60kHz)；cancel为10ms块FFT插值、频率优化、复幅度最小二乘；零频另测均值相减。
验证19407项，达标16431；完整失败列表在cancel_validation.csv（residual≤3dB且|off_line_delta|≤0.1dB）。
稳定但未全达标的线已试1ms块：{'lines_retried': 101, 'lines_improved': 3, 'extra_cases': 303}；最终选择见cancel_final_parameters.csv；仍失败项禁止自动对消。
续跑异常：refine_validation.py对无效清单行的空start_sample转int失败；新增run_refine_resume.py兼容入口，原脚本保留不改，仅有效文件进入验证。
每录制每线抽3个10ms块；重叠/不重叠类别按实际存在情况验证；首3块可能无瞬态线，低残留不等于成功消除瞬态信号。
重叠判据以线外150–300kHz邻域占用推断底层信号，存在边界不确定性；背景overlaps_signal=False只代表没有无人机标注，不代表没有其他无线电信号。
## 截取规划
固定输出61.44MS/s，通带±28MHz，过渡到±30.72MHz；data为1/2、v3为384/625重采样；合成±20/35MHz测试见crop_synthetic_tests.json。
crop_plan.csv给出每录制中心、占用比例、每候选处理决策；过渡带内不能假定已被带通完全消除。
中心与占用比例沿用原候选掩膜规划；有瞬态候选的无人机中心标center_status=review_required_candidate_mask，F前需复核，未为此重算已完成测量。
占用比例<0.9录制：002_5.8GHz_10M-signal__2026-09-20-19-40-26, 002_5.8GHz_20M-signal__2026-09-20-19-39-53, 004_5.8GHz_20M-signal__2026-09-20-20-00-09, 007_5.8GHz_20M-signal__2026-09-20-20-34-08, 008_5.8GHz_20M-signal__2026-09-20-20-45-25, F002_5.8GHz_20M-signal__2026-09-20-19-41-49, F002_5.8GHz_40M-signal__2026-09-20-19-42-33, F005_5.8GHz_40M-signal__2026-09-20-20-12-14, F006_5.8GHz_40M-signal__2026-09-20-20-21-39, 2026-07-09_urban_20260709-033602_evo2pro_v3, 20260709-031439_matrice4, 20260709-023828_mini3pro, 20260709-024423_mini3pro, 20260709-025842_mini3pro, 20260709-030107_mini3pro, 20260709-025325_mini5pro, 20260709-025658_mini5pro
背景候选数、可用范围、覆盖宽度见background_sessions.csv；中心列表与56MHz绝对窗见crop_plan.csv。
no_valid_center：20260709-044311_ambient, 20260710-085616_ambient, 20260710-085658_ambient, 20260710-114832_ambient, 20260710-114921_ambient, 20260710-115003_ambient, 20260710-115049_ambient, 20260710-115124_ambient, 20260710-115201_ambient, 20260710-115237_ambient, 20260710-115318_ambient, 20260715-075302_ambient@2367.06, 20260715-075302_ambient@5760.0, 20260715-075302_ambient@5881.56
## 同源分组与验收
source_groups.csv共5494行、266组；duplicate的97行单列source_duplicates_excluded.csv，兼顾逐文件排除记录与验收非重复行数。
data同录制所有epoch/seg同组；v3同raw session及_s0001/_s0002同组，扫频按中心频率细分；raw不另入表。
任务F必须以group_id为最小单位划分：同组所有片段（含重叠片段）必须进入同一个训练/验证/测试集合。
raw根目录20260709-030546_ambient、20260709-040756_mavic2与urban内重复；clean_export为空（任务给定）。
扫频对应：4/5重采样，100MS/s坐标偏移4000点，相关0.9999999952，通过=True。
verification.json：{"source_rows": true, "pairs": true, "crop": true, "persistent_coverage": true, "scan": true, "raw_unchanged": true, "png": true, "minimum_duration": true, "resume": true, "review_gate": true}
所有测量/消除失败按数值保留，任务E完成不表示所有线已满足自动清除条件；任务F需遵守review_required门禁。
