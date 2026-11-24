# 做法 B 阶段 1：特征与来源检查
实现检查：双实现 21 片段×19，最大绝对误差 2.21e-15，最大相对误差 8.87e-13，全部满足阈值；行对齐随机核对 180 行，通过。
5.8G 无 bgA，双实现该组不适用；共 21 个真实片段。Hann 采用对称 np.hanning；突发数仅计相邻 0→1，不把片段起点计为上升沿。
合成 2.4GHz（各N=1228800）：white shape=1.128378,c42n=-0.000369,c20=0.000660,flat=0.999618,duty=0.000；tone shape=1.000000,c42n=-0.693147,c20=0.000000,flat=0.000000,duty=0.000；burst shape=1.964925,c42n=2.766146,c20=0.000844,flat=0.996621,duty=0.100。
合成 5.8GHz（各N=2000000）：white shape=1.128244,c42n=-0.001161,c20=0.001816,flat=0.999734,duty=0.000；tone shape=1.000000,c42n=-0.693147,c20=0.000000,flat=0.000000,duty=0.000；burst shape=1.963878,c42n=2.763466,c20=0.001554,flat=0.997846,duty=0.100。
缓存：E:/fusionB_features_20261110/；每份原始 meta 行序完整保留，包括 bgA 全部5630行、两频段 rejected；列名 feature_names.json。 汇总图：[图15](图15_手工特征的来源与标签区分力.png)。
NaN/Inf（计数按特征与 dataset,label 见 nonfinite_counts.csv；缓存保留NaN，分析仅以相应训练集特征中位数填补）： 加噪缓存逐SNR分组计数见 noise_nonfinite_counts.csv，NaN合计0、Inf合计0。
- data/drone：NaN 全部19列均0；Inf=0。
- dataA_mirror/background：NaN 全部19列均0；Inf=0。
- v3/background：NaN 全部19列均0；Inf=0。
- v3/drone：NaN 全部19列均0；Inf=0。
来源单特征 AUC：括号为 n0/n1、方向；a组0=bgA、组1=v3实验室背景，b组0=data、组1=v3；方向均为填补后数值。
|特征|a 全split / 标记|a test|a′ 全背景|b 同型号（pooled及带宽分层）|
|---|---|---|---|---|
|shape_factor|0.996(5138/260,组1大)/接收机指纹|1.000(1287/60,组1大)|0.995(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.704(518/79,组0大); 2.4GHz:mini5pro:10M=0.938(164/20,组0大); 2.4GHz:mini5pro:20M=0.559(175/20,组1大); 2.4GHz:mini5pro:40M=0.726(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=1.000(477/68,组0大); 5.8GHz:mini5pro:10M=1.000(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=1.000(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=1.000(38/36,组1大)|
|log_kurt_a|0.991(5138/260,组1大)/接收机指纹|0.997(1287/60,组1大)|0.989(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.607(518/79,组1大); 2.4GHz:mini5pro:10M=0.613(164/20,组0大); 2.4GHz:mini5pro:20M=0.514(175/20,组1大); 2.4GHz:mini5pro:40M=0.955(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.515(477/68,组0大); 5.8GHz:mini5pro:10M=0.907(166/14,组1大); 5.8GHz:mini5pro:20M=0.584(155/15,组0大); 5.8GHz:mini5pro:40M=0.994(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=1.000(38/36,组1大)|
|papr999_db|0.995(5138/260,组1大)/接收机指纹|1.000(1287/60,组1大)|0.994(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.510(518/79,组1大); 2.4GHz:mini5pro:10M=0.662(164/20,组0大); 2.4GHz:mini5pro:20M=0.595(175/20,组0大); 2.4GHz:mini5pro:40M=0.777(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.803(477/68,组0大); 5.8GHz:mini5pro:10M=0.528(166/14,组1大); 5.8GHz:mini5pro:20M=0.994(155/15,组0大); 5.8GHz:mini5pro:40M=0.999(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=1.000(38/36,组1大)|
|c42n|0.996(5138/260,组1大)/接收机指纹|0.999(1287/60,组1大)|0.994(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.594(518/79,组1大); 2.4GHz:mini5pro:10M=0.690(164/20,组0大); 2.4GHz:mini5pro:20M=0.530(175/20,组1大); 2.4GHz:mini5pro:40M=0.937(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.952(477/68,组0大); 5.8GHz:mini5pro:10M=1.000(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=1.000(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=1.000(38/36,组1大)|
|c40n_abs|0.798(5138/260,组1大)/可疑|0.711(1287/60,组1大)|0.823(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.632(518/79,组0大); 2.4GHz:mini5pro:10M=0.718(164/20,组0大); 2.4GHz:mini5pro:20M=0.706(175/20,组0大); 2.4GHz:mini5pro:40M=0.559(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.568(477/68,组0大); 5.8GHz:mini5pro:10M=0.677(166/14,组1大); 5.8GHz:mini5pro:20M=0.798(155/15,组0大); 5.8GHz:mini5pro:40M=0.918(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.979(38/36,组1大)|
|c41n_abs|0.855(5138/260,组1大)/可疑|0.801(1287/60,组1大)|0.859(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.543(518/79,组1大); 2.4GHz:mini5pro:10M=0.666(164/20,组1大); 2.4GHz:mini5pro:20M=0.645(175/20,组1大); 2.4GHz:mini5pro:40M=0.527(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.851(477/68,组0大); 5.8GHz:mini5pro:10M=0.994(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=0.997(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.999(38/36,组1大)|
|c20n_abs|0.717(5138/260,组0大)/未见来源线索|0.756(1287/60,组0大)|0.659(5138/394,组0大)|2.4GHz:mini5pro:pooled=0.554(518/79,组1大); 2.4GHz:mini5pro:10M=0.777(164/20,组1大); 2.4GHz:mini5pro:20M=0.661(175/20,组1大); 2.4GHz:mini5pro:40M=0.815(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.732(477/68,组1大); 5.8GHz:mini5pro:10M=0.681(166/14,组1大); 5.8GHz:mini5pro:20M=0.526(155/15,组0大); 5.8GHz:mini5pro:40M=1.000(156/20,组1大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.975(38/36,组1大)|
|log_gamma_max|1.000(5138/260,组1大)/接收机指纹|1.000(1287/60,组1大)|0.999(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.843(518/79,组0大); 2.4GHz:mini5pro:10M=0.858(164/20,组0大); 2.4GHz:mini5pro:20M=0.826(175/20,组0大); 2.4GHz:mini5pro:40M=0.921(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.999(477/68,组0大); 5.8GHz:mini5pro:10M=1.000(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=1.000(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=1.000(38/36,组1大)|
|sigma_f|0.935(5138/260,组0大)/接收机指纹|0.941(1287/60,组0大)|0.923(5138/394,组0大)|2.4GHz:mini5pro:pooled=0.939(518/79,组1大); 2.4GHz:mini5pro:10M=0.990(164/20,组1大); 2.4GHz:mini5pro:20M=0.969(175/20,组1大); 2.4GHz:mini5pro:40M=0.742(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.685(477/68,组1大); 5.8GHz:mini5pro:10M=0.900(166/14,组1大); 5.8GHz:mini5pro:20M=0.902(155/15,组1大); 5.8GHz:mini5pro:40M=0.546(156/20,组1大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.998(38/36,组1大)|
|spec_flatness|1.000(5138/260,组0大)/接收机指纹|1.000(1287/60,组0大)|0.999(5138/394,组0大)|2.4GHz:mini5pro:pooled=0.902(518/79,组1大); 2.4GHz:mini5pro:10M=0.992(164/20,组1大); 2.4GHz:mini5pro:20M=0.998(175/20,组1大); 2.4GHz:mini5pro:40M=0.911(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=1.000(477/68,组1大); 5.8GHz:mini5pro:10M=1.000(166/14,组1大); 5.8GHz:mini5pro:20M=1.000(155/15,组1大); 5.8GHz:mini5pro:40M=1.000(156/20,组1大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=1.000(38/36,组1大)|
|occ_frac|0.885(5138/260,组1大)/可疑|0.858(1287/60,组1大)|0.879(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.652(518/79,组0大); 2.4GHz:mini5pro:10M=0.783(164/20,组0大); 2.4GHz:mini5pro:20M=0.929(175/20,组0大); 2.4GHz:mini5pro:40M=0.781(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.994(477/68,组0大); 5.8GHz:mini5pro:10M=1.000(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=0.997(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.701(38/36,组0大)|
|duty|0.890(5138/260,组1大)/可疑|0.867(1287/60,组1大)|0.901(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.785(518/79,组0大); 2.4GHz:mini5pro:10M=0.538(164/20,组1大); 2.4GHz:mini5pro:20M=0.937(175/20,组0大); 2.4GHz:mini5pro:40M=0.890(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=1.000(477/68,组0大); 5.8GHz:mini5pro:10M=1.000(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=1.000(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=1.000(38/36,组0大)|
|log1p_n_bursts|0.885(5138/260,组1大)/可疑|0.850(1287/60,组1大)|0.897(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.744(518/79,组1大); 2.4GHz:mini5pro:10M=0.793(164/20,组1大); 2.4GHz:mini5pro:20M=0.708(175/20,组1大); 2.4GHz:mini5pro:40M=0.723(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.613(477/68,组0大); 5.8GHz:mini5pro:10M=0.933(166/14,组0大); 5.8GHz:mini5pro:20M=0.770(155/15,组0大); 5.8GHz:mini5pro:40M=0.756(156/20,组1大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.526(38/36,组0大)|
|ctl_abs_power_db|1.000(5138/260,组1大)/接收机指纹|1.000(1287/60,组1大)|0.992(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.803(518/79,组0大); 2.4GHz:mini5pro:10M=0.819(164/20,组0大); 2.4GHz:mini5pro:20M=0.739(175/20,组0大); 2.4GHz:mini5pro:40M=1.000(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=1.000(477/68,组0大); 5.8GHz:mini5pro:10M=1.000(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=1.000(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=1.000(38/36,组0大)|
|ctl_floor_db|1.000(5138/260,组1大)/接收机指纹|1.000(1287/60,组1大)|0.990(5138/394,组1大)|2.4GHz:mini5pro:pooled=1.000(518/79,组1大); 2.4GHz:mini5pro:10M=1.000(164/20,组1大); 2.4GHz:mini5pro:20M=1.000(175/20,组1大); 2.4GHz:mini5pro:40M=1.000(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=1.000(477/68,组0大); 5.8GHz:mini5pro:10M=1.000(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=1.000(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.632(38/36,组0大)|
|ctl_dyn_range_db|1.000(5138/260,组1大)/接收机指纹|1.000(1287/60,组1大)|0.997(5138/394,组1大)|2.4GHz:mini5pro:pooled=0.813(518/79,组0大); 2.4GHz:mini5pro:10M=0.989(164/20,组0大); 2.4GHz:mini5pro:20M=0.737(175/20,组0大); 2.4GHz:mini5pro:40M=0.771(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=1.000(477/68,组0大); 5.8GHz:mini5pro:10M=1.000(166/14,组0大); 5.8GHz:mini5pro:20M=1.000(155/15,组0大); 5.8GHz:mini5pro:40M=1.000(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.632(38/36,组1大)|
|ctl_dc_rel|0.673(5138/260,组0大)/未见来源线索|0.674(1287/60,组0大)|0.629(5138/394,组0大)|2.4GHz:mini5pro:pooled=0.735(518/79,组0大); 2.4GHz:mini5pro:10M=0.852(164/20,组0大); 2.4GHz:mini5pro:20M=0.793(175/20,组0大); 2.4GHz:mini5pro:40M=0.704(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.924(477/68,组1大); 5.8GHz:mini5pro:10M=0.962(166/14,组1大); 5.8GHz:mini5pro:20M=0.952(155/15,组1大); 5.8GHz:mini5pro:40M=0.865(156/20,组1大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.723(38/36,组1大)|
|ctl_iq_power_ratio_db|0.521(5138/260,组0大)/未见来源线索|0.504(1287/60,组0大)|0.523(5138/394,组0大)|2.4GHz:mini5pro:pooled=0.506(518/79,组1大); 2.4GHz:mini5pro:10M=0.527(164/20,组0大); 2.4GHz:mini5pro:20M=0.551(175/20,组0大); 2.4GHz:mini5pro:40M=0.509(179/19,组1大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.707(477/68,组0大); 5.8GHz:mini5pro:10M=0.617(166/14,组0大); 5.8GHz:mini5pro:20M=0.532(155/15,组1大); 5.8GHz:mini5pro:40M=1.000(156/20,组0大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.569(38/36,组0大)|
|ctl_iq_corr|0.524(5138/260,组0大)/未见来源线索|0.508(1287/60,组1大)|0.518(5138/394,组0大)|2.4GHz:mini5pro:pooled=0.512(518/79,组1大); 2.4GHz:mini5pro:10M=0.548(164/20,组1大); 2.4GHz:mini5pro:20M=0.590(175/20,组1大); 2.4GHz:mini5pro:40M=0.539(179/19,组0大); 2.4GHz:mini5pro:unmatched=nan(0/20,不可估); 5.8GHz:mini5pro:pooled=0.999(477/68,组1大); 5.8GHz:mini5pro:10M=1.000(166/14,组1大); 5.8GHz:mini5pro:20M=1.000(155/15,组1大); 5.8GHz:mini5pro:40M=1.000(156/20,组1大); 5.8GHz:mini5pro:unmatched=nan(0/19,不可估); 2.4GHz:evo2pro_v3:pooled=0.548(38/36,组1大)|
带宽分层 b 结果及 JSON 来源见 source_same_model_auc.csv、v3_declared_bw.csv；unknown/null 不推断。5.8G 来源均标记“未能检验”。 pooled为全部合并；unmatched为其余带宽/unknown，单类时AUC不可估。分层数值：2.4GHz:mini5pro:10M=0.938(164/20,组0大)；2.4GHz:mini5pro:20M=0.559(175/20,组1大)；2.4GHz:mini5pro:40M=0.726(179/19,组0大)；2.4GHz:mini5pro:unmatched=nan(0/20,不可估)；5.8GHz:mini5pro:10M=1.000(166/14,组0大)；5.8GHz:mini5pro:20M=1.000(155/15,组0大)；5.8GHz:mini5pro:40M=1.000(156/20,组0大)；5.8GHz:mini5pro:unmatched=nan(0/19,不可估)。 declared.bw列表取唯一非空Hz值换算M；unknown组：v3:20260709-025658_mini5pro,v3:20260709-035445_mini5pro。b主分析仅含keep=True，rejected仅列附表。
来源多变量（19列仅来源诊断；标签模型仅用候选）：a_test/13列=1.000(1287/60,组1大)；a_test/19列=1.000(1287/60,组1大)。
b多变量 LOGO 与每折 C、跳过数见 source_multivariate_auc.csv；外层留出整组，C优先用剩余组的原val选，否则嵌套LOGO；内层不可估时C=1，单类别外层训练不拟合。 分层Mini每档v3仅1个group，EVO的data仅1个group；留出该group后训练仅单类，对应双类OOF AUC不可估，未改换划分。 AUC：2.4GHz:mini5pro:pooled/13列=0.949(518/79,组1大)；2.4GHz:mini5pro:pooled/19列=1.000(518/79,组1大)；2.4GHz:mini5pro:10M/13列=nan(164/0,不可估)；2.4GHz:mini5pro:10M/19列=nan(164/0,不可估)；2.4GHz:mini5pro:20M/13列=nan(175/0,不可估)；2.4GHz:mini5pro:20M/19列=nan(175/0,不可估)；2.4GHz:mini5pro:40M/13列=nan(179/0,不可估)；2.4GHz:mini5pro:40M/19列=nan(179/0,不可估)；2.4GHz:mini5pro:unmatched/13列=nan(0/0,不可估)；2.4GHz:mini5pro:unmatched/19列=nan(0/0,不可估)；5.8GHz:mini5pro:pooled/13列=1.000(477/68,组1大)；5.8GHz:mini5pro:pooled/19列=1.000(477/68,组1大)；5.8GHz:mini5pro:10M/13列=nan(166/0,不可估)；5.8GHz:mini5pro:10M/19列=nan(166/0,不可估)；5.8GHz:mini5pro:20M/13列=nan(155/0,不可估)；5.8GHz:mini5pro:20M/19列=nan(155/0,不可估)；5.8GHz:mini5pro:40M/13列=nan(156/0,不可估)；5.8GHz:mini5pro:40M/19列=nan(156/0,不可估)；5.8GHz:mini5pro:unmatched/13列=nan(0/0,不可估)；5.8GHz:mini5pro:unmatched/19列=nan(0/0,不可估)；2.4GHz:evo2pro_v3:pooled/13列=nan(0/36,不可估)；2.4GHz:evo2pro_v3:pooled/19列=nan(0/36,不可估)。
同接收机标签 AUC（组0背景/组1无人机）：完整train/val/test见 label_auc.csv；此表列 test。
|特征|2.4 A|2.4 B|5.8 B|
|---|---|---|---|
|shape_factor|1.000(1287/1120,组1大)|0.539(80/105,组0大)|0.683(60/130,组1大)|
|log_kurt_a|0.742(1287/1120,组1大)|0.553(80/105,组0大)|0.551(60/130,组1大)|
|papr999_db|1.000(1287/1120,组1大)|0.561(80/105,组1大)|0.608(60/130,组1大)|
|c42n|1.000(1287/1120,组1大)|0.511(80/105,组1大)|0.614(60/130,组1大)|
|c40n_abs|0.809(1287/1120,组1大)|0.548(80/105,组1大)|0.637(60/130,组1大)|
|c41n_abs|0.622(1287/1120,组1大)|0.511(80/105,组0大)|0.727(60/130,组1大)|
|c20n_abs|0.742(1287/1120,组0大)|0.590(80/105,组0大)|0.798(60/130,组1大)|
|log_gamma_max|0.999(1287/1120,组1大)|0.711(80/105,组1大)|0.884(60/130,组1大)|
|sigma_f|0.841(1287/1120,组0大)|0.753(80/105,组1大)|0.840(60/130,组0大)|
|spec_flatness|0.995(1287/1120,组0大)|0.540(80/105,组0大)|0.929(60/130,组0大)|
|occ_frac|0.945(1287/1120,组1大)|0.820(80/105,组1大)|0.859(60/130,组1大)|
|duty|0.996(1287/1120,组1大)|0.925(80/105,组1大)|0.893(60/130,组1大)|
|log1p_n_bursts|0.996(1287/1120,组1大)|0.983(80/105,组1大)|0.873(60/130,组1大)|
|ctl_abs_power_db|1.000(1287/1120,组1大)|0.688(80/105,组1大)|0.929(60/130,组1大)|
|ctl_floor_db|1.000(1287/1120,组1大)|0.913(80/105,组1大)|0.561(60/130,组1大)|
|ctl_dyn_range_db|0.999(1287/1120,组1大)|0.757(80/105,组1大)|0.870(60/130,组1大)|
|ctl_dc_rel|0.608(1287/1120,组0大)|0.593(80/105,组0大)|0.716(60/130,组0大)|
|ctl_iq_power_ratio_db|0.510(1287/1120,组1大)|0.528(80/105,组1大)|0.689(60/130,组1大)|
|ctl_iq_corr|0.509(1287/1120,组1大)|0.529(80/105,组1大)|0.789(60/130,组0大)|
噪声逐特征、逐SNR、n0/n1详表 noise_auc.csv（长表）、noise_auc_wide.csv（每个SNR一列）；图 fig_B1_auc_vs_snr_2.4GHz.png、fig_B1_auc_vs_snr_5.8GHz.png；5.8G A没有背景。
SNR定义=原片段平均功率/新增噪声功率；每片段seed=SHA256("2026|sample_id")低64位小端，按SNR序号派生SeedSequence；逐片种子和抽样见缓存 noise_meta.csv。
|接收机/特征（n0/n1）|clean|0|−5|−10|−15|
|---|---|---|---|---|---|
|2.4GHz:A/shape_factor(1287/400)|0.999|0.997|0.994|0.990|0.832|
|2.4GHz:A/c42n(1287/400)|1.000|0.999|0.995|0.993|0.859|
|2.4GHz:A/spec_flatness(1287/400)|0.989|0.989|0.989|0.990|0.992|
|2.4GHz:A/duty(1287/400)|0.990|0.834|0.615|0.552|0.502|
|2.4GHz:B/shape_factor(80/105)|0.539|0.512|0.509|0.510|0.505|
|2.4GHz:B/c42n(80/105)|0.511|0.511|0.511|0.513|0.505|
|2.4GHz:B/spec_flatness(80/105)|0.540|0.517|0.526|0.539|0.549|
|2.4GHz:B/duty(80/105)|0.925|0.785|0.579|0.695|0.613|
|5.8GHz:B/shape_factor(60/130)|0.683|0.637|0.624|0.614|0.502|
|5.8GHz:B/c42n(60/130)|0.614|0.612|0.615|0.610|0.523|
|5.8GHz:B/spec_flatness(60/130)|0.929|0.917|0.904|0.887|0.881|
|5.8GHz:B/duty(60/130)|0.893|0.525|0.598|0.508|0.500|
分类器：完整并列表 classifier_metrics.csv（两LR及J/H各2026/2027/2028；每个v3背景会话、bgA、nonG/G/含G、data/v3和所有型号召回；虚警均含k/N、单侧CP95上界），拟合参数 classifier_fits.csv，逐片预测 predictions_*.csv；筛出片段附表 rejected_classifier_metrics.csv。
|频段（虚警k/N[CP95]；漏检k/N）|LR13|LR_safe|J/H2026|J/H2027|J/H2028|
|---|---|---|---|---|---|
|2.4GHz|LR13:v3_all=3/80[0.0941],bgA_all=0/1287[0.0023],nonG=202/1088,includingG=202/1225|LR_safe:v3_all=54/80[0.7613],bgA_all=639/1287[0.5198],nonG=258/1088,includingG=307/1225|J2026:v3_all=1/80[0.0579],bgA_all=0/1287[0.0023],nonG=45/1088,includingG=57/1225|J2027:v3_all=0/80[0.0368],bgA_all=0/1287[0.0023],nonG=39/1088,includingG=138/1225|J2028:v3_all=0/80[0.0368],bgA_all=0/1287[0.0023],nonG=115/1088,includingG=214/1225|
|5.8GHz|LR13:v3_all=0/60[0.0487],nonG=73/784,includingG=81/818|LR_safe:v3_all=7/60[0.2080],nonG=98/784,includingG=109/818|H2026:v3_all=0/60[0.0487],nonG=12/784,includingG=24/818|H2027:v3_all=1/60[0.0766],nonG=9/784,includingG=15/818|H2028:v3_all=0/60[0.0487],nonG=9/784,includingG=16/818|
“未见来源线索”候选：c20n_abs；两频段使用同一(a)筛选清单，5.8G为迁移筛选，未检验其接收机来源。
小事项：特征分析保留G并标明；分类器主漏检排除G；LR C按val原向AUC最大选、并列取较小C；标准化不加样本权重，权重仅用于LR拟合；原有时频图预测直接读CSV，不重推。 异常已修复：混合bgA/原始meta列不同致CSV首次写出中止，保留已完成数组并按同种子恢复2462行元数据、全行clean特征精确核对；报告变量重名修正。加噪耗时由数组完成时间减原始阶段结束时间估算，含阶段启动开销。
耗时：全量特征2191.9s，加噪1301.3s，分析4.2s；逻辑IQ读取204.747GB（不含OS预读及检查21片）；缓存合计9.220MB。
保护目录：protected_before.json 与 protected_after.json 已核对3438个文件SHA256完全一致；不导入crop.py/common.py，不写H缓存，无网络训练/安装/提交/DSH。
**事实陈述，不下结论**
(a)阳性对照功率/底噪 AUC：1.000000(n0=5138,n1=260)/1.000000(n0=5138,n1=260)。
候选13列：(a)接收机指纹=7，可疑=5，未见来源线索=1。
2.4GHz LR13：v3_all=3/80；nonG=202/1088。
5.8GHz LR13：v3_all=0/60；nonG=73/784。
