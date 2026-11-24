# UAV Individual Identification via Distilled RF Fingerprints-Based LLM in ISAC Networks

## 基本信息

- 作者：Haolin Zheng、Ning Gao、Donghong Cai、Shi Jin、Michail Matthaiou 等
- 来源：IEEE Wireless Communications Letters, Vol. 14, No. 11, 2025, pp. 3769–3773
- 类型：同型号/多型号个体 RFF、大模型教师、PPO 动态知识蒸馏、轻量部署
- [源 PDF](<../UAV_Individual_Identification_RFF_LLM_2025.pdf>)

## 方法

论文把修改后的 GPT-2 作为 RF 指纹教师网络 RFF-LLM，将时频序列送入 Transformer/LSTM 组合以学习长短期结构；学生网络为 Lite-HRNet。蒸馏损失由真实标签交叉熵与教师—学生 KL 散度组成，PPO 控制器依据训练进度、准确率和 KL 变化动态调整蒸馏温度，试图避免固定温度的局部最优。

## 数据与结果

作者构建 DRFF-R1：USRP B210 采集 7 个机型、20 台商业无人机的上行 I/Q，户外高度为 10、30、50、70、90 m，并在 0 m 吸波材料条件下采集干扰较少的参考信号。STFT 后切成 2 万个样本，按 6:2:2 分割训练、验证、测试。

教师 RFF-LLM 准确率 97.17%，无蒸馏 Lite-HRNet 为 95.28%，动态蒸馏后的 Lite-HRNet 达 98.38%，高于最佳固定温度（96.62%）。学生仅 156,064 个参数，推理 2.74 ms；对比 ResNet 91.85%、LMSC 82.87%、ShuffleNet-v2 94.73%。

## 优点

- 同时关注精度、参数量、延迟和 RAM，而非只报准确率。
- 20 台设备、7 个机型比早期 RFF 研究更接近个体识别。
- “复杂教师离线学习、轻量学生在线部署”是工程上合理的结构。

## 局限与可信度判断

- 将 GPT-2 称作 LLM 容易造成概念夸大：它本质上是以 Transformer 为骨架的序列教师，并不意味着语言语义对 RF 必然有效。
- 论文篇幅短，6:2:2 是否按设备采集会话/日期严格分组并不清楚；若随机切片，98.38% 可能含会话泄漏。
- “无多普勒、接收机固定且唯一”的建模假设限制跨接收机和运动场景外推。
- PPO 控温增加训练复杂度；相较网格搜索、余弦温度或不确定性驱动蒸馏，是否必要需要更强消融。
- 只做闭集个体识别，非法未知设备到来时仍可能被强制归入已有身份。

## 对本项目的价值

适合作为后期轻量化研究，而不是第一基线。建议先证明严格跨会话、跨接收机和开放集条件下的教师特征确实更稳，再做蒸馏；同时用普通 Transformer、TCN、1D-ResNet 和固定温度 KD 做强消融，避免创新点仅剩“使用 LLM/PPO”的命名。
