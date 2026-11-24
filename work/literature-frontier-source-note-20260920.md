# 无人机 RF 前沿文献更新：Codex 证据说明（2026-09-20）

## 用途与边界

本文件为 `LIT-AUDIT1` 的受控外部证据输入。写作时应同时读取本地已有的 63 条元数据、34 条精选 BibTeX、10 篇精读和 6 份综合报告。不得把这里未核实的数值补成确定事实；不清楚的字段写 `NR`，并注明需要全文核验。

本轮目标不是逐篇全文翻译，也不声称完成 WoS、Scopus 或 Google Scholar 意义上的穷尽式系统综述。目标是对现有文献库进行可追溯的前沿补充、任务分层和项目差距审计。正文只能短引或转述，不能复制受版权保护的论文长段落。

## 已核实的新增或关键原始来源

### 1. CageDroneRF（2026，预印本 v2）

- 原文：[arXiv HTML](https://arxiv.org/html/2601.03302)，[代码/数据工具](https://github.com/DroneGoHome/U-RAPTOR-PUB)
- 证据状态：2026-03-19 的 arXiv v2；不是成熟同行评审结论。
- 关键事实：39 类、23 个无人机型号；受控 RF cage 与室外采集；20 MHz；在原始复数 I/Q 上做 SNR、干扰信号混合、频移，并同步重算 YOLO 标注；覆盖检测、分层分类、单标签分类和开放集。
- 对本项目的意义：它直接支持“干净小数据集的高准确率不能替代跨环境验证”，也为以后噪声/干扰实验提供信号域处理依据。本项目当前明确暂缓噪声方式实验，因此只列为后续阶段，不擅自开始。
- 已从全文表格核到的警示结果：indoor-only 到 outdoor 的 model-level accuracy 为 42.07%，indoor+outdoor 为 69.63%；单标签 ResNet 的 20 类结果为 53.49%；MetaMax 的 10 known + 4 unknown 总体约 75.15%、unknown detection 约 73.96%。这些数值写入正式报告前须在原文相应表格再次定位并给出来源层级。

### 2. RFUAV（2025，预印本/开源基准）

- 原文：[arXiv HTML](https://arxiv.org/html/2503.09033)，[官方仓库](https://github.com/kitoweeknd/RFUAV)
- 关键事实：约 1.3 TB 原始 RF/IQ，37 个 UAV；提供预处理和不同 SNR 的工具；两阶段路线为时频目标检测/定位后再分类。
- 对 STFT 参数的意义：论文比较多个 STFT 点数，结论只适用于其数据和任务，不能把“某一 n_fft 最优”外推为本项目固定定律。本项目已经用固定协议实测 512/1024/2048/4096，应报告为数据依赖的消融。

### 3. S3R（TIFS 2024）

- 原文：[IEEE](https://ieeexplore.ieee.org/document/10684814/)，DOI `10.1109/TIFS.2024.3463535`，[官方代码](https://github.com/DaftJun/S3R)
- 关键事实：STFT 后提取纹理与位置语义，做已知/未知判别，并对未知样本进一步聚类；公开 24 类、7 个品牌的真实频谱图数据。
- 意义：开放集是一项独立任务，不能用三分类闭集准确率代替；后续至少需要 MSP/energy/prototype/OpenMax 或 EVT 对照和多个 known/unknown 划分。

### 4. Open-RFNet（JSAC 2026）

- 原文：[IEEE](https://ieeexplore.ieee.org/document/11417759/)，DOI `10.1109/JSAC.2026.3669136`
- 关键事实：多域监督对比学习，ResNet 与 TransformerEncoder 融合，IG-OpenMax；论文报告 25 类型设置下 closed-set 95.12%、open-set 96.08%。
- 审计边界：这些数值不能和本项目三分类准确率直接横比；现有精读已记录其近距离子集、单一 known/unknown 划分和约 2.06 亿参数等限制，正式报告应链接该精读并复核。

### 5. 开放世界增量识别（2026，预印本）

- 原文：[arXiv HTML](https://arxiv.org/html/2603.24268)
- 关键事实：24 类数据，18 known + 6 unknown；先开放集拒识，再用 K-Means/GMM 发现新类，最后有限回放做增量学习。
- 边界：只作未来方向证据；固定一次 known/unknown 划分、真实复杂环境与跨域效度需要进一步审查。

### 6. DRFF-R2（2026，数据手稿）

- 原文：[arXiv](https://arxiv.org/abs/2603.00106)
- 关键事实：26 个物理 UAV、8 个型号、7 个子集，覆盖飞行状态、高度、速度、采集日期、接收机配置及多种信号组合。
- 意义：强调 model-level 分类、individual RFFI、跨天/跨接收机和干扰不是同一个问题；本项目目前只有型号级两机分类，没有证据支持个体识别。

### 7. 真实环境鲁棒性审计（Sensors 2024）

- 原文：[PMC 全文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11597967/)，DOI `10.3390/s24227339`
- 关键事实：1100 条 RF recordings，跨多天和多地点，通过真实距离改变接收 SNR。低 SNR 同环境时 CNN/XGB 接近 99/100；换到两个新 RF 环境后 CNN 为 81/53，XGB 为 44/1。
- 意义：对当前项目最直接的警示是：AWGN 条件和同源文件留出不能证明跨现场鲁棒性；PSD+SVM heldout 100% 也不能解释为部署能力。

### 8. DroneRFa 官方数据论文

- 原文：[电子与信息学报](https://jeit.ac.cn/article/doi/10.11999/JEIT230570)，DOI `10.11999/JEIT230570`
- 关键事实：公开 24 类无人机信号和背景，覆盖室内/室外与 915 MHz、2.4 GHz、5.8 GHz；原始文件的细粒度会话、日期、距离和接收机元数据并不等于完整可得。
- 本项目实际只使用 Background、DJI Phantom 4 Pro、DJI Air 2S 三类；不能把数据集全量 24 类写成已经训练使用。

## 已有主干来源（必须保留，不被新文献替代）

- Al-Sa'd et al., FGCS 2019，DOI `10.1016/j.future.2019.05.007`：DroneRF 早期层级闭集基线。
- DroneRF 数据论文，DOI `10.1016/j.dib.2019.104313`。
- Ezuma et al., OJ-COMS 2019，DOI `10.1109/OJCOMS.2019.2955889`：Wi-Fi/Bluetooth 干扰。
- Soltani et al., TVT 2020，DOI `10.1109/TVT.2020.3042128`：非标准波形 UAV 个体 RFFI 与跨突发问题。
- Reus-Muns and Chowdhury, TVT 2021，DOI `10.1109/TVT.2021.3081049`：前导码物理先验和跨状态泛化。
- Basak et al., TCCN 2022，DOI `10.1109/TCCN.2021.3099114`：联合检测、定位、分类。
- Cross-domain Drone RFFI，DOI `10.1109/TCCN.2025.3561305`：直接 UAV 跨域个体识别，但任务与型号分类不同。

## 明确排除或降级的来源

- `CrossRF`（arXiv:2505.18200）当前已撤稿，不能作为正面结果或 SOTA 证据。可在排除表中记录“withdrawn due to internal review/authorship concerns”，不要依赖其数值。
- 本地 `references_metadata.csv` 中题为 `Design Studies to Achieve Energy Optimal Attitude for a Solar-Powered Aircraft` 的 2019 条目与被动 UAV-RF 识别任务无关，即使历史 `track` 误标为 `passive_rf` 也必须排除，不能用于满足直接证据数量。
- 仅视觉、雷达、声学 UAV 论文不进入 RF 主矩阵；可在背景中简短说明多模态位置。
- 一般 LoRa/Wi-Fi/ADS-B RFFI 只能标作“相邻方法证据”，不得冒充直接 UAV 型号识别结果。
- 论文自称 SOTA、accuracy 或 accuracy 排名不能跨任务、跨类别数、跨数据集、跨切分方法直接比较。

## 必须和本项目实况对齐

- 数据：F 盘共 24 个 MAT，约 23.025 GiB；当前实验三类为 Background / DJI Phantom 4 Pro / DJI Air 2S。
- Stage 1 传统 clean 基线：能量阈值 heldout 文件等权 BA 98.30%，PSD+SVM 100.00%。
- Stage 1 n_fft 消融：512/1024/2048/4096 四条件文件等权 BA 跨 seed 均值为 99.04%/98.07%/97.40%/97.98%。
- Stage 2 两机型：选择 n_fft=512，四条件文件等权 BA 99.78%±0.09pp。
- 串联三分类：clean 99.75%、-10 dB 99.98%、-12 dB 99.63%、-15 dB 95.14%；-15 dB 的主要问题是 Air 2S 被 Stage 1 漏成 Background。
- 约束：heldout 已用于部分模型/n_fft 描述性分析；不同原始文件不证明独立会话；AWGN 档位不等于现场绝对 SNR；因此这些是已验证的工程基线，不是跨域前沿结论。

## 正式报告必须给出的项目顺序

1. 先做元数据与切分审计，建立 `group_file`，能证明时再建立 `cross_session/day/receiver/distance`；保留完全未用于选型的 final test。
2. 在现有三类上补直接 3-class 模型和同预算 1D Raw-IQ 基线，用来回答级联是否必要、STFT 是否占优。
3. 扩到当前本机已有的更多 DroneRFa 类，至少形成多类闭集和层级识别；不能长期停留在两机型。
4. 做开放集：多个 known/unknown 划分，报告 AUROC、OSCR、FPR95/ECE 等，阈值校准与最终未知类隔离。
5. 做跨域/真实条件：跨环境、跨距离、跨天、跨接收机或跨数据集；真实 Wi-Fi/Bluetooth/背景优先，合成 AWGN 只是受控对照。
6. 多信号场景只有在任务变成宽带连续流的“检测+定位”后才引入 YOLO；不要把单个已裁剪窗口的三分类问题包装成目标检测。
