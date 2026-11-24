# 文献调研：做法 B（ResNet18 时频嵌入 + IQ 手工特征融合）

日期：2026-11-10
范围：深度特征与手工特征怎么融合（位置、结构、归一化）、手工特征的清单和定义、低信噪比下手工特征表现如何、数据集/接收机偏差让手工特征变成捷径的问题。
原则：只写读到的内容。每篇都标了阅读程度。只读到摘要的，只写"摘要称……"，不用它下结论。引文保留英文原句，并给出章节和页码（页码按我读到的版本：arXiv 版或期刊版，下文注明）。

---

## 0. 阅读程度一览

| # | 论文 | 阅读程度 | 来源 |
|---|---|---|---|
| 1 | Ezuma et al., OJCOMS 2020（arXiv 1909.05429） | 全文 | arXiv v1 PDF |
| 2 | Ezuma et al., IEEE Aerospace 2019（arXiv 1901.07703） | 方法+实验+结论章节全文 | arXiv PDF |
| 3 | Ozturk, Erden, Guvenc, 2020（arXiv 2009.05519） | 全文 | arXiv v2 PDF |
| 4 | Medaiyese et al., 小波分析（arXiv 2102.11894） | 部分：摘要、7.2 节（SNR）、结论 | arXiv PDF |
| 5 | DroneRFa 数据集论文，电子与信息学报 46 卷 | 4.1–4.3 节全文 | 本地抽取文本 |
| 6 | Gao et al., Open-RFNet（JSAC 2026 稿） | 全文主体（I–VII 节） | 本地抽取文本 |
| 7 | Yang et al., Entropy 2021, 23(12):1678（FEG+MC-DNN） | 全文 | Europe PMC 全文 XML |
| 8 | Glüge et al., IEEE J-RFID 2024（低 SNR，VGG） | 全文 | 本地抽取文本 |
| 9 | Basak et al., TCCN 2022（Combined detection & classification） | 全文 | 本地抽取文本 |
| 10 | Soltani et al., TVT 2020（非标准波形 UAV 射频指纹） | 第 I–III 节（数据、训练流程、初步实验）全文；第 IV 节多分类器部分只读了开头 | 本地抽取文本 |
| 11 | Reus-Muns & Chowdhury, TVT 2021, 70(7)（前导码互相关特征） | 第 III–IV 节相关段落 + 摘要 | 本地抽取文本 |
| 12 | Han et al., Sensors 2021, 21(6):2117（AMC 特征融合） | 全文 | Europe PMC 全文 XML |
| 13 | Yin et al., FuseAMC（arXiv 2611.05937，未经同行评审） | 全文 | arXiv PDF |
| 14 | Shebert, Kirk, Buehrer（arXiv 2302.03749） | 全文主体（I–V 节） | arXiv PDF |
| 15 | Hanna et al., WiSig（arXiv 2112.15363） | I–VI 节全文 | arXiv PDF |
| 16 | Geirhos et al., Shortcut learning, NMI 2020（arXiv 2004.07780） | 摘要 + 6.2 节 | arXiv PDF |
| 17 | Wang, Tran, Feiszli, CVPR 2020（arXiv 1905.12681） | 摘要、第 1 节、2.1 节 | arXiv PDF |
| 18 | Fu, Zhang, He, Drones 2026, 10(2):124（VMD+多域特征） | **仅摘要**（Semantic Scholar 摘要 + 检索摘要）。MDPI 对本机返回 403，没拿到全文 | — |
| 19 | Shi, Shan, Dong, "Integrated detection of weak drone signals", DSP 168:105524 | **仅检索摘要片段**（ScienceDirect 摘要页的搜索摘录，不完整） | — |

至少读了全文或方法+实验章节全文的论文：1、2、3、5、6、7、8、9、12、13、14、15，共 12 篇。另外 10、11、17 读了关键章节。

另外查过、和本题无关的：本地库 `UAV_Individual_Identification_RFF_LLM_2025.txt` 里的"拼接"是两个深度分支的拼接，没有手工特征；`DRFF_R2_*` 里没有手工特征融合的内容。文献矩阵 `01_文献矩阵.csv` 中唯一相关的未读条目是 M014（就是本文第 7 篇，已读全文）和 M015（JESTCH 2022 频谱特征，没有获取，未读）。

---

## 1. 逐篇笔记

### A. 无人机 RF 领域

#### 1. Ezuma et al., "Detection and Classification of UAVs Using RF Fingerprints in the Presence of Interference"
- 作者/年份/刊物：M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, I. Guvenc；IEEE Open Journal of the Communications Society, vol. 1, pp. 60–76, 2020（arXiv 1909.05429v1, 2019-09）。
- 链接：https://arxiv.org/abs/1909.05429 ；IEEE 开放获取 https://ieeexplore.ieee.org/document/8913640
- 阅读程度：全文（arXiv v1，下文页码按 arXiv 版）。
- 任务与数据：一个多级系统。第 1 级是检测（"RF 信号" vs "噪声"），用两状态 Markov 模型和朴素贝叶斯；第 2 级排除 Wi-Fi/蓝牙（用带宽和 GFSK 调制特征）；第 3 级对 15 或 17 个遥控器做 ML 分类。采集设备是一台示波器（Keysight MSOS604A，20 GSa/s），配 2 dBi 全向天线或 24 dBi 栅格天线。"Due to space limitations, only the results of the indoor experiments will be reported."（Sec. VI, p.8）每个遥控器 100 段信号，每段 0.25 ms，随机 80/20 划分（Sec. VI, p.8–9）。
- 融合位置：**没有深度网络，也没有融合**。手工特征直接送 kNN/RandF/DA/SVM/NN。
- 手工特征与归一化（Sec. V, p.6–8，Table II p.7）：
  - 计算对象：不是原始 IQ。先做两级 Haar 小波预处理，再算谱图（128 点分段、120 点重叠、Hamming 窗、256 点 DFT），然后"the energy trajectory can be computed from the spectrogram by taking the maximum energy values along the time-axis"，在"normalized energy trajectory"上找能量突变点得到"energy transient"，最后"RF fingerprints (a set of 15 statistical features) are extracted"（p.7）。
  - 15 个特征（Table II）：mean、absolute mean、standard deviation、skewness、entropy、RMS、root、kurtosis、variance、peak value、peak-to-peak、shape factor、crest factor、impulse factor、clearance factor。公式在 Table II 里，但抽取文本中排版错位，公式不逐条转录。
  - 归一化：文中只提到能量轨迹是"normalized"，没有写归一化方式。没有读到对 15 维特征做 z-score 或 min-max 的说明。
  - 特征选择：用 NCA。"It turns out that the shape factor is the most discriminative feature in the feature set. The next significant feature is the kurtosis … Next are variance and standard deviation. On the other hand, entropy … is the least significant feature."（Sec. V-A, p.8）最后选了 shape factor、kurtosis、variance 三个。
- 关键结果（Table IV, p.10，25 dB SNR，10 次 Monte Carlo 平均）：

  | 控制器数 | 分类器 | 全部 15 特征 | NCA 选 3 特征 |
  |---|---|---|---|
  | 15 | kNN | 97.30 | 98.13 |
  | 15 | DA | 96.30 | 94.43 |
  | 15 | SVM | 96.47 | 91.67 |
  | 15 | NN | 96.73 | 96.13 |
  | 15 | RandF | 98.53 | 97.73 |
  | 17 | kNN | 95.62 | 95.53 |
  | 17 | DA | 92.77 | 88.12 |
  | 17 | SVM | 93.82 | 87.88 |
  | 17 | NN | 92.88 | 93.03 |
  | 17 | RandF | 96.32 | 95.18 |

  （表格排版错位，我按正文复述的数字核对过：RandF 98.53/96.32，kNN 97.30/95.62，选择后 kNN 98.13/95.53，RandF 97.73/95.18，DA 94.43/88.12，都和正文一致。）
  - 检测（Fig. 13, p.9）："at very low SNR, such as −10 dB, the detection accuracy is generally very low irrespective of the threshold."（Sec. VII-A, p.9）10 dB、δ=3.5σ 时检测准确率 99.8%，FAR 2.8%。
  - 分类与 SNR："In general, the accuracy of all the classifiers increases with SNR. Therefore, to ensure accurate identification of the UAV controller, it is best to operate the receiver at SNR above 15 dB"（Sec. VII-B, p.11–12）。各 SNR 的具体数字只在 Fig. 15 曲线里，正文没给，所以不转录。第 3 篇 Ozturk 的 Table I 把本文概括为"98.13% and 40% for 0 dB SNR"（那是 Ozturk 的转述）。
- 作者自己写的局限：只报告了室内结果；检测灵敏度受示波器输入阻抗限制（p.9："the input impedance of the oscilloscope places a fundamental limit on the sensitivity"）。
- 和我们任务的差别：他们分类的是遥控器型号，我们是背景 vs 无人机；他们只有一台接收机、在室内、按样本随机划分；特征算在能量包络的瞬态上，不是在原始 IQ 上；也没有任何深度网络融合。
- 对做法 B 的意义：这是"峰度、偏度、方差、形状因子"这类特征在无人机 RF 上最直接的出处，也说明 NCA 可以把 15 维压到 3 维而几乎不掉点。但 peak value、RMS、peak-to-peak 这类特征依赖绝对幅度，作者没有说明是否消除了接收机增益或距离的影响。到我们数据上，这类特征正好落在"接收机捷径"的风险区。

#### 2. Ezuma et al., "Micro-UAV Detection and Classification from RF Fingerprints Using Machine Learning Techniques"（IEEE Aerospace Conf. 2019）
- 链接：https://arxiv.org/abs/1901.07703
- 阅读程度：特征定义（Sec. V 及式 (11)）、NCA、实验与结果（Sec. VI）、结论都读了全文（按 arXiv PDF 页码）。
- 任务与数据：14 个遥控器，室内走廊，近场和远场（最远 130 m）。"Each RF signal is a vector of size 5000000 × 1 and has a time span of 0.25 ms"，随机 80/20 划分（p.9）。
- 手工特征（p.7，式 (11)）：在能量瞬态 fE(n) 上计算 4 个统计量："skewness (γ), variance (σ²), energy spectral entropy (H), and kurtosis (k)"，公式是标准的三阶、二阶、四阶中心矩（除以 σ³、σ⁴）和 Shannon 熵。
- NCA 结果（p.7–8）："kurtosis has the highest weight and so is the most important RF fingerprint for this test case. On the other hand, skewness has the lowest weight … because there is a correlation between the features skewness and kurtosis."
- SNR 结果（Sec. VI-B, p.9）：在这篇里，SNR 是靠改变距离得到的真实测量值。"At an SNR of 25 dB, corresponding to a distance of about 5 m, kNN achieves a classification accuracy of about 97.29% … When the SNR is 10 dB, corresponding to a distance of about 80 m, kNN, DA and SVM achieves a classification accuracy between 60-70% while NN performs below 50%."
- 局限：NN 没有调超参（"we did not optimize the hyper parameters of the NN algorithm"）；结论写了将来要做室外和多传感器。
- 和我们的差别：同第 1 篇。这里的低 SNR 来自真实的远距离，不是人工加噪。
- 意义：给出了一个真实数据点：只用手工统计特征时，10 dB 已经掉到 60–70%。

#### 3. Ozturk, Erden, Guvenc, "RF-Based Low-SNR Classification of UAVs Using Convolutional Neural Networks"
- 链接：https://arxiv.org/abs/2009.05519 （ITU J-FET 2021 发表；本文按 arXiv v2 页码）
- 阅读程度：全文。
- 任务与数据：沿用第 1 篇 Ezuma 的数据集，15 个遥控器，"the original data has an SNR of about 30 dB"，人工加白噪声到 −10…30 dB（Sec. III-B, p.6–7）。每类每个 SNR 100 张图（Sec. V, p.9）。
- 融合：**没有**手工特征融合。比较的是时序波形图 CNN 和谱图 CNN 两种输入。
- 关键结果：
  - 时序图（Table III, p.10）：30 dB 时 99.7%，20 dB 时 96.5%，10 dB 时 81.6%，5 dB 时 65.3%，0 dB 时 50.1%。原因（Sec. V-B, p.11）："CNNs gather distinctive features from both the transient … and the envelope of the RF signal. As the signal swamps into noise as SNR decreases, first the transient information disappears whereas the information carried in the signal envelope survives a little longer."
  - 谱图截断去噪（Table IV, p.10）：0 dB 时 99.5%，−5 dB 时 99.5%，−10 dB 时 92.0%。所有 SNR 合并训练的单一模型为 98.8%；用训练中没出现的 SNR 测试为 96.9%（p.13）。
  - Table I（p.2）把 Ezuma 的手工特征方法 [24] 概括为"98.13% and 40% for 0 dB SNR"。
- 和"捷径"直接相关的原句（Sec. V-B, p.11）："while using the time-series images, trained models extract features from the amplitude of the signals itself. However, amplitude of a received signal depends on the distance between the receiver and transmitter antenna. This is an obvious problem when the only distinctive difference between the time-series signal images of any two controllers is the difference in their amplitude".
- 局限与注意点：每个 SNR 的截断阈值都在验证集上选（Table IV），存在调参偏差；数据是单接收机、实验室环境。
- 意义：同一份数据上，基于包络和瞬态的信息（手工特征也正是从这里提取的）在 0 dB 附近崩掉，谱图在 −10 dB 仍有 92%。这说明在我们 0 到 −15 dB 的设置下，手工特征很可能大部分失效。不过这篇比较的是两种 CNN 输入，**不是**对手工特征本身做的消融。

#### 4. Medaiyese, Ezuma, Lauf, Guvenc, "Wavelet Transform Analytics for RF-Based UAV Detection and Identification System Using Machine Learning"
- 链接：https://arxiv.org/abs/2102.11894 （Elsevier 预印本）
- 阅读程度：**部分**，读了摘要、7.2 节"Performance at Difference SNR"、第 8 节结论。数据与特征提取章节（第 4–5 节）没有完整读。
- 任务：识别 6 个遥控器、2 个 Wi-Fi、2 个蓝牙；分为 group-device（UAV/Wi-Fi/蓝牙）和 specific-device 两种分类。在 30 dB 数据上训练，测试时加 AWGN（Sec. 7.2, p.24："we built our classifiers using signals captured at 30 dB SNR")。
- 关键结果（Sec. 7.2, p.26）："All the classifiers with coefficient based signatures are equal to or below the threshold of 60 accuracy from an SNR 5 dB and below"。小波散射图（scattergram）+ SqueezeNet "gives an accuracy that is above 60% at an SNR of 0 dB … At 10 dB, the accuracy stands as high as 98.9%."
- 作者结论（p.29）："using wavelet or scattering coefficient for feature extraction is sensitive to variation in SNR because some classifiers failed outrightly when the test signals are at an SNR different from the training signals despite having high accuracy at 30 dB."
- 意义：训练和测试 SNR 不一致时，手工系数特征加传统分类器会直接失效。我们会在多个 SNR 混合训练，条件不同，但这是一个警示。

#### 5. DroneRFa 数据集论文（电子与信息学报，第 46 卷）
- 阅读程度：4.1 节（样本预处理）、4.2 节（基于频谱可视统计特征的识别方法）、4.3 节（基于深度学习表征的识别方法）全文（期刊页码约 p.1151–1153）。
- 手工特征（4.2 节，p.1152）："定义跳频信号与图传信号的6种特征如下：跳频块带宽、跳频块时长、最邻跳频块时间间隔、最邻跳频块带宽间隔、图传信号周期、图传信号占空比"。表 4（p.1153）列出了每个机型的数值，不满足定义的标"/"。这些特征算在 STFT 时频图上，不是在 IQ 上。用法是"入库匹配"（规则或模板），**没有和网络融合**。
- 作者写的局限（4.3 节，p.1152–1153，原文）："一是在检测时延敏感的场景下需采集较短的信号片段，使得无法观测到完整的可视统计特征；二是许多同品牌无人机采用相同或相似的信号调制参数，使得统计特征区分度不大；三是当无人机种类很多时，通过遍历搜寻频谱中的跳频信号带来极大的计算开销；四是当无人机距离较远时由于电磁波的衰减特性使得在时频图上的无人机信号特征不显著"。
- 深度方案：ResNet-18，输入 (2, 1024, 1024)，样本约 10 ms，按 6:2:2 **随机**划分。
- 意义：对我们有两点直接相关。（a）"包络周期、占空比"这类特征需要片段里有足够多的周期。我们是 20 ms 片段，表 4 里图传周期 14 ms、跳频块间隔 12 ms，**20 ms 内最多只覆盖 1–2 个周期**，估计会很不稳定。（b）作者自己承认，远距离（低 SNR）时这些特征在时频图上"不显著"。

#### 6. Gao et al., "Multi-Domain Supervised Contrastive Learning for UAV Radio-Frequency Open-Set Recognition"（Open-RFNet，JSAC 2026 稿）
- 阅读程度：全文主体（I–VII 节，本地抽取文本，按稿件页码）。
- 任务与数据：在 DroneRFa 上做开集识别，24 个机型加 1 类背景（T0000），20 类已知、5 类未知。"we select the signals collected at near distances and indoors for experiment evaluation"，3 ms 片段，STFT 为 775×775（Sec. VII-A, p.10）。
- 融合位置与结构（Sec. IV-A, p.5–6）：**两路都是深度特征，没有手工特征**。ResNet-18 经 GAP 得到纹理特征 z̃a；TransformerEncoder 分别对时频图和它的转置提取"位置特征" z̃b、z̃c（各自先经过 MNLs，即多层非线性层）。融合方式是"z̃ = MNLs((z̃a ⊕ z̃b ⊕ z̃c))"，"⊕ represents the concatenation operation"（式 (12), p.6）。也就是嵌入层拼接后再接 MLP。正文没有写拼接前是否对各支做 BN/LN，也没给各支维度。
- 输入归一化（式 (4), p.5）：逐样本 dB 谱做 min-max 归一化。理由是"The placement of the UAVs at different locations makes the UAV signal power close to the RF receiver much larger than that of the remote UAVs. Thus, for the fairness of model training, we normalize each time-frequency information"（Sec. III-B, p.4–5）。
- 和"模态失衡"直接相关的原句（Remark, p.7）："The traditional cross-entropy loss function is prone to the unbalanced optimization problem for multiple features … When a certain feature, such as texture feature, has better prediction performance, it contributes more to the final prediction, and the corresponding gradient dominates in backpropagation."；可视化部分（Sec. VII-F, p.13）："when training with the cross-entropy loss function, an imbalance in multi-feature optimization occurs; that is, the model pays much attention to the texture features while ignoring the time-frequency position features".
- 消融（Table V, p.12）：闭集准确率 Open-RFNet 0.994、SupResNet 0.9929、TransNet 0.9929、ResNet 0.9958。也就是说**闭集上加了融合分支并没有超过单独的 ResNet**。增益体现在开集 UAR：Open-RFNet 0.9608，ResNet 0.9128。正文还写道（p.11）："when the TE or the supervised contrastive learning is introduced alone on the basis of ResNet, no performance improvement can be achieved."
- 去噪预处理（Table VI, p.12）：把信号强度低于阈值的子片段删掉再拼接；不去噪时开集 UAR 最高只有 66.08%。
- 局限：只用了近距离和室内数据，"some signals at medium and long distances are very weak or even non-existent, which makes these data samples untrainable"（p.10）；没有低 SNR 实验。
- 和我们的差别：开集机型识别，不是背景/无人机检测；两路都是深度特征。
- 意义：这是"嵌入层拼接 + MLP 融合"在 DroneRFa 上的直接先例，也是"交叉熵联合训练时强分支压制弱分支"的同领域证据。在我们的场景里，512 维 ResNet 嵌入就是强分支，64 维手工特征分支很可能被忽略；反过来，如果手工特征里有"接收机"这种极强的捷径，也可能是手工分支压制 CNN。

#### 7. Yang et al., "RF Signal-Based UAV Detection and Mode Classification: A Joint Feature Engineering Generator and Multi-Channel Deep Neural Network Approach"
- 刊物：Entropy 2021, 23(12):1678；doi:10.3390/e23121678；PMC8700519
- 阅读程度：全文。
- 任务与数据：DroneRF 数据集，10 类（背景 + 3 种无人机的飞行模式）。"Each category of data is collected by two RF receivers … two receivers record low-frequency and high-frequency components, respectively."（Sec. 5.1）用分层 K 折交叉验证（Sec. 4.2.1）。
- 这里的"特征工程"是**预处理**（截断、归一化、滑动平均、拼接），不是统计量特征，但其中的拼接和归一化经验可以直接借鉴：
  - Sec. 4.1.1 原文："The components may own different features and exhibit different power levels. Thus, normalizing two components together leads to that the frequency component with small values is dominated by the other one. This means that the small values are normalized to nearly zero, and the value changes become almost invisible. To address this issue, we truncate the two components into two sub-datasets. Each sub-dataset is normalized individually"。在 Algorithm 1 里每个分量各自除以自身最大值，然后直接拼接。
  - MC-DNN（Sec. 4.2.2）：两路输入各自接一部分第一隐层（N11、N12），之后再合并："The MC-DNN isolates the two frequency components, and better learns the respective classification features for each component."
- 消融（Table 2，累加式）：不预处理 + DNN 为 45.9%；截断+归一化后单独用低频/高频分量为 52.5%/85.4%；加滑动平均为 65.5%/90.6%；直接拼接为 97.3%；改用 MC-DNN 为 98.1%；再加余弦学习率为 98.4%。
- 局限：作者没有写局限。从论文本身能看出：K 折是随机划分（存在相邻片段泄漏的风险），没有做 SNR 实验，也没有讨论"两台接收机分别录低频和高频"会带来什么偏差。
- 意义：这是"拼接前**各自**归一化，否则数值小的那路被淹没"的直接例子，消融里拼接前后差距很大（各分量单独 65.5%/90.6%，拼接后 97.3%）；"每路先接一个独立的小层再合并"（等价于做法 B 里的小 MLP 投影）也有 +0.8 个点。不过它的两路都是频谱，不是"图像嵌入 + 统计量"。

#### 8. Glüge et al., "Robust Low-Cost Drone Detection and Classification Using Convolutional Neural Networks in Low SNR Environments"
- 刊物：IEEE Journal of Radio Frequency Identification, 2024
- 阅读程度：全文。
- 任务与数据：在暗室里录 6 种发射机，另录 Bluetooth/Wi-Fi 噪声和高斯噪声，按 SNR −20…30 dB 混合；分 7 类（含 Noise）。每段 2²⁰ 个采样（约 74.9 ms），USRP B210，VGG11–19（带 BN），分层 5 折。最后做了外场测试（110–670 m，三个天线朝向）。
- 融合：没有手工特征。
- **功率归一化**（Sec. II-B, p.3，式 (1)–(2)）：无人机信号按"carrier power of 1 per sample"归一化，只用 burst 部分算功率；噪声向量"normalised to a mean power of 1"；之后按式 (3) 以给定 SNR 混合。
- 接收机噪声（Sec. II-A, p.3）："receiver noise was recorded for 30 seconds from the USRP at various gain settings ([30,70]db in steps of 10dB) without the antenna attached. This should prevent the final model from misclassifying quantisation noise in the absence of a signal, especially at low gain settings."
- 关键结果：开发集上"we observe a drastic degradation in performance from −12dB down to near chance level at −20dB. The vast majority of misclassifications occurred between noise and drones and not between different types of drones."（Sec. IV-A, p.6）外场测试中，Drone/Noise 二分类在各距离、各朝向下的平衡准确率为 0.807–0.986（Table IX, p.8）；0° 朝向的 Noise 类只有 0.554（Table VIII），作者推测湖对岸可能真有无人机。
- 作者写的局限（Sec. V, p.8–9）：信号在实验室录制后人工加噪，"do not fully replicate the complexities encountered in real-world scenarios"；外场距离不够远，"not able to clearly demonstrate the effect of free space attenuation"。
- 意义：（a）低 SNR 下的错误主要是"无人机 vs 噪声"，正对应我们的第一阶段；（b）先做功率归一化再加噪，是去掉绝对功率捷径的标准做法；（c）专门录了不同增益下的接收机底噪放进噪声类，是一种针对"接收机/增益"偏差的数据层面做法。

#### 9. Basak, Rajendran, Pollin, Scheers, "Combined RF-based drone detection and classification"
- 刊物：IEEE TCCN 2022（本地是预印本）
- 阅读程度：全文。
- 任务与数据：暗室，USRP X310，单接收机，9 种遥控器/无人机 + 2 个 Wi-Fi 路由器，人工加 AWGN；"we calculated the signal SNR in the frequency domain"（Sec. V-2, p.5）。
- 方法：两种框架比较。两阶段：先用 Anderson-Darling 拟合优度检验（GoF，一种基于统计量的检测器）检测，再用 DRNN 分类。一体化：YOLO-lite。都不是"手工特征 + 深度"融合。DRNN 输入为 256×256 谱图，"We haven’t performed any batch normalization operation, since, we did not observe any additional improvement"（Sec. IV-A-2, p.4）。
- 关键结果：单信号时，YOLO 的 PD 在约 −3 dB 饱和于约 96%，GoF 在 3 dB 时达到 99.9%（Sec. VI-A, p.6）。多信号场景（Sec. VI-C, p.9）："At lower SNR regions, the GoF sensing showed high false alarm rate compared to the YOLO detection."原因是"if we have multiple signals with different SNR present in the spectrum, it impacts the noise floor estimation. This may result in an overestimation of the threshold."（p.8）
- 局限（p.9）：监督式方法对未知新信号不一定适用。
- 意义：基于统计量的检测器依赖底噪估计，在多信号和低 SNR 下会出问题。这对应我们"手工特征中有底噪/门限相关量"的风险，不过这里比较的是检测器，不是融合特征。

#### 10. Soltani et al., "RF Fingerprinting Unmanned Aerial Vehicles with Non-standard Transmitter Waveforms"
- 刊物：IEEE TVT 2020（本地是 accepted 版本）
- 阅读程度：第 I–III 节全文；第 IV 节只读了多分类器方案的开头。
- 任务：7 台同型号 DJI M100 的个体识别，暗室，单个 USRP X310，原始 IQ 输入 1D CNN。
- 标准化（Sec. III-C, p.3）："we calculate mean µ and standard deviation σ of the training set, in a preprocessing step, and use them later for normalizing the training and test batches"（式 (1)，z-score），测试批次也用训练集统计量（p.4）。
- 关键结果（Sec. III-D, p.4–5）：同一 burst 内随机划分时测试准确率为 97%；"Train on bursts 1, 2, and 3 Test on burst 4 … we get 49% accuracy with AlexNet1D … ResNet1D also yields marginal improvement with 50%"。作者归因于悬停引起的信道变化。
- 融合：第 IV 节用的是多个网络在**决策层**做两级分数聚合，不涉及手工特征。
- 意义：同一数据换一种划分方式，准确率从 97% 掉到 49%。说明随机划分的评估会严重高估性能，也直接支持"按录制段/时间分组划分"的检验方式。z-score 用训练集统计量，正是做法 B 打算采用的做法。

#### 11. Reus-Muns & Chowdhury, "Classifying UAVs With Proprietary Waveforms via Preamble Feature Extraction and Federated Learning"
- 刊物：IEEE TVT, vol. 70, no. 7, 2021
- 阅读程度：摘要 + 第 III 节（基线实验）和第 IV 节（互相关特征图与评估）的相关段落。
- 做法：手工设计的"前导码互相关特征图"（式 (1)–(7)，包括峰均比 γ 的归一化）在**输入层**替代原始 IQ，再送 3D CNN。
- 关键结果：原始 IQ 基线在静止状态下为 61.5%（payload）/85.88%（preamble），悬停状态下降到 56.75%/63.24%（Sec. III-C, p.6283）。互相关特征图"achieves an accuracy of 98.2%"，这是最差组合"static training, hovering testing"（p.6286）。为了排除个体差异，"To ensure that the learning model learns UAV-model-specific instead of device-dependant features, we test the performance of our model with different UAVs used for testing"（Sec. III, p.6283）。
- 意义：（a）这是"用手工设计的表示在输入层去掉信道影响"的例子，不是拼接融合；（b）它的检验方式是"训练和测试用同型号的不同个体"，以及"在静止数据上训练、在悬停数据上测试"，属于捷径检验的同类做法。

#### 12. Fu, Zhang, He, "Radio-Frequency-Based Drone Recognition via Variational Mode Decomposition and Multi-Domain Feature Fusion"
- 刊物：Drones 2026, 10(2):124；doi:10.3390/drones10020124
- 阅读程度：**仅摘要**。MDPI 网站对本机请求返回 403，PDF 和 HTML 都没拿到。
- 摘要称：先对 RF 信号做 VMD，然后"extracts time-domain, frequency-domain and time–frequency-domain features from the modes after VMD"；"a two-stage feature selection scheme based on ReliefF"；"a support vector machine (SVM) is constructed for UAV classification"；在 CardRF 数据集上"an average identification accuracy of over 74.7% at SNRs greater than −10 dB"。检索摘要还列出特征包括 mean、variance、kurtosis、skewness、RMS、Shannon entropy、shape factor、impulse factor。
- 能说的只有一点：摘要显示这里的"多域特征融合"是把手工特征拼接后交给 SVM，**没有深度网络分支**。融合细节、消融数字、归一化方式都没读到，不能下结论。建议用户在校园网下载全文后补读。

#### 13. Shi, Shan, Dong, "Integrated detection of weak drone signals: From parameter estimation to sensitive classification"
- 刊物：Digital Signal Processing, vol. 168, 105524（2026 年 1 月卷，2025 年在线）
- 阅读程度：**仅检索摘要片段**（ScienceDirect 摘要页的搜索摘录，不完整）。
- 摘要称：提出"integrates joint sensitive signal detection and precise classification"的框架；检测模块是"dual-task collaborative architecture that determines weak drone signal presence while concurrently extracting and compactly encoding communication parameters"；目标是研究"RF-based extracted hybrid features for high-sensitivity drone detection, communication parameter estimation, and classification"。
- 没读到具体的特征、融合位置和数字，不据此下任何结论。从题目和摘要看，它和"低 SNR 检测 + 参数特征"高度相关，值得在校内拿到全文。

### B. 调制识别 / 信号识别

#### 14. Han, Ren, Li, Zhu, "Automatic Modulation Classification Based on Deep Feature Fusion for High Noise Level and Large Dynamic Input"
- 刊物：Sensors 2021, 21(6):2117；PMC8003108
- 阅读程度：全文。
- 任务与数据：仿真 10 种数字调制，AWGN，采样率 50 MHz，每段 16384 点，训练 SNR 在 −20…20 dB 内连续随机取值（Sec. 4, Table 3）。
- 融合位置：**特征层拼接**（"all the extracted features are combined to form a joint feature vector"，Sec. 3.3），然后送 PNN（概率神经网络）分类，即**分开训练**：先训练 CNN/SAE 取特征，再训练 PNN。
- 三路特征：
  - F：FFT 谱 → 自建 CNN（Table 2：最后一个全连接层为 64 维 SeLU；卷积层用 dropout 0.5）；
  - W：Welch 功率谱 → 无监督 SAE；
  - S：统计特征（Sec. 3.2.3，式 (15)–(26)），11 维 ζ = [γmax, σa, σφ, σφ̃, σf, C40, C41, C42, C60, C63, C80]，即瞬时幅度谱峰 γmax、中心化瞬时幅度标准差、相位标准差（两种）、瞬时频率（相位导数）标准差，以及 4、6、8 阶累积量。
  - 统计特征的计算对象：复基带 IQ，"after the quadrature demodulation and normalization, with complex value and zero-mean"，xI、xQ ∈ [−1, 1]（式 (12)），即先做幅度归一化。拼接前是否对三路特征再做标准化，**正文没有写**。
- 消融（Sec. 4，Fig. 6，仅定性描述，正文没给具体数字）："the classification performance of CNN-based features is much better than other two types under high SNR (when SNR ≥−2 dB) … the SAE-based features are better for low SNR (when SNR ≤−4 dB). By feature fusing, the methods FW, FS and FWS can improve the overall classification accuracy. However, the WS method is best for the case of extremely low SNR (when SNR≤−14 dB)."摘要称 0 dB 时达到 99.8%。
- 局限（作者，Sec. 5）："more modulation types should be considered"。只有 AWGN 仿真，没有真实接收机。
- 意义：这是"深度特征 + 累积量/瞬时统计量在特征层拼接"的直接例子，并且有 F/W/S/FW/FS/WS/FWS 的组合消融。不过它是分开训练、用 PNN 分类，不是端到端；也没有说明尺度如何对齐。

#### 15. Yin et al., "Deep Meets Classical: A CPU-Efficient Decision-Fusion Model for Automatic Modulation Classification"（FuseAMC）
- 链接：arXiv 2611.05937v1（2026-11-04）。**未经同行评审**。另外它的参考文献 [7][8][13] 是作者自己的、和 AMC 无关的论文，质量需要打折看待。
- 阅读程度：全文。
- 任务与数据：RadioML 2016.10a 子集，11 类，SNR −20…18 dB；每个（调制，SNR）格子分 100 训练/50 验证/200 测试；5 个随机种子。
- 归一化（Sec. III-A）："Each frame is normalized to unit L2 norm across both channels before being passed to any model or feature extractor, so that every method … operates on directly comparable, power-normalized inputs".
- 两路：约 198k 参数的残差 CNN（原始 IQ）；16 维手工特征（"normalized second-, fourth-, and sixth-order cumulants (C20, C21, C40, C41, C42, C60, C63) together with instantaneous amplitude, phase, and frequency statistics"）+ GBT。"The two streams are trained independently and never share parameters or gradients"（Sec. III-B）。
- 融合位置：**决策层**。两路后验拼接成 22 维，再用逻辑回归元分类器，在与测试集不相交的验证集上拟合（Sec. III-C，式 (1)）。
- 关键结果（Table I）：

  | 方法 | Overall | Low（≤0 dB） | High（≥6 dB） |
  |---|---|---|---|
  | Cumulant+SVM | 0.507 | 0.241 | 0.843 |
  | CNN（原始 IQ） | 0.526 | 0.336 | 0.762 |
  | 后验平均 | 0.557 | 0.332 | 0.837 |
  | FuseAMC | 0.558 | 0.337 | 0.835 |

  摘要原文："the CNN is markedly more robust at low SNR, where hand-crafted statistics collapse toward chance, while the cumulant classifier is more accurate at high SNR"。消融（Sec. IV-D）："Dropping the CNN lowers low-SNR accuracy from 33.7% to 23.6% … dropping the GBT lowers high-SNR accuracy from 83.5% to 76.2%"。融合规则之间差别很小："every soft rule … lands within 0.4 points overall"（Table III）。
- 分布偏移（Table V，Sec. IV-F）：在测试时加入载波偏移、衰落、额外噪声后，融合模型的准确率下降 0.08–0.24。并且"under fading a fixed posterior average … is more robust than the tabulated combiner (0.470), whose weights over-trust the CNN once the channel changes"。
- 局限（作者）：只用了缩减的数据集；没有做跨数据集和 OTA 实验。
- 意义：同一篇里有"手工特征在低 SNR 下趋向随机、高 SNR 下更强"的定量证据，还有"融合增益 + 去掉任一路的消融 + 不同融合规则对比"。不过它是决策层融合，并且元分类器在分布偏移下会过度信任某一路。

#### 16. Shebert, Kirk, Buehrer, "Open Set Wireless Signal Classification: Augmenting Deep Learning with Expert Feature Classifiers"
- 链接：arXiv 2302.03749（2023）
- 阅读程度：全文主体（I–V 节）。
- 任务：识别无线标准（LTE/5G/WiFi6/BLE/NB-IoT），开集。
- 融合方式：**级联/决策层**。CNN 先给出 top-2 候选，再只运行对应标准的"专家特征分类器"（同步序列 + CRC 校验）（Sec. IV-E）。
- 归一化（Sec. II-C-3, p.4）原文："The final preprocessing step scales signals between 0 and 1 to ensures that the model does not use the signal power as a feature."
- 增广（Sec. III-D, p.8）：训练时加入 Rayleigh/Ricean 衰落、AWGN、"I/Q power imbalances … randomly chosen between -3 and 3 dB"、"Spurious and DC Components"。这是把"接收机硬件缺陷"当作扰动注入训练数据，让模型对它不敏感。
- SNR 结果（Sec. IV-D, p.12）：专家特征分类器"approach 100% accuracy as the in-band SNR exceeds 20 dB and approach 0% accuracy below -5 dB SNR"。
- 意义：（a）明确把"去掉功率信息"作为预处理目标；（b）把 IQ 不平衡、直流、杂散作为增广，这和我们担心的捷径正好是同一组量；（c）专家特征在低 SNR 下失效。

### C. 偏差、捷径与方法学

#### 17. Hanna, Karunaratne, Cabric, "WiSig: A Large-Scale WiFi Signal Dataset for Receiver and Channel Agnostic RF Fingerprinting"
- 链接：arXiv 2112.15363（2022）
- 阅读程度：I–VI 节全文。
- 数据：174 个 WiFi 发射机、41 个 USRP 接收机，4 天采集。
- 偏差来源（Sec. I, p.1）："captured signals contain confounding factors besides the transmitter fingerprint … receivers suffer from manufacturing variability same as transmitters, hence, the receiver RF fingerprint is also embedded in the signal. Data-driven approaches like deep learning are easily confounded by both the channel and receivers."
- 检验方法（Sec. VI-A, p.6–7，Fig. 10）：两个测试集，"Same Rx"和"Diff. Rx"（5 个训练中完全没出现过的接收机）。结果："using only one Rx for training results in a drop in accuracy from 99% to less than 33%"。训练接收机增多后泛化变好，"the improvement starts to saturate at about 20 Rx"。
- 跨天（Sec. VI-B）：同一天测试 >99%，换一天测试就下降；均衡可以减轻但不能消除下降。
- 功率作为位置指纹（Sec. VI-E, p.8）："This power received by all Rx creates a location fingerprint, which we removed in the previous results by normalization."并且"All signals were normalized to have unit average power prior to using them"（Sec. VI, p.6）。
- 意义：这是"接收机身份会被模型学进去"的最直接实证，检验方法（留出接收机）可以直接照搬。我们只有两台接收机，没法做"留出多台接收机"，见第 4 节的讨论。

#### 18. Geirhos et al., "Shortcut learning in deep neural networks"
- 刊物：Nature Machine Intelligence 2020（arXiv 2004.07780）
- 阅读程度：摘要 + 6.2 节"Detecting shortcuts: towards o.o.d. generalisation tests"。
- 定义（摘要）："Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions".
- 检验建议（Sec. 6.2，arXiv p.12–13）："If model performance is assessed only on i.i.d. test data, then we are unable to discover whether the model is actually acquiring the ability we think it is"。检测分布差异的方法："training a classifier to distinguish samples in dataset A from samples in dataset A’ can reveal a distribution shift."
- 意义：对我们来说可以直接这样做：只用手工特征训练一个分类器去**预测接收机编号**（最好只在背景样本上，或在同一标签内），准确率越高，说明这些特征里的"接收机信息"越多。

#### 19. Wang, Tran, Feiszli, "What Makes Training Multi-modal Classification Networks Hard?"
- 刊物：CVPR 2020（arXiv 1905.12681）
- 阅读程度：摘要、第 1 节、2.1 节。
- 结论（摘要）："the best uni-modal network often outperforms the multi-modal network … multi-modal networks are often prone to overfitting due to their increased capacity. Second, different modalities overfit and generalize at different rates, so training them jointly with a single optimization strategy is sub-optimal."
- 证据：Kinetics 上把各模态特征在最后一层前拼接做"late fusion"，结果低于最好的单模态（Table 1，例如 A+RGB 71.4 对 RGB 72.6）。dropout 和中层拼接只带来很小的提升（Fig. 1）。他们的解法 Gradient-Blending 是给每个模态加一个单独的分类头，按各模态的过拟合/泛化比加权监督信号（Sec. 2，Fig. 2c）。
- 意义：这是方法学上对"嵌入层拼接 + 单一交叉熵端到端训练不一定比单分支好"的证据。和第 6 篇 Open-RFNet 的观察一致。对做法 B 的启示：一定要和"只用 ResNet18"做公平对比，并考虑给手工特征分支加辅助头，或采用分阶段训练。

---

## 2. 针对 5 个问题的综合回答

### 问题 1：深度特征和手工特征在文献里怎么融合？理由和比较是什么？

读到的做法分四类：

1. **输入层**（手工设计的表示代替或作为网络输入）：Reus-Muns 2021 用互相关特征图作 3D CNN 输入，理由是去掉信道影响；Entropy 2021 用的是"预处理后的频谱分段归一化再拼接"。**我读到的文献里，没有一篇是把"时频图展开成一维，再把归一化的标量特征追加在后面"（老师原话那种方式）喂给网络的。**最接近的是 Entropy 2021 的两段频谱直接拼接。它的经验是：两段**各自**归一化以后再拼，拼完的输入再进 MC-DNN，用两个独立的第一隐层分开处理效果更好（97.3% 到 98.1%）。
2. **嵌入层（特征层）拼接**：Han 2021 把 CNN 64 维、SAE、11 维统计特征拼接后送 PNN，分开训练；Open-RFNet 把 ResNet 和 Transformer 两路深度嵌入拼接后接 MLP（两路都是深度特征）。
3. **决策层**：FuseAMC 把两路后验拼接，再用逻辑回归元分类器；Soltani 2020 做多网络分数聚合；Shebert 2023 用 CNN top-2 加专家分类器级联。
4. **不融合，只用手工特征加传统分类器**：Ezuma 2019/2020、Fu 2026（仅摘要）、DroneRFa 4.2 节。

有比较或消融的：
- Han 2021：F/W/S 单独以及两两、三者组合（Fig. 6，正文只有定性描述）。融合后整体准确率最高，但在极低 SNR（≤−14 dB）时最好的是不含 CNN 特征的 WS 组合。
- FuseAMC：决策层融合比最好的单路高 3.2 个点（CNN）和 5.1 个点（累积量）；各种软融合规则之间差别在 0.4 个点以内。
- Entropy 2021：拼接前各自归一化 vs 拼接后，以及单通道 vs 多通道的累加式消融。
- Open-RFNet：闭集上融合（0.994）没有超过单 ResNet（0.9958）；提升只体现在开集。
- Wang 2020（视频领域）：late-fusion 拼接常常不如最好的单模态。

**没读到任何一篇在同一个任务上直接比较"输入层拼接 vs 嵌入层拼接 vs 决策层融合"三种位置的消融。**

### 问题 2：手工特征有哪些，怎么定义，在什么数据上算，怎么归一化、怎么选择？

- 统计矩和形状类（Ezuma）：15 个，算在谱图最大值得到的能量轨迹的瞬态段上。NCA 选出形状因子、峰度、方差三个。会议版只用 4 个（偏度、方差、熵、峰度），峰度权重最高。
- 瞬时量和累积量（Han 2021）：γmax（中心化瞬时幅度的 DFT 峰值）、σa、σφ、σφ̃、σf（解卷绕相位导数的标准差，即瞬时频率标准差）、C40、C41、C42、C60、C63、C80。算在**幅度归一化后的零均值复 IQ**上。FuseAMC 是 16 维，包括 C20–C63 的归一化累积量和瞬时幅度/相位/频率统计量，算在 L2 归一化后的帧上。
- 协议和结构类（DroneRFa）：跳频块带宽、时长、间隔、带宽间隔、图传周期、占空比，算在时频图上。作者自己指出短片段观测不全、同品牌机型区分度不大、远距离时不显著。
- 小波系数类（Medaiyese）：在训练和测试 SNR 不一致时失效。
- 功率归一化：Glüge（按 burst 部分功率归一）、WiSig（单位平均功率）、FuseAMC（单位 L2）、Han（幅度归一化到 [−1, 1]）、Shebert（缩放到 [0, 1]，理由是不让模型用功率）。理由一致，都是为了去掉距离、增益、功率信息。Ezuma 没有说明峰值、RMS 这类绝对量是否做了归一化。
- 特征标准化：唯一明确写"用训练集均值和方差做 z-score，并用于测试"的是 Soltani 2020，但它标准化的是原始 IQ，不是手工特征。**我读到的文献都没有写明手工特征向量本身是 z-score、min-max 还是取对数后送入融合网络。**
- 维度：3–15（Ezuma）、11（Han）、16（FuseAMC）。
- 特征选择：NCA（Ezuma）；ReliefF 两阶段（Fu 2026，仅摘要）。

### 问题 3：融合网络的结构细节（拼接前归一化、维度比例、投影层、dropout、训练方式）

- 拼接前各自归一化：Entropy 2021 有明确理由和消融（见第 7 篇）。Open-RFNet 和 Han 2021 都**没有写**拼接前是否加 BN/LN。
- 维度比例：Han 2021 中 CNN 特征 64 维、统计特征 11 维，SAE 维度没给。Open-RFNet 没给各支维度。**没读到任何关于"512 对 64"这类比例怎么选的讨论或实验。**
- 投影层：Open-RFNet 的位置分支先过 MNLs，拼接后再过 MNLs；Entropy 2021 的 MC-DNN 给每路一个独立的第一隐层。这两者对应做法 B 的"小 MLP 到 64 维"。
- dropout：Han 2021 的 CNN 用 0.5；Basak 的 DRNN 在全连接层后用 50%；Wang 2020 指出 dropout 对 late fusion 的过拟合只有小幅缓解。
- 训练方式：端到端联合训练的有 Open-RFNet（交叉熵加监督对比，作者强调交叉熵会导致"强特征主导梯度"）和 Wang 2020（同样观察到失衡，提出 Gradient-Blending 和辅助头）；分开训练的有 Han 2021（先 CNN/SAE，再 PNN）和 FuseAMC（两路独立，元分类器在独立验证集上拟合）。

### 问题 4：低信噪比下手工特征更有用还是失效？有没有数据？

多数证据指向**失效或明显变弱**：
- FuseAMC：在 ≤0 dB 时累积量加 GBT 为 0.236–0.241，CNN 为 0.336；在 ≥6 dB 时反过来，累积量 0.843 高于 CNN 0.762。两者大约在 0 dB 交叉。
- Ezuma 会议版：真实距离下 10 dB 时已经只有 60–70%（kNN/DA/SVM）。Ozturk 把 Ezuma 的方法概括为 0 dB 约 40%。
- Ozturk：基于包络和瞬态的时序图 CNN 在 0 dB 为 50.1%，谱图 CNN 在 −10 dB 仍有 92.0%。
- Medaiyese：系数类特征在 ≤5 dB 时都低于 60% 门限，训练和测试 SNR 不一致时直接失效。
- Shebert：专家特征在 −5 dB 以下接近 0%。
- DroneRFa：作者承认远距离时特征"不显著"。

反例或部分有用：
- Han 2021：在 ≤−14 dB 的极低 SNR 下，最好的组合是 Welch 谱 SAE 加统计特征（WS），不是 CNN 组合。不过正文只有定性描述，而且是仿真 AWGN 的调制识别。
- 摘要称高阶累积量"insensitive to Gaussian noise"（Han 2021 第 3.2.3 节给出的理论动机）。但这只对足够长的样本成立，估计方差随 SNR 降低而增大。FuseAMC 的数据显示，在 128 点短帧上低 SNR 时并不鲁棒。

结论：在我们 0 到 −15 dB 的范围里，文献预期手工特征的独立判别力会接近随机。融合的收益如果有，更可能出现在高 SNR 或原始数据段。这一点必须分 SNR 报告才能看清，只看总体准确率会掩盖它。

### 问题 5：有没有论文讨论数据集、接收机或环境偏差让特征变成捷径？怎么检验？

有，而且证据很强，但都来自射频指纹或通用机器学习领域，**没有一篇是在无人机"背景 vs 无人机"检测里专门讨论"接收机和标签相关"的**。
- WiSig：接收机指纹会被学进去；留出接收机测试时从 99% 降到 <33%；功率是"位置指纹"，靠归一化去掉。
- Soltani 2020：随机划分 97%，按 burst 留出 49%。
- Reus-Muns 2021：同型号换个体测试，静止数据训练、悬停数据测试。
- Ozturk：幅度取决于距离，"obvious problem"。
- Shebert：归一化"ensures that the model does not use the signal power as a feature"，并把 IQ 不平衡、DC、杂散作为增广。
- Glüge：录了不同增益下的接收机底噪放进 Noise 类。
- Geirhos：只看 i.i.d. 测试发现不了捷径；可以训练分类器区分 A 和 A′ 来检测分布差异。
- FuseAMC Table V：测试时的信道/硬件偏移下，学到的融合权重会"over-trust"某一路。

可以照搬的检验方法：（a）留出接收机、日期或录制段做测试；（b）用特征预测接收机编号（Geirhos 的 A/A′ 思路）；（c）在同一 SNR、同一接收机内分层报告；（d）对 IQ 不平衡、DC、增益做增广或扰动，看预测是否改变。

---

## 3. 做法 B 设计选择对照表

| 做法 B 的设计选择 | 支持的文献 | 反对或做法不同的文献 | 备注 |
|---|---|---|---|
| 在嵌入层拼接（512 + 64），而不是在输入层展开拼接 | Open-RFNet（嵌入拼接 + MLP，同数据集）；Han 2021（特征层拼接）；Entropy 2021（每路独立第一层优于直接合并） | FuseAMC（决策层，各种规则差别 <0.4 点）；Wang 2020（late-fusion 拼接常不如单模态） | 没有任何文献支持把时频图展开成一维再追加标量特征的输入层做法，也没有文献直接比较三种位置 |
| 手工特征先 z-score（用训练集统计量） | Soltani 2020（训练集 μ/σ 用于测试，但对象是 IQ） | — | 文献没写手工特征本身怎么标准化，需要自己定 |
| 先对 IQ 做功率归一化，再算特征 | Glüge、WiSig、FuseAMC、Han、Shebert（理由：去掉功率信息） | Ezuma 没说明峰值/RMS 是否归一化 | 对我们是必须的，因为接收机和标签相关 |
| 手工特征经小 MLP 投影到 64 维再拼接 | Open-RFNet（MNLs）；Entropy 2021（MC-DNN 分路第一层） | Han 2021 直接拼接原始统计向量 | 64 维是否合适没有文献依据 |
| 拼接前各分支做归一化（BN/LN） | Entropy 2021（拼接前各自归一化，否则小值被淹没） | Basak 发现 BN 无帮助（单分支网络） | 嵌入层 BN/LN 的效果文献没有覆盖 |
| 端到端联合训练 | Open-RFNet（交叉熵 + 监督对比） | Wang 2020、Open-RFNet Remark（交叉熵下强分支主导）；Han 2021 和 FuseAMC 都是分开训练 | 建议至少加辅助头或做分阶段训练对比 |
| 特征清单：峰度、峰均比、累积量、瞬时频率标准差等 | Ezuma（峰度、形状因子）；Han 和 FuseAMC（累积量、瞬时统计量） | 低 SNR 失效：FuseAMC、Ezuma、Medaiyese、Shebert | 0 到 −15 dB 下预期接近随机 |
| 占空比、包络周期类特征 | DroneRFa 表 4（图传周期 14 ms、占空比） | DroneRFa 作者：短片段观测不全 | 20 ms 片段只覆盖 1–2 个周期 |
| 特征选择 | Ezuma（NCA，15 维到 3 维几乎不掉点）；Fu 2026（ReliefF，仅摘要） | — | 选择应在训练集内做 |

---

## 4. 文献没有覆盖、需要我们自己实验决定的地方

1. 三种融合位置（输入层展开拼接、嵌入层拼接、决策层）在同一任务上的直接比较。没有任何文献做过。老师的方案和做法 B 都需要我们自己跑对比。
2. 手工特征向量本身的标准化方式（z-score、min-max、先取对数）以及重尾特征（峰度、峰均比）的处理。
3. 拼接前是否对 512 维嵌入和 64 维特征分别做 BN/LN，以及 512:64 的比例和投影维度怎么选。
4. "接收机和标签强相关"的二分类检测场景下，手工特征会不会成为捷径。文献只在射频指纹里证明过接收机指纹会被学到。我们只有两台接收机，没法像 WiSig 那样留出多台。可行的替代办法：（a）用手工特征预测接收机编号（只用背景样本，或在同一标签内）；（b）如果每台接收机都既有背景又有无人机数据，就分接收机报告指标，并做"训练用 A、测试用 B"的交叉测试；（c）对增益、DC、IQ 不平衡做扰动，看预测是否变化。
5. 20 ms 片段上占空比、包络周期这类特征的估计稳定性。
6. 低 SNR（0 到 −15 dB）下融合带来的增益是否为正。文献预期手工特征的独立判别力接近随机，所以融合可能只是增加噪声输入，必须分 SNR 报告。
7. 在多 SNR 混合训练的条件下，手工特征分布随 SNR 的漂移会不会让网络把它当成"SNR 指示器"，而不是判别特征。

---

## 5. 获取记录

- 本机 WebFetch 对所有域名（mdpi.com、export.arxiv.org、ieeexplore 等）都返回"无法验证域名"，所以改用 curl 下载。arXiv（export.arxiv.org）和 Europe PMC 全文 XML 可以下载；mdpi.com 和 mdpi-res.com 返回 403；Hindawi（Tong 2022 SEI）返回 403；Semantic Scholar API 第二次请求被限流（429）。
- 下载文件放在 `C:C:/Users/Administrator/AppData/Local/Temp/litdl_20261110/`（临时目录，不在项目内）。
- 建议用户在校园网补读：Fu et al. Drones 2026（全文）、Shi et al. DSP 2026（全文）、M015 JESTCH 2022、Tong et al. 2022（WCMC，doi:10.1155/2022/9342085，检索结果显示它把手工特征分三个通道输入 CNN，并有组合消融；全文没拿到，**未读，不引用数字**）。
