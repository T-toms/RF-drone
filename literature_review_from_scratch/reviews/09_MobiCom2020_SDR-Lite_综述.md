# SDR Receiver Using Commodity WiFi via Physical-Layer Signal Reconstruction

## 基本信息

- 作者：Qiben Yan、Yung Yi、Song Min Kim
- 来源：ACM MobiCom 2020
- DOI：10.1145/3372224.3419189
- 类型：低成本接收系统；无人机检测只是七个应用之一
- [源 PDF](<../04_MobiCom2020_SDR_Lite_Physical_Layer_Reconstruction.pdf>)

## 核心贡献

SDR-Lite 不改 Wi-Fi 网卡硬件，而由另一个发射端伪造一个可被网卡接受的 Wi-Fi PHY 头，使网卡开始“解码”随后到来的环境信号。系统再逆转扰码、卷积编码、交织、调制和 IFFT 等过程，从输出比特重建环境 I/Q。重建波形与 USRP 原信号的相关性超过 85%，采样率约 17 MS/s，明显高于普通 CSI 的包级观测。

## 无人机实验

作者选择 Intel Aero、DJI Mavic Pro 与 3DR Solo 三种无人机，使用 1D CNN 做两项任务。存在性检测用 Aero、Solo 各 400 个训练样本和 400 个噪声样本，测试还包含另一台 Aero 和训练未见的 Mavic Pro；SDR-Lite 最终约 98%，接近 USRP 的 99.6%。三机型分类每类 450 个训练、200 个测试样本，USRP 达 99.3%，SDR-Lite 达 93.7%；重建失真使相似的 Aero 与 Mavic Pro 更难区分。

## 权威性与相关性

MobiCom 是移动计算/无线系统领域顶级会议，这篇论文的系统贡献非常强。但它不是专门的无人机 RF 识别论文；无人机只是用于证明通用物理层重建能力的应用。因此，它适合作为“接收端成本与城市基础设施复用”的权威依据，而不应作为无人机识别算法 SOTA 的主要证据。

## 局限

- 依赖 2.4 GHz Wi-Fi 网卡、特定 OFDM 模式、伪头触发器和网卡可输出的解码比特，频段与信号长度受限。
- 重建引入幅相畸变，细粒度同型号个体 RFF 可能比机型分类受影响更大。
- 无人机样本与类别较少，未覆盖低 SNR、远距离、多源重叠和开放集系统评估。
- 需要额外 Initiator 发射伪头，并非完全被动的单盒接收器。

## 对本项目的价值

若项目目标包含低成本部署，可把 SDR-Lite 作为第二条系统创新路线：先用标准 USRP 训练稳定算法，再研究特征在波形重建失真下能否迁移。若当前目标是先做算法论文，则它不是最短路径。
