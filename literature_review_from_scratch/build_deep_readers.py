from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "deep_research_redo" / "readers"
EVIDENCE = ROOT / "deep_readers" / "_evidence"


PAPERS = [
    {
        "id": "01_DroneRFa",
        "title": "DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集",
        "source": "《电子与信息学报》46(4), 2024；数据集与基线论文",
        "pdf": "DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf",
        "pages": 10,
        "evidence": "01_dronerfa_key_evidence.png",
        "thesis": "论文的主要贡献是把无人机通信 RF 的类别、频段、距离和原始 I/Q 组织为较大规模公开数据，而不是证明某个网络已经解决真实泛化。",
        "task": "24 个无人机/飞控器类别加背景类的检测与机型/链路类别识别；不等同于 24 个同型号物理个体的 RFFI。",
        "data": "NI USRP-2955，100 MS/s、14 bit、双通道；915 MHz、2.4 GHz、5.8 GHz。户外近/中/远约 20–40、40–80、80–150 m，室内约 2 m。每类至少 12 个独立片段，每片段不少于 1 亿采样点。",
        "method": "先用跳频块带宽、时长、间隔、图传周期和占空比做规则观察；再把 100 万点（约 10 ms）变为双通道 STFT，输入 ResNet18 做 25 类闭集分类。",
        "results": "随机 6:2:2 窗口切分下约 97.73%，推理约 53 fps；减少时间跨度比降低频率分辨率造成更明显下降。",
        "audit": "相邻 10 ms 窗口可来自同一长片段/会话，随机切分可能泄漏；室内与户外类别、中心频率和接收条件并非完全对称；采—存间断使长周期研究不能假设连续流。",
        "counter": "若按原始片段/会话/距离分组后仍保持近似性能，泄漏担忧会减弱；否则原 97.73% 只能看作同分布窗口识别上限。",
        "project": "作为机型识别与跨距离基准；先重建 group split，并把绝对频率、功率与场景作为待审计混杂变量。",
        "quotes": [
            (1, "本文公开一个面向低空无人机侦测的大规模无人机射频信号数据集 DroneRFa。", "The paper releases DroneRFa, a large-scale UAV RF dataset for low-altitude drone sensing."),
            (4, "接收端采用 USRP-2955，以 100 MS/s 对无人机与飞控器通信链路进行双通道采集。", "A USRP-2955 receiver records the UAV–controller links with two channels at 100 MS/s."),
            (8, "基于时频图和 ResNet18 的分类实验用于验证数据可分性。", "A spectrogram–ResNet18 experiment is used as a separability baseline for the dataset."),
        ],
        "repro": ["从文件名/头部恢复类别、距离、片段和通道键", "按原始片段分组，不让相邻窗口跨集合", "统一每类频带与功率处理", "分别报告室内、户外和距离域", "记录采—存断点对长窗的影响"],
    },
    {
        "id": "02_OpenRFNet",
        "title": "Multi-Domain Supervised Contrastive Learning for UAV Radio-Frequency Open-Set Recognition",
        "source": "IEEE Journal on Selected Areas in Communications, Vol. 44, 2026, pp. 4083–4098；DOI 10.1109/JSAC.2026.3669136。本地 PDF 为 arXiv:2508.12689v2 版本。",
        "pdf": "02_JSAC2026_Multi_Domain_Contrastive_RF_Open_Set.pdf",
        "pages": 15,
        "evidence": "02_openrf_key_evidence.png",
        "thesis": "Open-RFNet 将多域时频表示、监督对比学习、生成式代理未知和极值开放集决策连成完整流程；它是有价值的方法候选，但当前证据不足以支持跨距离/跨接收机泛化。",
        "task": "DroneRFa 25 类中的 20 已知/5 未知开放集识别；同时优化 known accuracy 和 unknown rejection。",
        "data": "只选室内与户外近距离；每个 I/Q 片段 3 ms、30 万点，STFT 约 775×775。中远距离因弱信号被排除。",
        "method": "ResNet 纹理分支 + Transformer 时频位置分支，经 MD-SupContrast 学表示；WGAN-GP 生成已知分布边缘样本并把误分类者作为代理未知；冻结编码器后重训头，IG-OpenMax 以 Weibull 尾部拟合输出未知概率。",
        "results": "报告 KAR 95.12%、UAR 96.08%；去噪对 UAR 有巨大影响。模型约 2.06 亿参数、约 54.21 ms，对照轻量 ResNet 约 14.41 ms。",
        "audit": "只有一个 20/5 划分；代理未知不保证代表新协议/新干扰；近距离选择削弱低 SNR 外推；若 3 ms 窗口未按原始文件分组，会继承 DroneRFa 泄漏；发表状态必须按 arXiv 处理。",
        "counter": "需要五组以上未知类留出、文件分组、远距离/真实干扰、跨接收机和独立校准。若增益只在去噪后的单划分出现，则方法结论很窄。",
        "project": "只把它当开放集上限对照。优先实现小模型 CE/SupCon + energy/prototype/EVT，再判断生成式未知和巨型双分支是否必要。",
        "quotes": [
            (1, "Unauthorized flight incidents occur frequently, posing a severe security threat to LA-ISAC networks.", "未经授权的飞行事件频繁发生，对低空通感网络构成严重安全威胁。"),
            (1, "We propose a multi-domain supervised contrastive learning framework for UAV RF open-set recognition.", "论文提出面向 UAV RF 开放集识别的多域监督对比学习框架。"),
            (14, "The current framework primarily considers a single UAV emitter.", "当前框架主要考虑单一 UAV 发射源。"),
        ],
        "repro": ["核实正式发表信息，引用时保留 arXiv 状态", "从 DroneRFa 原始文件生成 group split", "复现固定 20/5 后增加多组留出", "阈值仅用校准未知，不触碰测试未知", "报告参数、吞吐、窗口延迟和去噪消融"],
    },
    {
        "id": "03_LowSNR",
        "title": "Robust Low-Cost Drone Detection and Classification Using CNNs in Low SNR Environments",
        "source": "IEEE Journal of Radio Frequency Identification 8, 2024；DOI 10.1109/JRFID.2024.3487303",
        "pdf": "02_Robust_Low-Cost_Drone_Detection_Low_SNR_2024.pdf",
        "pages": 11,
        "evidence": "03_lowsnr_key_evidence.png",
        "thesis": "这篇论文最重要的不是比较 VGG 深度，而是给出低 SNR 崩溃区间、低成本端到端原型和最长约 670 m 的现场测试。",
        "task": "6 个无人机/控制器类别与噪声类的检测/分类；强调低 SNR 与现场可部署性。",
        "data": "USRP B210 + 对数周期天线；消声室采信号并记录 Wi-Fi、Bluetooth、接收机/放大器噪声，再加入 AWGN。56 MHz 降采样至 14 MS/s，1,048,576 点约 74.9 ms 一窗，SNR −20 到 30 dB。",
        "method": "连续 FFT 形成双通道频谱图，比较 BatchNorm VGG11/13/16/19；分层五折；现场系统为普通 GPU 笔记本、B210 与 8.5 dBi 定向圆极化天线。",
        "results": "开发集 balanced accuracy 约 93.2%–93.9%，各深度接近；SNR 高于 −12 dB 时平均至少 85%，到 −20 dB 接近随机。110/340/560/670 m 场测约 79.2%–81.1%。",
        "audit": "实验室信号后混噪声不能覆盖空口近远效应、频选衰落、饱和和互调；单一现场；若窗口跨折仍可能泄漏；窗口分类不能估计同时发射源数量。",
        "counter": "在独立会话、真实共信道干扰和第二地点重复相同 SNR 曲线，才能把 −12 dB 视为稳定工作边界。",
        "project": "定义工程验收线：balanced accuracy–SNR 曲线、真实干扰、误报/小时、每窗与端到端延迟；以 VGG11/轻量 ResNet 作基线。",
        "quotes": [
            (1, "We focus on the robustness of the models to low SNRs, as this is the most relevant aspect for a real-world application of the system.", "作者把低信噪比鲁棒性视为现实系统最关键的方面。"),
            (1, "We evaluate a low-cost drone detection system consisting of a standard computer, SDR, and antenna in a real-world field test.", "作者在真实场测中评估由普通计算机、SDR 和天线组成的低成本系统。"),
            (8, "All models achieved an average balanced classification accuracy of at least 85% for SNRs above approximately −12 dB.", "在约 −12 dB 以上，各模型的平均平衡准确率至少为 85%。"),
        ],
        "repro": ["保留 14 MS/s 与约 75 ms 基线", "按原始记录分组五折", "分开 AWGN、真实干扰和纯背景", "报告 SNR 分层置信区间", "场测记录距离、方向、增益和误报"],
    },
    {
        "id": "04_TCCN2022",
        "title": "Combined RF-Based Drone Detection and Classification",
        "source": "IEEE Transactions on Cognitive Communications and Networking 8(1), 2022；DOI 10.1109/TCCN.2021.3099114",
        "pdf": "Combined_RF_Based_Drone_Detection_Classification_2022.pdf",
        "pages": 11,
        "evidence": "04_tccn_key_evidence.png",
        "thesis": "论文把‘有没有信号、信号在哪、是什么’放在同一系统比较，是从窗口分类走向多源频谱监测的关键主干文献。",
        "task": "2.4 GHz 宽带 RF 中的单/多信号检测、时频定位和 10 类分类。",
        "data": "消声室，USRP X310、100 MS/s、约 7 m；9 个商业无人机/控制器源加 Wi-Fi 类。低 SNR 通过 AWGN，多信号主要通过实测信号数字叠加。",
        "method": "方案 A：Anderson–Darling GoF 盲检测后，256×256 时频图送 DRNN；方案 B：YOLO-Lite 一步输出存在性、时频边界框和类别。",
        "results": "单信号约 −3 dB 达到约 97% F1；3 dB 时 DRNN 约 99%。YOLO 在多信号定位更稳，DRNN 分类略强。YOLO 约 0.552 ms/样本但约 4154 万参数，DRNN 约 1.89 ms、约 46.9 万参数。",
        "audit": "真实 OTA 多源中的时钟差、功率近远、接收机非线性与互调未被数字混合覆盖；YOLO 是闭集监督器，会把未知协议强制分类；不同物理设备/会话泛化不足。",
        "counter": "用 DRFF-R2 真实双/三机和未知干扰验证实例级 mAP、未知召回与功率不平衡，才能支持真实多源主张。",
        "project": "采用盲候选检测 + 监督实例定位 + 开放集分类的级联系统；保留 GoF 作为未知 RF 入口。",
        "quotes": [
            (1, "The presence of a drone should be detected and classified beforehand.", "在定位或处置之前，必须先检测并分类无人机。"),
            (1, "Detecting and classifying a drone automatically is a challenging task.", "自动检测与分类无人机是一个具有挑战性的任务。"),
            (9, "The YOLO-based framework can jointly detect, localize, and classify multiple signals in the time-frequency plane.", "基于 YOLO 的框架可以在时频平面联合检测、定位并分类多个信号。"),
        ],
        "repro": ["复现 GoF 阈值和 DRNN 分类", "标清信号级 SNR 而非混合窗平均 SNR", "区分数字混合与 OTA 多源", "增加未知协议/干扰类", "分别报告检测 mAP、分类 F1 与端到端误报"],
    },
    {
        "id": "05_TVT2020_RFF",
        "title": "RF Fingerprinting Unmanned Aerial Vehicles With Non-Standard Transmitter Waveforms",
        "source": "IEEE Transactions on Vehicular Technology 69(12), 2020；DOI 10.1109/TVT.2020.3042128",
        "pdf": "05_TVT2020_RF_Fingerprinting_Nonstandard_UAV_Waveforms.pdf",
        "pages": 14,
        "evidence": "05_tvt20_key_evidence.png",
        "thesis": "这篇论文用同型号 7 台 M100 直接证明：随机混合相邻 I/Q 切片产生的高准确率，在跨突发信道变化下会严重崩溃。",
        "task": "同型号物理 UAV 个体识别和训练未见个体检测，不是机型分类。",
        "data": "7 台 DJI M100，USRP X310，消声室悬停，6/9/12/15 ft；每距离 4 个相隔约 10 s 的突发组，共 >13k 样本，平均约 9.2 万复数点。",
        "method": "AlexNet1D/ResNet1D 输入长度 200 的 I/Q；把训练数据拆为不重叠子集训练 12 个 AlexNet1D，两级分数/多数投票；随机复高斯 FIR 进行信道增强。",
        "results": "随机近邻切片 AlexNet1D 约 97%；前三突发训练、第四突发测试时约 49%（ResNet 约 50%）；多分类器到 91%，再加增强到 95%；未知个体从 68% 到约 99%。",
        "audit": "只有一个型号、7 台设备、单场地和单接收机；随机 FIR 是粗信道模型；阈值与集成规则需独立校准；训练 12 个网络成本较高。",
        "counter": "跨日、跨接收机和真实运动多径若仍能保持提升，才能把方法称为设备指纹而非某次接收链特征。",
        "project": "把跨突发/会话留出设为 RFFI 的最低门槛；可用单编码器、多域增强和原型头替代 12 网络，但不能删除严格切分。",
        "quotes": [
            (1, "We propose RF fingerprinting for this purpose, where a neural network learns subtle imperfections present in the transmitted waveform.", "论文用神经网络学习发射波形中的细微硬件不完美，以实现射频指纹识别。"),
            (1, "Since the wireless channel changes constantly, the network trained with a previously collected dataset performs poorly on the test data.", "由于无线信道持续变化，先前采集数据训练的网络在测试数据上表现很差。"),
            (13, "The robust multi-classifier approach achieves an overall accuracy of 95% under an unseen channel.", "鲁棒多分类器在未见信道条件下达到约 95% 总体准确率。"),
        ],
        "repro": ["按 burst/session 建组", "保留同型号多物理个体", "比较随机 split 与跨突发 split", "单独验证 FIR 增强和集成贡献", "未知个体轮流留一并独立校准阈值"],
    },
    {
        "id": "06_TVT2021_Preamble",
        "title": "Classifying UAVs With Proprietary Waveforms via Preamble Feature Extraction and Federated Learning",
        "source": "IEEE Transactions on Vehicular Technology 70(7), 2021；DOI 10.1109/TVT.2021.3081049",
        "pdf": "12_TVT2021_Proprietary_Waveforms_Preamble_Federated_Learning.pdf",
        "pages": 12,
        "evidence": "06_tvt21_key_evidence.png",
        "thesis": "论文把不可解码专有波形中的前导码结构转成互相关特征图，为‘通信物理先验 + 深度学习’提供了最直接的 UAV-RF 证据。",
        "task": "4 种 DJI 机型/6 台设备的型号识别、一个新型号检测，以及非 IID 联邦学习模拟。",
        "data": "M100×2、M600×2、Phantom 3、Mavic Pro 2；USRP B210；4/7/10/13 ft；每 UAV–距离 4 组、每组 2 s；室内静态与室外悬停，共 8.9 GB。",
        "method": "搜索周期性候选前导码，切成子片段并构造跨相关特征图，交给 CNN；未知类用 softmax 向量到已知参考集的欧氏距离；联邦部分为 FedAvg + 少量共享数据 + 加权交叉熵。",
        "results": "悬停训练/测试 99.97%；室内静态训练→室外悬停测试 98.2%；一个 DJI Air 2 未知类约 99%。40 用户非 IID 模拟中，共享总数据 0.5%/1% 后约 99.28%/99.51%。",
        "audit": "已知类全 DJI、距离短、未知仅一类；前导码搜索在新协议/低 SNR 下可能失败；联邦为离线模拟，未覆盖真实接收机异质、通信开销和攻击。",
        "counter": "跨厂商、跨接收机、远距离和多未知类条件下，前导码分支若仍提供增益，才证明先验可泛化。",
        "project": "优先复现随机 I/Q、整帧 STFT、前导码相关图三种输入；把前导码可检测率也作为指标，而不是只统计成功裁剪后的分类。",
        "quotes": [
            (1, "This paper proposes a deep learning method to detect the type/model of the UAV using the transmitted RF signals, even when these signals follow proprietary medium access protocols.", "即使 RF 信号采用专有接入协议，论文仍尝试从发射信号识别 UAV 类型/型号。"),
            (1, "The preamble portion of the packet is better suited for learning subtle protocol-specific differences.", "包的前导码部分更适合学习细微的协议特定差异。"),
            (1, "The model trained on static data achieves 98.2% accuracy when tested under hovering conditions.", "在静态数据上训练的模型，在悬停条件测试时达到 98.2%。"),
        ],
        "repro": ["公开前导码候选定位规则", "按采集组/环境分离训练测试", "统计定位失败样本", "多个未知型号轮流留出", "联邦实验加入真实接收机域与通信成本"],
    },
    {
        "id": "07_DRFF_R2",
        "title": "A Multi-Scenario UAV RF Dataset with Real-World Acquisition and Signal Processing Benchmarking",
        "source": "arXiv:2603.00106v1；数据 DOI 10.57760/sciencedb.36815；当前按数据手稿处理",
        "pdf": "DRFF_R2_Multi_Scenario_UAV_RF_Dataset_2026.pdf",
        "pages": 12,
        "evidence": "07_drffr2_key_evidence.png",
        "thesis": "DRFF-R2 的价值是把多日、双接收机、多状态、多 UAV 和 Wi-Fi 混合放进统一元数据，而不是其单一 EfficientNet 基线。",
        "task": "8 种 DJI 机型、26 台物理设备的状态/机型/个体识别，并支持跨域、干扰与多源研究。",
        "data": "两台 USRP-2943、5 dBi 双频全向天线、100 MS/s；每段约 1.4 亿复数样本。户外 5.745 GHz，室内 Wi-Fi 混合 2.437 GHz。七组子集覆盖飞行状态、多机、多日/接收机、双频/跳频、吸波材料、Wi-Fi 混合和背景。",
        "method": "Hann 窗 STFT，FFT 1024、hop 256 展示数据；在 dataset-1 用 EfficientNet-B0 做 26 个细粒度状态/设备组合的可分性验证。",
        "results": "论文提供时频可视化和混淆矩阵，但没有形成完整的跨日、跨接收机、单机→多机、洁净→干扰定量基准。",
        "audit": "全为 DJI；部分机型只有 1–2 台，型号与个体难解耦；不同场景中心频率不同，频段可能成为捷径；当前同行评审状态不明。",
        "counter": "如果元数据完备且能从同一类别构造接收机/日期留出，它是最适合证伪域泛化主张的数据；否则只能用作场景展示。",
        "project": "优先构造 USRP1→USRP2、日 1→日 2、单机→双/三机、洁净→Wi-Fi、已知→未知个体五个协议。",
        "quotes": [
            (1, "We present a real-world multi-scenario unmanned aerial vehicle radio frequency dataset, namely DRFF-R2.", "论文发布真实世界多场景无人机射频数据集 DRFF-R2。"),
            (1, "The dataset is systematically organized into seven well-defined subsets corresponding to different operational and signal composition scenarios.", "数据被系统组织为七个子集，对应不同运行与信号组合场景。"),
            (9, "The benchmark visualizations demonstrate distinct time-frequency structures under single-UAV, multi-UAV, and Wi-Fi-mixed conditions.", "基准可视化展示单 UAV、多 UAV 和 Wi-Fi 混合条件下不同的时频结构。"),
        ],
        "repro": ["先核对许可证、数据 DOI 和手稿版本", "建立 device/model/date/receiver/frequency 键", "防止频段与类别绑定", "每个跨域协议单独报告", "不把 EfficientNet 随机划分当最终基线"],
    },
    {
        "id": "08_RFF_LLM",
        "title": "UAV Individual Identification via Distilled RF Fingerprints-Based LLM in ISAC Networks",
        "source": "IEEE Wireless Communications Letters 14(11), 2025, 3769–3773",
        "pdf": "UAV_Individual_Identification_RFF_LLM_2025.pdf",
        "pages": 6,
        "evidence": "08_llm_key_evidence.png",
        "thesis": "论文真正可迁移的贡献是动态知识蒸馏与轻量学生，而不是‘LLM’这个名称；其高准确率仍受数据划分独立性不清的限制。",
        "task": "DRFF-R1 中 7 型号、20 台 UAV 的物理个体闭集识别与边缘部署。",
        "data": "USRP B210；户外高度 10/30/50/70/90 m，0 m 吸波材料参考；STFT 后约 2 万样本，6:2:2 划分，但短文未充分说明是否按会话/日期分组。",
        "method": "修改 GPT-2 的 Transformer/LSTM 教师 RFF-LLM；Lite-HRNet 学生；CE + KL 蒸馏；PPO 根据训练状态动态控制温度。",
        "results": "教师 97.17%，无蒸馏学生 95.28%，动态蒸馏学生 98.38%；学生 156,064 参数、约 2.74 ms。最佳固定温度约 96.62%。",
        "audit": "GPT-2 骨架不意味着语言语义；6:2:2 可能随机窗口泄漏；单接收机、无开放集；PPO 是否优于简单温度日程缺少足够强消融。",
        "counter": "在跨会话/接收机和开放集条件下，普通 Transformer/TCN、固定/余弦温度与 PPO 对照后仍有增益，才能支持动态 KD 主张。",
        "project": "放到主方法验证后的压缩阶段；教师必须先在严格跨域开放集上可靠，学生同时蒸馏表示与校准而非只蒸馏 logits。",
        "quotes": [
            (1, "We propose a novel dynamic knowledge distillation-enabled wireless radio frequency fingerprint large language model framework for UAV ID identification.", "论文提出动态知识蒸馏驱动的 UAV 个体射频指纹‘大语言模型’框架。"),
            (1, "Radio-frequency fingerprint originates from hardware imperfections during the device manufacturing processes.", "射频指纹源于设备制造过程中形成的硬件不完美。"),
            (6, "The lightweight student achieves 98.38% identification accuracy with 156,064 parameters.", "轻量学生以 156,064 个参数达到 98.38% 识别准确率。"),
        ],
        "repro": ["核实 split 是否按会话/个体分组", "复现普通教师与无 PPO 温度日程", "报告训练成本而非只报推理", "增加未知个体拒识", "检验蒸馏后的 ECE/OSCR"],
    },
    {
        "id": "09_SDR_Lite",
        "title": "SDR Receiver Using Commodity WiFi via Physical-Layer Signal Reconstruction",
        "source": "ACM MobiCom 2020；CCF-A；DOI 10.1145/3372224.3419189",
        "pdf": "04_MobiCom2020_SDR_Lite_Physical_Layer_Reconstruction.pdf",
        "pages": 14,
        "evidence": "09_sdrlite_key_evidence.png",
        "thesis": "这是高质量低成本接收系统论文：通过伪造 Wi-Fi PHY 头触发网卡并逆向重建环境 I/Q；无人机只是七个应用之一。",
        "task": "通用物理层信号重建；UAV 子实验包括存在检测和 3 机型分类。",
        "data": "普通 Wi-Fi 网卡接收重建波形，约 17 MS/s；对照 USRP。UAV 包含 Intel Aero、DJI Mavic Pro、3DR Solo，训练/测试样本规模较小。",
        "method": "Initiator 发送可被网卡接受的 PHY 头；网卡输出随后环境信号的解码比特；系统逆转扰码、卷积编码、交织、调制和 IFFT 重建 I/Q，再用 1D CNN。",
        "results": "重建波形与 USRP 相关性 >85%；UAV 存在检测约 98%（USRP 99.6%），三机型分类约 93.7%（USRP 99.3%）。",
        "audit": "依赖 2.4 GHz Wi-Fi 与额外 Initiator，不是完全被动单盒；重建幅相失真可能破坏细粒度 RFF；UAV 类别、低 SNR、多源与开放集覆盖有限。",
        "counter": "若标准 USRP 训练的表征能在重建波形上少量适配，并保留跨域/开放集性能，才构成 UAV-RF 低成本部署证据。",
        "project": "作为第二条系统创新：先建立可靠算法，再研究接收硬件替代与域迁移；不把它放在识别算法主线最前。",
        "quotes": [
            (1, "Physical-layer signal analysis has become critically beneficial across distinctive domains including security and privacy.", "物理层信号分析对安全与隐私等多个领域具有重要价值。"),
            (1, "SDR is known to be highly effective, but it is rarely deployed or used at scale due to cost.", "SDR 很有效，但因成本很少被大规模部署。"),
            (12, "For drone classification, SDR-Lite achieves 93.7% while the USRP baseline achieves 99.3%.", "无人机分类中，SDR-Lite 为 93.7%，USRP 对照为 99.3%。"),
        ],
        "repro": ["复现 PHY 头触发与重建链", "量化重建幅相误差", "把硬件域作为明确变量", "比较 USRP→SDR-Lite 迁移", "UAV 实验扩展到开放集与同型号个体"],
    },
    {
        "id": "10_ResNet",
        "title": "Deep Residual Learning for Image Recognition",
        "source": "CVPR 2016；CCF-A；通用视觉基础论文，非 UAV-RF",
        "pdf": "03_Deep_Residual_Learning_ResNet_CVPR2016.pdf",
        "pages": 9,
        "evidence": "10_resnet_key_evidence.png",
        "thesis": "ResNet 通过学习 F(x)+x 的残差映射缓解深层网络优化退化；它支持选用稳定基线，但不证明任何 UAV-RF 泛化或开放集结论。",
        "task": "ImageNet/CIFAR/COCO 图像识别与检测，不涉及无线信号。",
        "data": "ImageNet、CIFAR-10、COCO；与 UAV-RF 的采样、信道、频偏和接收机问题无直接对应。",
        "method": "恒等 shortcut 令若干层学习残差 F(x)，输出 y=F(x)+x；18/34 层基本块和 50/101/152 层瓶颈结构。",
        "results": "34 层残差网优于 plain 深网；152 层集成在 ILSVRC 2015 达 3.57% top-5 error；CIFAR 上 1202 层可优化但泛化不优于 110 层。",
        "audit": "把 STFT 当普通图像可能引入不合适的平移不变性；ImageNet 预训练的收益需单独消融；更深不自动更好。",
        "counter": "若同参数量 1D-IQ 或频率感知网络在严格域留出明显更好，ResNet 只应保留为参考基线。",
        "project": "使用 ResNet18 作为 2D-STFT 基线，与参数量匹配的 1D 残差网比较；创新必须来自数据、物理先验、开放集或跨域。",
        "quotes": [
            (1, "Deeper neural networks are more difficult to train.", "更深的神经网络更难训练。"),
            (1, "We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions.", "作者把层改写为相对于输入学习残差函数，而不是直接学习无参照映射。"),
            (6, "These results suggest that the degradation problem is well addressed and we manage to obtain accuracy gains from increased depth.", "结果说明退化问题得到缓解，增加深度可以带来准确率收益。"),
        ],
        "repro": ["只复现 ResNet18/34 作为稳定基线", "明确输入通道和归一化", "不默认使用 ImageNet 预训练", "与同参数量 1D 模型比较", "报告深度、参数和计算量的公平消融"],
    },
]


def render_paper(p: dict) -> str:
    quote_blocks = []
    block_ids = []
    for idx, (page, original, translation) in enumerate(p["quotes"], 1):
        sid = f"S{idx:03d}"
        block_ids.append((sid, page))
        quote_blocks.append(
            f"### {sid} · p.{page}\n\n"
            f"**Source / 原文**\n\n> {original}\n\n"
            f"**Translation / 对照译文**\n\n{translation}\n"
        )
    repro = "\n".join(f"- [ ] {x}" for x in p["repro"])
    return f"""# {p['title']}

> **Reader status:** draft deep reader（方法、数据与核心结果完整覆盖；不是逐句全文翻译）  
> **Source:** {p['source']}  
> **PDF:** [打开源 PDF](../../../{p['pdf']})  
> **Length:** {p['pages']} pages

## 核心判断

{p['thesis']}

## 任务定义

{p['task']}

## 数据与独立实验单位

{p['data']}

这里真正应视为统计独立的单位不是任意窗口，而是设备、原始突发/文件、采集会话、日期或接收机。若论文没有把这些单位隔离，窗口级准确率只能解释为同分布可分性。

## 方法重构

{p['method']}

## 核心结果

{p['results']}

![关键结果或图表摘录](assets/key_evidence.png)

> 图像来自源 PDF 的对应结果页，经裁剪用于定位证据；字号/字体渲染存在轻微替代，数值引用应回源 PDF 核对。

## 源文—译文锚点

{''.join(quote_blocks)}
## 关键主张审计

### 论文支持什么

{p['thesis']}

### 论文没有证明什么

{p['audit']}

### 最小反证实验

{p['counter']}

## 对本项目的角色

{p['project']}

## 复现清单

{repro}

## 阅读时要追问的六个问题

1. 类别是协议族、型号还是同型号物理个体？
2. 窗口是否跨越同一原始记录并被分到不同集合？
3. 信道、接收机、频率、功率和背景是否与类别绑定？
4. 低 SNR/多源是真实空口还是后期数字混合？
5. 未知类阈值是否接触了测试未知数据？
6. 结果能否在第二会话、第二接收机或第二数据集重现？

## 覆盖索引

- 摘要/问题：已覆盖
- 数据采集与预处理：已覆盖
- 模型/算法：已覆盖至可复现层级；复杂公式以源 PDF 为准
- 核心实验、表格和图：已覆盖关键页
- 局限与外推边界：已做独立审计
- 参考文献：结构性索引，不逐条翻译
"""


def source_map(p: dict) -> dict:
    blocks = []
    for idx, (page, original, translation) in enumerate(p["quotes"], 1):
        blocks.append({
            "block_id": f"S{idx:03d}",
            "type": "paragraph_excerpt",
            "page": page,
            "section": "source-translation anchor",
            "source_text": original,
            "translation": translation,
            "confidence": "medium-high",
        })
    blocks.append({
        "block_id": "F001",
        "type": "figure_or_table_crop",
        "page": next((x[0] for x in p["quotes"][::-1]), None),
        "asset": "assets/key_evidence.png",
        "confidence": "high for placement; verify numerical transcription in source PDF",
    })
    return {
        "title": p["title"],
        "source_pdf": p["pdf"],
        "reader_status": "draft deep reader",
        "coverage": {
            "abstract": True,
            "problem": True,
            "data": True,
            "method": True,
            "core_results": True,
            "limitations": True,
            "full_line_by_line_translation": False,
            "reference_list_translation": False,
        },
        "blocks": blocks,
    }


def notes(p: dict) -> str:
    return f"""# Translation and extraction notes

- Source: `{p['pdf']}`
- Status: **draft deep reader**.
- Coverage: abstract, research question, acquisition/data, method, principal results, limitations, reproduction audit, and project implications.
- The complete background prose and reference list are not translated line by line. References are treated as a structural index.
- Multi-column PDF extraction can reorder text. The paired excerpts in `paper.md` were normalized for spacing and obvious ligatures; use the page anchor and source PDF for quotation-grade verification.
- Equations were not re-typeset when extraction confidence was insufficient. This avoids silently changing symbols; inspect the source page directly.
- The key evidence image is a crop from a rendered source page. Missing embedded fonts can cause minor visual substitutions, but the crop preserves source layout.
- Analytical claims outside the paired excerpts are reader synthesis, not author quotations.
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = ["# 逐篇精读索引", "", "每篇目录都含 `paper.md`、`source_map.json`、`translation_notes.md` 和关键证据图。", ""]
    for p in PAPERS:
        folder = OUT / p["id"]
        assets = folder / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        (folder / "paper.md").write_text(render_paper(p), encoding="utf-8")
        (folder / "source_map.json").write_text(json.dumps(source_map(p), ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "translation_notes.md").write_text(notes(p), encoding="utf-8")
        shutil.copy2(EVIDENCE / p["evidence"], assets / "key_evidence.png")
        index.append(f"- [{p['title']}]({p['id']}/paper.md)")
    (OUT / "README.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"built {len(PAPERS)} readers in {OUT}")


if __name__ == "__main__":
    main()
