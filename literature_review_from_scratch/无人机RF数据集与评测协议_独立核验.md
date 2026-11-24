# 无人机射频（RF）深度学习检测与识别：数据集与评测协议（独立核验）

核验日期：2026-08-15。范围限于**被动截获无人机与遥控器之间通信链路的射频（RF）信号**，包括控制、遥测和图传发射；不把雷达回波、声学或视觉数据混入本表。本文件不读取、不使用、也不评价本地既往探索性实验及其结果。

## 使用结论（先看这里）

没有单一数据集能够同时证明型号泛化、个体指纹、低信噪比、真实干扰、多机重叠和跨地点可迁移性。应把数据集按问题拆开使用：

|优先问题|较适合的数据候选|不应做出的外推|
|---|---|---|
|可复现实验与经典闭集检测/型号/模式|DroneRF、VTI_DroneSET_FFT|不能据此声称跨地点、跨接收机或未知型号泛化|
|Wi-Fi / Bluetooth 共频干扰|CardRF、DroneDetect V2|不能把受控干扰等同于开放城市频谱环境|
|原始 I/Q 与较多型号/链路形态|Tampere Zenodo、KU Leuven Drone RF、DroneRFa（取得官方授权后）|暗室/半暗室采集不能代替野外迁移验证|
|同型号个体（RFF）与多信号问题|UAVSig、DroneRFb-DIR、Drone RC RF|须以采集会话和设备隔离，不能随机切片|

“真实采集”只表示信号由真实设备实际接收，并不自动代表野外、跨站点或可推广。

## 已核验数据集矩阵

|数据集|RF 信号 / 频段或来源|适配任务|真实采集|官方入口 / DOI|已知局限（应如实写入实验设计）|
|---|---|---|---|---|---|
|**DroneRF**|两台 NI-USRP2943R 截获无人机与飞控模块通信；覆盖 2.4 GHz ISM 全带宽（两台接收机各捕获一半）；公开文件是时域幅度 CSV，而非复数 I/Q。|无人机存在检测、3 类机型识别、连接/悬停/飞行/录像等运行状态。共 227 段（含背景）。|是，实验室受控采集。|[Mendeley Data](https://data.mendeley.com/datasets/f4c2b4n755/1)，DOI [10.17632/f4c2b4n755.1](https://doi.org/10.17632/f4c2b4n755.1)；数据论文 [10.1016/j.dib.2019.104313](https://doi.org/10.1016/j.dib.2019.104313)。|机型很少；实验室受控；非复数 I/Q，无法公平评估相位信息方法；公开说明未建立多站点/多日期/多接收机独立性。因此仅宜作经典可复现基准，不能作为泛化主证据。|
|**Drone Remote Controller RF Signal Dataset**|被动监听遥控器（RC）至无人机的 2.4 GHz 控制信号；17 个 RC、8 个厂商、每个 RC 约 1000 条、单条 0.25 ms；无人机在采集中处于 idle。|遥控器/发射机射频指纹、控制链路检测与分类、低 SNR 消融。|是，受控 BYOD 采集。|[AERPAW 官方页](https://aerpaw.org/dataset/drone-remote-controller-rf-signal-dataset/)，IEEE DataPort DOI [10.21227/SS99-8D56](https://doi.org/10.21227/SS99-8D56)。|目标是 RC 发射机而不是飞行中无人机全链路；没有飞行状态、图传链路或多机实况；同一控制器大量短片段，随机切片极易泄漏。DataPort 实际下载权限须在使用前核验。|
|**Cardinal RF（CardRF）**|户外采集 UAV 遥测/控制、Wi-Fi 与 Bluetooth 的 2.4 GHz 信号；MAT 格式；每条 5M 采样点、0.25 ms。|有/无 UAV 与 UAV/Wi-Fi/Bluetooth 区分；干扰鲁棒识别；层级检测与识别。|是，户外受控采集。|[AERPAW 官方页](https://aerpaw.org/dataset/cardinal-rf-cardrf-an-outdoor-uav-uas-drone-rf-signals-with-bluetooth-and-wifi-signals-dataset/)，IEEE DataPort DOI [10.21227/1XP7-GE95](https://doi.org/10.21227/1XP7-GE95)。|虽有真实 Wi-Fi/BT 干扰，但仍是单个受控采集体系；不应当作跨城市频谱泛化证据。数据库索引曾标为 closed，下载许可/可访问性须先确认。|
|**DroneDetect / DroneDetect V2**|BladeRF SDR 的原始 I/Q；公开关联研究描述 7 个机型、开机/悬停/飞行 3 状态，并含 Wi-Fi、Bluetooth、二者同时及无干扰条件。|受干扰机型/模式分类；时频检测器与表征鲁棒性比较。|是，SDR 实测、受控条件。|IEEE DataPort DOI [10.21227/5JJJ-1M32](https://doi.org/10.21227/5JJJ-1M32)；关联的干扰研究 [10.1109/ICUAS51884.2021.9476867](https://doi.org/10.1109/ICUAS51884.2021.9476867)。|必须以下载版本的 README 为准核对频段、标签和是否有独立背景负类；不能假设每一版本都支持完整的“有/无无人机”检测。公开使用材料显示其采集会话/地点/接收机多样性有限，不能把随机窗口高分视为跨域表现。|
|**UAVSig**|真实 OTA 无人机及 RC 射频信号；原始文件、元数据、可视化/下载脚本均在数据仓库，具体通道/采集参数见数据说明文档。|宽带信号检测、时频定位、同型号多架无人机的个体指纹（RFF）。|是，OTA 实测。|[UCLA Dataverse](https://dataverse.ucla.edu/dataset.xhtml?persistentId=doi:10.25346/S6/LVRRAE)，DOI [10.25346/S6/LVRRAE](https://doi.org/10.25346/S6/LVRRAE)，关联论文 [10.1109/MILCOM61039.2024.10773837](https://doi.org/10.1109/MILCOM61039.2024.10773837)。|UCLA 官方页明确说明 B205mini SDR 导致发布的原始 I/Q 中存在随机缺段；不能把缺段当作信号特征。Dataverse 批量下载可能不稳定，宜使用其官方脚本；站点/日期/接收机外推能力仍应按元数据逐项检查。|
|**DroneRFa**|SDR 监测无人机—遥控器通信；覆盖 3 个 ISM 频段；城市户外运动无人机 9 类、室内 15 类、背景 1 类；每类不少于 12 段、每段超过 1 亿采样点，并提供户外飞行距离和工作频段标注。|宽带 RF 检测、机型/品牌识别、室内外与距离条件分析。|是，城市户外与室内实测。|数据论文与说明：[10.11999/JEIT230570](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT230570)；[官方数据入口](https://jeit.ac.cn/web/data/getData?dataType=Dataset3)。|官方入口要求手机/邮箱验证码，故是“有官方发布入口”而非匿名一键开放；取得授权前不应把它列为可立即复现实验的数据。不要使用既往本地试验结论；本研究只能按新下载数据、重新制定切分和指标。|
|**VTI_DroneSET_FFT**|3 个 DJI 机型（Phantom IV、Mavic Zoom、Mavic 2 Enterprise）及飞控通信；版本 3 同时提供 2.4/5.8 GHz 原始 RF、预处理 RF 和 DNN 格式文件，说明文件含单/多机及飞行模式标签。|检测、型号/模式分类、单机与多机情形的受控比较。|是，真实设备实测。|[Mendeley Data](https://data.mendeley.com/datasets/s6tgnnp5n2/3)，DOI [10.17632/s6tgnnp5n2.3](https://doi.org/10.17632/s6tgnnp5n2.3)。|仅 3 个 DJI 机型；跨地点、跨日期和跨接收机元数据有限。预处理/DNN 派生包不能替代从原始记录重新按组切分，否则可能把同一连续录制的相关片段泄漏到训练与测试。|
|**Radio-Frequency Control and Video Signal Recordings of Drones（Tampere）**|10 款消费/准专业无人机的原始 I/Q；控制与图传；2.44 GHz 和 5.8 GHz（Yuneec Typhoon H 为 5.7 GHz）；采样率 120/200 MSps。|协议/跳频形态研究、机型或链路类别识别、原始 I/Q 与时频表征比较。|是，Tampere 大学隔离暗室，转台上的无人机与遥控器。|[Zenodo 官方记录](https://zenodo.org/records/4264467)，DOI [10.5281/zenodo.4264467](https://doi.org/10.5281/zenodo.4264467)。|暗室、固定几何与近距离控制器位置；每机记录长度短，某些跳频序列分散在两条非连续记录；Parrot Mambo 控制/图传不能同时干净录得。适合波形研究，不宜独立主张真实野外检出率。|
|**Drone RF Dataset（KU Leuven / RMA）**|USRP X310 原始 I/Q，中心频率 2.44 GHz、100 MSps；遥控器与图传文件；接收天线距无人机/遥控器 7 m。|原始 I/Q 的检测、分类、异常/新颖性识别。|是，半暗室实测。|[KU Leuven RDR](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/HZRVNZ)，DOI [10.48804/HZRVNZ](https://doi.org/10.48804/HZRVNZ)。|单一中心频率、半暗室、固定 7m/接收机配置；约 43.5 GB，需按文件或 Globus 获取；许可为 CC BY-NC 4.0，商业用途需另行处理。|
|**DroneRFb-DIR**|城市场景 2.40–2.48 GHz 原始 I/Q；6 类 DJI 无人机、每类 3 个个体，另有背景；含飞控、图传和周边干扰，标注 LoS/NLoS。|同型号/跨个体识别（DIR/RFF）、背景干扰下检测。|是，城市实测。|数据论文 [10.11999/JEIT240804](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT240804)；期刊给出的 [ScienceDB 数据入口](https://www.scidb.cn/detail?dataSetId=84cf9101e739402784b1396783881202&version=V1&code=j00173)。|品牌集中于 DJI；官方已给出训练/测试划分，但研究者仍应核验该划分是否会话、日期、接收机和个体隔离。不要用提供的预定义切分替代独立的组隔离审计。|

### 可作为补充、但不能替代原始 I/Q 基准的候选

**DroneRFb-Spectra**（IEEE DataPort DOI [10.21227/wv7h-sv64](https://doi.org/10.21227/wv7h-sv64)）发布的是由 50 ms 原始 I/Q 经 STFT 下采样得到的 512×512 频谱图，描述覆盖 3 个 ISM 频段、7 个品牌、14,460 张图。它适合图像式深度学习的复现，但不应替代保留相位信息的原始 I/Q 实验；且所有谱图必须按原始连续记录而非按图像随机切分。正式纳入主实验前，应在 DataPort 页面重新核验许可、原始录制分组与标签来源。

## 独立、严格的评测协议

### 1. 先建立“采集单元”而不是“切片单元”

数据清单中每个样本至少应有：`dataset_id, raw_recording_id, session_id, device_id, model_id, controller_id, site, date/time, receiver/antenna, band/channel, sampling_rate, distance, LoS/NLoS, flight_state, interferer, number_of_emitters, raw_start/end`。未知字段必须显式标为 `unknown`，不能编造。

将从同一连续原始记录切出的所有窗口放进同一组。若一个记录产生 W 个 STFT 图或 I/Q 窗口，这 W 个样本**必须全部在同一训练/验证/测试划分**。归一化参数、去噪器、频段筛选阈值、数据增强范围也只能在训练集拟合。

### 2. 把任务分开报告

|任务|正确定义|主指标|禁止的替代说法|
|---|---|---|---|
|RF 活动/无人机检测|目标发射存在 vs 背景及非无人机干扰；以连续事件或时间窗标注。|PR-AUC、ROC-AUC、召回率@指定虚警率、每小时虚警数、检测延迟。|只报切片 accuracy 或只在“已有无人机信号”上做分类。|
|闭集机型/品牌识别|测试类别在训练阶段已知，测试采集组未见。|宏 F1、平衡准确率、逐类 P/R、混淆矩阵、会话组 bootstrap 95% CI。|把大量重叠窗口的 accuracy 当作独立样本精度。|
|飞行状态识别|同一机型不同状态；应控制机型或在未见机型上单列测试。|宏 F1、每状态召回、跨机型/跨会话差值。|将设备指纹当作状态特征。|
|个体/遥控器指纹（RFF/DIR）|识别特定发射机个体；测试必须来自未见会话/日期/通道。|宏 F1；验证式任务另报 EER、FAR/FRR。|“同一录制随机切片”上的识别率。|
|开放集|测试含训练时从未出现的整机型/整品牌/整设备家族；阈值用独立验证未知类确定。|已知类宏 F1、unknown-vs-known AUROC、OSCR、FPR@95%TPR。|把未知类加入训练后仍称开放集。|
|多机重叠|一个时频片中可以有 0、1、≥2 个发射源，标签为多标签/事件集合。|每源 AP/mAP、集合 F1、漏检率、虚警率、源数量 MAE。|把合成混合数据的单标签分类当作多机检出。|

### 3. 最低合格的切分层级

1. **诊断层（不能做论文主结论）**：分层随机窗口切分，仅用于调试代码和验证数据管线。
2. **最低主结果**：`raw_recording/session` 组隔离。推荐 `GroupKFold` 或预注册的 train/val/test 组清单。
3. **条件泛化结果**：留出完整 `date`、`site`、`receiver/antenna`、`channel` 或 `distance/LoS` 条件；每个条件轮流作为测试域（leave-one-domain-out）。
4. **设备/机型泛化结果**：若每个机型有多个物理个体，测试个体未出现在训练中；再做 leave-one-model-out / leave-one-manufacturer-out 的开放集评测。
5. **外部验证**：训练于一个原始 I/Q 数据集、无微调地在另一个独立采集体系测试。两者的采样率和频段不同，应分别报告“直接迁移”和“预先声明的重采样/频带对齐”结果，绝不把标签强行拼成同一随机池。

若公开数据没有 `session/site/date/receiver` 元数据，只能声称“记录级隔离”，不能声称完成了相应的跨域验证。

### 4. 低 SNR、真实干扰与多机应独立成协议

- **低 SNR**：优先使用原生低 SNR / 不同距离 / LoS-NLoS 记录。若加入 AWGN，只能作为受控补充，必须单独画出 SNR 曲线，并写明 SNR 估计方法、信号窗和噪声窗；不能以人工加噪结果代替真实干扰结论。
- **干扰**：分别报告 Wi-Fi、Bluetooth、二者同时、其他占用频谱和纯背景；训练和测试都要按采集组隔离。CardRF、DroneDetect 可以支撑此类试验，但仍需说明它们是受控干扰。
- **多机**：优先真实同时发射录制；没有时，可用原始 I/Q 在线性叠加产生可追溯混合，但训练、验证、测试所用的源记录必须彼此组隔离，且“合成”与“真实”结果分开报告。
- **静默/自主飞行边界**：被动 RF 方法只能检测可接收的通信发射。没有 RF 发射的无人机不在这类数据及这类评测的可保证检测范围内；文中应把此写成能力边界，而非模型失败或模型优点。

### 5. 可复现性与统计报告

- 发布 split manifest（原始记录级 ID、hash、标签版本），而不是只发布随机种子。
- 切片窗口长度、步长、STFT 参数、频带选择、归一化、增强、早停和阈值，都写入可机器读取的配置文件。
- 每个深度学习配置至少报告多个随机种子；95% CI 以**会话/原始记录或设备**为重采样单位，而不是以高度相关的窗口为单位。
- 报告参数量、MACs、窗口级推理时延、端到端（含 STFT）吞吐量，以及测量硬件；边缘部署结论必须基于实际目标硬件。
- 首先给出能量检测/CFAR 或传统特征分类器、1D 原始 I/Q CNN、2D 时频 CNN 的同切分基线；新模型必须在完全相同的分组和阈值协议下比较。

## 数据准入与选型原则

1. DOI 或网页“可见”不等于可立即下载。DroneRFa 需要验证码；部分 IEEE DataPort 数据可能需要登录或授权；应在项目登记表保存访问日期、许可、版本和下载清单。
2. 有“真实采集”也不等于可评估泛化。暗室、半暗室、单接收机或固定距离数据适合表征/消融，不适合单独证明外场可用性。
3. 先选定一个可公开复现的核心集（建议 UAVSig + DroneRF / VTI 中之一），再把 CardRF / DroneDetect 的干扰、DroneRFa / DroneRFb-DIR 的场景或个体维度作为**预注册的外部验证**；不要把不同数据集的切片混合后随机切分。

## 核验来源（均为数据提供方、官方实验室/项目页或原始数据论文）

- DroneRF：[Mendeley Data](https://data.mendeley.com/datasets/f4c2b4n755/1)；[Data in Brief 数据论文](https://doi.org/10.1016/j.dib.2019.104313)。
- Drone RC RF 与 CardRF：[AERPAW Drone RC 页面](https://aerpaw.org/dataset/drone-remote-controller-rf-signal-dataset/)；[AERPAW CardRF 页面](https://aerpaw.org/dataset/cardinal-rf-cardrf-an-outdoor-uav-uas-drone-rf-signals-with-bluetooth-and-wifi-signals-dataset/)。
- UAVSig：[UCLA CORES 实验室页](https://cores.ee.ucla.edu/downloads/datasets/uavsig/)；[UCLA Dataverse](https://dataverse.ucla.edu/dataset.xhtml?persistentId=doi:10.25346/S6/LVRRAE)。
- DroneRFa / DroneRFb-DIR：[JEIT DroneRFa](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT230570)；[JEIT DroneRFb-DIR](https://jeit.ac.cn/cn/article/doi/10.11999/JEIT240804)。
- VTI_DroneSET_FFT：[Mendeley Data](https://data.mendeley.com/datasets/s6tgnnp5n2/3)。
- Tampere：[Zenodo](https://zenodo.org/records/4264467)。
- KU Leuven：[KU Leuven RDR](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/HZRVNZ)。
