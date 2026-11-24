# DroneRFa 项目共享摘要

更新时间：2026-09-21。用户要求沿用 deepseek-workflow；Codex 规划与验收，DSH Desktop DeepSeek 做具体实现和文档。任务串行执行，不自动提交/推送/发布，不恢复已取消的定时器或 heartbeat。

## 当前授权与任务

- 最近任务：`LIT-AUDIT1` 已验收，整理了可追溯的前沿文献矩阵，并把已有基线与前沿论文逐项对齐。
- 任务文件：`work/deepseek-tasks/20260920-literature-frontier-audit.md`
- 外部证据说明：`work/literature-frontier-source-note-20260920.md`
- 交付目录：`literature_review_from_scratch/frontier_update_20260920/`；本次交接：`work/deepseek-handoffs/20260920-190804-183f7c6f.json`。
- 本任务只做文献与实验路线图，不训练、不下载数据、不改实验代码、权重、正式结果或 F 盘文件。

## 数据与现有模型边界

- 合并数据根：`F:\DroneRFa数据集`；原始数据：`F:\DroneRFa数据集\original`；heldout：`F:\DroneRFa数据集\heldout_20260917`。
- 共 24 个 MAT，约 23.025 GiB；当前实际实验只使用三类：Background、DJI Phantom 4 Pro、DJI Air 2S。
- 每个分类窗为 1,000,000 个复数采样点、100 MS/s、RF0；不同源文件不自动等于独立采集会话。
- 当前已验收的是工程基线。heldout 已用于部分模型和 n_fft 描述性分析，不可称全新未触碰最终测试；AWGN 档位不等于现场绝对 SNR。

## 已验收实验

- `PREP12`：heldout 12 文件、1820 窗，背景 560、Phantom 610、Air2S 650；见 `work/heldout_prep_20260917/codex-acceptance-20260918.json`。
- `EVAL6`：六个第一阶段检查点冻结评价；见 `dronerfa_baseline/outputs/heldout_evaluation_20260918/independent_verification.json`。
- `BASE2`：能量阈值 clean heldout 文件等权 BA 98.30%；PSD+SVM 100.00%；见 `work/base2-codex-independent-verification-20260919.json`。
- `NFFT4`：n_fft 512/1024/2048/4096 四条件文件等权 BA 跨 seed 均值为 99.04%/98.07%/97.40%/97.98%；见 `work/nfft4-codex-acceptance-20260919.json`。
- `STAGE2-BASE`：Phantom/Air2S 二分类，旧 val 选中 n_fft=512；heldout 四条件文件等权 BA 99.78%±0.09pp；见 `work/stage2-codex-verification.json`。
- `CASCADE3`：三级串联 clean/-10/-12/-15 dB 为 99.75%/99.98%/99.63%/95.14%；-15 dB 主要错误是 Air2S 在 Stage 1 漏成 Background；见 `work/cascade3-codex-acceptance-20260920.json`。
- `INFER-UI1`：本地分析界面运行于 `http://127.0.0.1:8765/`，桌面快捷方式为“无人机信号分析”；见 `work/infer-ui1-codex-acceptance-20260920.json`。
- `PACK-RESULTS1`：F 盘成果包 `F:\DroneRFa项目近期工作汇总_20260920` 已验收；154 项清单哈希与大小通过，MAT 原始数据只引用不复制。

## 文献资产与本轮新增重点

- 已有候选元数据 63 条：`literature_review_from_scratch/references_metadata.csv`。
- 已有精选 BibTeX 34 条：`literature_review_from_scratch/deep_research_redo/references_无人机RF_34篇.bib`。
- 已有 10 篇逐篇精读：`literature_review_from_scratch/deep_research_redo/readers/`。
- 已有综合总报告、证据等级、领域地图、实验协议和 22 周路线图，位于 `literature_review_from_scratch/deep_research_redo/`。
- 本轮补充/强化：CageDroneRF 2026、RFUAV 2025、S3R TIFS 2024、Open-RFNet JSAC 2026、DRFF-R2 2026、开放世界增量识别 2026、真实跨环境鲁棒性 Sensors 2024。
- `CrossRF` arXiv:2505.18200 已撤稿，只能进排除表。

## 当前科学判断

1. 现有结果证明在当前三类固定协议下信号可分，不能证明开放集、跨环境、跨接收机、跨会话或部署鲁棒性。
2. 论文中的高准确率跨数据集、类别数、切分和任务不可直接排名；活动检测、时频定位、型号分类、个体 RFFI、开放集必须分开。
3. 最优先缺口是可证明独立性的元数据与 group split，以及未参与选型的 final test。
4. 之后应补直接三分类与同预算 Raw-IQ 基线，再扩展更多 DroneRFa 机型，然后进入开放集与跨域。
5. YOLO 只适合连续宽带流的多信号定位，不是当前已裁剪三类窗口分类的必选升级。
6. 用户已明确暂缓噪声方式实验；路线图可记录，但不能自动开始。

## LIT-AUDIT1 验收结果

- 矩阵 77 条、19 个字段、50 条 direct_uav_rf、10 条 dataset、9 条 review、8 条 adjacent_rffi；Import-Csv 解析成功、重复 ID=0、必读来源齐全、CrossRF 不在矩阵。
- 产物包括主报告、检索纳排与证据台账、六阶段路线图、术语边界、排除/降级来源和 README；旧 `deep_research_redo` 未改动。
- Codex 独立抽查修正了两个表述：M059 的 dataset 字段改为 `Robustness dataset（1100 条 RF 记录）`；X005 的会议发表状态改为“待核验”，不再写未经核实的 ICC 接收信息。
- 报告保留限制：本轮是结构化前沿审计，不是 PRISMA 穷尽综述；CageDroneRF/Sensors 关键数值尚待逐表复核；大量 sample/split/code 字段为 NR；当前项目仍不能声称跨会话、开放集或个体 RFFI。
- 唯一推荐下一实验：`META-SPLIT`，先做文件元数据、group_file 划分与未参与选型的 final test 冻结；用户暂缓的噪声方式实验不恢复。

## 工作流状态

- 上一任务 `PACK-RESULTS1` 已验收，当前无活动运行。
- DSH Desktop 健康检查已通过，工作区可写。
- `LIT-AUDIT1` 已由 DSH 完成并由 Codex 验收；当前无活动运行。下一项需用户继续授权后派发 `META-SPLIT`，依赖当前审计已满足。

## REPORT-DOC1 验收结果

- 已生成两页中文进度报告：`reports/DroneRFa项目进展汇报_结果版_20260921.docx`。当前版本只保留实验主线和核心结果，用连续段落说明检测阶段、阶段 2 和完整三分类；插入 n_fft/SNR/阶段 2 结果汇总图，以及正常条件与 −15 dB 的三分类混淆矩阵。
- 内容按普通读者理解顺序整理为已完成工作、主要结果、结果边界、当前六项问题和下一步优先级；解释了 dB、人工白噪声、I/Q 数据等必要术语，没有把当前结果写成现场通用准确率。
- DOCX 压缩包结构有效，含一个明确分页符；文档表格数、段落边框数和底纹数均为 0，可见文字只使用黑色。两页本地排版镜像均已逐页检查，无文字截断。验收记录见 `work/progress-report-codex-acceptance-20260921.json`。
- 本次检查时 DSH Desktop 运行时发现记录缺失，Codex 为避免阻塞用户已授权的文档请求，直接完成并验收该报告；没有创建新的 DSH 会话或恢复定时器。


### 2026-09-21 导师汇报最新修订
当前交付：`reports/DroneRFa进展汇报_导师版.docx`。按用户要求改为7个小点，每项工作紧接结果；仅嵌入已有召回率图及−15 dB混淆矩阵，原图字节一致，以灰度显示。原始CSV核对检测和机型识别的SNR数据。通过WPS原生分页及PDF导出确认2页，逐页目视检查通过；无表格、装饰线和有色文字。验收记录见 `work/progress-report-codex-acceptance-20260921.json`。

用户最新修正：实验图片保留原始彩色，正文黑色。已移除文档图片灰度效果，原图文件未改；WPS原生渲染确认仍为2页，逐页检查通过。

### 2026-09-22 F盘数据目录统一

- `DATA-MERGE1` 已验收。原 `F:\DroneRFa数据集\original` 与 `F:\DroneRFa数据集\heldout_20260917` 的24个MAT现统一位于 `F:\DroneRFa数据集\original`，共24个唯一文件、24,723,083,329字节，T0000/T0010/T0101各8个。
- 下载日志和断点没有混入数据目录；原 heldout 目录已重命名为 `F:\DroneRFa数据集\download_records_20260917`，其中保留 `downloads`。
- 统一清单为 `F:\DroneRFa数据集\dataset_manifest.csv`。原分组语义由 `former_partition` 保存：`development_original` 12条，`evaluated_holdout` 12条。后者已参与既有评估，不能称为未触碰最终测试集。
- 历史冻结配置和CSV中的旧绝对路径不批量改写；需要重跑时按 manifest 或 `work/dataset-unification-20260922.json` 的12条路径映射解析新位置。
- Codex独立验收：manifest 24行、文件名唯一、路径缺失0、大小不符0；记录见 `work/data-merge1-codex-acceptance-20260922.json`。

### 2026-09-22 启动12机型扩充下载

- `DOWNLOAD12-START` 已验收“启动阶段”：从本地 DroneRFa RAR索引冻结10种新增机型 × 8个S段，共80个文件。候选总块79,572,949,889字节、packed总79,572,944,417字节、解压总82,727,100,398字节；清单和dry-run均通过。
- 新增目标：T0001、T0011、T0100、T0110、T0111、T1000、T1001、T1010、T1011、T1100。前7种文件名带D00；T1010/T1011/T1100在官方归档文件名中不带距离字段，不能伪标D00。
- 后台逐文件断点下载已启动，PID=2244，记录根 `F:\DroneRFa数据集\download_records_12models_20260922`，解压目标仍是 `F:\DroneRFa数据集\original`。最新只读检查仍为0/80个MAT完整落位，第1个 `T0001_D00_S0000.mat` 已下载约416 MiB/1.17 GiB（34.63%），错误日志为0字节；不要启动第二个进程。
- `F:\DroneRFa数据集\dataset_manifest.csv` 尚未修改；80个文件全部通过HDF5与大小校验后，才新增为 `split_status=unassigned`，随后另行冻结12机型训练/验证/测试划分。
- 选择清单见 `work/dronerfa-12model-download-candidates-20260922.json`；Codex启动验收见 `work/download12-start-codex-acceptance-20260922.json`。这只代表下载已启动，不代表80个文件已完成。
