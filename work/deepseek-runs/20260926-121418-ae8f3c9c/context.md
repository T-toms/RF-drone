# DroneRFa 项目共享摘要

更新时间：2026-09-24。用户要求沿用 deepseek-workflow；Codex 规划与验收，DSH Desktop DeepSeek 做具体实现和文档。任务串行执行，不自动提交/推送/发布，不恢复已取消的定时器或 heartbeat。以下历史章节的 `F:\DroneRFa数据集` 是旧盘符，现已变为 `G:\DroneRFa数据集`；新 F 盘是另一个物理盘。

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

### 2026-09-24 数据范围改为12机型（最新决策）

- 用户撤回“下载完整 DroneRFa”的方案，明确选择“停止整包，改为只准备 12 机型所需数据”。新 F 盘整包下载 PID 16800 已停止；截至停止时 `F:\DroneRFa完整数据集\DroneRFa.rar.part` 与状态前缀均为 1,551,892,480 字节（0.2519%），暂存不删除。不得自动续传整包。
- `FULL-ARCHIVE1` 的 DSH 实现和 26 项测试曾完成，但 Codex 验收发现写入返回值未核对、每块状态保存前未 fsync、中途未复查空间；原返工任务 `work/deepseek-tasks/20260924-full-archive-hardening.md` 因用户改变范围而暂不派发。其下载器不能直接视为已验收的长期下载方案。
- 当前旧 G 盘 `G:\DroneRFa数据集\original` 有 62 个 MAT：原三类基线24个，12机型扩充目标中已完成38个。冻结候选清单 `work/dronerfa-12model-download-candidates-20260922.json` 为新增10机型×8段共80个，尚缺42个。G 盘还有5个 `.part` 约1.50 GiB、8个 `.rar` 约7.53 GiB，待严格盘点；不得把未验证文件当训练数据。
- 新目标是在新 F 盘建立受卷身份保护的12机型数据目录，先只读验收并复制G盘可用MAT，再补齐冻结清单缺少的文件；保留既有训练/heldout来源标记，新增样本标为 unassigned，之后另定不泄漏的分组划分。G 盘任何旧数据只在 F 盘对应文件完成且验证后才考虑清理；保留实验结果、报告、代码和日志。

### 2026-09-24 14:30 用户再次更改目标盘（覆盖上一段）

- 用户明确要求**仍下载到 G 盘，不转移到新 F 盘**，并已停止 DSH 的 `SELECT12-F1 新F盘准备12机型数据` 会话。Codex 核对该会话 `nativeStatus=idle`、原核验进程已退出、`F:\DroneRFa12机型` 不存在；没有发生新 F 盘复制。
- 现在仅在 `G:\DroneRFa数据集` 核验和补齐冻结12机型目标。已有62 MAT原位保留，42个缺失 MAT 定向补下载到 `G:\DroneRFa数据集\original`；旧5个 `.part`、8个 `.rar` 只读审计后决定能否安全复用。新 F 盘完整压缩包断点继续保持停止，暂不删除。
- 旧 `work/deepseek-tasks/20260924-select12-to-new-f.md` 被用户撤销，不得继续执行；新的 G 盘任务须另行派发。已使用的原始基线/heldout来源标记必须保留，新增样本标为 unassigned，避免训练/验证/测试泄漏。

### 2026-09-24 14:54 下载废料清理

- 用户授权清理无用缓存与下载报废文件。Codex 删除了新 F 盘不再使用的整包 `.part`（1,551,892,480 字节）及状态/空锁文件，保留日志与卷标记；未重启整包下载。
- G 盘 `download_records_12models_20260922/downloads` 中8个无对应 MAT 的 `.rar` 经逐一 `UnRAR t` 全部退出代码3（CRC错误），删除共 8,087,215,641 字节；另删3个16字节空 `.part` 及其显示下载0字节的配对状态。两份约1.5 GiB真实内容的 `.part` 和全部62个MAT保留。共释放 9,639,109,429 字节。
- 验证与删除清单：`work/cleanup-audit-20260924/rar-results.json`、`work/cleanup-audit-20260924/cleanup-result.json`。DSH `SELECT12-G1` 仍运行，后续 G 盘清理需避开执行中的下载器；当前压缩包数0、内容断点数2。

### 2026-09-24 17:45 SELECT12-G1 验收发现需返工

- DSH 完成62个现有MAT的尺寸、HDF5四通道和SHA256审计，均合格；冻结104目标，尚缺42。实现下载器并通过34项模拟测试，但真实文件未成功落位（G盘仍62 MAT）。交接见 `work/deepseek-handoffs/20260924-143812-682f2402.json`。
- Codex 查出核心错误：下载器写入的是官方归档的**裸文件块**，却按从 RAR5 签名开始的完整RAR验证，所以两个文件的“缺少RAR5签名”是代码结构错误，不能据此断言服务器损坏。且目标条目 `packed_size < unpacked_size`，不能把压缩载荷复制为MAT。错误重试已浪费约2.1 GB流量，须修复后先单文件真实验收。
- Codex 已停止旧下载 worker PID 22252 和 launcher PID 404，保留了 G 盘本轮断点；停止时 `T0011_D00_S0000.mat.rar.part.restart` 与状态均为251,658,240字节。修复方案及限制见 `work/deepseek-tasks/20260924-select12g-wrapper-fix.md`。未验证前不得恢复42文件批量下载，不改回完整RAR方案。

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
# 2026-09-24 18:33 Codex 架构修复与下一步

SELECT12-G1R 会话在下载完整 `T0011_D00_S0000.mat` 裸 RAR 文件块后，因 `verify_wrapped_archive` 只预读 256 KiB、未跳到约 1 GB 后的结束块，误报包装块类型 `[1,2]`；会话停止且未写交接文件。Codex 修正了包装校验：按文件块声明的载荷长度 seek 到真实结束块，且文件块长度按相对偏移检查。`dronerfa_baseline/test_download_select12g.py` 56 项通过。真实 `tmp/diag-wrapper/real-wrapped.rar` 结构校验为 `[1,2,5]`、头部 CRC 全通过，`UnRAR t -idq -p-` 退出码 0。G 盘 `T0011_D00_S0000.mat.rar.part.restart` 已完整，仍未解出第 63 个 MAT。下一任务见 `work/deepseek-tasks/20260924-select12g-one-mat-finish.md`，仅处理这个文件并完成 MAT 校验，禁止放开批量。

## 2026-09-24 18:42 单文件真实验收通过

DSH 的 SELECT12-G1R2 会话因额度不足未执行；Codex 直接按限定范围完成。`G:\DroneRFa数据集\original\T0011_D00_S0000.mat` 大小 1,096,484,492 字节，SHA256 `a0328981f038de41f9e89324e10c61d8214bc0e9c3592f7bb7aaa541c3538ee0`。下载状态记录包装块 `[1,2,5]`、各块头 CRC、UnRAR t/e 退出码均 0、4 个 HDF5 通道各 160,000,000 采样、有限值探针通过。G 原始目录有效 MAT 从 62 增至 63；剩余 41 个目标文件。真实断点、旧数据和 F 盘均保留。`inspect_mat` 改为只读取 1000 采样而非全量数组，且禁止覆盖已有 MAT。见 `G:\DroneRFa数据集\download_records_12models_20260922\select12g-state.json`。下一步允许仅针对剩余 41 个清单文件在 G 盘恢复可续传下载，不改数据划分或训练集。

## 2026-09-25 SELECT12-G2 验收与数据治理决定

G `original` 现有冻结目标 104/104 个 MAT，文件名和清单预期大小逐一匹配；下载 state 中新增的 42/42 个条目为 completed，2026-09-25 最后续传 16 个文件的 run failures=0，单实例锁不存在。历史 state.errors 的 2 条来自先前已修的包装校验失败，不是当前 MAT 损坏证据。旧 `G:\DroneRFa数据集\dataset_manifest.csv` 仍只有 24 行，`unified_path` 指向旧 F 盘；24 行分别为 development_original 12、evaluated_holdout 12。决策：先做全量结构/哈希审计，再生成 104 行 G 盘清单；新增 80 行为 split_status=unassigned，原 24 行保留曾用于开发/评价的历史标记，不能洗成新 final test。META-SPLIT 需先查采集会话/重复关系，至少以源文件为不可跨 split 的最小组；仅凭当前 104 文件不能声称 13 类拥有完全未参与选型的 final test，因为原 24 个文件已被使用。39.7 GiB 成功 `.restart` 暂不删，待审计与清单验收后只清理已验证成功且可重构的冗余裸块，保留未解决断点/错误证据。下一项 SELECT12-G3 审计与清单提案，不训练、不冻结 split、不清缓存。

## 2026-09-25 SELECT12-G3 验收与发布

DSH 只读结构审计 104/104 valid、0 missing/unknown/problems，总字节 107,450,183,727；独立 SHA256 复算 104 文件，62 个旧文件与旧哈希记录全相同、104 个哈希互异。候选清单 `work/sel12g/dataset_manifest_candidate_104_20260925.csv` 经 Codex 独立核对：104 行、13 类各 8 文件、80 新行 unassigned、24 旧行保留原非路径字段、路径全部指向 G，无 F 路径，SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`。Codex 已用带哈希检查的原子替换发布为 `G:\DroneRFa数据集\dataset_manifest.csv`，旧 24 行原文件备份于 `work/sel12g/dataset_manifest_legacy24_backup_20260925.csv`（SHA256 `009e27e62a7232b19825f0121812ae881a4329fcddad669278f28956c78e3246`）。

发布后，Codex 逐一核对 42 个 `.rar.part.restart` 均对应 state completed、现有 MAT 与哈希报告，精确清理 42 个冗余裸块共 42,597,539,738 字节；`work/sel12g/restart-cleanup-audit-20260925.json` 保存删除清单与哈希依据。两份旧 `.rar.part` 仍保留。清理后 104 MAT、104 行正式清单、0 `.restart`、2 旧 `.part`，正式清单哈希未变。旧 F 盘下载器 `download_select12.py` 的形状检查潜在缺陷与当前 G 流程无关，未改；并发生成的 `sel12g-mat-audit-20260925-final.json` 非本轮 DSH 产物，仅作为交叉证据，不作为主验收依据。下一步 META-SPLIT 先核查采集会话/来源元数据及近重复，再提可追溯分组方案；不能凭现有 104 文件声称 13 类完全未参与选型的 final test。

## 2026-09-25 META-SPLIT-1 决策

DSH 的 `work/meta_split1/META-SPLIT-1-report.md` 与数字复算显示：104 个 MAT 的文件头 Created on 时间使 13 类各落在单一短时间簇、整体为两个采集夜；14 个本地 RAR 头 HTIME 与 MAT 时间相差 ≤2 秒。Codex 接受这些为**文件时间/批次代理**，但 RAR 时间可由原文件修改时间继承，不能称作独立证明的无线电采集时间或会话 ID；“同一次连续采集”的强断言须降为“时间上相邻，疑似同批”。每类仅一个时间簇，当前 104 文件不能做 13 类批次独立或全新未参与选型 final test。新 80 文件未出现在此前模型/数据产物中，旧 24 文件均已暴露；文件级分组只可称同批次文件留出，不写 cross-session。近重复 L1 阈值失去判别力，不据此认定重复或独立。

已批准只对官方归档索引里旧三类尚未下载的 88 个成员做每个最多 128 字节的 RAR 成员头区间读取，验证远端 ETag/总长、块头 CRC、文件名和时间字段，形成候选日期表；不下载载荷、不改 G 数据。选择口径：先补证并挑选最少必要的新来源文件，再考虑 13 类 final test；10 新类同批次文件级划分可作开发基线，旧三类作已暴露对照，不冻结为 final。夜级未见类开放集探针暂不作为下一阶段主实验，因为未知类与夜/设备条件完全混杂；以后若做，只能标探索性混杂检验。

## 2026-09-26 META-HEADER1 验收与划分决定

`work/meta_header1/` 对旧三类 88 个未下载成员完成只读 RAR 头筛查：88/88 块头、文件名、大小、ETag 与时间字段通过，网络仅接收 6,812 字节、0 载荷，62 项数字断言 PASS；正式 G 清单哈希未变。T0000 唯一有不同文件日期（2022-12-13 vs 现有 2022-11-27），但它也从 initial_comm_band 换为 switched_band，不能隔离日期效应；T0010 的 40 个候选与现有文件同日且连续，T0101 的 40 个也同日连续，06:00 夜界产生的键变化是人为切分，不是新批次。时间仍只是文件时间代理，不能当采集会话证据。

Codex 决定不为所谓 13 类跨批次 final test 下载这些候选，也不探测另外十类的 293 个头：T0010/T0101 缺口已使该目标在本归档内不可闭合。T0000 的新日期/新频段文件可留待独立的背景域移位实验提案，当前不下载。现阶段仅冻结**开发用途**的 13 类源文件级分组：新 10 类每类 5 train/1 val/2 file-holdout test；旧 3 类各 4 个 development_original 进 train、4 个 previously evaluated 分为 2 val/2 descriptive test。共 62 train/16 val/26 test 文件，其中新 10 类测试的 20 文件此前未用于模型选型但仍与训练同时间簇，旧 3 类测试的 6 文件此前已暴露；不得报告全 13 类 fresh final 或 cross-session。独立最终测试以后需要可核实的新采集/外部来源。下一项 META-SPLIT2 只生成候选划分与验收报告，未验收前不训练、不改 G 清单。

## 2026-09-26 META-SPLIT2 验收与正式冻结

Codex 接受 `work/meta_split2/split_candidate_104_20260926.csv` 为**13 类开发用途源文件级划分的冻结工件**；生成时的 candidate 文件名和 JSON 内候选状态是生成记录，正式冻结以 `work/meta_split2/CODEX-FREEZE-20260926.json` 的 CSV SHA256 为准，不改写已验收工件。104 文件按既定规则分为 62 train、16 val、20 `development_test`、6 `development_test_exposed`；每个源 MAT 只占一组，未来 1M 窗继承源文件角色。Codex 复跑独立核验 21/21 和反向对照 13/13，G 正式清单哈希仍为 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`。没有训练、下载或改动 G 数据。

定调：`work/meta_split1/source_table_104.csv` 保持原始审计证据不修改，其 `prior_use_role` 对新增 80 文件只表示归档 S 码分段，不能作为本项目既往用途。后续代码和报告**只能从冻结划分表读取** `prior_use_role`（真实既往用途）和 `archive_segment_role`（上游分段），不得直接使用旧表同名列判定暴露。新十类 20 个测试文件可单列为同批次文件级开发主指标；旧三类 6 个测试文件只能作已暴露对照；13 类整体仅称开发评估，不能称独立最终测试或跨会话泛化。下一阶段如需训练，应另定任务并固定该 CSV 哈希和源文件继承规则。

## 2026-09-26 用户授权三类两阶段基线按新划分重训

用户确认采用“按新划分重新训练两阶段”，目标是背景 T0000、Phantom 4 Pro T0010、Air 2S T0101。先建冻结 CSV 驱动的 24 MAT 窗口索引，再按旧架构训练阶段1和阶段2，最后串联评价；旧三类 6 个 test MAT 是已暴露开发对照，不得称全新 final。正式 G 清单 SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`，冻结划分 CSV SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`。Codex 已在 G 盘各取一个 test MAT 的首个 1M 窗做旧模型 CUDA 冒烟运行，六个检查点哈希均匹配，三个种子在背景/两机型上均 3/3 一致；仅证明运行链路可用，不是新训练结果。当前任务 `work/deepseek-tasks/20260926-cascade3-newsplit-index.md` 先建数据入口。

CASCADE3-NEWSPLIT-A 已验收（2026-09-26）：冻结三类 24 MAT 索引 3630 窗，train 1810 / val 930 / development_test_exposed 890；阶段2视图 1210 / 650 / 610。索引 SHA256 `deed870f9693c5ebe93fb80a9647b452e90680dd5b1f21372e108ded4bf4d231`，阶段2视图 SHA256 `76001e671cf3ebe492b191af87a85b5368caed85e03c84c4c249ccc5081642de`。DSH 完整哈希 24/24 两次核对、独立核验 47/47、反向对照 10/10、测试 16 项、24 文件各首/中/末窗 72 次真实 RF0 读取通过；Codex 独立复核索引 24 文件零跨角色、无冻结表外文件、哈希一致。用户授权“跑一次”，Codex 决定先按全 3630 窗跑 seed 2026 的两阶段 5 epoch，不抽样；训练任务 `work/deepseek-tasks/20260926-cascade3-newsplit-train.md`。
