# LIT-AUDIT1：无人机 RF 前沿文献矩阵与项目差距审计

## 目标

在不破坏现有综述的前提下，创建一个带日期的新版本目录，完成可追溯的前沿文献矩阵，并把文献证据与本项目已经完成的 Stage 1、Stage 2 和三级串联实验逐项对齐。用户需要能直接看懂：现有工作处于领域哪一层，哪些结论已经成立，哪些尚未成立，以及接下来按什么顺序补实验。

本任务是文献与项目文档工作，不训练模型、不下载数据、不修改现有实验结果、不改分析界面。

## 必读输入

1. `work/literature-frontier-source-note-20260920.md`
2. `work/deepseek-plan.md`
3. `work/deepseek-context.md`
4. `literature_review_from_scratch/references_metadata.csv`（63 条候选元数据）
5. `literature_review_from_scratch/deep_research_redo/references_无人机RF_34篇.bib`
6. `literature_review_from_scratch/deep_research_redo/00_深度研究总报告.md`
7. `literature_review_from_scratch/deep_research_redo/01_证据等级_权威性与检索记录.md`
8. `literature_review_from_scratch/deep_research_redo/02_外部权威文献扩展与领域地图.md`
9. `literature_review_from_scratch/deep_research_redo/04_可实现方法_实验协议与里程碑.md`
10. `literature_review_from_scratch/deep_research_redo/05_分类阅读清单与阅读顺序.md`
11. `literature_review_from_scratch/deep_research_redo/readers/*/paper.md` 与对应 `source_map.json`
12. `work/base2-codex-independent-verification-20260919.json`、`work/nfft4-codex-acceptance-20260919.json`、`work/stage2-codex-verification.json`、`work/cascade3-codex-acceptance-20260920.json`

## 允许新增或修改

只允许新增或修改：

- `literature_review_from_scratch/frontier_update_20260920/README.md`
- `literature_review_from_scratch/frontier_update_20260920/00_前沿文献矩阵与项目差距报告.md`
- `literature_review_from_scratch/frontier_update_20260920/01_文献矩阵.csv`
- `literature_review_from_scratch/frontier_update_20260920/02_检索纳排与证据台账.md`
- `literature_review_from_scratch/frontier_update_20260920/03_下一阶段实验路线图.md`
- `literature_review_from_scratch/frontier_update_20260920/04_术语与任务边界.md`
- `literature_review_from_scratch/frontier_update_20260920/05_排除与降级来源.md`
- 本次 `work/deepseek-handoffs/<runId>.json`

不得修改旧版 `deep_research_redo`、旧 BibTeX、实验代码、权重、结果 CSV 或 F 盘文件。

## 文献矩阵要求

1. 至少 45 条经筛选条目；其中至少 25 条为直接 UAV-RF 证据。可以吸收 63 条候选元数据，但不能为凑数纳入与任务无关的论文。
2. 每行至少包含以下 CSV 字段，字段值不能包含未经正确转义的换行：
   - `id`
   - `year`
   - `title`
   - `venue_or_status`
   - `doi_or_url`
   - `evidence_type`（direct_uav_rf / adjacent_rffi / dataset / review）
   - `task_level`（activity_detection / tf_localization / model_classification / individual_rffi / open_set / incremental_open_world / multi_emitter）
   - `dataset`
   - `class_scope`
   - `input_representation`
   - `sample_unit_or_length`
   - `split_unit`
   - `domain_conditions`
   - `main_metrics`
   - `code_or_data`
   - `evidence_status`
   - `main_limit`
   - `project_relevance`
   - `verification_level`
3. 不清楚写 `NR`；不得猜测窗口长度、切分单位、采样率、类别数或发表状态。
4. 明确区分型号分类和个体 RFFI；明确区分存在检测、时频定位和已裁剪信号分类；不能将三分类 accuracy 与开放集、多类别、跨域论文作横向排行榜。
5. 至少包含 CageDroneRF、RFUAV、S3R、Open-RFNet、DRFF-R2、DroneRFa、DroneRF、Sensors 2024 鲁棒性论文、TCCN 2022 联合检测分类、TVT 2020 非标准波形 RFFI、TVT 2021 前导码、2026 开放世界增量论文。
6. `CrossRF` 撤稿只进入排除/降级表，不能进入正面核心证据。

## 主报告要求

1. 开头用一页内回答用户最关心的问题：之前并非只看四篇；本轮实际覆盖多少候选、多少矩阵条目、多少直接 UAV-RF、多少逐篇精读；哪几个前沿方向证据最强。
2. 给出清晰任务分层，解释为什么论文中的 99% 与本项目 99% 不能直接比较。
3. 逐项审计当前项目：
   - 已完成且证据成立；
   - 完成了工程基线但证据边界有限；
   - 尚未完成；
   - 当前不该做或应推迟。
4. 必须把当前三类、两机型、不同文件不等于独立会话、heldout 已参与选型、AWGN 非现场 SNR、-15 dB Air 2S Stage 1 漏检等事实写准确。
5. 对 `PSD+SVM heldout 100%` 和串联 `clean 99.75%` 给出科学解释：它们说明当前固定数据协议可分，不说明开放集、跨环境或部署鲁棒性。
6. 给出“最小可发表贡献”和“扩展版贡献”两条路线；每条必须有进入条件、数据需求、对照、主指标、停止条件和风险。
7. 引用优先直链论文、DOI、官方数据页或官方代码；二次综述只用于发现文献。所有关键数值紧邻来源链接和证据状态。
8. 结论必须明确推荐下一个实际任务，且与 `03_下一阶段实验路线图.md` 一致。

## 路线图要求

按依赖顺序给出阶段、目标、数据、模型/基线、划分、主指标、预计计算成本、产物、通过条件和停止/转向条件。至少覆盖：

1. `META-SPLIT` 元数据、组划分与最终测试集冻结；
2. `BASELINE-COMPLETE` 直接三分类与 Raw-IQ 同预算基线；
3. `MULTICLASS` 扩展 DroneRFa 更多机型/层级分类；
4. `OPENSET` 多个 known/unknown 划分及校准；
5. `CROSSDOMAIN` 跨会话/环境/距离/接收机/数据集；
6. `STREAM-DETECT` 仅在有连续宽带流与目标框标注后做 YOLO/多信号定位。

路线图要将用户已暂缓的噪声方式实验标为“待用户以后恢复”，不能自动排入立即执行任务。

## 检索与证据说明

- 截止日期写 `2026-09-20`。
- 如实说明本轮使用本地 Crossref/OpenAlex 元数据快照、原始论文/出版商页面、arXiv、官方数据/代码页面；未完整访问 WoS、Scopus、Google Scholar，因此是“结构化前沿审计”，不是可声称穷尽的 PRISMA 系统综述。
- 保存纳入/排除标准、证据等级、未核实项和撤稿项。
- 不复制论文长段落。只用短引或中文转述，并保留来源链接。

## 验证与交接

1. 用 PowerShell `Import-Csv` 成功解析矩阵；检查总行数、直接证据数、各任务层级数和空 DOI/URL 数。
2. 检查所有本地相对 Markdown 链接指向存在文件；网络链接只做格式和域名检查，不宣称全部可访问。
3. 检查主报告所有项目数值与现有验收 JSON/计划一致。
4. 检查必备 12 篇/数据集已进入矩阵；CrossRF 仅在排除表。
5. 在 README 中给出推荐阅读顺序与各文件用途。
6. 完成后写 `completed` 交接，列出实际条目数、直接证据数、NR 较多的字段、检查结果和唯一推荐的下一实验任务，然后停止，不自行开始实验。
