# DroneRFa 项目共享摘要

更新时间：2026-09-20。用户要求沿用 deepseek-workflow；Codex 规划与验收，DSH Desktop DeepSeek 做具体实现和文档。任务串行执行，不自动提交/推送/发布，不恢复已取消的定时器或 heartbeat。

## 当前授权与任务

- 当前任务：`LIT-AUDIT1`，整理可追溯的前沿文献矩阵，并把已有基线与前沿论文逐项对齐。
- 任务文件：`work/deepseek-tasks/20260920-literature-frontier-audit.md`
- 外部证据说明：`work/literature-frontier-source-note-20260920.md`
- 允许改动仅限新目录 `literature_review_from_scratch/frontier_update_20260920/` 和本次交接文件。
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

## 工作流状态

- 上一任务 `PACK-RESULTS1` 已验收，当前无活动运行。
- DSH Desktop 健康检查已通过，工作区可写。
- 当前需要派发并完成 `LIT-AUDIT1`；执行器交接后由 Codex 检查 CSV、链接、条目数量、关键数值和下一任务建议，再更新计划和状态。
