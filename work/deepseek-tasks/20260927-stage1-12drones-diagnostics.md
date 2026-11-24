# STAGE1-12DRONES-DIAG：一阶段验证错误诊断

## 目标

用户要求继续审视一阶段，在进入第二阶段前确认当前 seed 2026 的表现与问题。基于已经完成且验收的 `STAGE1-12DRONES-A` 产物，做**只读、可复算**的验证集错误诊断，给 Codex 一份具体结论和下一轮训练建议。本任务不启动新训练、不读取测试角色的原始 IQ、不用测试结果选型。

## 冻结依据

- 划分：`work/meta_split2/split_candidate_104_20260926.csv`，SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`。
- 索引：`dronerfa_baseline/outputs/stage1_12drones_20260926/window_index_stage1_12drones.csv`，SHA256 `5973a2fa5b8476a507fcc7a665365daf8d572f569ce5908d54f5c4bc84c48bd6`。
- 已验收训练：`dronerfa_baseline/outputs/stage1_12drones_20260926/seed_2026/`。选中 epoch 3，`best_model.pt` SHA256 `37dbb815450b8ec0f798a113a7f922858fcb2be604a863eb7594eb5c52ff814b`。
- 主要数据：`validation_scores_epoch_01.csv` 至 `validation_scores_epoch_05.csv`、`validation_metrics.csv`、`training_history.csv`、`selection.json`、`summary.json`。每轮的验证窗口只来自 16 个 val MAT；四种条件 clean/-10/-12/-15 dB。

## 允许工作与交付

1. 只新增独立的诊断脚本、报告和 JSON/CSV 到 `dronerfa_baseline/` 的新文件及 `dronerfa_baseline/outputs/stage1_12drones_20260926/diagnostics/`；可在 `reports/` 生成简明中文 Markdown。不要修改已验收训练脚本、checkpoint、逐窗评分、冻结划分或 G 盘数据。
2. 从逐窗评分**独立复算**每个 epoch×条件的背景特异度、无人机召回率、平衡准确率、FP/FN 和文件等权指标，并与已落盘的 `validation_metrics.csv` 对照，差异需明确记录。
3. 对选中 epoch 3，按 12 个无人机 T 码、16 个 val 文件、四个噪声条件列出样本数、漏检数、召回率、margin 分布摘要；背景两文件列出虚警数及特异度。指出错误最集中的机型、文件和噪声条件。数值中要清楚区分窗口加权和文件等权，避免把 12 个 T 码误当 12 类分类准确率。
4. 分析五轮验证轨迹：epoch 3 与其他轮次在各条件、背景/无人机、文件之间的变化。重点判断第 4/5 轮均值下降是哪些条件或文件造成，是否有过拟合迹象；只把可从记录支持的事实写为结论，不凭单种子断言原因。
5. 输出下一步建议：是否需要多随机种子全量重复、需要哪些最小诊断或改动，以及成本估计。训练建议不得依赖测试角色数据。本任务不实际训练。
6. 独立核验评分每轮每条件恰有全部 2,360 个 val 窗、16 个 val 文件，键唯一，角色和标签与冻结索引一致；任何缺失、重复、跨角色或差异要失败并上报。只补能够捕获真实错误的必要测试/负向对照。

## 验收条件

- DSH 提交 `completed` 交接，包含产物路径、关键数字、所做检查和问题。
- Codex 能从报告及机器可读明细追溯最差机型/文件、噪声条件及 epoch 变化；汇总与原验证指标一致。
- 本任务未新训练，未读测试 IQ，未改变已验收的一阶段产物。
