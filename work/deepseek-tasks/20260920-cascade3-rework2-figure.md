# CASCADE3 验收返工 2：修复召回率图横轴标签

## 问题

Codex 已用项目 `.venv`/CUDA 复跑独立验证器，`work/cascade3-codex-verification-after.json` 为 `passed=true, failures=0, formal_environment`，48 项测试全过；数值、预测、混淆矩阵和返工1根因均已通过。

视觉抽查发现 `dronerfa_baseline/outputs/cascade3_evaluation_20260919/recall_by_condition.png` 只显示 `clean` 横轴标签，虽然 heldout 的 clean/-10/-12/-15 四组数据点均已画出。根因在 `run_cascade3_evaluation.py` 的 dataset 循环内反复调用 `axis.set_xticks(positions, conditions)`：最后处理的 old_test 只有 clean，覆盖了 heldout 四档刻度。

## 修复与验收

1. 只修绘图逻辑，使横轴明确显示 `clean`、`-10`、`-12`、`-15`；old_test 仍只在 clean 位置显示，不改变任何数据。
2. 重新生成 `recall_by_condition.png` 及必要的 `figure_evidence.json`、`experiment_manifest.json` 哈希；不要重跑正式推理，不改预测、指标、混淆矩阵或五个 digest 文件。
3. 增加一个能阻止“后处理单条件数据集覆盖四档刻度”的有意义测试。
4. 复跑相关测试、manifest核对和 `--no-write` 独立验证器。保存返工前后正式非图表产物哈希不变证据。
5. 完成后交接并停止执行器。不得扩展研究范围、训练模型、恢复定时器、提交、推送或发布。
