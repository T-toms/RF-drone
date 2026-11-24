# CASCADE3 验收返工 1：修复跨进程原始 IQ 重建验证

## 背景

CASCADE3 正式推理和静态产物已完成，但 Codex 在一个新的普通 Python 进程中复跑独立验证器失败，因此当前不能验收。保留现有正式预测、摘要和汇总，先查清验证重建差异；只有证据证明正式预测错误时才允许重跑全量推理。

失败命令（项目根目录）：

```powershell
python dronerfa_baseline\verify_cascade3_evaluation.py --no-write --report work\cascade3-codex-verification.json
```

实际结果：`passed=False failures=4`，约 51.6 秒：

- 255 个目标特征 digest 与 CSV 不同；
- Stage1 最大 logit 差 0.002524852752685547，大于 1e-4；
- Stage2 最大 logit 差 0.007019996643066406，大于 1e-4；
- 242 个 decision/rule mismatch 计数。

完整证据：`work/cascade3-codex-verification.json`。同一次检查中，23,160 行身份与静态决策规则、全部汇总回算、六权重/两索引/14冻结输入、五组正式摘要对既有 Stage1 摘要、Stage1 发布 logits parity 均通过；失败只在 `target_recomputation`。

## 必须完成

1. 用上面的**直接命令**复现，不得只通过 `work/_c3_driver.py` 或引用先前日志。说明为什么先前交接宣称同一验证为全零，而新进程会失败。检查 Python/torch/CUDA 环境、GPU STFT/cufft 状态、前端调用顺序、batch=4 构成、数据 epoch/seed 和缓存路径。
2. 把 `rule_mismatches` 拆成可审查的独立计数：Stage1 执行门变化、Stage1 margin 超差、Stage2 最终类变化、Stage2 margin 超差、Stage2 batch id 错误。不要把数值超差称为“决策变化”。
3. 验证原始 IQ 与加噪输入的确定性应独立于 GPU STFT：加入每目标窗或足够强的输入 tensor SHA256 复算证据，并核对正式运行已记录的 input probe。不得以放宽 logit 容差替代输入核对。
4. 若 GPU STFT/AMP 跨进程存在小数值非确定性，给出至少两次全新进程复跑的实际差异分布，确定有依据的特征/logit比较方法，并单独要求 Stage1执行门、Stage2最终机型和最终三分类标签在目标窗上一致。不得静默把阈值调到刚好掩盖本次错误。
5. 若可以通过正确复现正式 batch/warm-up/调用顺序得到稳定结果，优先修正验证器以复现。若不能位级复现，报告限制并用“输入精确一致 + 数值有界 + 离散决策一致”替代不可靠的 float32 字节哈希判据；保留正式特征摘要与既有 Stage1 摘要逐字节一致这一独立历史证据。
6. 从项目根目录连续运行两次直接验证命令，两次都必须 `passed=true/failures=0`。复跑相关单元测试。将 Codex 的失败 JSON 保留为首次验收失败证据，另写修复后报告，不覆盖它。
7. 核对正式 `window_predictions.csv`、指标 CSV、五个正式 digest 文件的 SHA256 未因返工意外改变。若无需重跑正式推理，明确记录没有改变；若确需重跑，先在报告中解释证据和范围。

## 范围

允许修改 CASCADE3 的验证器、测试、报告/manifest 及必要的验证证据。不要训练模型、调阈值、改变冻结配置、扩展类别或噪声实验、恢复定时器、提交/推送/发布。完成后写新的 completed/needs_codex 交接并停止执行器，等待 Codex 复验。
