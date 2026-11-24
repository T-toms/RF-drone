# CASCADE3-NEWSPLIT-C：新划分两阶段串联评价与数据划分报告

依赖 CASCADE3-NEWSPLIT-B 经 Codex 验收：seed 2026 两个阶段的最佳 checkpoint 均已根据新 val 冻结。此项才允许读取三类 `development_test_exposed` 测试窗。

## 串联评价

- 输入固定为新索引 SHA256 `deed870f9693c5ebe93fb80a9647b452e90680dd5b1f21372e108ded4bf4d231` 的 890 个已暴露测试窗，并使用 B 验收的两个新 checkpoint；仅 seed 2026。
- 复用旧 CASCADE3 的 RF0/1M/100 MS/s、去均值、RMS、Hamming STFT n_fft=512/hop=256、256×256 特征，阶段 1 和阶段 2 对同一窗口共享同一个特征张量。
- Stage 1 margin > 0 执行 Stage 2，否则 Background；Stage 2 margin > 0 为 Air 2S，否则 Phantom 4 Pro。阈值零固定，不准根据测试改。
- 先评 clean 条件，完成完整三分类窗口级预测 CSV、按文件等权指标、混淆矩阵、阶段 1 漏检与阶段 2 错分分解，以及输入/权重/配置 SHA256。若用户没有进一步授权，不自动恢复旧 AWGN 多档实验。
- 独立从逐窗预测重算指标，检查 890 唯一窗全覆盖、两个阶段决策与最终类一致、source_role 全部为 `development_test_exposed`。新结果独立目录，不覆盖旧实验。

## 用户要的详细 Markdown 数据划分说明

在 `reports/DroneRFa_数据集划分与三类二阶段复跑_20260926.md` 编写可直接交付的详细中文 Markdown：

1. 从项目正式 G 清单与 `work/meta_split2/CODEX-FREEZE-20260926.json` 阐明 104 个 MAT、13 类（12 机型+背景）、各类 8 文件、源文件级最小分组、总 62/16/26 文件、约 100.07 GiB，G 盘路径与冻结 SHA256。
2. 逐类表列类别码、名称、train/val/test 的 S 码及数量、所在文件时间夜；旧三类为各 4/2/2，新十类为各 5/1/2，分别说明 `development_test` 与 `development_test_exposed`。
3. 写明新十类 20 个同时间簇文件级测试可作开发主指标；旧三类 6 个测试文件以前参加实验，只能作已暴露对照；不存在 13 类跨采集会话或全新 final test。MAT 文件时间只是批次代理，不是采集会话证明；不把窗口随机拆到不同角色。
4. 单列本次三类复跑 24 MAT、3630 窗的训练/验证/测试文件数和各类窗口数，阶段 1 与阶段 2 标签语义、1M/10ms 非重叠窗口规则，历史旧索引与新索引规模及角色变化。
5. 报告新单种子两阶段训练配置、验证选型、clean 串联测试结果与局限；如果未完成某项，如实标出。

## 交接

写 `work/deepseek-handoffs/<runId>.json`，列出 completed/needs_codex/needs_user、产物、核验和需 Codex 处理的问题。不得改 G 原数据、冻结划分 CSV、旧实验权重/预测/报告，不提交或发布。
