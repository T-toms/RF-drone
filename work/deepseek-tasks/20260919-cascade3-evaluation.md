# CASCADE3：第一阶段与第二阶段直接串联三分类评价

## 目标

在不重训、不调门槛的前提下，把已验收的第一阶段 Background/Drone 检测器与第二阶段 Phantom/Air2S 分类器直接串联，得到 Background、DJI Phantom 4 Pro、DJI Air 2S 三分类结果。完整实现、报告和必要测试交给 DSH Desktop；Codex 最终独立验收。

冻结配置以 `work/cascade3-config-20260919.json` 为唯一依据。开始正式推理前验证该文件、两个索引和六个检查点的 SHA256；正式推理后配置不得改变。

## 架构与推理规则

1. 只用 `n_fft=512, hop_length=256, image_size=256`。RF0、1,000,000 点窗和既有前端保持一致。
2. 三套同种子级联：2026 的 Stage1 对 2026 的 Stage2，2027 对 2027，2028 对 2028。不做跨种子投票或集成。
3. 每个数据集、条件和窗口只生成一次噪声和一次 `1x256x256` 特征。噪声 RNG 位置按完整 split 的行号计算，背景行也占位置。同一张特征张量先给 Stage1；若通过，再原样给 Stage2。
4. Stage1 margin=`logit_drone-logit_background`。只有 margin>0 才执行 Stage2；否则最终标签为 Background，并设置 `stage2_executed=false`，Stage2 logits/margin 留空或 NaN。
5. Stage2 margin=`logit_air2s-logit_phantom`。margin>0 输出 Air2S，否则输出 Phantom。
6. 禁止通过连接已有 Stage1/Stage2 预测 CSV 实现级联。必须从原始 IQ 重算；旧 Stage2 oracle 结果使用无人机过滤后位置，不能代表真实级联输入。

## 数据与规模

- 新 heldout：1820 窗，四档 `clean/-10/-12/-15`，三个种子级联，共 21,840 条预测。
- 旧 test：440 窗，只做 clean，三个种子级联，共 1,320 条预测。
- heldout 类别：Background 560、Phantom 610、Air2S 650。
- 噪声 seed=22026、epoch=0。clean 不加噪声，其他档沿用既有复 AWGN 实现。
- heldout 已参与此前消融，因此这里是描述性留出评价，不能称全新未触碰最终测试或跨会话泛化。

## 实现与产物

建议新增：

- `dronerfa_baseline/run_cascade3_evaluation.py`
- `dronerfa_baseline/test_cascade3_evaluation.py`
- `dronerfa_baseline/verify_cascade3_evaluation.py`

统一输出到 `dronerfa_baseline/outputs/cascade3_evaluation_20260919/`，至少包含：

- `window_predictions.csv`：数据集、seed、condition、完整窗口身份、真实类、Stage1 两 logits/margin/decision、`stage2_executed`、Stage2 两 logits/margin、最终预测。
- 每个数据集/seed/condition 的 3x3 混淆矩阵和汇总 CSV/JSON。
- 每源文件统计，支持文件等权主指标。
- 跨种子均值/样本标准差与同种子表。
- `report.md` 中文报告。
- clean、-10、-12、-15 的三分类混淆矩阵图；类别召回率随条件变化图。可加入简洁流程图，不生成逐窗 PNG。
- 输入验证、配置 SHA、特征摘要、运行环境、执行日志和 `verification.json`。

## 指标

每个 seed/condition 至少报告：

- pooled overall accuracy、macro recall/balanced accuracy、macro precision、macro F1；
- 每类 recall；
- 文件等权主指标：先计算每个源文件的该类正确率，再在类内对源文件等权，最后三个类等权；
- Stage1 背景误报率、无人机漏检率；
- 真实无人机且通过 Stage1 的样本上，Stage2 条件分类准确率；
- 最终错误分解：Background→Phantom、Background→Air2S、Phantom→Background、Air2S→Background、Phantom↔Air2S。

## 验证与拒绝用例

1. 六个检查点、两个索引和既有冻结输入哈希一致；评价后再次核对，禁止修改。
2. heldout 恰好 21,840 行，old test 恰好 1,320 行；主键唯一，每个组合按索引完整覆盖。
3. 每窗口保存特征 digest；同一窗口的 Stage1/Stage2 输入必须同 digest。Stage2 未执行时不得伪造 logits。
4. `stage2_executed` 当且仅当 Stage1 margin>0；最终标签映射逐行复算必须一致。所有已执行 logits/margin 有限。
5. 汇总混淆矩阵、指标、文件等权口径可从预测 CSV 独立复算。
6. 用构造数据测试四类关键路径：背景正确；背景误报后分到两机型；无人机被 Stage1 漏掉；两机型在 Stage2 互相混淆。
7. bounded smoke 只能在正式输出之外；不得以小样本覆盖正式产物。
8. 可对 Stage1 n_fft512 的少量或完整结果与既有相同协议预测作严格一致性核对；Stage2 不得要求与旧 oracle CSV 一致，因为 RNG 行位置不同。

## 边界

本任务不训练模型、不用 heldout 选配置或门槛、不做噪声方式实验、不添加未知类、不制作任意外部信号上传界面、不提交/推送/发布、不恢复定时器。完成后以 `completed/needs_codex/needs_user` 交接报告上报并停止执行器，等待 Codex 验收。
