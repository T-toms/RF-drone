# STAGE2-12-BASE：一次十二机型正式训练

状态：已授权路线中的待派发项。依赖 STAGE1 冻结与 STAGE2-12-CONTRACT 经 Codex 验收；未正式派发不得开始。

## 目标与边界

训练一个已知无人机输入下的十二机型分类器。仅真实无人机 train/val，不使用 Stage1 的预测筛选样本，不含背景，不做新 seed/n_fft/模型搜索。本步不读任何测试 IQ；最终 oracle/cascade 测试是后续锁定评价任务。

## 运行前冻结

- 数据和完整 T code→0..11 映射、full_split_position/noise RNG 依 `20260927-stage2-12class-contract.md` 的已验收版本；输入/协议/代码哈希写入新 run_config。
- seed2027，完整5 epoch，train8990/val2080；RF0 1M窗、512/hop256/256x256；单通道 ResNet18 输出12，batch4、Adam lr1e-3 wd0、FP16 AMP。项目 .venv/CUDA，与正式 Stage1 相同设备；仅使用 Codex 已批准的加载配置。
- 本 Stage2 基线沿用传统 weighted CE：`w_c=8990/(12*n_train_c)`，PyTorch 默认按批内权重和归一化。权重只来自 train。Stage1 单次归一化对照不外推为 Stage2 最佳损失。
- 训练和评价噪声、顺序按 contract 预声明规则；12类训练指标独立实现 multiclass，不能返回 binary_metrics。
- 主选型分数：每条件先算每个 val 源文件的正确率，再同类别的文件等权平均，再12类别等权平均，最后四条件等权平均；最高者胜，同分取最早 epoch。旧两种各有2个val文件，不能直接平均14文件使其双倍加权。
- 辅助报 pooled accuracy、支持类宏 recall/F1、12x12混淆矩阵、每类/文件召回和最差条件；辅助结果不改变选型规则。

## 产物、恢复与验收

新增训练入口和必要独立核验；正式输出 `dronerfa_baseline/outputs/stage2_12drones_20260927/seed_2027/`，拒绝覆盖既有结果。保存五轮全部 val 12维 logits/预测、逐文件统计、权重、选型、初始化与顺序哈希、环境和代码身份、实际读取事件/消费及日志。已有其他任务代码及成果保持兼容。

保存 epoch 边界 resume checkpoint，包括 model/frontend/optimizer/scaler/Python NumPy torch CPU CUDA RNG/epoch/历史/最佳选择/身份；复用已验收实现，不重跑无关恢复测试。旧只含权重的 checkpoint 不当成可精确续训状态。

常规预算：训练原始读取8990×5=44950窗；val缓存2080×4=8320窗。任何真实额外读取如 smoke/重载独立列账，不伪造固定总量；78个train/val中仅无人机72文件（58+14），测试和背景原始IQ零读取。

独立核验全量完成、每epoch/condition键唯一且覆盖2080窗、标签映射与12logits正确、所有指标和选epoch可复算、audit逐窗/文件预算精确、模型可重载且预测一致、冻结输入和旧权重未变。可用缓存核验重载，无须反复打开MAT。

输出中文 Markdown 简报，明确单seed、同批次文件留出、开发验证及局限。低成绩不能触发擅自多跑；NaN/协议冲突等交 needs_codex。完成唯一handoff后等待 Codex 验收，再锁定测试。
