# STAGE1-12DRONES-A：104 MAT 的背景 vs 12 机型一阶段训练

## 用户目标与本阶段范围

用户已澄清要在既有二阶段基线结构上使用当前 **12 种无人机 + 背景**的全部数据，并明确要求先开始**第一阶段训练**、使用 deepseek-workflow。本阶段交付：冻结划分驱动的 104 MAT 窗口索引（必要数据入口）、第一阶段 Background vs Drone 训练与验证、checkpoint 和可审查验收证据。不要训练第二阶段，不对任何测试角色执行评价，不改已有三类权重/产物。

## 冻结输入和数据语义

- G 正式清单 `G:\DroneRFa数据集\dataset_manifest.csv`，SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`。
- 冻结决策 `work/meta_split2/CODEX-FREEZE-20260926.json` 与 CSV `work/meta_split2/split_candidate_104_20260926.csv`，CSV SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`。
- 104 MAT，13 类；源文件角色 62 train、16 val、20 `development_test`、6 `development_test_exposed`。其中背景 T0000 为 4 train / 2 val / 2 exposed test；12 机型为 58 train / 14 val / 20 dev test / 4 exposed test。
- Stage1 标签：T0000=Background(0)，其余 12 个 T 码=Drone(1)。从冻结 CSV `role` 读取划分，绝不由 S 码或旧 `source_table_104.csv` 的 `prior_use_role` 推断。
- 不修改 G 数据、正式清单和冻结 CSV。训练/验证不能读取 26 个测试 MAT 的 RF0；元数据索引可记载测试文件，但原始 IQ 读取审计必须证明只有 train/val。

## 已验收的协议参考

- 参考 `dronerfa_baseline/run_cascade3_newsplit_training.py`、`build_newsplit_index.py`、`dronerfa_data.py`、`model.py`、`train_resnet18.py`。保留三类脚本与产物不变，在新脚本/新输出目录实现 13 类 Stage1。
- RF0 I/Q；每窗 1,000,000 采样、非重叠、尾段不足一窗丢弃；每窗去均值/RMS 归一化；Hamming STFT n_fft=512、hop=256，256×256 图；ResNet18 单通道输入、2 类输出。
- 基线训练口径：seed 2026；5 epochs；batch size 4；Adam lr=1e-3、weight decay=0；类别加权交叉熵；CUDA AMP float16；训练噪声口径沿用 `clean,10,5,0,-5,-10,-12` dB 的 A_original 规则与既有确定性播种方法。验证只用 val 文件，在 clean/-10/-12/-15 dB 四条件、seed 12026 下固定噪声；以四条件 pooled balanced accuracy 均值选 epoch，同分取最早。
- 训练使用全部 train/val 窗，不抽样，不用测试指标/文件调阈值或选型。推理判 Drone iff float32(logit_drone-logit_background)>0。
- 现有 GPU 为 RTX 4060 Laptop；任务可较久，请在 DSH Desktop 原生会话保持可见并产生日志/进度。

## 实现建议与产物

1. 新建 `dronerfa_baseline/outputs/stage1_12drones_20260926/` 等独立路径。构造全 104 MAT 窗口索引（CSV）、摘要（JSON）、源身份与核验（JSON），按冻结 CSV 完整覆盖和准确角色断言。可重用索引构建逻辑，但不要修改旧三类索引。只读 HDF5 元数据推算窗口数，并针对 train/val 做有界真实 RF0 读取探针；已验收 G 清单包含 MAT SHA256，避免为了文档或索引再全量重哈希 100 GiB，除非核验确有必要。
2. 训练前完成索引/标签/角色门禁：104 源文件全集，13 类每类 8，零跨角色，所有窗口继承源 MAT，train/val/test 文件集合互斥。记录输出索引 SHA256 和实际窗口数。
3. 实施第一阶段训练、验证、按规则选最佳 epoch；保存 checkpoint、逐 epoch 训练/验证指标、run config、输入身份、环境与读取审计。checkpoints 采用新目录，明确模型类别映射。
4. 独立核验：训练/验证读取审计无测试文件；checkpoint 可重新加载；窗口索引和冻结表一致；验证指标/选型可从明细复算；有针对性的负向对照用于检出角色污染或标签错误。补必要的回归测试。可用旧协议验证脚本作参考，但新验证不能仅信任训练脚本的自报。
5. 若实现中遇到真实技术问题，报告 needs_codex，说明已经尝试的方案。不要因耗时而自行抽样、缩短 epoch、改变协议或混入测试数据。

## 验收边界

- 第一阶段完成的定义：全量训练结束、最佳 checkpoint 存在且可重载、独立验证和读取审计通过，DSH 提交 `completed` 交接供 Codex 验收。
- 本阶段不能报告对 12 机型的最终识别准确率：那属于尚未开始的 Stage2 和串联评价。Stage1 的 val 指标仅是开发选型指标。D00 与无 D 字段机型及不同文件时间簇可能让二分类学习到非机型因素，报告中需说明这一限制。
