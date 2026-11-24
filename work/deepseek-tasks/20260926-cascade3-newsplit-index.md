# CASCADE3-NEWSPLIT-A：新冻结划分的三类窗口索引

用户要求按 2026-09-26 冻结的 104 文件开发划分，重新训练原来的“背景 + Phantom 4 Pro + Air 2S”两阶段基线。本项先完成可审计的数据入口，不训练模型。

## 权威输入与身份

- `work/meta_split2/CODEX-FREEZE-20260926.json` 指定的 CSV 和 SHA256 是角色唯一来源；CSV 为 `work/meta_split2/split_candidate_104_20260926.csv`，SHA256 `d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9`。
- 数据来自 `G:\DroneRFa数据集\original`，正式清单 `G:\DroneRFa数据集\dataset_manifest.csv` SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`。
- 只取 T0000、T0010、T0101，24 个 MAT。各类 S0000/S0001/S0010/S0011 为 train，S0100/S0101 为 val，S0110/S0111 为 `development_test_exposed`。旧三类验证/测试均已用于此前实验，绝非全新 final test。
- 不得从 `work/meta_split1/source_table_104.csv` 的旧 `prior_use_role` 判断使用历史。

## 实现范围

1. 在 `dronerfa_baseline/` 新增独立的新划分索引生成代码，不改历史 `outputs/window_index.csv`、heldout 索引、旧权重、旧实验输出，也不改 G 盘原数据或清单。新产物位于 `dronerfa_baseline/outputs/cascade3_newsplit_20260926/`。
2. 复用项目现有 RF0、100 MS/s、1,000,000 采样/窗、非重叠窗口规则。每个源文件尽可能使用全部完整 1M 窗；尾段不足 1M 丢弃。建立窗口级索引，标明源文件、源文件角色、目标类、起点、长度、窗口序号、阶段1二元标签、最终三类标签。阶段2只从两个无人机类读取。
3. 任何同一个源 MAT 不能跨 train/val/test。对所有 24 个 G 路径实际存在、文件大小匹配冻结表、索引边界/无重叠、类别及角色数、索引覆盖和 SHA256 的身份约束做可复算核验。若全部完整窗口使训练显著超出既有规模，报告规模和估时，先不要任意抽样或改变规则；留给 Codex 决定。
4. 输出简短数据方案说明及机器可读清单，列出各类各角色 MAT 数和窗口数。任何测试文件不得用于训练、选 epoch、选阈值或选超参数。
5. 保持原两阶段架构、n_fft=512/hop=256、256×256、三个种子 2026/2027/2028、5 epoch、旧训练噪声方案的后续可复用性；本项只建入口，不训练。

## 验收

- 冻结 CSV 和 G 清单身份吻合，24 文件、每类 4/2/2 源文件；源文件零交叉。
- 全部窗口由 G 盘实际 RF0 源读取规则可达；索引中没有 F 盘或旧 junction 路径。
- 新产物与旧正式实验隔离；提供命令和检查结果。
- 在 `work/deepseek-handoffs/<runId>.json` 写 completed/needs_codex/needs_user 交接，列出实际文件、检查、窗口量及后续训练注意点。
