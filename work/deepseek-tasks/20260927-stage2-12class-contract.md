# STAGE2-12-CONTRACT：十二机型索引、标签与共享输入接口

状态：用户已授权后续路线；待 Stage1 对照验收和版本冻结后由 Codex 正式派发。本文件不是当前 R1 的新增执行范围。本任务只准备接口与有界 train/val smoke，不启动完整训练、不读测试 IQ。

## 冻结来源与映射

读取冻结 104 MAT 表与已验收 Stage1 全窗口索引，哈希见 `work/deepseek-context.md`。不能重新划分、重新截窗或直接调用旧两机型训练任务。保留所有源列，额外建立 Stage2 无人机视图及显式映射。

| stage2_label | target_code | train窗 | val窗 | development_test窗 | development_test_exposed窗 |
|---:|---|---:|---:|---:|---:|
| 0 | T0001 | 760 | 140 | 270 | 0 |
| 1 | T0010 | 590 | 310 | 0 | 300 |
| 2 | T0011 | 770 | 150 | 300 | 0 |
| 3 | T0100 | 760 | 160 | 310 | 0 |
| 4 | T0101 | 620 | 340 | 0 | 310 |
| 5 | T0110 | 780 | 150 | 280 | 0 |
| 6 | T0111 | 750 | 150 | 310 | 0 |
| 7 | T1000 | 730 | 140 | 290 | 0 |
| 8 | T1001 | 870 | 150 | 290 | 0 |
| 9 | T1010 | 670 | 110 | 280 | 0 |
| 10 | T1011 | 840 | 150 | 290 | 0 |
| 11 | T1100 | 850 | 130 | 330 | 0 |

共 96 无人机 MAT / 14,630 窗；四角色文件数 58/14/20/4，窗口数 8990/2080/2950/610。T0000 不进入 Stage2 训练/验证视图。未知 T code、重复键、计数不符均报错，不可静默丢行。

现有 `class_index` 是 T 码数字化，现有 `label` 是 Stage1 0/1，均不能拿来作为十二类标签。保留 `row_index`、原始角色、文件和窗位置等原列，新增 `stage2_label`、`cascade_label=stage2_label+1` 和 `full_split_position`。完整 cascade 映射另含 T0000=0。

## 输入身份与噪声接口

- 原始窗口键为 `(source_role, file_path, start, length)`，另保留原 row_index。
- `full_split_position` 是先将冻结全索引按原 row_index 顺序取某角色（包含背景），再从 0 开始编号；在排除背景之前确定，不能重新编号。
- Stage2 训练顺序可对无人机视图用 `torch.randperm(8990, generator=2027+epoch)`；训练加噪 RNG 固定为 `default_rng(2027 + epoch*9590 + full_split_position)`，epoch 从 1 开始。档位抽取/加噪和去均值、RMS 的顺序与 Stage1 相同。此规则显式记录，不能误称完全复用旧两机型的过滤位置规则。
- val 固定 epoch=0、noise_seed=12026，RNG 为 `default_rng(12026+full_split_position)`。clean 不加人工噪声。训练噪声 clean/10/5/0/-5/-10/-12；评价 clean/-10/-12/-15。
- 后续两个测试角色各自的 full_split_position 同上述定义；预先固定 noise_seed：development_test=22026，development_test_exposed=32026。无须现在打开测试 IQ。
- 共用 RF0、1M 非重叠窗、512/hop256、Hamming、256x256 前端；验证/测试每个窗口和条件只生成一个预处理 IQ/谱图，oracle 与 cascade 后续共用，保存可比 digest。不得依据 Stage1 判决重抽噪声。

## 实现范围与预检

允许新增 `dronerfa_baseline/stage2_12class_data.py`、索引/协议生成入口、相关必要测试和新输出目录 `outputs/stage2_12drones_20260927/contract/`。旧 `run_stage2_classification.py`、旧两机型产物只作参考，不改写。可复用已验收基础前端/网络与读取接口，不能照搬旧二分类 metrics 或 softmax[:,1]。

输出显式 mapping.json、stage2_index.csv、protocol.json、源文件零交叉核验、预期窗口/文件计数、输入身份与噪声坐标说明。用实际 train/val 的少量有界样本证明新视图与同坐标 Stage1 输入逐字节一致（同条件），验证12输出shape/argmax映射。测试元数据可用，但测试 MAT 零打开。

必要负向用例：非连续 class_index 被误用、漏掉一类、混入背景、源文件角色移动、full_split_position 被过滤后重编号、映射打乱。完成交接后由 Codex 验收本接口，再派完整训练。
