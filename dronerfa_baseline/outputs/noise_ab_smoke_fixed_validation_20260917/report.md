# 原噪声范围与增加 −15 dB：配对试验

本轮只使用训练集和验证集。选模规则在运行前固定；本报告为探索性验证结果。

A：clean,10,5,0,−5,−10,−12；B：上述范围加 −15。各档等概率抽样，因此 B 同时降低了原各档的抽样概率。
两组初始权重及每轮训练窗口顺序一致；同一份 float32 验证特征用于所有模型。验证 AWGN seed=12026，保留完整验证索引。
模型选择：clean、−10、−12、−15 四档平衡准确率的算术平均，分数相同时取最早轮次；同时记录只按 clean 选择的结果。
默认门槛保持 logit 差 > 0（数学上等价于 drone 概率 > 0.5）。

| 组别 | 选模方式 | 轮次 | 附加噪声 | 平衡准确率 | Air 检出 | Phantom 检出 | 背景误报 |
|---|---|---:|---|---:|---:|---:|---:|
| A_original | clean_only | 1 | clean | 50.00% | 3/3 | 3/3 | 3/3 |
| A_original | clean_only | 1 | -10dB | 50.00% | 3/3 | 3/3 | 3/3 |
| A_original | clean_only | 1 | -12dB | 50.00% | 3/3 | 3/3 | 3/3 |
| A_original | clean_only | 1 | -15dB | 50.00% | 3/3 | 3/3 | 3/3 |
| A_original | multi_noise | 1 | clean | 50.00% | 3/3 | 3/3 | 3/3 |
| A_original | multi_noise | 1 | -10dB | 50.00% | 3/3 | 3/3 | 3/3 |
| A_original | multi_noise | 1 | -12dB | 50.00% | 3/3 | 3/3 | 3/3 |
| A_original | multi_noise | 1 | -15dB | 50.00% | 3/3 | 3/3 | 3/3 |
| B_plus_minus15 | clean_only | 1 | clean | 50.00% | 3/3 | 3/3 | 3/3 |
| B_plus_minus15 | clean_only | 1 | -10dB | 50.00% | 3/3 | 3/3 | 3/3 |
| B_plus_minus15 | clean_only | 1 | -12dB | 50.00% | 3/3 | 3/3 | 3/3 |
| B_plus_minus15 | clean_only | 1 | -15dB | 50.00% | 3/3 | 3/3 | 3/3 |
| B_plus_minus15 | multi_noise | 1 | clean | 50.00% | 3/3 | 3/3 | 3/3 |
| B_plus_minus15 | multi_noise | 1 | -10dB | 50.00% | 3/3 | 3/3 | 3/3 |
| B_plus_minus15 | multi_noise | 1 | -12dB | 50.00% | 3/3 | 3/3 | 3/3 |
| B_plus_minus15 | multi_noise | 1 | -15dB | 50.00% | 3/3 | 3/3 | 3/3 |

## 解释边界

- 这是单随机种子的配对试验，不能据此声称稳定增益或统计显著。
- 验证集每类仅一个源文件；同一文件的窗口及同窗口的各噪声档互相关联。
- 用于选模的数据也用于本报告描述，结果可能偏乐观；需新的独立采集检验泛化。
- −15 dB 指额外加入的 AWGN 条件，不是现场绝对信噪比。
- 本轮显式固定了配对训练顺序，不能把 A 与历史 pilot 的差异仅归因于选模方式。
- 每轮检查点含模型、优化器、AMP 和随机数状态；selection.json 指向所选轮次。
