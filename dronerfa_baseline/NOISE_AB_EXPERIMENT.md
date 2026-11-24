# 多噪声验证与训练噪声范围配对试验

运行入口是 `run_noise_ab.py`，复用原来的数据预处理、STFT、ResNet18 和训练步骤。
原 `train_resnet18.py` 保留历史行为；新实验通过单独入口执行。

## 固定设计

- A 的训练档位：clean、10、5、0、−5、−10、−12；B 再加入 −15。
- 每个训练窗口在每轮中等概率抽取一个档位。增加档位会改变整个抽样分布。
- 两组各 5 轮、batch size 4、seed 2026；Adam，学习率 0.001，无权重衰减。
- 显式固定每轮样本顺序，并在结果中核对两组初始权重和顺序的哈希。
- 验证条件固定为 clean、−10、−12、−15，每轮使用相同原始窗口和噪声实现。
- 以四档平衡准确率的算术平均选模型，同分选最早轮次。同时保留 clean-only 选择结果。
- 平衡准确率 =（无人机检出率 + 背景正确拒绝率）/ 2。无人机检出率合并两种机型；同时单列每种机型检出率。
- 判定规则固定为 float32(logit_drone) − float32(logit_background) > 0。
- 验证频谱由原前端计算为 float32 并缓存于内存；训练频谱仍逐批实时计算。
- 验证随机种子默认 seed+10000；`--validation-seed` 可独立固定验证噪声。跨训练种子复现统一使用 12026，索引保持完整验证集中的原位置。各档使用相同噪声实现的不同幅度。
- 不执行测试集推理。现有验证文件已经用于多次探索，正式结论需要新的独立录制。

## 从工作区根目录运行

在 PowerShell 中，新输出目录必须尚不存在，以防覆盖既有实验。

```powershell
.\.venv\Scripts\python.exe -u dronerfa_baseline/run_noise_ab.py --epochs 5 --output-dir dronerfa_baseline/outputs/my_noise_ab
.\.venv\Scripts\python.exe dronerfa_baseline/summarize_noise_ab.py dronerfa_baseline/outputs/my_noise_ab
.\.venv\Scripts\python.exe dronerfa_baseline/verify_noise_ab_inputs.py dronerfa_baseline/outputs/my_noise_ab
```

小规模连通性检查可使用 `--epochs 1 --max-train-samples 12 --max-val-samples 9`；这种结果不能用于性能结论。
`verify_noise_ab_inputs.py` 针对完整验证集的实际试验，不用于上述小规模检查。

## 产物

- `experiment_manifest.json`：运行前参数、源代码/索引哈希以及完成状态。
- `validation_cache_manifest.json`：各档实际验证索引与频谱哈希。
- 每组 `training_history.csv`、`validation_metrics.csv` 和逐轮 `validation_scores_epoch_XX.csv`。
- 每组 `epoch_XX.pt`：各轮模型、优化器、AMP 及随机数状态；脚本当前未提供自动续训入口。
- 每组 `selection.json`：clean-only 与 multi-noise 选中的轮次。
- 直接输入复核后，每组生成 `best_model.pt`（multi-noise）和 `best_model_clean.pt`；两者是兼容原诊断脚本的精简推理检查点。
- `report.md`、`selected_model_comparison.csv` 和两张结果图。
- `independent_verification.json`：逐窗口重算混淆矩阵、选模规则与窗口对应关系。
- `raw_input_verification.json`：从原始 IQ 重算所选模型的部分完整批次，核对缓存推理及原文件哈希。

每个验证源文件中取一批中间窗口，在四档噪声下复核；这是抽样输入复核。逐窗口 CSV 统计复核覆盖全部轮次。
训练顺序已显式固定，与历史 pilot 的随机顺序不完全相同；应在本轮 A/B 内比较训练噪声范围，在每组内比较选模方式。

## 多训练种子复现

`run_noise_replicates.py` 固定新增训练种子为 2027、2028，验证噪声为 12026；每个种子各训练 A/B 两组各 5 轮，并执行逐窗口统计复核与原始 IQ 抽查。第一组计算的验证频谱在第二组中复用，且与 2026 原试验逐档核对哈希与索引。

```powershell
.\.venv\Scripts\python.exe -u dronerfa_baseline/run_noise_replicates.py --reference-run dronerfa_baseline/outputs/noise_ab_5epochs_20260917 --output-dir dronerfa_baseline/outputs/my_noise_replicates
.\.venv\Scripts\python.exe dronerfa_baseline/summarize_noise_replicates.py dronerfa_baseline/outputs/my_noise_replicates
```

单独运行某个种子时，同样明确指定 `--validation-seed 12026 --reference-validation-run dronerfa_baseline/outputs/noise_ab_5epochs_20260917`。
跨种子报告列出每个种子的配对差值及其描述性均值、样本标准差，不把重复使用的验证窗口汇总为新的独立样本量。

每次运行保存 `source_snapshot`，按原始文件哈希核对对应运行版本。2026 原试验也已补存与其原 manifest 哈希一致的源代码快照；源代码后续升级不会抹去其运行依据。
