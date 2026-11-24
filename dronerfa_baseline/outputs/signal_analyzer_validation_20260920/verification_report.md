# INFER-UI1 任意IQ信号分析界面 —— 独立验证报告

- 生成时间（UTC）：2026-09-20T06:02:03.654738+00:00
- 项目：`C:\Users\Administrator\Desktop\数据集处理`
- 运行环境：`C:\Users\Administrator\Desktop\数据集处理\.venv\Scripts\python.exe`，torch 2.12.1+cu130，device=cuda（NVIDIA GeForce RTX 4060 Laptop GPU），项目 .venv=True，AMP float16=True

## 结果总览

| 检查 | 结论 | 关键证据 |
| --- | --- | --- |
| 冻结产物未改动 | 通过 | window_predictions.csv SHA256 `59321de7be48aae7…`，6 个权重 hash 与冻结配置一致 |
| 三种子多数决复算 | 通过 | 7720 个 dataset/condition/window 键，1-1-1 共 0 窗 |
| 真实原始IQ走新核心路径 | 通过 | 每数据集 3 个目标窗（背景/Phantom/Air2S），逐种子离散决策与 CASCADE3 完全一致 |
| 复现 CASCADE3 的 batch=4 组成 | 通过 | 6 组连续 4 窗，Stage1 logits 最大差 0 |
| 格式/窗口/路径/取消 单元测试 | 通过 | 12 项（+ 7 项 HTTP 服务集成测试，`test_signal_analyzer.py` 共 19 项），失败 0，错误 0 |
| 真实 HTTP 流程 | 通过 | health → inspect → analyze → poll → download → stft；并发第二次提交被 409 拒绝 |
| 三个真实文件有界 smoke | 通过 | Background=['Background', 'Background']；DJI Phantom 4 Pro=['DJI Phantom 4 Pro', 'DJI Phantom 4 Pro']；DJI Air 2S=['DJI Air 2S', 'DJI Air 2S'] |
| 界面渲染证据 | 通过 | 截图 `C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\outputs\signal_analyzer_validation_20260920\ui_screenshot.png`；离线快照 `C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\outputs\signal_analyzer_validation_20260920\ui_snapshot.html` |
| 启动器与真实服务 | 通过 | `launch_signal_analyzer.ps1` 实测启动于 127.0.0.1:50220，model=ready，device=cuda |

## 1. 固定多数决规则的独立复算

从已验收的 `window_predictions.csv` 独立重算每个 `dataset/condition/window` 的三种子最终标签多数决（1-1-1 记为 uncertain，不强制归类）：

| 数据集/条件 | 窗口数 | 背景 | Phantom | Air2S | uncertain(1-1-1) | 三种子一致率 | 集成准确率 | 无人机窗比例 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heldout/-10 | 1820 | 560 | 610 | 650 | 0 | 99.95% | 100.00% | 69.23% |
| heldout/-12 | 1820 | 567 | 610 | 643 | 0 | 99.45% | 99.62% | 68.85% |
| heldout/-15 | 1820 | 637 | 607 | 576 | 0 | 93.35% | 95.66% | 65.00% |
| heldout/clean | 1820 | 560 | 610 | 650 | 0 | 99.23% | 100.00% | 69.23% |
| old_test/clean | 440 | 140 | 139 | 161 | 0 | 97.50% | 99.77% | — |

heldout 四档集成准确率均值：98.82%；clean=100.00%，-10=100.00%，-12=99.62%，-15=95.66%

这只验证既有集成规则，不据此调整任何规则；heldout 仍是描述性评价（仅 12 个源文件，不能称跨会话泛化）。

## 2. 真实原始IQ → 新核心路径 与 CASCADE3 离散决策一致性

| 数据集 | 类别 | 位置 | 文件 | 起始样点 | 逐种子决策 | 最大 logit 差 |
| --- | --- | --- | --- | --- | --- | --- |
| heldout | Background | 0 | `T0000_D00_S0100.mat` | 0 | 2026:0/ref0, 2027:0/ref0, 2028:0/ref0 | 0 |
| heldout | DJI Phantom 4 Pro | 560 | `T0010_D00_S0100.mat` | 0 | 2026:1/ref1, 2027:1/ref1, 2028:1/ref1 | 0.000976562 |
| heldout | DJI Air 2S | 1170 | `T0101_D00_S0100.mat` | 0 | 2026:2/ref2, 2027:2/ref2, 2028:2/ref2 | 0 |
| old_test | Background | 0 | `T0000_D00_S0011.mat` | 0 | 2026:0/ref0, 2027:0/ref0, 2028:0/ref0 | 0 |
| old_test | DJI Phantom 4 Pro | 140 | `T0010_D00_S0011.mat` | 0 | 2026:1/ref1, 2027:1/ref1, 2028:1/ref1 | 0.00012207 |
| old_test | DJI Air 2S | 280 | `T0101_D00_S0011.mat` | 0 | 2026:2/ref2, 2027:2/ref2, 2028:2/ref2 | 0 |

batch 构成：每个目标窗独立成 batch=1 进入前端（窗口自身就是原来的 batch 成员之一），Stage1 用该 float32 特征张量，Stage2 仅在该 seed 自身 margin>0 时用同一张量执行。数值比较按正式 CUDA 环境记录，未使用 CPU 解释器静默复验。

### 3b. 复现 CASCADE3 的 batch=4 连续窗口组成

CASCADE3 的前端缓存按冻结顺序以 `batch_size=4` 连续分块；先前 Codex 的 batch-parity 探针已证明不同 batch 组成不能数值互换。此处的界面核心路径被喂入**同一批连续 4 个窗口**（`start_sample` = 该组首窗，`max_windows=4`），因此其 batch 与已发布分块一致，Stage1 logits 应与已发布值完全一致：

| 数据集/类别 | 锚点位置 | 首窗样点 | 每种子最大 Stage1 logit 差 | 特征摘要一致 | 离散决策 |
| --- | --- | --- | --- | --- | --- |
| heldout/Background | 0 | 0 | seed 2026: 0 | 是 | 全部一致 |
| heldout/Background | 0 | 0 | seed 2027: 0 | 是 | 全部一致 |
| heldout/Background | 0 | 0 | seed 2028: 0 | 是 | 全部一致 |
| heldout/DJI Air 2S | 1170 | 0 | seed 2026: 0 | 是 | 全部一致 |
| heldout/DJI Air 2S | 1170 | 0 | seed 2027: 0 | 是 | 全部一致 |
| heldout/DJI Air 2S | 1170 | 0 | seed 2028: 0 | 是 | 全部一致 |
| heldout/DJI Phantom 4 Pro | 560 | 0 | seed 2026: 0 | 是 | 全部一致 |
| heldout/DJI Phantom 4 Pro | 560 | 0 | seed 2027: 0 | 是 | 全部一致 |
| heldout/DJI Phantom 4 Pro | 560 | 0 | seed 2028: 0 | 是 | 全部一致 |
| old_test/Background | 0 | 0 | seed 2026: 0 | 是 | 全部一致 |
| old_test/Background | 0 | 0 | seed 2027: 0 | 是 | 全部一致 |
| old_test/Background | 0 | 0 | seed 2028: 0 | 是 | 全部一致 |
| old_test/DJI Air 2S | 280 | 0 | seed 2026: 0 | 是 | 全部一致 |
| old_test/DJI Air 2S | 280 | 0 | seed 2027: 0 | 是 | 全部一致 |
| old_test/DJI Air 2S | 280 | 0 | seed 2028: 0 | 是 | 全部一致 |
| old_test/DJI Phantom 4 Pro | 140 | 0 | seed 2026: 0 | 是 | 全部一致 |
| old_test/DJI Phantom 4 Pro | 140 | 0 | seed 2027: 0 | 是 | 全部一致 |
| old_test/DJI Phantom 4 Pro | 140 | 0 | seed 2028: 0 | 是 | 全部一致 |

结论：复现正式 batch 组成时 Stage1 logits 最大绝对差 0，特征摘要一致，离散决策一致。

## 3. 有界端到端 smoke（真实文件，产物仅写 scratch）

| 类别 | 文件 | 可选窗口 | 分析窗口 | 尾部丢弃 | 集成标签 | 每种子最终标签 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Background | `T0000_D00_S0100.mat` | 140 | 2 | 138000000 | ['Background', 'Background'] | 2026:[0, 0] / 2027:[0, 0] / 2028:[0, 0] | 0.25s |
| DJI Phantom 4 Pro | `T0010_D00_S0100.mat` | 160 | 2 | 158000000 | ['DJI Phantom 4 Pro', 'DJI Phantom 4 Pro'] | 2026:[1, 1] / 2027:[1, 1] / 2028:[1, 1] | 0.259s |
| DJI Air 2S | `T0101_D00_S0100.mat` | 160 | 2 | 158000000 | ['DJI Air 2S', 'DJI Air 2S'] | 2026:[2, 2] / 2027:[2, 2] / 2028:[2, 2] | 0.26s |

## 4. 界面与服务

- {'step': 'health', 'status': 200}
- {'step': 'static index', 'status': 200, 'bytes': 8365}
- {'step': 'inspect', 'status': 200, 'input': 'F:\\DroneRFa数据集\\heldout_20260917\\T0010_D00_S0100.mat', 'windows': 6, 'allowed': True}
- {'step': 'analyze', 'status': 202, 'response': {'ok': True, 'job': {'job_id': '20260920-140212-001', 'status': 'queued', 'stage': 'queued', 'stage_text': '排队等待', 'progress': {}, 'created_at': '2026-09-20T14:02:12', 'started_at': None, 'finished_at': None, 'cancel_requested': False, 'error': None, 'output_dir': None, 'source_path': None, 'log': []}}}
- {'step': 'job id', 'job_id': '20260920-140212-001'}
- {'step': 'queue limit', 'status': 409}
- {'step': 'review artifacts', 'copied': ['window_predictions.csv', 'ensemble_predictions.csv', 'analysis.json', 'summary.md', 'model_hashes.json', 'timeline.png', 'waveform_preview.png', 'stft_preview_window_0.png']}
- {'step': 'poll', 'status': 'completed', 'windows': 6, 'dominant': '无人机占主导'}
- {'step': 'downloads', 'status': 200}
- {'step': 'stft', 'status': 200, 'rows': 512, 'columns': 1302}

启动器实测：`C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\launch_signal_analyzer.ps1`（UTF-8 BOM=True）在 127.0.0.1:50220 启动成功，health 返回 host=127.0.0.1、weights_verified=True；验证结束时已终止该临时进程。

```
015d8769eac793b9f811b53992e6def209 (0.0s)
  hash start [stage2 seed 2027] best_model.pt
  hash done  [stage2 seed 2027] best_model.pt 44767819 bytes sha256=31653ff02328c584b97d7c957ab77dc861c54b57bb9c1e0a067324dede1cfcaf (0.0s)
  hash start [stage1 seed 2028] best_model.pt
  hash done  [stage1 seed 2028] best_model.pt 44766923 bytes sha256=af3cb6c006254a38f8cd5e3c1913027751bcdc154b25841262a99a33bad2f59a (0.0s)
  hash start [stage2 seed 2028] best_model.pt
  hash done  [stage2 seed 2028] best_model.pt 44767819 bytes sha256=04e4bd790b1c41706dfcf4737ed277f6a7959cc909be767d1c8ee64177cacbbc (0.0s)
```

界面截图：`C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\outputs\signal_analyzer_validation_20260920\ui_screenshot.png`（headless `chrome.exe`）。
离线渲染快照：`C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\outputs\signal_analyzer_validation_20260920\ui_snapshot.html` —— 内联 CSS/JS 并注入一次真实任务的完整结果，可直接用浏览器打开（file://）复核结果区的渲染。
- 截图尝试 live：returncode=0，字节=172816，成功=True

## 5. 未改动与边界

- 全程未训练、未调阈值、未改动 `cascade3-config-20260919.json`、索引、六个权重与正式 CASCADE3 预测；完成前后 hash 一致。
- 未添加噪声、未重采样；采样率未确认 100 MHz 或短于一窗时只预览、不分类。
- 本界面只支持当前三类，未知设备可能被分到已知类；margin 不是概率。
- 服务器只监听 127.0.0.1，未改防火墙，未绑定 0.0.0.0，未安装任何新依赖。

## 6. 问题清单（0）

- 无。
