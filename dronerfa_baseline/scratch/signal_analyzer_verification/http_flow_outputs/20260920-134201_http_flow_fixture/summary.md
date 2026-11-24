# 任意IQ信号分析报告：http_flow_fixture

- 分析时间（UTC）：2026-09-20T05:42:02.243356+00:00
- 输入文件：`C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\scratch\signal_analyzer_verification\fixtures\http_flow_fixture.mat`
- 检测格式：mat_v73；总点数：3000011；文件大小：96004448 字节
- 采样率：100 MS/s（用户确认：True）；协议：RF0；未添加任何噪声
- 窗口：1000000 点 = 10 ms；起点 0，步长 1000000，已分析 3 窗（可选 3 窗）；尾部丢弃 11 点（不足一窗不分析、不补零）
- 环境：cuda（NVIDIA GeForce RTX 4060 Laptop GPU），torch 2.12.1+cu130，AMP float16=True，项目 .venv=True
- 冻结配置 SHA256：`24a253c41a8421a3071dca1d17823909ba64c8eb33164b32c66a4cbec2c366f0`（匹配=True）；六权重校验：True

## 结论

- 主导结论：**混合或不确定**
- 是否出现过无人机窗：否（无人机窗 0 / 3 = 0.0%）
- 机型描述：无无人机窗
- 模型一致率（三种子全部一致的窗口比例）：0.0%
- 背景 Background：0 窗（0.0%）
- DJI Phantom 4 Pro：0 窗（0.0%）
- DJI Air 2S：0 窗（0.0%）
- 模型意见不一致：3 窗（100.0%）
- 总耗时：0.63 秒

## 边界说明

- 本界面只支持当前三类（背景 / DJI Phantom 4 Pro / DJI Air 2S）与 100 MS/s、RF0 协议。
- 决策分数（margin）是 logit 差值，**不是概率**，未做任何校准。
- 未知设备可能被强行分到已知类别；结果仅代表这三种已训练类别。
- 文件级结论是已分析窗口的多数描述，不是经过校准的事件检测器。
- 短于 1,000,000 点的信号直接拒绝；采样率非 100 MHz 时只允许预览，不做分类。

## 产物

- `window_predictions.csv`
- `ensemble_predictions.csv`
- `timeline.png`
- `waveform_preview.png`
- `stft_preview_window_0.png`
- `model_hashes.json`
- `analysis.json`
- `summary.md`
