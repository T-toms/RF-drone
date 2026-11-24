# DroneRFa 任意IQ信号分析界面 —— 使用说明

本界面是一个**只监听本机 127.0.0.1** 的中文浏览器页面：选择一个本机 IQ 文件（或上传文件），
检查格式与采样参数，按 1,000,000 复数点流式切窗，调用**已验收的 CASCADE3 六个冻结检查点**
给出每个窗口的类别，并展示时间轴、I/Q 波形、幅度和 STFT。

- 不训练、不调阈值、不添加噪声、不重采样。
- 三级串联：Stage1 判定 Background/Drone，判为 Drone 的窗口再进入 Stage2 区分
  DJI Phantom 4 Pro 与 DJI Air 2S；三个训练种子（2026/2027/2028）各出一套结论，
  最终标签取三者多数决，**1-1-1 记为“模型意见不一致”，不强行归类**。
- 界面完全离线：不使用 CDN 字体、脚本或图片；不联网安装任何新依赖。

## 最便捷的打开方式

桌面快捷方式：`C:\Users\Administrator\Desktop\无人机信号分析.lnk`。

直接双击“无人机信号分析”即可：如果服务已经运行，启动器会直接打开
`http://127.0.0.1:8765/`，不会重复加载模型；如果服务没有运行，启动器会在后台启动服务，
等待模型就绪后自动用 **Google Chrome** 打开。若本机以后找不到 Chrome，才回退到系统默认浏览器。
分析结果仍写到 `F:/DroneRFa数据集/analysis_outputs`。

快捷方式调用 `dronerfa_baseline/open_signal_analyzer.ps1`。启动失败时会显示中文提示，详细日志位于
`dronerfa_baseline/scratch/open_signal_analyzer.log`、`signal_analyzer_service_stdout.log` 和
`signal_analyzer_service_stderr.log`。

### 一键启动器的行为与退出码

- 先做短超时健康检查（默认最多 3 次），再判断端口：**只要 8765 端口已有服务在监听，就只等待
  同一个服务，绝不会启动第二个**（服务偶尔忙、健康检查暂时无响应时也不会误判成“没运行”）。
- 只有端口空闲时才隐藏启动 `launch_signal_analyzer.ps1 -NoBrowser -Quiet`，日志分 stdout/stderr
  写入 `dronerfa_baseline/scratch/`；等待上限默认 90 秒（`-ReadyTimeoutSeconds`）。
- 快速连点两次：命名 mutex（`Local\DroneRFaSignalAnalyzerLauncher_<端口>`）保证只有一个进程去启动。
- 不结束任何进程、不改防火墙、不联网；只记录自己启动的子进程 PID 到
  `dronerfa_baseline/scratch/signal_analyzer_service.pid` 便于排查。

退出码：`0` 成功（浏览器已打开，或被 `-NoOpenBrowser` 抑制）；`2` 等待就绪超时；
`3` 后台启动进程提前退出或无法启动；`4` 该端口服务在跑但模型加载失败；`5` 项目环境缺失；
`6` 参数不合法或端口被其它程序占用。

排错/测试用法（不打开浏览器、不弹窗、日志写到临时目录）：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File dronerfa_baseline\open_signal_analyzer.ps1 `
    -NoOpenBrowser -NoMessageBox -Quiet -ReadyTimeoutSeconds 120 `
    -LogDirectory "C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\scratch\_launcher_tests\manual"
```

自动化测试：

```powershell
.venv\Scripts\python.exe dronerfa_baseline\test_open_signal_analyzer.py
```

覆盖：BOM/5.1 解析静态检查、服务已就绪、加载中→就绪、等待超时、模型加载失败、端口被占用、
后台启动失败（含子进程参数转发核对）、真实后台启动 + 只结束自己创建的进程、
端口在监听但健康检查无响应。验证产物在
`dronerfa_baseline/outputs/signal_analyzer_launcher_validation_20260920/`。

## 1. 启动

```powershell
# 默认端口 8765，输出到 F:/DroneRFa数据集/analysis_outputs
pwsh -File dronerfa_baseline\launch_signal_analyzer.ps1

# 换端口、不自动开浏览器
pwsh -File dronerfa_baseline\launch_signal_analyzer.ps1 -Port 8899 -NoBrowser

# 把结果写到项目 scratch（测试用，不污染 F 盘）
pwsh -File dronerfa_baseline\launch_signal_analyzer.ps1 -OutputRoot "C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline\scratch\analysis_outputs"
```

也可以直接运行：

```powershell
.venv\Scripts\python.exe dronerfa_baseline\signal_analyzer_server.py --port 8765
```

启动器固定使用项目 `.venv`（CUDA PyTorch），**不会**回退到 anaconda 的 CPU PyTorch；
只允许 `127.0.0.1`，拒绝绑定 `0.0.0.0`，不修改防火墙。

启动日志会打印配置 hash 匹配情况和权重校验结果；页面顶部也持续显示
“服务 / 运行设备 / 六权重校验”。如果只能用 CPU，页面会显著提示：速度明显变慢，
且数值环境与正式 CUDA float16 AMP 不同。

## 2. 选择信号

**本机路径**（推荐大文件）：选择允许根目录 → 列出目录 → 点击文件。
默认允许根目录为 `F:/DroneRFa数据集` 与项目根目录，可用 `-AllowedRoot` 增加。
路径在服务端解析真实路径后必须仍落在允许根目录内，目录穿越会被拒绝。

**浏览器上传**：按 4 MiB 分块写入项目 `dronerfa_baseline/scratch/signal_analyzer_uploads`；
取消或失败会删除不完整文件。多 GB 文件请优先用本机路径。

### 格式与参数

| 格式 | 自动识别 | 说明 |
| --- | --- | --- |
| MAT v7.3 (HDF5) | 是（含 512 字节用户块） | 默认 `RF0_I`/`RF0_Q`，可改键名；支持 `[1,N]` 或 `[N]`；切片读取 |
| NPY | 是 | 一维复数，或浮点 `[2,N]` / `[N,2]`（第 0 行/列为 I）；内存映射切片 |
| NPZ | 是 | 复数键 `iq`，或成对键 `i`/`q`；压缩包需整体解压，已提示风险 |
| RAW | 需显式参数 | `dtype`（float32/float64/complex64/complex128）、字节序、`IQ`/`QI` 顺序 |

窗口参数：`start sample`（默认 0）、`stride`（默认 1,000,000）、
最多窗口数（留空=全部）、采样率（默认 100,000,000 Hz）。

> **必须勾选“我确认该信号的采样率为 100 MHz，且使用 RF0 通道”** 才会启用“开始分析”。
> 未确认、采样率不是 100 MHz、或文件短于 1,000,000 点时，只能“检查文件”预览，
> 分类按钮保持禁用并给出中文原因；短信号**不补零**。

先点“检查文件”，确认格式、可选窗口数、尾部丢弃点数，再点“开始分析”。

## 3. 看结果

- **主导结论**：无人机窗严格超过 50% → “无人机占主导”；背景严格超过 50% → “背景占主导”；
  其余 → “混合或不确定”。同时始终显示“是否出现过无人机窗”和无人机窗比例，
  避免多数规则掩盖短时片段。
- **类别统计**：Background / Phantom / Air2S / uncertain 的窗口数与比例、模型一致率、总耗时。
- **时间轴**：按窗口着色的条带 + 下方“是否出现无人机窗”条带；点击任意窗口可查看细节。
- **窗口细节**：I/Q 波形、幅度 |IQ|、STFT（n_fft=512、hop=256、fftshift、相对通道中心），
  以及三个种子各自的 Stage1/Stage2 logits、margin、是否执行第二阶段、该种子最终标签。
- **逐窗表**：可点表头排序、按类别筛选、只看无人机窗。
- **下载**：`window_predictions.csv`（每种子一行）、`ensemble_predictions.csv`、
  `analysis.json`、`summary.md`、`model_hashes.json`、`timeline.png`、`waveform_preview.png`、
  `stft_preview_window_N.png`，以及打包 ZIP。

结果默认写到 `F:/DroneRFa数据集/analysis_outputs/<时间戳>_<安全文件名>/`。

## 4. 重要边界

- 只支持当前三类：Background、DJI Phantom 4 Pro、DJI Air 2S。
- 只支持 100 MS/s、RF0 协议。其他采样率只能预览。
- **决策分数（margin）是 logit 差值，不是概率**，未做任何校准。
- 未知设备可能被强行分到已知类别；本界面不做未知类检测。
- 文件级结论是已分析窗口的多数描述，不是经过校准的事件检测器。
- 用户信号不添加噪声、不重采样；结果不代表现场绝对 SNR。
- 单 GPU 作业队列：同一时间只运行一个分析任务；取消后不会发布不完整结果。

## 5. 自检

```powershell
# 单元测试与真实 HTTP 流程（会启动本地服务器并自动关闭）
.venv\Scripts\python.exe dronerfa_baseline\test_signal_analyzer.py

# 一键启动器（桌面快捷方式）测试
.venv\Scripts\python.exe dronerfa_baseline\test_open_signal_analyzer.py

# 独立验证：多数决复算 + 真实IQ一致性 + 单元测试 + HTTP + 三文件 smoke + 冻结哈希复核
.venv\Scripts\python.exe dronerfa_baseline\verify_signal_analyzer.py
```

验证产物在 `dronerfa_baseline/outputs/signal_analyzer_validation_20260920/`。
