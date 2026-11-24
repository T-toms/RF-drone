# LAUNCHER1：DroneRFa 分析界面一键启动器

## 目标

在不改变现有推理、模型或服务器行为的前提下，为已验收的 INFER-UI1 增加适合 Windows 桌面快捷方式调用的一键启动入口。

用户体验：双击桌面快捷方式后，如果 `http://127.0.0.1:8765/api/health` 已就绪，立即用默认浏览器打开界面；如果未运行，则在后台启动现有 `launch_signal_analyzer.ps1 -NoBrowser -Quiet`，等待模型就绪后自动打开界面。用户不需要打开 PowerShell，也不需要输入命令。

## 允许修改

- 新增 `dronerfa_baseline/open_signal_analyzer.ps1`
- 必要时新增只针对该启动器的轻量测试脚本
- 更新 `dronerfa_baseline/signal_analyzer_README.md` 的一键启动说明
- 验证产物写入 `dronerfa_baseline/outputs/signal_analyzer_launcher_validation_20260920/`
- 交接文件由桥接写入指定 `work/deepseek-handoffs/` 路径

不要创建桌面 `.lnk`；Codex 验收后负责把快捷方式安装到用户桌面。不要修改已验收推理代码、六个权重、配置、索引、正式预测或现有输出。

## 实现要求

1. 兼容 Windows PowerShell 5.1；脚本编码需保证中文路径与中文提示正常。
2. 固定服务地址 `127.0.0.1:8765`，只调用现有 `launch_signal_analyzer.ps1`，默认输出仍为 `F:/DroneRFa数据集/analysis_outputs`。
3. 先对 `/api/health` 做短超时检查；若服务 `ok=true` 且模型 `state=ready`，直接打开主页，不启动第二个服务。
4. 未就绪时后台隐藏启动现有启动脚本，等待最长建议 90 秒；就绪后打开主页。启动日志写入项目 `dronerfa_baseline/scratch/`，stdout/stderr分开，便于失败排查。
5. 防止快速连续双击引发重复服务。可用当前用户会话内命名 mutex；另一个启动器发现已有启动过程时只等待相同 health。
6. 启动失败或超时必须给出可见中文提示，包含日志位置；不能静默失败。成功时不弹多余提示。
7. 提供测试友好的参数，例如 `-NoOpenBrowser`、可覆盖端口或等待时限；默认桌面行为仍是自动打开。测试不能联网、不能修改防火墙或安装依赖。
8. 不结束当前已运行的服务，不扫描或结束其他 Python 进程。

## 验收条件

- PowerShell 5.1 语法/编码可用。
- 服务已运行场景：检测 ready，不拉起第二个服务；测试模式退出码0。
- 服务未运行场景：可在临时端口后台启动、等待ready、测试模式退出码0，并只停止测试自己创建的临时进程。
- 超时/失败路径有明确中文消息和非零退出码，可通过测试参数避免真实弹窗。
- README说明桌面快捷方式由Codex创建，给出目标脚本位置。
- 现有 `test_signal_analyzer.py` 继续通过；冻结文件哈希不变。

完成后写 completed/needs_codex/needs_user 交接报告，列出改动、命令、结果和遗留问题。
