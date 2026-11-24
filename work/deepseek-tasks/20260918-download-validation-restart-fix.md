# 原会话下载续传修复（仅在用户处理当前审批之后继续）

同一run/session，不创建重复任务。当前网页命令会Stop-Process全部curl；Codex自动审批禁止代替用户拒绝，用户需先拒绝该命令。

之后先通过按现有流程获准的Get-CimInstance Win32_Process，核实ParentProcessId=18080且CommandLine包含F盘本任务scratch的curl PID。在停止主进程之前捕获确定子PID，只按这些明确PID结束本次下载。不得按进程名称全部杀，不处理其他Python/curl，不改变审批或权限设置。

download_heldout_files.py的两处unpacked_size//4推算采样数已由Harness修正。Codex真实文件检查通过：T0000_D00_S0100.mat大小908097315，与清单一致，四通道各140000000，向量长度一致，RF0切片有限，existing_mat_is_valid返回True。之前RAR结构校验、unrar t、解压大小检查已通过。

保留所有MAT、RAR、part/state及日志，结束本次旧代码进程后以修复版恢复all/4workers/16MiB后台任务，F盘原目录，隐藏窗口，独立日志名。首个有效MAT跳过重下、第2文件从part续传。下载记录需要由新进程更正，不并发编辑正在写的download_records.json。确认进程、首个文件跳过和第2文件字节增长。

其他索引、重复性分析、频段、模型评价、全面测试和环境优化保持暂停。只修复实际下载阻碍，不扩大任务。
