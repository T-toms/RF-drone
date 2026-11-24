# SELECT12-G2：在 G 盘续传剩余 41 个目标 MAT

第一个真实文件 `T0011_D00_S0000.mat` 已由 Codex 验收：G 盘 `original` 现有 63 个 MAT，该文件大小 1,096,484,492 字节，SHA256 `a0328981f038de41f9e89324e10c61d8214bc0e9c3592f7bb7aaa541c3538ee0`；包装块 `[1,2,5]`、UnRAR t/e、HDF5 四通道检查均通过。`dronerfa_baseline/download_select12g.py` 及 `work/sel12g/sel12g-config-20260924.json` 是当前程序/冻结清单。修复代码已跑 56 项测试通过。前一 DSH 会话因额度不足未执行，不需要重复解压这个文件。

本任务：先只读核实无现存 `download_select12g.py` 后台 worker、G 盘卷身份和空闲容量，然后启动一个可持续运行的 G 盘后台下载进程，处理冻结清单中剩余 41 个 `origin=to_download` 且尚未落位的目标 MAT。使用既有断点；不要 `--restart-from-zero`。每个文件仍必须经过块头/包装、UnRAR t/e、解压大小及 HDF5 校验才放进 `G:\DroneRFa数据集\original`。每个文件校验失败时保留断点和证据，不反复盲目重试；建议 `--file-attempts 1 --retry-rounds 1`。可用 2–4 个范围请求 worker，避免过量并发。保留此前旧 `.part` 证据，禁止写入新 F 盘、禁止整包下载、禁止改变 train/val/test 划分或开展训练。

建立清楚的工作区内启动记录和日志路径，启动后检查 PID 确实存活、日志有远程身份和首个目标/断点进度、G 盘文件增长。长任务可以独立于 DSH 会话继续运行；确认已启动后及时写 `completed` 交接，说明 PID、命令、日志、状态文件、首个目标及如何复查，不必等待 41 个文件全部完成，也不要用长时间前台循环耗尽 DSH 额度。后台 `Start-Process` 必须使用 `-WindowStyle Hidden`。如已有 worker，不启动第二个，报告其 PID 和状态。若身份/空间/权限/网络检查未通过，停止并报告 `needs_codex`，不改去 F 盘。
