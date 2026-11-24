# SELECT12-G1R2：完成一个真实 MAT 的解压验收

Codex 已接管并修复前一轮真实文件包装校验的偏移问题。前一轮 DSH 会话已停止，未写交接文件；不要复用该会话的未完成状态。当前 `T0011_D00_S0000.mat.rar.part.restart` 在 G 盘已完整达到冻结块长 1,057,934,381 字节。Codex 已用该块生成的 `tmp/diag-wrapper/real-wrapped.rar` 验证结构 `[1,2,5]`、全部块头 CRC，且 `C:\Program Files\WinRAR\UnRAR.exe t -idq -p-` 退出码 0。`python -m unittest dronerfa_baseline.test_download_select12g -q` 为 56 项通过。

仅处理 `T0011_D00_S0000.mat`，不启动 42 文件批量下载。用修好的 `dronerfa_baseline/download_select12g.py` 从 G 盘完整断点继续，完成包装、`UnRAR t`、`UnRAR e`、解压 MAT 大小 1,096,484,492 字节、HDF5 四通道与有限值检查，最终将 MAT 安全放入 `G:\DroneRFa数据集\original`。保持旧 62 个 MAT、G 盘其他旧 `.part` 和所有 F 盘数据原样。若仍发生校验失败，停止并报告具体证据，不增加重试轮数、不要从零重新下载。

建议执行参数：`--config work/sel12g/sel12g-config-20260924.json --only T0011_D00_S0000.mat --workers 1 --file-attempts 1 --retry-rounds 1 --unrar "C:\Program Files\WinRAR\UnRAR.exe"`。验收时记录实际文件路径、大小、SHA256、HDF5 通道及计数，确认 G 原始目录有效 MAT 从 62 到 63，并检查无后台批量 worker。检查 `verify_wrapped_archive` 针对大文件的修复是否保持稳健；如需小范围实现/测试修正可直接完成。结束前按工作流写 completed/needs_codex/needs_user 交接文件。
