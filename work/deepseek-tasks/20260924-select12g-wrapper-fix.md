# SELECT12-G1R：修正单文件 RAR 包装与解压，先验证一个真实文件

## Codex 已确诊的架构错误

上一任务 `20260924-143812-682f2402` 的 `download_select12g.py` 下载官方 RAR 的**单个文件块**（从 manifest 的 `block_offset` 起，长度 `block_size`），把裸文件块 `.part.restart` 重命名为 `.rar`，然后 `verify_wrapper()` 从偏移0要求 `RAR5_SIGNATURE`。这会必然报“缺少RAR5签名”，与网络是否故障无关。Codex 实际读取 `T0011_D00_S0000.mat.rar.part.restart` 前96字节为 `EF-D1-13-1D-42-02-03-0B...`，属于文件块头，当然不是 `52-61-72-21-1A-07-01-00` 的 RAR 签名。`verify_wrapper_prefix()` 也错误要求 `.part.restart` 从 RAR 签名开始，因此旧进程崩溃后不能续传。

第二个错误：该条目 `packed_size=1,057,934,310`、`unpacked_size=1,096,484,492`，说明载荷确有压缩；不能使用 `extract_stored_payload()` 直接复制压缩字节当 MAT，不能要求压缩载荷 SHA256 等于解压 MAT SHA256。当前代码的“stored”注释与验证条件均不成立。

Codex 已在 2026-09-24 17:43 左右停止旧后台 worker PID 22252 和 launcher PID 404，保留 `T0011_D00_S0000.mat.rar.part.restart` 及状态，两者当时均为 251,658,240 字节。旧 G 盘原有62个MAT未改；不要运行或重启旧代码。

## 修复方案

1. 保留当前 `.part.restart` 的**裸文件块**格式和状态 `committed_bytes` 语义；改 `verify_wrapper_prefix()` 为在偏移0解析/校验 RAR5 `HEAD_FILE` 块头 CRC、条目名、打包长度，无需 RAR 签名。状态与磁盘长度一致、远端总长/ETag/G卷身份一致时可从已提交字节续传；任何不一致硬停，不默默忽略。
2. 完成单个原始文件块后，不要直接重命名为 `.rar`。参考项目已成功用于38个文件的 `download_dronerfa_subset.py::stream_selected_archive`：在 G 盘受控 scratch 内按 `RAR5_SIGNATURE + MINIMAL_MAIN_HEADER + 原始文件块 + MINIMAL_END_HEADER` 生成**临时单文件 RAR**，全量核对每次写入返回值并 fsync；检查包装后的预期长度。保留裸块断点直到新 RAR 校验通过，避免失败后丢失证据。不要复制整个官方归档。
3. 在正确包装的临时 RAR 上验证唯一 HEAD_FILE 的名字、块头 CRC、packed_size、unpacked_size；运行 `UnRAR t`。随后用 `UnRAR e` 或等价命令真正解压到受控 scratch，核对 MAT 文件名、manifest `unpacked_size` 和 HDF5 四通道及读取探针。只有全部通过才原子放到 `G:\DroneRFa数据集\original`。不再使用“压缩载荷直接复制为 MAT”及其 SHA256 相等判据；可分别记录压缩块/解压 MAT 的哈希作溯源。
4. `PROJECT_ROOT = Path(__file__).resolve().parents[2]` 看起来指向项目的上级 `Desktop`，请改成项目根（`parents[1]`），核对默认配置/日志路径在本项目内。
5. 旧代码会在内容校验失败后重下整文件。先关闭或限制失败重试，避免尚未验证修复时重复消耗流量。若包装正确后 `UnRAR t` 仍报 CRC，再保留精确文件、区间、哈希和服务端响应证据，交给 Codex 判断端点质量；不要因旧“缺少签名”错误就宣称端点损坏。
6. 加针对性测试：裸文件块首字节不是签名却能通过头部校验、包装 RAR 结构和大小、压缩条目必须用 UnRAR 解压、掉电断点续传、错误 CRC 阻止 MAT 落位。测试用小 fixture；不要用与实现同样错误假设的模拟 fixture。旧34项如需更新，说明改变原因。
7. 先只跑 `T0011_D00_S0000.mat` 一件真实文件的端到端续传和 UnRAR t/e/HDF5验收；Codex 只在看到 G 盘有效 MAT 从62增到63且验收链完整后才允许恢复42文件批量下载。**本任务不得直接启动批量长期进程**。如端点真实内容损坏，报告 `needs_codex` 并保留证据，不增加重试轮数。

## 边界

- 只写项目工作区及 `G:\DroneRFa数据集` 下受控路径，不写新 F，不下载完整 RAR。
- 原有62个MAT、两个旧大 `.part`、实验结果、报告不移动/删除。用户已授权清理的8个损坏旧 `.rar` 与3个空桩由Codex处理完毕，无需恢复。
- 不创建定时任务/heartbeat，不改权限设置，不提交/推送代码。
