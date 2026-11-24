# SELECT12-F1：只准备12机型所需 DroneRFa 数据到新 F 盘

## 最新用户决策

用户已明确停止完整 RAR 下载，改为只准备12机型实验所需数据。Codex 已停止完整 RAR 下载 PID 16800；新 F 盘保留 1,551,892,480 字节 `.part` 但不得续传。旧 F 盘现为 G 盘；新 F 盘是物理新盘。不得创建定时器/heartbeat，不训练模型，不删 G 盘或旧 F 整包断点。

## 冻结目标

- `work/dronerfa-12model-download-candidates-20260922.json` 是唯一新增目标清单：10个新增无人机机型 × 8个 S 段 = 80 MAT。既有两个机型 T0010/T0101 各8 MAT，加 Background T0000 的8 MAT，总目标 104 MAT（12种无人机机型 + 背景）。不要擅自扩大为整个归档或其它距离条件。
- G 盘 `G:\DroneRFa数据集\original` 当前62 MAT：既有24 + 新增目标中38；尚缺约42 MAT。另有旧 `.part` 5个约1.50 GiB、`.rar` 8个约7.53 GiB，必须只读盘点其身份及可复用性。不能凭扩展名/大小宣称有效。
- 官方完整归档 URL、长度、ETag、局部 RAR 索引均见 `dronerfa_manifest.json` 和 `work/full-archive-config-20260924.json`。`http.client` 的 206/Content-Range/ETag 探测可用，多连接更快；旧 `curl` 单文件下载故障多，不应原样重启。
- 新目标根 `F:\DroneRFa12机型`，最终 MAT 为 `F:\DroneRFa12机型\original\<filename>.mat`，下载中间物和记录放 `F:\DroneRFa12机型\download_records`，项目内代码和验收记录放工作区。创建带 UUID、F 卷序列号/容量的卷标记；每次启动和续传都核对，不能误写 G。

## 工作与验收

1. 冻结104文件的目标表，标明机型、官方归档索引的文件名/大小、已有24个的 `former_partition`（从 G 的 `dataset_manifest.csv` 读，不要将已评估 heldout 当成新 final test）、新增文件 `split_status=unassigned`。先只读检查 G 盘62个 MAT 与目标表是否逐一匹配。逐文件核对字节数和 HDF5 四通道结构；可对重要源文件加 SHA256。发现不一致立即隔离到问题清单，不把它复制为有效数据。
2. 只复制经验证的 G 盘 MAT 到新 F，不移动、不删除 G；复制后逐文件核对源/目标 SHA256、大小、HDF5 结构。记录新 F 已有的有效数、每类数和仍缺的文件名。运行中不要把输出混进 G 盘或旧实验结果目录。
3. 审计 G 盘旧 `.part`/`.rar`：精确路径、长度、对应候选、断点元数据、文件头/区间身份、是否已经有对应 MAT。能安全复用的先复制到新 F 工作目录并独立验证；无法证明连续性或身份的只报告为“不可安全复用”，不删除 G。优先级低于已完整的 MAT。
4. 为缺失 MAT 使用**目标归档条目的 Range 区间**构建可续传下载。可复用 `download_heldout_files.py` 的 RAR wrapper/UnRAR/HDF5 逻辑以及 `download_full_archive.py` 的 Python `http.client`、并行 Range 校验，但要修复此前验收发现的写入返回值与 fsync 顺序问题。每次响应在写入前严格核对 206、请求起止、归档总长、ETag 和精确正文长度；只允许按顺序落盘到单个连续断点；每完整块全量写入+fsync后再原子更新状态。协议/身份变化立即停止，DNS/短暂故障有界重试。单实例锁。中途定期检查 F 盘身份与空间。
5. 每个 MAT 完成后验证 RAR wrapper（或对应条目）、运行 `UnRAR t`、解压、核对 MAT 名称/字节数/HDF5 四通道，之后才标记完成并更新新 F 清单。避免将 `split_status=unassigned` 的文件自动加入训练或测试，后续另行冻结分组划分。
6. 写针对性测试：错 Range/ETag拒绝、短写/崩溃断点、续传、锁、卷变化、空间不足、已有完整 MAT 跳过。可先只实现和完成一个缺失文件的端到端验证，再启动独立的长期下载进程补剩余文件。`Start-Process` 必须 `-WindowStyle Hidden`。已知普通沙箱会回收分离子进程，只对固定已授权命令申请必要提升，不更改系统权限。
7. 交接时明确：104目标中已完成/未完成数量、已经实际复制和下载的字节、新进程 PID、跨两次检查的增长证据、错误数、哪些 G 残留可复用/不能复用。`completed` 只代表迁移验收与长期定向下载启动；不能声称104 MAT全部到位，不能清理 G。若无法安全续传旧部分，丢弃其复用计划但保留 G 原文件，继续从该目标条目的起点下载。

## 严格边界

- 不续传 `F:\DroneRFa完整数据集\DroneRFa.rar.part`，不解压整包，不下载清单以外的条目。
- 不删除、移动、覆盖 G 盘任何文件；不清理新 F 的整包断点（后续另议）。
- 不改变旧冻结训练/验证/heldout 结果和模型。
- Windows 任何将来递归移动/删除都要先核对绝对目标路径在明确授权根内；本任务没有这类操作。
