# DOWNLOAD12-START：冻结并启动 DroneRFa 12机型扩充下载

## 目标

为现有两种无人机（T0010 Phantom 4 Pro、T0101 Air 2S）补充10种 DroneRFa 无人机，每种下载前8个 S 段，共80个MAT。使用已有逐文件断点下载器，全部数据最终落入统一目录 `F:\DroneRFa数据集\original`。

本任务的完成定义是：选择清单冻结并验收、dry-run通过、真实后台下载进程已经启动且观察到有效日志或断点增长。80文件全部下载完成属于后续持续状态，不得在仅启动时宣称全部完成。

## 冻结机型与文件规则

### 带 D00 字段的7种（每种8文件）

- T0001：DJI Phantom 3
- T0011：DJI Matrice 200
- T0100：DJI Matrice 100
- T0110：DJI Mini 3 Pro
- T0111：DJI Inspire 2
- T1000：DJI Mavic Pro
- T1001：DJI Mini 2

文件名严格为 `Txxxx_D00_Syyyy.mat`。

### 官方归档不带距离字段的3种（每种8文件）

- T1010：DJI Mavic 3
- T1011：DJI Matrice 300
- T1100：DJI Phantom 4 Pro RTK

文件名严格为 `Txxxx_Syyyy.mat`。这3种必须在计划中记录 `distance_code=null` / `distance_status=not_provided_in_archive_filename`，禁止伪写成D00。

### 每种机型的8个S段

`S0000,S0001,S0010,S0011,S0100,S0101,S0110,S0111`

不得加入其他target、D01/D10或S1000及以后文件。

## 已由 Codex 核对的精确常量

- 文件数：80
- 唯一文件名：80
- 当前 `F:\DroneRFa数据集\original` 中同名文件：0
- 所选RAR块总字节：79,572,949,889
- packed_size总和：79,572,944,417
- 解压总字节：82,727,100,398
- 解压单文件范围：741,013,512～1,503,936,244字节
- 当前F盘可用约452.63 GiB；现有24个MAT约23.03 GiB
- 远程完整RAR预期大小：615,955,805,466字节

## 产物与路径

- 冻结选择清单：`work/dronerfa-12model-download-candidates-20260922.json`
- 启动/预检证据：`work/dronerfa-12model-download-start-20260922.json`
- 下载记录根：`F:\DroneRFa数据集\download_records_12models_20260922`
- 暂存/断点目录：`F:\DroneRFa数据集\download_records_12models_20260922\downloads`
- 下载器逐文件记录：`F:\DroneRFa数据集\download_records_12models_20260922\download_records.json`
- 标准输出日志：`F:\DroneRFa数据集\download_records_12models_20260922\download_12models_20260922.log`
- 错误日志：`F:\DroneRFa数据集\download_records_12models_20260922\download_12models_20260922.error.log`
- PID记录：`F:\DroneRFa数据集\download_records_12models_20260922\download_12models_20260922.pid.json`
- 最终数据目录：`F:\DroneRFa数据集\original`

## 执行步骤

1. 从项目根 `dronerfa_manifest.json` 生成冻结候选JSON，沿用已有candidate格式并增加：schema、created_at、target_labels、selection_rule、distance元数据、80条原始archive entry、三种总字节数、源manifest SHA256。必须断言上述精确常量全部相等。
2. 再次解析所有绝对路径，确认所有F盘路径严格位于 `F:\DroneRFa数据集` 下；不得写C盘数据临时文件。
3. 检查目标80个文件在 `original` 中0个已存在；若不为0，列出并停止，不覆盖。
4. 检查F盘可用空间至少为 `解压总字节 + 最大单块字节 + 20 GiB`。禁止使用忽略空间检查参数。
5. 使用项目 `.venv\Scripts\python.exe` 调用现有 `dronerfa_baseline\download_heldout_files.py` 做 dry-run，显式传入新候选、staging/extract/records路径，断言输出恰好80项且无缺失。此脚本虽保留旧文件名，但本次按显式candidate运行；不要改默认旧heldout产物。
6. 真实命令固定为相同脚本，加：`--files all --workers 4 --chunk-mib 16`。使用 `Start-Process -WindowStyle Hidden -PassThru` 在后台启动，标准输出和错误分别重定向到上述日志。不得使用 `--no-skip-existing`、`--keep-archive`，不得使用覆盖参数。
7. PID JSON记录PID、命令参数、候选SHA256、开始时间、日志、错误日志、records、staging、extract目录。
8. 等待20～40秒后核实：进程仍存在，或者若已退出则退出码为0并有真实完成证据；日志已通过远程RAR大小/ETag检查并进入第1个文件；错误日志为空。如进程异常退出，写needs_codex，保留断点和日志，不盲目重启。
9. 写启动证据JSON，区分 `download_started` 与 `download_completed`。本任务通常只能写前者，不能把80文件写成已完成。

## 数据与实验安全

- 下载过程中不修改 `F:\DroneRFa数据集\dataset_manifest.csv`；只有完整MAT经HDF5和大小校验成功后，后续验收任务才把它原子加入清单并标记 `split_status=unassigned`。
- 不修改已有24个MAT、历史冻结索引、模型、实验结果或 `analysis_outputs`。
- 逐文件下载器必须保持：Range响应验证、RAR结构和完整性测试、HDF5四通道检查、SHA256记录、成功后清理当个临时RAR、断点幂等续传。
- 下载失败保留 `.part` 与state，不删除有效MAT，不扩大到完整573.65 GiB压缩包。
- 不创建heartbeat、automation或定时唤醒。

## 允许修改

- `work/dronerfa-12model-download-candidates-20260922.json`
- `work/dronerfa-12model-download-start-20260922.json`
- 本次交接文件
- 上述 `F:\DroneRFa数据集\download_records_12models_20260922` 记录树
- `F:\DroneRFa数据集\original`（只能由既有下载器逐个写入冻结清单中的80个MAT）

## 验收与交接

- 选择清单80项、10个目标各8项、总字节完全匹配。
- dry-run无缺项。
- 后台下载启动后有真实PID和日志证据。
- 交接 `completed` 仅代表 DOWNLOAD12-START 已完成；摘要必须明确实际已完成文件数、正在下载文件、压缩下载进度（如可得）和进程状态。
- 权限、网络、索引或启动失败写 `needs_codex`/`needs_user` 并停止，不创建第二个下载进程。
