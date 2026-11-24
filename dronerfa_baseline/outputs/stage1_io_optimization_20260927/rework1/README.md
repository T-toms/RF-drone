# STAGE1-IO-OPT-R1 返工产物索引

本目录只包含 2026-09-27 第一轮返工新增的独立复核产物；上一轮 STAGE1-IO-OPT 的产物仍在
上一级目录（`../benchmark.json`、`../smoke_serial_refactor/` 等），本轮没有覆盖它们。

| 路径 | 内容 | 关键结论 |
| --- | --- | --- |
| `equivalence/equivalence_equiv-20260927T112604Z.json`、`equivalence_summary.csv` | 同版本 `legacy_serial`/`workers0`/`workers2`/`workers4` 两轮真实训练步对照（真实 G 盘 train 窗） | 24 个 batch 的输入摘要/标签/loss/梯度/参数、每 epoch optimizer+GradScaler 状态、最终模型全部逐位相同（loss 最大差 0.0，final delta 0.0）→ **R3 通过** |
| `benchmark/benchmark_bench-20260927T114431Z.json`、`benchmark_summary.csv` | 真实训练步吞吐基准，父进程 torch 线程 2，48 窗 × 2 epoch × 3 迭代，轮换顺序，warmup 1 | 8.35 / 6.57（×0.786）/ 4.62（×0.552）窗/秒 → **0 workers 最快** |
| `benchmark/digest_cost.json` | 逐批数据摘要的单项成本（真实窗张量实测） | 23.8 ms/批 ≈ 0.285 s/epoch ≈ epoch wall 的 4.96%；三个配置同等承担，不改变相对结论 |
| `benchmark_threads_default/benchmark_bench-20260927T114758Z.json` | 同上但父线程取默认（10） | 8.30 / 6.65（×0.801）/ 4.67（×0.563）→ 结论相同 |
| `benchmark_diagnostic_no_audit/benchmark_bench-20260927T115149Z.json` | 诊断：关闭 worker 事件流 | 8.45 / 6.56 / 4.60 → 审计开销约 1%，不是变慢原因；该配置无读取证据，不得作为吞吐配置 |
| `smoke_serial/` | 新串行 smoke（`--num-workers 0 --torch-threads 2`） | 身份审计 40/40、验证缓存 64/64、未消费 0、未读 0；`loader_consumption.csv` 104 行 |
| `smoke_parallel2/` | 新 2-worker smoke | 同上；worker 事件按 `worker<id>@pid<pid>` 分组，父进程读取 0 |
| `verification/smoke_serial_loader_audit.json` | 顶层核验（串行 smoke） | 41/41 通过，篡改对照 10/10 |
| `verification/smoke_parallel2_loader_audit.json` | 顶层核验（并行 smoke） | 41/41 通过，篡改对照 11/11 |
| `verification/tampered_smoke_parallel2/`、`verification/tampered_smoke_parallel2_loader_audit.json` | 篡改副本（`scratch/tamper_loader_audit.py`）及其核验 | rc=1，P0/P7/P8/P10/P11/P13 失败 → **R2 通过** |
| `verification/verification_seed_2027_after_rework.json` | 用返工后的核验器复验已验收 seed 2027 运行（换新输出路径，未覆盖原文件） | 35/35 通过、对照 8/8：串行分支无回归（--replay-windows 96） |
| `superseded/` | 返工过程中一版"每区域同步"计时口径的中间运行，**不作为任何结论依据**（说明见其中的 README） | — |
| `formal_run_hashes_before_seed_2026.json` / `_after_seed_2026.json` | seed 2026 正式运行全树哈希（返工前后各一次） | `combined_sha256` 均为 `a3773fdf…` → 未改动 |
| `formal_run_hashes_before_seed_2027.json` / `_after_seed_2027.json` | seed 2027 正式运行全树哈希 | `combined_sha256` 均为 `4fe7b8ef…` → 未改动 |

复现命令见 `reports/DroneRFa_一阶段数据加载优化返工R1_20260927.md` 第 4.4 节。

审计检查项共 19 条（P0–P18，其中 P8b 为串行专用顺序检查）：文件集合、窗口计数、多重集与顺序摘要、
重复读取、交付/完成计数与顺序、遍历完整性、批次完成、事件丢失、worker 进程证据、父进程未读、
两份相位报告一致，以及 P18 的**loader 注册/关闭生命周期**（覆盖"事件退出与队列清理可靠"）。

证据边界：基准是 48 窗有界样本（单机单盘），不能外推为整轮定标；并行审计的"已消费未读到"
能发现事件丢失，但不能证明 worker 崩溃前的物理读取次数。本轮未启用 workers 于任何正式训练。
