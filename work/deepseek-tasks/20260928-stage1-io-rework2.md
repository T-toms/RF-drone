# STAGE1-IO-OPT-R2：补齐内存记录并修正报告（第二轮，限定范围）

## Codex 决策

R1 run `20260927-190015-e514cf21` 已停止并交接 completed。Codex 独立查原始产物：同版本 2 epoch/24 batch 的 legacy/0/2/4 完整训练步对照均逐位相同；串/并行顶层核验各 41/41 通过，并行 11/11 篡改对照检出；真实训练步基准 0/2/4 workers 为 8.353/6.567/4.615 窗每秒，默认线程复测方向相同；旧两份正式训练前后哈希一致。这些结果足以决定**后续正式训练采用 `num_workers=0`**，不再尝试加 worker。R1 的剩余验收点仅在下文，完成即可放行下一任务。

## 两个必须修复的证据问题

1. 基准报告宣称记录了进程内存，但 `rework1/benchmark/benchmark_latest.json` 和 `benchmark_threads_default/benchmark_latest.json` 的三档 `configurations.<workers>.memory` **全部是 `{}`**。`benchmark_summary.csv` 也没有内存列。`benchmark_stage1_io.py:109` 的 `process_memory_mb()` 广泛吞异常或失败并返回空对象，不能将其视为成功记录。
   - 查明 Win32 `GetProcessMemoryInfo` 失败原因并修复；显式声明正确函数签名/返回值，发生错误时给可诊断信息，不能吞成 `{}` 并报告成功。记录至少父进程 working set 与 peak working set，单位 MiB；如能在 worker 活跃期间可靠采样，再分列 worker 合计。若只能记录父进程，要在字段/报告中明确，不能声称总进程内存。
   - 增加针对“内存采样非空、有限、正值”和失败可见的必要小测试。重新跑**同样有界**的真实训练步基准，输出独立 `rework2/` 新运行（原始逐轮、环境/配置、JSON/CSV一致）；比较新旧速度结论，若变化明显先分析，再交 Codex。不得改写或删除 R1 原始结果。无需增加 9590 窗全量基准，也不进行正式训练。
2. `reports/DroneRFa_一阶段数据加载优化返工R1_20260927.md` 第0节写并行 smoke **40/40＋10/10**，第2节及真实 `smoke_parallel2_loader_audit.json` 是 **41/41＋11/11**。修正报告，使所有数字由已存实际证据导出。第3节“差异不是加载器等价性问题，而是不同代码版本”属于过度因果判断：只可说旧两个 smoke **不同版本，所以旧差值不能用于判断加载器等价性**；同版本新对照证明当前版本等价，无法恢复旧代码逐行归因。报告中引用新内存与基准数据，旧测量留为历史版本。

## 范围与验收

只改 `dronerfa_baseline/benchmark_stage1_io.py`、相关必要测试、R1中文报告以及新 `dronerfa_baseline/outputs/stage1_io_optimization_20260927/rework2/`。其他源码和旧正式产物只读。无需再次改加载模块、核验器或复跑 104 项全套测试；跑内存测试、必要基准与摘要一致性核对即可。确认 R1 的串/并行 smoke 与等价 JSON 保持原哈希，旧 seed2026/2027 正式产物保持不变。

用户已授权既定路线，但本轮不得擅自开始 `STAGE1-LOSS-NORM`；交接完成后由 Codex 验收，选择 `num_workers=0` 并单独派下一任务。多进程测试如触发 DSH 原生审批，按正常机制处理，不绕过权限。
