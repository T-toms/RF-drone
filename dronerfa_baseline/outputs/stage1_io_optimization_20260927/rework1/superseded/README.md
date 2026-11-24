# 被取代的中间运行（不作为任何结论依据）

这三个 JSON 由本轮返工**过程中**的一版基准脚本产生：那一版在**每个区域边界都同步**
（`torch.cuda.synchronize()`），因此 wall time 里没有 CPU/GPU 重叠，不能当作端到端吞吐。
最终版把"吞吐"与"分段诊断"分开：吞吐用异步流水 wall time（每 epoch 一次同步），分段耗时
另跑一次同步诊断并单独报告。

因此这三个文件只作为过程留痕，任何报告数字都必须取自已移除这些文件的最终 run：

| 文件 | 说明 |
| --- | --- |
| `benchmark_bench-20260927T113418Z.json` | 父线程 2、每区域同步口径 |
| `benchmark_bench-20260927T113727Z.json` | 父线程默认、每区域同步口径 |
| `benchmark_bench-20260927T114050Z.json` | 关闭事件流诊断、每区域同步口径 |

最终口径的运行见 `../benchmark/`、`../benchmark_threads_default/`、
`../benchmark_diagnostic_no_audit/`。
