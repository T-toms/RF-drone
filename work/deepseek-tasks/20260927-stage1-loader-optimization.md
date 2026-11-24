# STAGE1-IO-OPT：用户授权的数据加载性能优化（当前会话追加任务）

用户明确要求“继续优化”，重点解决 GPU 等待串行 CPU/HDF5 读取和预处理的问题。沿用当前 DSH 原生会话，原 STAGE1-12DRONES-B seed 2027 的完整训练与验收目标仍然有效。不要另起竞争训练任务。

## 现状与接续决策

Codex 实测当前 GPU 利用率 1–24%，吞吐约 5 窗/s；训练和验证缓存构造的 DataLoader 都是 num_workers=0，CPU 完成 HDF5 读取、RMS 归一化及 AWGN；GPU 做 STFT 和 ResNet18。seed 2027 已完成第 1 轮并在第 2 轮继续。现有 epoch checkpoint 不含 optimizer/scaler/RNG，不能假装从这些权重无损恢复训练。

保留当前训练进程完成其已开始的工作，不注入或热改运行进程，不删除/覆写其产物。可以在同一会话内准备独立优化代码和轻量逻辑检查；真实 I/O/GPU 对照基准必须与正式训练串行运行，避免磁盘争用影响结论。新的加载器在独立 smoke/benchmark 中实际运行通过后才称已启用；必须明确当前 seed 2027 仍为串行旧加载器，不谎称加速已应用到它。

## Codex 接口和实现约束

1. 新增独立加载优化模块，保留 num_workers=0 参考路径。CLI 支持 --num-workers、--prefetch-factor，Windows spawn，每 worker 独立懒打开 HDF5；优先比较 workers 0/2/4，有界预取（初始 factor 2）。参数默认必须保持旧调用兼容；根据实测再确定推荐命令。batch size 4、样本顺序、seed/epoch/position 的 AWGN 规则、模型、学习率、AMP、验证噪声种子及全部训练协议保持一致。
2. worker epoch 状态必须正确传播。最简单可靠方式是在每个 epoch 设好 dataset.epoch 后重建非 persistent workers 的 DataLoader；若采用 persistent workers，必须证明跨 epoch 的噪声种子更新有效。不得因为多 worker 复制而共享已打开 HDF5 对象。
3. 读取审计不能只留父进程的空计数，也不能把索引推算数伪装成实际读取。为各 worker 记录真实文件打开/成功读取事件或计数，完整汇总；区分预取读取与实际消费窗口，记录异常中断/未消费预取。只允许 train/val 源文件，测试 IQ 零读取。保持已有审计语义可追溯，兼容旧验证器或明确提供新版本核验。
4. 验证：同一 seed、epoch、样本顺序下，0 vs 2/4 workers 的原始预处理后输入张量逐字节一致，标签/次序一致，至少覆盖两个 epoch、两类及不同噪声抽取；短 CUDA 训练步的损失/参数对照使用同初始化、同 batch、同 RNG 和相同环境，按数值模式合理比较。验证异常退出的 workers 清理及审计落盘。测试应针对真实风险，不反复运行无关大测试。
5. 性能：待原训练完成后，用同一组真实 train/val 窗与同样顺序做有界交替基准，分开记录读取/CPU 预处理、传输/前端、模型步耗时，至少比较 workers 0/2/4。记录 warmup、冷/热缓存限制、samples/s、进程内存和 GPU 利用率，单独考虑 Windows 启动开销；不要为了 GPU 数值好看增加无意义计算。只采用确实提升端到端吞吐且等价验证通过的方案。若无增益，解释测量结果和下一步，不能虚报优化成功。
6. 先不复制几十 GB 原始数据到 SSD、不引入全量特征缓存（会冻结随机噪声）、不增加 batch size。若简单并行仍受限，再向 Codex 报告实际分段计时，由 Codex 决定进一步架构。

## 产物和交接

- 独立优化模块、必要训练入口参数与审计/核验适配、必要测试；保持 seed 2026、2027 已开始/已验收产物不变。
- `dronerfa_baseline/outputs/stage1_io_optimization_20260927/` 保存等价性、实际性能、环境、读取审计、可复制的启用命令与短 CUDA 验证运行；中文简明报告给出优化前后实测数字和当前正式训练是否应用优化。
- 不新开 seed 2028 全量训练，不启动第二阶段。当前会话完成 seed 2027 验收证据后串行完成本优化，合并更新原 handoff 文件（run 20260927-142720-1a3455e4）为 completed 或 needs_codex，明确两部分完成情况。
