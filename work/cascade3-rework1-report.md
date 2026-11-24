# CASCADE3 验收返工 1 报告：跨进程原始 IQ 重建验证修复

- 任务：`work/deepseek-tasks/20260920-cascade3-rework1.md`
- 执行会话：`session-d1a06e50-6be6-4dd9-8db2-179c39b7528a`，run `20260920-105246-475b889f`
- 时间：2026-09-20（+08:00）
- 修改范围：只改 CASCADE3 验证器与测试（`dronerfa_baseline/verify_cascade3_evaluation.py`、`dronerfa_baseline/test_cascade3_evaluation.py`）。
  **未训练、未调阈值、未改冻结配置、未重跑正式推理、未改动任何正式产物、未提交、未恢复定时器。**

## 1. 结论（先说结果）

1. **正式预测没有错误，因此没有重跑全量推理。** 23,160 行 `window_predictions.csv`、指标 CSV 与五个正式摘要文件的 SHA256 与 Codex 首次验收证据逐个一致（18 个文件里 17 个有可比对记录的全部匹配，唯一无记录的是 manifest 自身）。
2. **Codex 那次失败的原因不是预测错误，而是那次验证跑在 CPU 上**：`python` 在本机 PATH 上解析为 `D:\anaconda\python.exe`（torch **2.12.0+cpu**），而正式推理用的是项目 `.venv`（torch **2.12.1+cu130**，CUDA）。用**未修改的旧验证器**把 CUDA 关掉复跑，四条失败信息与 Codex 的报告**逐位相同**。
3. **旧交接里的 `passed=true failures=0` 是真的**，只是在 `.venv`+CUDA 环境下得到的；缺陷在于旧验证器既不记录解释器/设备，也会在 CUDA 不可用时静默回退 CPU，然后把「这个解释器无法复现正式运算」报成「预测不符」。
4. 验证器已修复：记录并比对运行环境、按环境分层判定、必要时用项目解释器重跑同一条命令；跨环境时用「输入精确一致 + 数值有界 + 离散决策一致」替代不可靠的 float32 字节摘要判据，并保留正式摘要与既有第一阶段摘要逐字节一致这一独立历史证据。
5. 修复后：**从项目根目录连续两次直接运行均为 `passed=True failures=0 tier=formal_environment`**（92.8s / 59.0s）；CPU 解释器下也通过（跨环境层），且 156 个目标窗的**执行门、最终机型、最终三分类标签零变化**。
6. 输入侧证据：**7,720 个窗口的正式 `input_probe_sha256` 全部逐位复现**（原始 IQ + 复 AWGN，全程不涉及 GPU STFT），156 个目标窗的输入张量又与验证器自写的读取/加噪链逐字节一致。

## 2. 根因（含复现）

### 2.1 直接复现失败

失败命令（任务给定）：

```powershell
python dronerfa_baseline\verify_cascade3_evaluation.py --no-write --report work\cascade3-codex-verification.json
```

本机 `python` 解析顺序（`Get-Command python -All`）：`D:\anaconda\python.exe`、`D:\Python\python.exe`、WindowsApps。
其中 `D:\anaconda` 只有 **torch 2.12.0+cpu**（无 CUDA），`D:\Python` 没有 torch。正式运行 `environment.json` 记录的是 **torch 2.12.1+cu130 / device cuda / RTX 4060 Laptop**，项目 `.venv` 正是这个组合。

用**修改前的旧验证器**、同一 `.venv` torch，但把 CUDA 关掉：

```powershell
$env:CUDA_VISIBLE_DEVICES="-1"
.venv\Scripts\python.exe dronerfa_baseline\verify_cascade3_evaluation.py --no-write --report work\_c3_repro_cpu_venv.json
```

结果与 Codex 的报告完全一致（`work/_c3_repro_cpu_venv.json`，53.0s vs Codex 51.6s）：

| 失败项 | Codex | CPU 复现 |
| --- | --- | --- |
| window digest 不符 | 255 | 255 |
| stage-1 logit 最大差 | 0.002524852752685547 | 0.002524852752685547 |
| stage-2 logit 最大差 | 0.007019996643066406 | 0.007019996643066406 |
| decision/rule mismatch | 242 | 242 |

同时，**同一验证器在 `.venv`+CUDA 下（正式环境）对全部 156 个目标窗复算也是位级一致的**（§4 的第 1/2 行：摘要差 0、logit 差 0.0）。`work/_c3_rebuild_probe.py` 进一步在同一 torch 下直接对比同一组目标窗在两种设备上的重建结果：

```powershell
.venv\Scripts\python.exe work\_c3_rebuild_probe.py --dataset heldout --condition clean --seed 2026 --device cuda --out work\_c3_probe_venv_cuda.json
# env torch=2.12.1+cu130 device=cuda cuda=True
# targets=11 digest_mismatches=0 max_stage1=0.000000000 max_stage2=0.000000000 gate_flips=0 final_flips=0

.venv\Scripts\python.exe work\_c3_rebuild_probe.py --dataset heldout --condition clean --seed 2026 --device cpu --out work\_c3_probe_venv_cpu.json
# env torch=2.12.1+cu130 device=cpu cuda=True
# targets=11 digest_mismatches=11 max_stage1=0.002120018 max_stage2=0.007019997 gate_flips=0 final_flips=0
```

### 2.2 失败数字的完整分解（要求 2）

旧计数器把「数值超差」和「决策变化」混在一个 `rule_mismatches` 里。新验证器把它们拆开，跨环境（CPU）复算结果：

| 计数 | 值 | 说明 |
| --- | --- | --- |
| stage-1 特征摘要不符 | 156 / 156 | 纯数值，CPU STFT 与 CUDA STFT 末位不同 |
| stage-2 特征摘要不符 | 99 / 99 | 同上（99 = 目标窗中执行了第二阶段的窗数） |
| 旧 255 = 156 + 99 | ✅ | 与 Codex 的 255 完全对应 |
| stage-1 margin 超 1e-4 | 147（最大 0.0040683746337890625） | 数值漂移 |
| stage-2 margin 超 1e-4 | 95（最大 0.011341094970703125） | 数值漂移 |
| 旧 242 = 147 + 95 | ✅ | 与 Codex 的 242 完全对应 |
| **stage1_gate_changes（执行门变化）** | **0** | 真正的决策变化 |
| **stage2_class_changes（最终机型变化）** | **0** | 真正的决策变化 |
| **stage2_batch_id_errors** | **0** | 批次组成不同 |
| stage-1 logit 最大差 | 0.002524852752685547 | |
| stage-2 logit 最大差 | 0.007019996643066406 | |

也就是说：Codex 那次失败里**没有任何一个窗口的分类决策发生变化**，全部是 CPU/GPU 计算路径的末位差异被 1e-4 判据放大出来的。

### 2.3 为什么旧交接会报全零

旧验证器的 `check_targets` 自己选设备（`cuda if available else cpu`），且报告里不记录解释器、torch 版本或设备。执行器通过 `work/_c3_driver.py` 在 `.venv`（CUDA）下运行，得到的确实就是零差异（`work/_c3_verify.log` 55.2s、`work/_c3_final_check.log` 51.2s）。两件事同时为真：

- 正式环境复算 = 位级一致（现在已由两次全新进程复跑证明，见 §4）；
- 非正式环境（CPU）复算 = 大量摘要不符。

旧工具无法区分这两者，于是 Codex 在普通 `python` 进程里得到了「失败」，而执行器在 `.venv` 里得到了「通过」。**问题在验证器的环境盲区，不在正式推理，也不在旧交接的诚实性。**

### 2.4 与 batch/缓存/seed 无关的排除项

`work/_c3_rebuild_probe.py` 与两次正式环境复跑共同确认：batch=4 组成、`default_rng(22026 + 完整 split 行号)`、epoch=0、共享特征张量、stage-2 按升序选中行切批、以及 F 盘缓存路径都与旧运行一致；环境差异只体现在 `device` 与 torch 构建上。

## 3. 验证器修改（要求 2/3/4/5）

文件：`dronerfa_baseline/verify_cascade3_evaluation.py`

1. **环境记录与分层**：新增 `runtime_environment()` / `comparison_tier()`，报告里同时写入正式运行 `environment.json` 与本次复算的解释器、torch、CUDA、numpy、device、device_name，并给出
   - `formal_environment`：torch 构建与 device 都与正式运行一致 → 严格判据（逐窗摘要逐字节相同、|Δlogit| ≤ 1e-4、离散决策一致）；
   - `cross_environment`：不一致 → 摘要判据标记为 `not_applicable_cross_environment`（只记录不判失败），改用输入精确一致 + |Δlogit| ≤ `CROSS_ENV_LOGIT_BOUND = 5e-2` + 离散决策一致。
   判据与阈值都写进报告（`criterion`、`tolerance`、`cross_environment_logit_bound`），不静默放宽容差。
2. **必要时正确复现正式运算**：`maybe_reexec()` / `plan_reexec()`。当前解释器无法复现正式运算且项目 `.venv` 存在时，用**同一条命令**在 `.venv` 下重跑（子进程 stdio 直接继承，不捕获），并在报告中记录 `checks.environment.reexec.parent`。`--no-reexec` 可关闭；`--device auto|cuda|cpu` 可显式指定。
3. **rule_mismatches 拆分**：`target_rule_differences()` 独立计数 `stage1_gate_changes`、`stage1_margin_out_of_tolerance`、`stage2_class_changes`、`stage2_margin_out_of_tolerance`、`stage2_batch_id_errors`，并保留总数 `rule_mismatches`。
4. **输入侧独立证据**：新增 `build_input_tensor()`（验证器自写的 RF0 切片 → 去复均值 → RMS 归一化 → `default_rng(noise_seed + 0*len + 完整 split 行号)` 复 AWGN，纯 numpy，无 GPU）与 `iter_group_inputs()`；每个目标窗同时记录「冻结读取器输入张量摘要」「独立重建摘要」（探针前 4096 点与整张量两种）。
5. **正式 input probe 复核**：新增 `--input-probe-groups`，按正式规则重算整段 split 的滚动 `input_probe_sha256` 并与 `prediction_cache_manifest.json` 比对。
6. 逐目标窗记录（`checks.target_recomputation.targets`）含输入摘要、复算特征摘要、logit/margin 差、gate/class/batch 标志，便于跨进程直接比对。
7. 报告新增顶层 `numeric_comparison_tier` 与 `checks.environment`；旧字段语义未变，`--fixture` 模式仍保持原严格行为（自检篡改用例照旧全部被拒）。

文件：`dronerfa_baseline/test_cascade3_evaluation.py`（新增 11 项，共 48 项）

- `EnvironmentTierTests`：分层判定（同环境/CPU/异 torch/无记录）、重跑计划（缺 venv、已有标记、已是项目解释器、同环境不重跑均不触发）、`target_failures` 在两种层下的失败集合、`target_rule_differences` 对漂移/门变化/机型变化/批次错的拆分。
- `InputRebuildTests`：合成 HDF5 上独立读取链与冻结读取器在 clean/-10/-12 下逐字节一致、错误的噪声位置不复现、`probe_payload` 前导点数、`iter_group_inputs` 全 split 顺序。

## 4. 实际运行的检查（要求 1/4/6/7）

| # | 命令（项目根目录） | 结果 | 证据 |
| --- | --- | --- | --- |
| 0 | `$env:CUDA_VISIBLE_DEVICES="-1"; .venv\Scripts\python.exe dronerfa_baseline\verify_cascade3_evaluation.py --no-write --report work\_c3_repro_cpu_venv.json`（旧验证器） | 失败 4 项，数字与 Codex 逐位相同（255 / 0.002524852752685547 / 0.007019996643066406 / 242，53.0s） | `work/_c3_repro_cpu_venv.json` |
| 1 | 连续直接运行 1：`.venv\Scripts\python.exe dronerfa_baseline\verify_cascade3_evaluation.py --no-write --report work\cascade3-rework1-verify-direct1.json` | `passed=True failures=0 tier=formal_environment`（92.8s）；156 目标窗摘要不符 0、输入不符 0、logit 最大差 0.0、五项规则计数全 0 | `work/cascade3-rework1-verify-direct1.json` |
| 2 | 连续直接运行 2（紧接上一条，无其他前台命令） | `passed=True failures=0 tier=formal_environment`（59.0s），逐目标窗记录与第 1 次位级相同 | `work/cascade3-rework1-verify-direct2.json` |
| 3 | 同 torch、强制 CPU：`--device cpu --no-reexec` | `passed=True failures=0 tier=cross_environment`（66.2s）：摘要 156+99 不符按设计不判失败，输入不符 0，|Δlogit| 0.0025/0.0070 ≤ 0.05，**执行门/机型/批次零变化** | `work/cascade3-rework1-verify-cpu-venv.json` |
| 4 | 裸 PATH python（torch 2.12.0+cpu）`--no-reexec` | `passed=True failures=0 tier=cross_environment`（68.9s），同上 | `work/cascade3-rework1-verify-cpu-anaconda.json` |
| 5 | 裸 PATH python（默认） | 自动 `re-executing with .venv\Scripts\python.exe` → `passed=True failures=0 tier=formal_environment`（50.2s），报告记录父进程为 `D:\anaconda\python.exe / torch 2.12.0+cpu / cpu` | `work/cascade3-rework1-verify-reexec.json` |
| 6 | `--input-probe-groups heldout:clean,heldout:-10,heldout:-12,heldout:-15,old_test:clean` | `passed=True failures=0 tier=formal_environment`（1721.0s）：5 组共 7,720 窗（1820×4 + 440）的滚动输入摘要与正式 `input_probe_sha256` **全部逐位相同**（clean `c77614c8…`、-10 `081d2281…`、-12 `2908f6e9…`、-15 `b66b3322…`、old_test clean `087953e9…`） | `work/cascade3-rework1-input-probe.json` |
| 7 | `python work\_c3_repeatability.py formal=…direct1 formal=…direct2 cross=…cpu-venv cross=…cpu-anaconda --out work\cascade3-rework1-repeatability.json` | 正式层两次全新进程：156/156 输入摘要相同、156/156 复算特征摘要相同且都等于 CSV、漂移差 0 → `bit_identical_recomputation=true`；两个**不同 torch 构建**的 CPU 进程之间 156/156 复算特征摘要相同（但与 CSV 全不相同）→ CPU 路径本身也是位级可复现的系统性差异；形式层 vs 交叉层：输入 156/156 相同、gate/class 标志零分歧 | `work/cascade3-rework1-repeatability.json` |
| 8 | `.venv\Scripts\python.exe test_cascade3_evaluation.py` | `Ran 48 tests … OK`（16.7s，无跳过；含 145 窗真实有界 smoke 与正式产物复算） | 见 §3 新增测试；`PublishedOutputTests` 同时断言正式预测 CSV 未被测试改动 |
| 9 | `.venv\Scripts\python.exe work\_c3_artifact_hashes.py` | 18 个正式文件，17 个有可比对记录的**全部匹配**（`window_predictions.csv` = `59321de7be48aae7…`，与 Codex 首次证据一致），无记录的是 manifest 自身（manifest 不列自身哈希） | `work/cascade3-rework1-formal-artifacts.json` |

**最终三分类标签的一致性**：验证器在两种层下都硬性要求 `stage1_gate_changes = 0`（执行门）与 `stage2_class_changes = 0`（通过第一阶段的窗的最终机型）；未通过第一阶段的窗按冻结规则输出 Background 且 `final_prediction` 由静态规则单独复核。因此「执行门 + 机型」两项全零即等价于 156 个目标窗的**最终三分类标签零变化**，本次 CPU 复算与正式层复跑均满足。

补充说明（要求 1 的环境检查）：在**本 DSH 会话**里直接 `python -c "import torch"`（`D:\anaconda\python.exe`）会以 `OMP: Error #15 … libiomp5md.dll already initialized` 中止；设 `KMP_DUPLICATE_LIB_OK=TRUE` 后可导入 torch 2.12.0+cpu，本报告 §4 的第 4/5 行即在该设置下运行。这不是项目代码问题，但说明「裸 `python`」在本机既不是正式解释器、也不总是能起来；正式与复验都应以项目 `.venv` 为准（验证器现在会自动这样做）。

## 5. 正式产物未变（要求 7）

- 未重跑正式推理：`evaluation_run.json`、`full_run.log`、`window_predictions.csv`、五张指标表、五个 `feature_digests/*.bin`、七张图、`report.md`、`experiment_manifest.json`、`verification.json` 全部保持 2026-09-20 10:36–10:44 的原状。
- 本轮所有验证器调用都带 `--no-write`，因此没有覆写 `outputs/cascade3_evaluation_20260919/verification.json`；发布树里的 `verification.json` 仍是执行器 GPU 那次的历史记录（不带新的环境字段），本轮新增证据都在 `work/` 与 `--report` 指向的 JSON 中。
- 提醒：不带 `--no-write` 直接对正式根目录运行验证器会重写该 `verification.json`，从而与 `experiment_manifest.json` 记录的哈希不一致；复验请沿用任务里的 `--no-write` 形式（本次未改变这一既有行为）。

## 6. 边界与未主张的内容

- 逐窗摘要与 logits 是**设备与 torch 构建相关**的：它们在记录的 CUDA 环境下位级可复现，在任意 CPU 解释器下不可复现；这是数值事实，不是缺陷。
- 跨环境的「有界数值漂移 + 决策一致」只在验证器固定的 156 个目标窗上验证过，不能外推为整张 23,160 行表在 CPU 上逐窗安全的普适保证。
- 本轮没有重训、没有调阈值、没有改冻结配置、没有新增类别或噪声方式实验、没有恢复任何定时器、没有提交/推送/发布。
- 第二次验收失败 JSON `work/cascade3-codex-verification.json` 原样保留，未被覆盖。

## 7. 建议 Codex 复验步骤

1. 用任务里的直接命令（建议显式用项目解释器）复跑：
   `& .venv\Scripts\python.exe dronerfa_baseline\verify_cascade3_evaluation.py --no-write --report work\cascade3-codex-verification-after.json`
   期望 `passed=True failures=0 tier=formal_environment`。
2. 用裸 `python`（CPU torch）复跑同一条命令：期望看到 `environment: … re-executing with …\.venv\Scripts\python.exe` 且仍然通过；若要检查**跨环境层**本身，加 `--no-reexec`。
3. 抽查 `work/cascade3-rework1-verify-cpu-venv.json` 的 `checks.target_recomputation`：确认 255/242 的分解、五项规则计数全 0、以及每个目标窗的输入摘要一致。
4. 抽查 `work/cascade3-rework1-repeatability.json`：确认两次正式环境复跑 `bit_identical_recomputation=true`。
5. 用 `work/_c3_artifact_hashes.py` 复核正式产物未变；用 `work/cascade3-rework1-rootcause.json` 核对根因链条。
