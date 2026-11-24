# 12 个未使用 DroneRFa 源文件：准备与核查报告

生成时间：2026-09-18。范围：下载、完整性核查、重复性核查、活动频段核查、生成稳定评价索引。
**本次未训练任何模型、未做模型选择、未调整阈值、未运行性能评价。**

---

## 1. 结论摘要

| 项目 | 结果 |
|---|---|
| 指定文件 | 12 个（T0000/T0010/T0101 × D00 × S0100/S0101/S0110/S0111） |
| 下载完成 | **12/12**，全部通过 RAR 结构与 CRC、`unrar t` 完整性测试、解压大小核对 |
| 落位位置 | `F:\DroneRFa数据集\heldout_20260917`（新数据与临时文件均在此盘，未写原 Documents 数据目录） |
| 文件字节合计 | 12,382,967,102 字节（11.53 GiB），与候选清单**逐字节一致 12/12** |
| 可用 10 ms 窗 | **1820**（背景 560 / Phantom 610 / Air 2S 650） |
| 全样本非有限值 | **0**（逐文件、逐通道完整扫描，非抽样；每通道 140M/150M/160M/180M 按文件分别记录） |
| 与原数据重复 | 逐字节重复 0 对；固定位置原始 IQ 指纹命中 0 对；文件名重叠 0 |
| 活动频段 | 无人机类固定窗 RF0 活动 12/12；背景类 5/12（间歇，符合背景定义）；RF1 新旧共 72 窗全部无活动 |
| 机器核查 | `verification.json` 中 8 项检查**全部 passed** |
| 阻塞 | 无 |

---

## 2. 实际下载的文件

| 文件名 | 字节数 | 每通道样本 | 采集块 | 索引窗 |
|---|---:|---:|---:|---:|
| T0000_D00_S0100.mat | 908,097,315 | 140,000,000 | 14 | 140 |
| T0000_D00_S0101.mat | 907,985,834 | 140,000,000 | 14 | 140 |
| T0000_D00_S0110.mat | 907,843,381 | 140,000,000 | 14 | 140 |
| T0000_D00_S0111.mat | 905,521,637 | 140,000,000 | 14 | 140 |
| T0010_D00_S0100.mat | 1,133,423,170 | 160,000,000 | 16 | 160 |
| T0010_D00_S0101.mat | 1,060,199,333 | 150,000,000 | 15 | 150 |
| T0010_D00_S0110.mat | 1,061,636,161 | 150,000,000 | 15 | 150 |
| T0010_D00_S0111.mat | 1,066,075,408 | 150,000,000 | 15 | 150 |
| T0101_D00_S0100.mat | 1,091,103,154 | 160,000,000 | 16 | 160 |
| T0101_D00_S0101.mat | 1,226,409,950 | 180,000,000 | 18 | 180 |
| T0101_D00_S0110.mat | 1,091,427,695 | 160,000,000 | 16 | 160 |
| T0101_D00_S0111.mat | 1,023,244,064 | 150,000,000 | 15 | 150 |

合计 **12,382,967,102 字节（11.53 GiB）**，**1820 窗**。

注：各文件存储长度并不一致（140M/150M/160M/180M 样本），因此窗数按各文件实际长度推导，未硬编码固定窗数。

每文件完整 SHA256 见 `verification.json` 的 `file_sha256`。

---

## 3. 远程档案核查（下载前）

`remote_probe.json`：

- RAR5 签名 `526172211a070100` 校验通过。
- 档案总长 **615,955,805,466 字节**，与预期一致；ETag `6774350c-8f69d3b51a`，下载全程未变。
- HTTP **206** 与 `Content-Range` 正常；内部区间 1024–1535 返回区间与长度均正确。
- **12/12 候选块头**核查通过：块内文件名、`block_offset`、`packed_size`、`block_size`、`unpacked_size` 与缓存清单**全部一致**。

---

## 4. 下载执行与断点

- 工具：`dronerfa_baseline/download_heldout_files.py`。选择、RAR5 封装与断点记账**复用** `download_dronerfa_subset.py` 的 `stream_selected_archive`（未改写其旧行为），传输后端为 curl 范围请求。
- 传输校验：每次范围响应先校验**同一次 GET** 的状态 206、`Content-Range` 的 start/end/total、`Content-Encoding`、落盘字节数，且写盘上限为请求长度 +1；协议违规 0 次。
- 逐文件流程：下载 → RAR 签名/主头/文件块/结束块结构校验 → CRC → `unrar t` → 解压 → **字节数与清单严格匹配** → HDF5 读取校验 → 仅清理该文件本轮的临时压缩包。
- 断点续传多次生效：首文件从 576 MiB 断点续传，第 10 个文件从 939,524,112 字节断点续传，均未重下已完成部分。
- 限流：最后一次 **r3** 运行共 161 次范围请求、24 次重试，其中 **23 次为 HTTP 429**，由退避重试自动化解；该计数只覆盖 r3 这一段，不是整轮下载的累计值（整轮请求数因多次断点续传而更大，各轮日志分别保留）。
- 运行日志（逐轮独立，均保留）：`download_heldout.log`、`download_heldout_20260918*.log`（含 `_resume`、`_r2`、`_r3`）及对应 `.error.log`（全部为空）、`.pid.json`。
- 暂停记录：`work/heldout_prep_20260917/下载暂停记录_2026-09-17.md`。

### 下载期间修正的两个脚本缺陷

1. **HDF5 校验误判（已由 Codex 定位）**：原条件 `source_samples == unpacked_size // 4` 把 HDF5 容器字节数当成采样数推算，导致第 1 个文件虽已正确解压却被判失败。已改为按实际内容判断：四通道均为向量、长度一致且非空、dtype 为浮点、RF0 经共享读取接口回读 1000 点且全部有限；**文件字节数与清单的严格匹配保持独立存在**。
2. **窗数硬编码**：核查脚本原写死「每文件 140 窗」，与实际长度不符，导致 `complete_blocks_and_windows` 误报失败。已改为由各文件实际样本数推导。

---

## 5. 结构与非有限值核查

对 12 个文件、4 个通道（RF0_I/RF0_Q/RF1_I/RF1_Q）**逐样本完整扫描**（每次读取 2,000,000 样本，不整块载入）：

- 四通道均存在，均为 `(1, N)` 向量，`dtype=float64`，各通道长度一致。
- **非有限值 0 个**（12 文件 × 4 通道全部）；每通道扫描长度按文件实际值记录（140M/150M/160M/180M），`verification.json` 的 `scanned_samples_per_channel_by_file` 保留逐文件数字。
- 每通道长度均可被 10,000,000 整除：**尾部未索引样本 0**。
- 原始 IQ 幅度量级约 ±0.07 以内，各通道平均功率见 `heldout_dataset_summary.json`。
- **容器内无采集元数据**：仅有四个数据集（外加 MATLAB 格式标记 `MATLAB_class`），无中心频率、增益、天线或采集时间。因此下文频率只以「相对通道中心的偏移」表述，不宣称绝对频段。

### 采集中心频率的元数据情况

- MAT 内**不存在**中心频率字段（`acquisition_metadata.present_in_mat = false`）。
- 文献本地提取文本（`01_DroneRFa_Dataset_JEIT2024.txt` 第 243–250 行）记载两种配置：默认 RF0 中心 2440 MHz、RF1 中心 5800 MHz；若设备仅支持 915 MHz 与 2.4 GHz，则 RF0 中心 915 MHz、RF1 中心 2440 MHz。同文第 194–195 行把 S0000–S0111 划为「初始通信频段」组。
- 两种配置下的绝对范围仅作**兼容性列出**，不代表已确认实际频段。

---

## 6. 重复性核查（`duplicate_check.json`）

| 检查 | 结果 |
|---|---|
| 新-旧 逐字节完全相同文件对 | **0** |
| 新-新 逐字节完全相同文件对 | 0 |
| 旧-旧 逐字节完全相同文件对 | 0 |
| 新-旧 固定位置原始 IQ 指纹命中 | **0** |
| 新-新 固定位置指纹命中 | 0 |
| 文件名与旧集合重叠 | **无**（旧 S0000/S0001/S0010/S0011，新 S0100/S0101/S0110/S0111） |

- 逐字节 SHA256 覆盖全部 12 个新文件与 12 个旧文件。
- 固定位置指纹：每文件在**首块、中间块、末块**各取**两个固定 10 ms 窗**的起始处 10,000 个样本，对四个通道的原始存储字节取哈希。
- **局限（必须保留）**：抽样指纹**不能**保证全面排除重录或重编码副本；MAT 内无会话/日期/地点元数据，因此**不同源文件不构成已证明的采集会话独立性**，本报告一律称「未使用源文件」，不称「跨会话泛化」。

---

## 7. 活动频段核查（`activity_band_check.json`）

取窗规则预先固定，新旧文件完全相同：**首块首窗、中间块中间窗、末块末窗**，共 3 个固定 10 ms 窗/文件。未参考任何模型表现，未因文件困难而剔除。

方法：每窗先去均值，再用 **Welch 平均周期图**（nfft=1024、50% 重叠、约 1950 个子段）估计 PSD；峰值搜索排除最外侧 1/64 频点，避免切片边界伪峰。活动判据为峰值高出中位数谱 **≥ 6 dB**。

### 新 12 文件

| 类别 | RF0 活动窗 | RF0 峰值偏移均值 | RF0 峰值偏移范围 | RF1 活动窗 |
|---|---:|---:|---|---:|
| T0000 背景 | **5/12** | −20.21 MHz | −40.04 … +7.81 MHz | **0/12** |
| T0010 Phantom 4 Pro | **12/12** | +2.35 MHz | −27.34 … +36.72 MHz | **0/12** |
| T0101 Air 2S | **12/12** | +7.85 MHz | −13.96 … +28.03 MHz | **0/12** |

### 与原 12 文件对比

| 类别/通道 | 新活动比例 | 旧活动比例 | 新峰值偏移均值 | 旧峰值偏移均值 | 差值 |
|---|---:|---:|---:|---:|---:|
| T0000 RF0 | 0.42 | 0.92 | −20.21 MHz | −4.59 MHz | −15.62 MHz |
| T0010 RF0 | 1.00 | 1.00 | +2.35 MHz | +4.61 MHz | −2.25 MHz |
| T0101 RF0 | 1.00 | 1.00 | +7.85 MHz | +5.54 MHz | +2.31 MHz |
| T0000/0010/0101 RF1 | 0.00 | 0.00 | 约 −1.7 … −3.7 MHz | 约 −1.8 … −3.2 MHz | <0.5 MHz |

### 如何解读

- **无人机两类在全部固定窗中 RF0 均有活动，背景类仅 5/12 且为间歇**——与「背景无无人机通信」的定义一致。**T0000 背景不会因缺少无人机活动而被判不合格**，此处不做任何此类判定。
- **RF1 在新旧两套共 72 个固定窗中全部无活动**；其噪声底平均功率约 1.7–2.2×10⁻⁶，而 RF0 在无人机文件中约 1.5–6.1×10⁻⁵。这与论文所述 RF0 为 2.4 GHz 接收通道、RF1 为 5.8 GHz 通道、本组录制为初始通信频段相符，**支持继续使用现有 RF0 前端**。
- 无人机两类的活动比例新旧**完全一致（12/12）**，峰值偏移同量级，占用带宽可比，未见协议级不匹配。
- 待 Codex 审查的一处观察：T0010 的 RF0 95% 占用带宽中位数，新数据 6.25 MHz、旧数据 24.56 MHz，差异较大（活动比例与峰值位置仍一致）。仅作为现象记录，不作结论，也未据此排除任何文件。
- 绝对频段：因缺少中心频率元数据，仅列出两种已声明配置下的兼容范围（默认 2390–2490 MHz 或变体 865–965 MHz），**不宣称实际频段**。

图：`activity_band_heldout.png`（新 12 文件）、`activity_band_original.png`（原 12 文件）。

---

## 8. 稳定评价索引（`heldout_window_index.csv`）

- **1820 窗**，字段与现有索引完全一致：`file_path, filename, target_code, target_name, distance_code, segment_code, split, label, block_index, window_index, start, length, source_samples`。
- `split` 一律为 **`heldout`**；标签沿用二分类规则：T0000→0（Background），T0010/T0101→1（Drone）。
- 切窗规则与现有基线一致：**严格在每个 10,000,000 点采集块内切 10 个 1,000,000 点窗，不跨块、不重叠、起点按 1,000,000 对齐**。
- 索引哈希：`6fe32faeb5d583a359d9e8aa17caa52a7c4567d545272ded25266767cd2cffcf`。
- **其他已有索引一律未修改**：与 `outputs/window_index.csv` 的文件名交集、路径交集**均为 0**；旧集合为 S0000/S0001/S0010/S0011（1810 窗），新集合为 S0100/S0101/S0110/S0111（1820 窗）。

### 类别计数

| target_code | 名称 | 文件数 | 窗数 | 标签 |
|---|---|---:|---:|---:|
| 0000 | Background | 4 | 560 | 0 |
| 0010 | DJI Phantom 4 Pro | 4 | 610 | 1 |
| 0101 | DJI Air 2S | 4 | 650 | 1 |
| 合计 | | 12 | **1820** | |

---

## 9. 机器核查（`verification.json`）

| 检查项 | 结果 |
|---|---|
| `manifest_size_match` | **passed**（12/12） |
| `four_channels_present` | **passed** |
| `lengths_consistent` | **passed** |
| `all_samples_finite` | **passed**（全样本扫描，非抽样） |
| `complete_blocks_and_windows` | **passed**（窗数由实际长度推导） |
| `index_block_boundaries` | **passed**（无跨块、无重叠） |
| `index_disjoint_from_existing` | **passed** |
| `download_records_consistent` | **passed** |

`verification.json` 另含：文件与索引哈希、类别计数、与原集合交叉检查，以及**被冻结的评价配置**（六个所选 A/B 检查点、现有 RF0/10 ms/STFT 前端、默认 logit 差 > 0 判定、clean/−10/−12/−15 四档、**噪声种子 22026**、按机型源文件等权的主汇总并保留计数、不利用新数据调整门槛/轮次/噪声范围）。

### 明确标记为「未核查」的项目

- 采集会话、日期、地点的独立性：**元数据缺失，无法证明**。
- 接收机设置（增益、天线）：MAT 内无元数据。
- 跨文件属于不同录制批次：无法证明。

以上均如实记录为未完成，未计为通过。

---

## 10. 实际运行命令

工作目录 `C:\Users\Administrator\Desktop\数据集处理`，解释器 `.venv\Scripts\python.exe`。

```powershell
# 远程可用性核查（RAR5签名/大小/206/Content-Range/12个候选块头）
python dronerfa_baseline\probe_heldout_remote.py

# 下载（断点续传，逐文件校验与解压）
python dronerfa_baseline\download_heldout_files.py --files all --workers 4 --chunk-mib 16 `
  --staging-dir "F:\DroneRFa数据集\heldout_20260917\downloads" `
  --extract-dir "F:\DroneRFa数据集\heldout_20260917"

# 结构 + 非有限值全扫描 + 生成索引与核查记录
python dronerfa_baseline\scan_heldout_files.py --workers 4

# 重复性核查（逐文件SHA256 + 固定位置原始IQ指纹）
python dronerfa_baseline\check_heldout_duplicates.py

# 活动频段核查（固定首/中/末窗，Welch PSD）
python dronerfa_baseline\check_heldout_activity_band.py

# 失败用例与真实窗回读
python dronerfa_baseline\test_heldout_prep.py
python dronerfa_baseline\test_heldout_index.py
```

### 测试结果

- `test_heldout_prep.py`：**17 项通过**。含 4 个拒绝用例（HTTP 200 不得读取正文、`Content-Range` 区间不符、正文超长、意外 `Content-Encoding`）、5 个路径安全用例（越界删除、删除非本轮 MAT、递归删除授权根、越界落位、本轮产物可删）、2 个 `(1,N)`/`(N,1)` 向量读取用例，以及真实小范围响应检查（签名与总长、候选偏移返回自身块头）。
- `test_heldout_index.py`：**8 项通过**。含索引不跨块、不重叠且对齐、不超文件长度、窗数与块数一致、`split` 仅 heldout、与旧集合文件名/路径无交集、标签与 target_code 一致，以及 **12 文件 × 首/中/末共 36 个真实窗经现有读取接口回读且全部有限**。

---

### 记账修正（2026-09-18，评价前）

本次只改记录表述，未重跑下载/扫描/去重/PSD，未改索引、MAT 哈希或信号统计：

- `download_records.json`：总体 `status` 由 `running` 改为 `completed`，`completed_at` 取 12 个文件 `finished_at` 的最大值（2026-09-18T04:02:50Z，即北京时间 12:02:50）；8 个本轮完成、4 个已存在校验通过。
- `verification.json`：删除固定值 `scanned_samples_per_channel=140000000`，改为按文件记录 `scanned_samples_per_channel_by_file`（140M/150M/160M/180M）。
- 本报告第 4 节的 161 次请求/24 次重试/23 次 429 明确标注为最后一次 r3 运行的数据，不再称整轮。

## 11. 失败与阻塞记录

| 事项 | 处理 |
|---|---|
| 沙箱默认拦截 Windows TLS 凭据（`schannel SEC_E_NO_CREDENTIALS`、Python `SSL UNEXPECTED_EOF`）且不允许写项目外 F 盘 | 按权限流程逐次申请更宽权限并在说明中写明用途；**TLS 证书校验全程保持开启**，未使用跳过校验的参数 |
| `pip` 无法访问 PyPI（同一 TLS 问题），`requests` 未装入 .venv | 未伪造模块；改为在导入前临时把基础运行时的纯 Python `requests` 载入 `sys.modules` 后立即移出该路径，经验证 **venv 自身的 h5py 3.16.0 / numpy 2.5.1 / torch 2.12.1+cu130 未被降级** |
| 后台下载进程两次被静默回收（PID 4012、10816） | 断点均完好，重跑同命令续传，未重下已完成数据 |
| HDF5 校验误判、窗数硬编码、`old_segments` 解析错误 | 均已修正并按实际内容/长度重新推导，修正后重跑通过 |
| 活动频段核查首版超时 | 原因是对 gzip 单块数据集做整文件 RMS 统计代价过大；规格只需固定窗，已移除该统计并改为按通道整块读取 |
| 首版峰值恒为 −50.00 MHz（FFT 边界） | 切片边缘不连续造成的伪峰；改用 Welch 多子段平均并排除最外侧频点 |

**当前无阻塞。**

---

## 12. 产物清单

目录：`dronerfa_baseline/outputs/heldout_preparation_20260917/`

| 文件 | 内容 |
|---|---|
| `heldout_window_index.csv` | 1820 窗稳定索引（split=heldout） |
| `verification.json` | 文件/索引哈希、类别计数、交叉检查、8 项核查、冻结评价配置 |
| `heldout_dataset_summary.json` | 逐文件结构、四通道扫描统计、索引审计、交叉检查 |
| `duplicate_check.json` | 全文件 SHA256 与固定位置 IQ 指纹对比 |
| `activity_band_check.json` | 固定窗 PSD 统计、新旧协议对比、频段兼容性 |
| `activity_band_heldout.png` / `activity_band_original.png` | 固定窗活动比例图 |
| `download_records.json` | 逐文件下载/校验记录（12 条） |
| `remote_probe.json` | 远程档案与 12 个候选块头核查证据 |

新增脚本：`probe_heldout_remote.py`、`download_heldout_files.py`、`scan_heldout_files.py`、`check_heldout_duplicates.py`、`check_heldout_activity_band.py`、`test_heldout_prep.py`、`test_heldout_index.py`（均在 `dronerfa_baseline/`）。

---

## 13. 未做与边界

- **未训练、未选模型、未调阈值、未做任何性能评价**；新数据未用于重选轮次、门槛或训练噪声范围。
- 新文件仅用于**预先冻结**的后续评价。
- 「不同源文件」不自动等于「跨会话/跨地点泛化」；本报告未作此声称。
- 活动频段一节的频率均为**相对通道中心的偏移**；绝对频段因元数据缺失而未确认。
- 按指示未编辑 `work/deepseek-context.md` 与 `work/deepseek-workflow.json`，未提交、推送或发布。
