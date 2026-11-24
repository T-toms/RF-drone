# DroneRFa 第一阶段：无人机有无检测与低 SNR 基线

## 1. 当前研究目标

本阶段只解决一个问题：

> 对一段 2.4 GHz 频段的 RF0 复数 IQ 信号，判断其中是否存在无人机通信信号。

类别定义如下：

| 类别 | DroneRFa 标签 | 当前文件 |
|---|---|---|
| 背景（0） | T0000 | 4 个文件 |
| 无人机（1） | T0010、T0101 | Phantom 4 Pro 和 Air 2S，共 8 个文件 |

当前暂不做：

- Phantom 4 Pro 与 Air 2S 的机型分类；
- Wi-Fi、蓝牙等干扰类型的单独分类；
- 多架无人机重叠检测；
- 同型号不同实体的射频指纹识别；
- 实时 USRP 部署。

这些问题需要在“有无无人机”的数据划分和评估可信之后逐步增加。

## 2. 数据到底是什么

每个 `.mat` 是 MATLAB v7.3/HDF5 文件，包含：

| 数组 | 含义 |
|---|---|
| `RF0_I`、`RF0_Q` | RF0 接收通道的同相、正交采样 |
| `RF1_I`、`RF1_Q` | RF1 接收通道的同相、正交采样 |

复数基带信号由 `IQ = I + jQ` 重建。根据 DroneRFa 官方说明，采样率为 100 MS/s；当前 S0000-S0011 文件的设备初始通信位于 2.4 GHz，因此本阶段使用中心频率约 2.440 GHz 的 RF0。RF1 是同步采集的另一频段通道，本阶段不作为主输入。

`T0000` 不是理想高斯白噪声。实际图中可见窄带和突发信号，符合官方“背景含蓝牙、Wi-Fi”的说明。因此当前任务比“白噪声/无人机”更接近真实频谱监测。

文件名示例 `T0010_D00_S0001.mat`：

- `T0010`：DJI Phantom 4 Pro；
- `D00`：20-40 m；
- `S0001`：一次信号片段编号。

## 3. 方法的论文依据

### 3.1 DroneRFa 原始论文

俞宁宁等在 DroneRFa 原论文中使用 STFT 和 ResNet-18 验证数据的可识别性。论文基准使用 1,000,000 个采样点作为一个信号样本，STFT 分辨率为 1024，并报告 ResNet-18 深度学习结果。论文表 5 的基准组准确率为 0.9773，但它对应原论文完整识别实验，不能直接作为当前 3 类文件二分类实验的目标值。

本工程保留：

- 1,000,000 点窗口，即 100 MS/s 下 10 ms；
- STFT `n_fft=1024`；
- Hamming 窗；
- 50% 重叠，即 `hop_length=512`；
- ResNet-18；
- Adam，初始学习率 0.001；
- 交叉熵损失。

本工程主动修改：

- 不使用随机窗口 6:2:2 划分，而按原始 `S` 文件划分，避免相邻采样泄漏；
- 只使用当前活动频段 RF0；
- 将 1024 频率单元的时频矩阵缩放为 256×256，以适配 8 GB 显存；
- 做二分类，并用类别权重处理背景/无人机数量不平衡。

### 3.2 低 SNR 论文

Glüge 等在 IEEE Journal of Radio Frequency Identification 发表的工作比较了多个 CNN 在低 SNR 无人机 RF 识别中的表现。其方法将 IQ 转成二维频谱表示，按 SNR 分层评估，并使用 balanced accuracy；论文覆盖到 -20 dB，指出约 -12 dB 以下性能开始明显下降。

本工程据此采用：

- 在 IQ 域加入复高斯白噪声，而不是在 PNG 上加像素噪声；
- 训练时随机抽取 `clean, 10, 5, 0, -5, -10, -12 dB`；
- 测试时分别报告 `clean, 0, -5, -10, -12, -15, -20 dB`；
- 主要看 balanced accuracy、召回率、特异度和 F1，而不只看 accuracy；
- 每个 SNR 保存独立混淆矩阵。

重要限制：DroneRFa 是城市实采数据，不存在可分离的“纯净无人机源信号”。这里的 SNR 是对现有窗口再次加入 AWGN 的**附加退化 SNR**，不能宣称为采集现场的绝对物理 SNR。代码对背景类也施加同等级噪声，防止模型把“是否加噪”当成类别捷径。

### 3.3 ResNet-18 与后续扩展

ResNet-18 按 He 等 2016 年论文的标准残差基本块实现。Basak 等在 IEEE TCCN 论文中比较了 GoF+DRNN 两阶段方案和基于 YOLO 的时频联合检测分类；它更适合未来的多信号时频区域定位，不是当前单窗口二分类的必要复杂度。

## 4. 数据划分和防泄漏

原始数据以 10,000,000 点为一次连续采集保存块，块与块之间可能不连续。本工程：

1. 每个 10,000,000 点块切成 10 个不重叠的 1,000,000 点窗口；
2. 绝不让窗口跨越采集块边界；
3. 不复制原始 IQ，只在 CSV 中保存文件路径、起点和长度；
4. 训练时由 `h5py` 按需读取。

按文件划分：

| 集合 | S 编号 | 窗口数 | 背景 | 无人机 |
|---|---|---:|---:|---:|
| 训练集 | S0000、S0001 | 920 | 300 | 620 |
| 验证集 | S0010 | 450 | 160 | 290 |
| 测试集 | S0011 | 440 | 140 | 300 |
| 合计 | 12 个文件 | 1810 | 600 | 1210 |

这种划分比随机打散 1810 个窗口更严格。否则同一原始文件中连续的两个窗口很可能分别进入训练集和测试集，得到偏高且不可复现的结果。

## 5. 信号处理和模型流程

```text
.mat/HDF5
  -> 按索引读取 RF0_I、RF0_Q
  -> IQ = I + jQ
  -> 去复数均值、按 RMS 归一化
  -> 训练时随机加入复 AWGN
  -> STFT (1024 点 Hamming, hop 512)
  -> 复频谱功率取 10*log10
  -> 单样本标准化
  -> 双线性缩放到 1×256×256
  -> ResNet-18
  -> Background / Drone
```

RMS 归一化会减少模型仅凭整体幅度或距离猜类别的可能性，让模型更多学习跳频块、图传带宽、持续时间等时频结构。

## 6. 工程文件

| 文件 | 作用 |
|---|---|
| `prepare_index.py` | 检查 `.mat` 结构并建立虚拟窗口索引 |
| `audit_dataset.py` | 检查边界、重复窗口、划分泄漏和随机读取 |
| `preview_signals.py` | 生成包络、PSD、STFT 和低 SNR 图 |
| `dronerfa_data.py` | HDF5 按需读取、归一化和复 AWGN |
| `model.py` | STFT 前端和标准 ResNet-18 |
| `train_resnet18.py` | 训练、验证、按 SNR 测试和保存结果 |
| `requirements.txt` | 除 PyTorch 外的依赖 |

原始数据保持在：

```text
C:\Users\Administrator\Documents\DroneRFa数据集
```

工程和结果位于：

```text
C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline
```

## 7. 已执行的数据检查

已经生成：

- `outputs/window_index.csv`：1810 个虚拟窗口；
- `outputs/dataset_summary.json`：12 个文件的数组形状和窗口统计；
- `outputs/audit_report.json`：审计通过；
- `outputs/figures/signal_overview.png`：3 类信号的包络、PSD、STFT；
- `outputs/figures/low_snr_stft.png`：Phantom 4 Pro 在不同附加 SNR 下的时频图。
- `outputs/smoke_test`：24 个训练窗口的全链路 GPU 冒烟测试；
- `outputs/pilot_5epochs`：全部 1810 个窗口的 5 轮试验结果和模型。

审计结果：

- 跨 1000 万点边界：0；
- 重复窗口：0；
- 同一源文件跨集合：0；
- 抽查读取失败或非有限值：0。

### 7.1 已完成的 5 轮试验

本次试验使用全部 920 个训练窗口、450 个验证窗口、440 个测试窗口，时频输入为 256×256，批量大小为 4。最佳检查点来自第 2 轮验证结果。

| 附加 SNR | Balanced accuracy | Precision | Recall | Specificity | F1 | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| Clean | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| 0 dB | 0.9929 | 0.9934 | 1.0000 | 0.9857 | 0.9967 | 2 | 0 |
| -10 dB | 0.9879 | 0.9933 | 0.9900 | 0.9857 | 0.9917 | 2 | 3 |
| -12 dB | 0.9781 | 0.9966 | 0.9633 | 0.9929 | 0.9797 | 1 | 11 |
| -15 dB | 0.7898 | 0.9944 | 0.5867 | 0.9929 | 0.7379 | 1 | 124 |

结论：

- 当前子集在 clean 至 -12 dB 下可以稳定区分背景和无人机；
- -15 dB 时主要问题是漏检：300 个无人机窗口中漏掉 124 个；
- 背景误报一直较少，说明性能下降不是模型随机乱猜，而是无人机结构被噪声淹没；
- clean 结果达到 100% 不能解释为“真实环境通用性能”。当前只有同一距离档、两个机型和有限片段，模型也可能利用采集批次或设备状态的稳定特征；
- 下一步必须增加 D01/D10、更多 S 片段或跨日期数据，并至少重复 3 个随机种子。

完整数值位于 `outputs/pilot_5epochs/test_metrics.csv`，曲线位于 `snr_robustness.png` 和 `training_curves.png`。

## 8. 运行方法

### 8.1 环境

项目使用独立虚拟环境，避免当前 Anaconda 中两个 Intel OpenMP 运行库冲突。

```powershell
cd "C:\Users\Administrator\Desktop\数据集处理"
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r .\dronerfa_baseline\requirements.txt
```

### 8.2 重建索引和图

```powershell
cd "C:\Users\Administrator\Desktop\数据集处理\dronerfa_baseline"
..\.venv\Scripts\python.exe .\prepare_index.py
..\.venv\Scripts\python.exe .\audit_dataset.py
..\.venv\Scripts\python.exe .\preview_signals.py
```

### 8.3 最小冒烟测试

这一步只验证 GPU、读取、STFT、反向传播、模型保存和 SNR 评估是否全链路可运行，不用于汇报精度。

```powershell
..\.venv\Scripts\python.exe .\train_resnet18.py `
  --epochs 1 `
  --batch-size 2 `
  --image-size 64 `
  --max-train-samples 24 `
  --max-eval-samples 12 `
  --eval-snrs "clean,-10" `
  --output-dir .\outputs\smoke_test
```

### 8.4 第一轮正式基线

```powershell
..\.venv\Scripts\python.exe .\train_resnet18.py `
  --epochs 20 `
  --batch-size 4 `
  --image-size 256 `
  --output-dir .\outputs\baseline_20epochs
```

8 GB 显存若不足，先把 `--batch-size` 改为 2；不要先缩短信号窗口，因为 DroneRFa 论文表 5 已显示信号长度下降会明显损失性能。

## 9. 如何看结果

每次训练目录包含：

| 输出 | 应如何使用 |
|---|---|
| `run_config.json` | 保存数据量、SNR、参数、GPU 和 PyTorch 版本 |
| `best_model.pt` | 验证集 balanced accuracy 最优模型 |
| `training_history.csv` | 每轮训练/验证指标 |
| `training_curves.png` | 判断是否收敛或过拟合 |
| `test_metrics.csv` | 各附加 SNR 的最终测试指标 |
| `confusion_*.png` | 查看漏检和误报 |
| `snr_robustness.png` | 查看性能随 SNR 的退化曲线 |

判断优先级：

1. 先看测试集 balanced accuracy，避免类别不平衡掩盖问题；
2. 再看 recall，漏检无人机对应 FN；
3. 看 specificity，背景误报对应 FP；
4. 看 clean 到 -12 dB 是否平滑下降；
5. 不要用测试集选择模型或调整阈值。

已经完成的 5 轮试验用于建立第一版趋势，正式结论仍应运行 20 轮和 3 个随机种子。当前只有近距离、两种大疆机型和四个片段，结果只能说明模型在这一子集上的能力，不能外推到所有无人机或真实远距离场景。

## 10. 下一阶段进入条件

满足以下条件后再做机型分类：

- 数据审计持续通过；
- 至少完成 3 个随机种子的重复训练；
- clean 测试集和低 SNR 曲线稳定；
- 混淆矩阵没有退化成总猜“无人机”；
- 增加 D01/D10 或其他日期数据后仍有可接受结果。

之后将二分类头改为 3 类：

```text
Background / Phantom 4 Pro / Air 2S
```

再后面才是时频区域定位和多信号问题。届时输入应改为较长频谱画布，并参考 Basak 等的 YOLO 时频目标检测框架，而不是把多个信号强行压成一个窗口标签。

## 11. 参考文献

1. YU N., MAO S., ZHOU C., et al. DroneRFa: A Large-scale Dataset of Drone Radio Frequency Signals for Detecting Low-altitude Drones. *Journal of Electronics & Information Technology*, 2024, 46(4): 1147-1156. DOI: [10.11999/JEIT230570](https://doi.org/10.11999/JEIT230570)
2. Glüge S., Nyfeler M., Aghaebrahimian A., et al. Robust Low-Cost Drone Detection and Classification Using Convolutional Neural Networks in Low SNR Environments. *IEEE Journal of Radio Frequency Identification*, 2024, 8: 821-830. DOI: [10.1109/JRFID.2024.3487303](https://doi.org/10.1109/JRFID.2024.3487303). [作者复现代码](https://github.com/sgluege/Robust-Drone-Detection-and-Classification)
3. He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition. *CVPR*, 2016: 770-778. [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
4. Basak S., Rajendran S., Pollin S., Scheers B. Combined RF-Based Drone Detection and Classification. *IEEE Transactions on Cognitive Communications and Networking*, 2022, 8(1): 111-120. DOI: [10.1109/TCCN.2021.3099114](https://doi.org/10.1109/TCCN.2021.3099114)
