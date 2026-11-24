# RFUAV 数据处理流程说明

## 1. 原始数据是什么

下载的 `.iq` 文件是射频采集设备保存的复数 IQ 信号，不是图片，也不是普通音频。

数据排列方式通常是：

```text
I, Q, I, Q, I, Q ...
```

每两个 float 组成一个复数点：

```text
x[n] = I[n] + jQ[n]
```

旁边的 XML 文件记录采集参数，例如：

```text
SampleRate: 100 MHz
CenterFrequency: 5.76 GHz / 5.77 GHz
ScaleFactor: 85 或 90
ReferenceSNRLevel: 原始参考 SNR
Drone: 机型名
```

所以第一步是：

```text
读取 XML
读取 IQ
恢复复数信号
除以 ScaleFactor
```

## 2. 为什么要切片

一个 1 秒 `.iq` 文件在 100 MHz 采样率下包含：

```text
100,000,000 个复数点
```

数据量太大，不能直接作为 CNN 输入。

因此需要切成较短片段，例如：

```text
每段 100 ms
步长 50 ms
```

这样一个 1 秒文件可以得到 19 个切片：

```text
0.00~0.10s
0.05~0.15s
0.10~0.20s
...
0.90~1.00s
```

每个切片后续单独加噪、单独生成时频图。

## 3. 为什么要定位信号带

采样率是 100 MHz，所以每个切片覆盖的基带频率范围是：

```text
-50 MHz 到 +50 MHz
```

但无人机信号通常不会占满整个 100 MHz。

例如：

```text
DAUTEL EVO NANO 图传信号约 20 MHz
DJI AVATA2 VTSBW=40 图传信号约 40 MHz
```

也就是说，真正有用的信号只在某一段频带中。

如果不定位信号带，而直接用整个 100 MHz 计算功率，会把大量无信号区域也算进去，导致 SNR 不能代表“有效无人机信号”的质量。

因此方法 B 的核心是：

```text
先找到真正有信号的频带
再基于该频带估计 SNR 和加噪
```

## 4. 信号带怎么找

当前使用的方法是：

```text
对切片计算平均频谱 PSD
用固定宽度窗口在频率轴上滑动
寻找平均能量最大的一段
```

例如 DAUTEL 设定信号宽度为 20 MHz：

```text
使用一个 20 MHz 宽的窗口
从 -45 MHz 滑到 +25 MHz
寻找能量最高的位置
```

AVATA2 VTSBW=40 则使用 40 MHz 窗口。

同时需要限制：

```text
不要选择最边缘 ±50 MHz 附近
```

因为接收机边缘可能存在滤波滚降或假峰。

现在采用的是：

```text
slice-aware signal band
```

也就是每个 100 ms 切片独立定位信号带。

原因是 DAUTEL 的信号带会随时间变化，不能简单用一个固定频带代表整个 1 秒文件。

## 5. 噪声带怎么找

确定信号带之后，还要找到一段“没有信号”的频带作为噪声参考。

例如信号带是：

```text
-30 MHz 到 +10 MHz
```

噪声带不能紧贴信号带，而应满足：

```text
距离信号带至少 2 MHz
不能靠近 ±50 MHz 接收机边缘
选择平均能量最低的一段 1 MHz 频带
```

每个切片都会记录：

```text
signal_low_hz
signal_high_hz
noise_low_hz
noise_high_hz
```

这些信息是方法 B 中非常重要的元数据。

## 6. 原始 SNR 怎么估计

有了信号带和噪声带后，计算：

```text
P_signal+noise = 信号带内平均功率
P_noise = 噪声带内平均功率
P_signal = P_signal+noise - P_noise
```

然后：

```text
SNR = 10log10(P_signal / P_noise)
```

这比直接使用整段 IQ 平均功率更接近“有效信号 SNR”。

注意：

```text
XML 中的 ReferenceSNRLevel 是整包参考值
当前估计的是每个 100 ms 切片、指定信号带和噪声带下的有效 SNR
```

因此二者不一定相同。

## 7. 怎么加噪

目标是把原始切片加噪到：

```text
0, -5, -10, -15, -20 dB
```

添加的是复高斯白噪声：

```text
noise = a * (randn + j * randn)
noisy_signal = original_signal + noise
```

关键是确定噪声系数 `a`。

当前流程是：

```text
估计原始信号带 SNR
计算达到目标 SNR 所需的噪声强度
添加复高斯白噪声
重新用同一信号带和噪声带估计 SNR
必要时微调噪声强度
```

每个样本都记录：

```text
target_snr_db
measured_snr_db
```

其中：

```text
target_snr_db   = 目标 SNR
measured_snr_db = 加噪后重新估计得到的 SNR
```

## 8. 为什么加噪难

加噪的难点主要有四个。

### 8.1 信号不是一直存在

无人机信号可能包含：

```text
跳频短脉冲
图传宽带信号
二者混合信号
```

有些信号只在某些时刻出现，因此局部很强，但整段平均可能较弱。

### 8.2 信号带可能变化

尤其是 DAUTEL，在整个 1 秒内，最强的 20 MHz 频带并不是始终固定的。

### 8.3 噪声带不好选

如果噪声带选到接收机边缘，噪声可能被低估，导致 SNR 虚高。

### 8.4 低 SNR 下估计不稳定

在 `-20 dB` 附近：

```text
P_signal+noise - P_noise
```

可能非常小，甚至由于随机波动变成负数。

因此低 SNR 下的 `measured_snr_db` 更难精确。

## 9. 怎么生成时频图

加噪后的 IQ 切片仍然是一维复数信号，不能直接输入 CNN。

需要做 STFT：

```text
Short-Time Fourier Transform
短时傅里叶变换
```

STFT 会把一维信号转换为二维图像：

```text
横轴：时间
纵轴：频率
颜色：能量强弱
```

当前研究三个 NFFT：

```text
512
1024
2048
```

它们代表不同的时频分辨率：

```text
NFFT 小：时间分辨率较好，频率分辨率较差
NFFT 大：频率分辨率较好，时间分辨率较差
```

你的研究问题就是：

```text
不同 SNR 和不同 NFFT/切片长度对 CNN 分类准确率有什么影响
```

## 10. CSV 应该记录什么

每张图都必须能追溯来源。

建议 `labels.csv` 至少记录：

```text
filename
class_name
source_file
slice_index
slice_start_s
target_snr_db
measured_snr_db
nfft
signal_low_hz
signal_high_hz
noise_low_hz
noise_high_hz
estimated_raw_snr_db
snr_method
status
```

这样后续如果模型结果异常，可以回头检查：

```text
是否某个切片信号带选错
是否某个 -20 dB 样本 SNR 没有对齐
是否某个机型原始 SNR 太低
```

## 11. 当前最大的难点

当前最大的难点不是 CNN，而是数据定义。

核心问题是：

```text
什么叫 -10 dB？
```

### 方法 A：整段平均 SNR

```text
整个 IQ 切片的平均 SNR = -10 dB
```

这种方法简单稳定，但对于跳频信号，强脉冲在低 SNR 下仍可能很清楚。

### 方法 B：有效信号 SNR

```text
有效信号带里的 SNR = -10 dB
```

这种方法更接近论文，也更适合研究低信噪比下有效信号被噪声淹没的情况。

你现在选择的是方法 B。

因此最难的是：

```text
如何为每个切片稳定、合理地找到有效信号带和噪声带
```

## 12. 当前已经完成的内容

目前已经完成一个小规模方法 B 数据集：

```text
DAUTEL EVO NANO: 1 秒数据
DJI AVATA2 VTSBW=40: 1 秒数据

100 ms 切片
50 ms 步长
5 个 SNR
3 个 NFFT
共 570 张图
```

生成脚本：

```text
generate_signal_aware_dataset.py
```

小数据集作用是验证流程：

```text
信号带能否自动定位
噪声带是否合理
加噪目标是否准确
时频图是否符合预期
```

## 13. 一句话总结

完整数据处理流程是：

```text
IQ 原始数据
→ 按 100 ms 切片
→ 每个切片定位有效信号带
→ 找噪声参考带
→ 估计原始有效 SNR
→ 加噪到目标 SNR
→ 生成不同 NFFT 的时频图
→ 用 CSV 记录所有元信息
→ 训练 CNN 并分析准确率
```

当前最核心的难点是：

```text
信号带和噪声带是否选得合理
```

因为它直接决定了：

```text
-20 dB 到 0 dB 是否是真正意义上的有效信号 SNR
```
