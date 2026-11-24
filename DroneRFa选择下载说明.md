# DroneRFa 选择下载说明

ScienceDB网页只提供完整的`DroneRFa.rar`，大小约573.65 GiB。本目录中的
`download_dronerfa_subset.py`利用服务器的HTTP Range功能，按RAR内部文件的
字节区间下载，生成一个可由WinRAR正常打开的小型RAR。

## 默认下载内容

| 编号 | 类别 | 距离 | 片段 |
| --- | --- | --- | --- |
| T0000 | 背景（无无人机） | D00 | S0000、S0001、S0010、S0011 |
| T0010 | DJI Phantom 4 Pro | D00 | S0000、S0001、S0010、S0011 |
| T0101 | DJI Air 2S | D00 | S0000、S0001、S0010、S0011 |

共12个MAT文件，压缩数据约11.9 GB。下载并解压都放在当前磁盘时，建议至少
保留30 GB可用空间。

## 使用方法

在PowerShell中进入本目录：

```powershell
cd "C:\Users\Administrator\Desktop\数据集处理"
```

先检查文件清单和空间需求，不下载数据：

```powershell
python .\download_dronerfa_subset.py --dry-run
```

确认后开始下载、校验和解压：

```powershell
python .\download_dronerfa_subset.py
```

不联网查看本地下载进度：

```powershell
python .\download_dronerfa_subset.py --status
```

脚本在Windows上默认使用系统`curl`传输大文件分块，使用4路并行下载，每完成
16 MiB保存一次断点。运行期间不要让电脑进入睡眠。网络中断或手动按`Ctrl+C`
后，重新运行相同命令即可继续，不会重下已经完成的分段。

下载已经完成但解压中断时，也可以重新运行相同命令。脚本会核对
`.selection.json`，跳过下载并重新执行完整性检查和解压。

## 输出文件

- `dronerfa_manifest.json`：远程RAR目录缓存，约几百KB。
- `DroneRFa_pilot.rar`：只包含选中12个MAT文件的小型RAR。
- `DroneRFa_pilot.rar.selection.json`：已完成压缩包的文件选择记录。
- `DroneRFa_pilot`：解压后的数据目录。
- `DroneRFa_pilot.rar.part`和`.state.json`：未下载完成时的断点文件。

首次运行需要建立远程RAR目录索引，通常需要3至6分钟，只读取约40 MiB头部
信息。后续运行直接使用本地索引。

## 自定义选择

只选择背景和Phantom 4 Pro：

```powershell
python .\download_dronerfa_subset.py --targets T0000,T0010
```

选择更多片段：

```powershell
python .\download_dronerfa_subset.py `
  --segments S0000,S0001,S0010,S0011,S0100,S0101,S0110,S0111
```

只下载并生成RAR，不解压：

```powershell
python .\download_dronerfa_subset.py --no-extract
```

只重新建立目录索引：

```powershell
python .\download_dronerfa_subset.py --index-only --refresh-index
```

不要随意使用`--overwrite`，它会清除当前输出RAR和断点并从头下载。

ScienceDB连接不稳定时可以降为2路；网络稳定且速度仍低时最多可尝试8路：

```powershell
python .\download_dronerfa_subset.py --workers 2
python .\download_dronerfa_subset.py --workers 8
```

## 第一阶段标签

进行“有无无人机”二分类时：

- `T0000`标记为`0`（无无人机）。
- `T0010`和`T0101`标记为`1`（有无人机）。

进行机型分类时，只使用无人机文件：

- `T0010`：DJI Phantom 4 Pro。
- `T0101`：DJI Air 2S。

低SNR实验不需要下载额外文件。应在切分训练集、验证集和测试集之后，仅对训练
样本按目标SNR在线加入AWGN，验证集和测试集分别保留干净版本及固定SNR版本。
