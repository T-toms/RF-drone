# SRTP中期答辩PPT：R2内容重写与验收报告

## 当前交付

主交付物：`../中期答辩_模板稿.pptx`，24页。最终预览：`../ppt_preview/r2_01.png` 至 `r2_24.png`（1920×1080），总览 `contact_r2.png`。本报告覆盖旧版REPORT，以R2任务书逐页内容为准。

已完成：九个问题的因果链、统一的发现/证实/依据/解决/结果结构、文献原图与出处、图旁解释、每页备注、阶段成果与局限。仅保留第6/9/20页三张过渡页。未安装包、未导出PDF、未调用DSH、未发送heartbeat。

## 每页版式与用图

| 页 | 版式与内容组织 | 配图 |
|---|---|---|
| 1 | 学校模板封面 | 无，使用学校模板或可编辑图形/表格 |
| 2 | 学校模板目录 | 无，使用学校模板或可编辑图形/表格 |
| 3 | 任务与难点双栏 + 主线结论 | 无，使用学校模板或可编辑图形/表格 |
| 4 | 上方文献对标表，下方两组原文/原图 | r2_assets/soltani97.png、soltani49.png、wisig_fig10.png |
| 5 | 公开数据与实测数据两个底框，9节点蛇形链 + 进行中节点 | 无，使用学校模板或可编辑图形/表格 |
| 6 | 模板章节页贰 | 无，使用学校模板或可编辑图形/表格 |
| 7 | 左侧五段论证，右上文献表、右下本项目时频图 | r2_assets/drone_table4.png、m100_window.png |
| 8 | 左侧四段论证，右侧平移单元格与信道前后时频图 | r2_assets/move_focus.png、channel_before.png、channel_after.png |
| 9 | 模板章节页叁 | 无，使用学校模板或可编辑图形/表格 |
| 10 | 四行审核表 + 结论与规模 | 无，使用学校模板或可编辑图形/表格 |
| 11 | 左五段论证，右侧谱线位置 + 同陷波表说明 | r2_assets/line2426.png |
| 12 | 左五段论证，右侧底噪探针柱图 + 关键反例 | r2_assets/source_probe.png |
| 13 | 左四段论证，右侧背景替换两子图 + 关键数字 | background.png（图B上两子图） |
| 14 | 左四段论证，右侧可编辑镜像示意 + 虚警子图 | r2_assets/bgA_fa.png + 可编辑镜像示意 |
| 15 | 左依据/做法/结果，右侧原文公式 + J/La/Lb图 | r2_assets/gluge_mix3.png、wifi_focus.png |
| 16 | 两幅折线图 + 设计、取舍与统计限制 | r2_assets/window_miss.png、window_recall.png |
| 17 | 左文献表与设计/检验说明，右网络原图局部与特征箱图 | r2_assets/openrf_concat.png、mirror_feature.png |
| 18 | 左四段论证，右融合结果图；箭头与大数字强调下降 | fusion_focus.png |
| 19 | 左侧发现/依据/做法/遗留问题，右正常/合并反例 | normal.png、merged.png |
| 20 | 模板章节页肆 | 无，使用学校模板或可编辑图形/表格 |
| 21 | 左首轮→当前指标表，右方法与工作积累 | 无，使用学校模板或可编辑图形/表格 |
| 22 | 四项“文献→我们→证据”卡片 | 无，使用学校模板或可编辑图形/表格 |
| 23 | 左四项不足，右三项行动，计划一页带过 | 无，使用学校模板或可编辑图形/表格 |
| 24 | 学校模板尾页 | 无，使用学校模板或可编辑图形/表格 |

## 文献原图：文件、页码和裁切

全部PDF使用pdftoppm渲染，PIL只打开PNG并裁切，未用Python打开PDF。采用200 dpi渲染；先查看整页，再查看裁切结果。坐标为渲染PNG像素，左/上/右/下。

| PPT页 | 原始PDF | 页/图/式 | PNG与裁切范围 | 用途与校正 |
|---|---|---|---|---|
| 4 | Desktop/文献/05_TVT2020_RF_Fingerprinting_Nonstandard_UAV_Waveforms.pdf | PDF p.4–5, Sec.III-D | soltani200-04.png，(858,1775,1575,2075)；soltani200-05.png，(133,150,841,400) | 97%和49%实际在正文，不在同一结果图表。采用原文证据截图，明确写Sec.III-D，不冒称图号。 |
| 4 | C:/tmp/lit_wisig/wisig.pdf | p.6, Fig.10 | wisig200-06.png，(870,145,1555,515) | 同接收机/跨接收机曲线随训练接收机数变化，证明需留出评测。 |
| 7 | Desktop/文献/DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf | PDF p.7，刊页1153，表4局部 | drone200-07.png，(133,218,796,726) | 表4提供周期统计框架。M100行没有填图传周期，因此未声称14 ms来自表4；14 ms明确为本项目测量、由R2第7页提供。 |
| 15 | Desktop/文献/02_Robust_Low-Cost_Drone_Detection_Low_SNR_2024.pdf | p.3，式(3) | gluge200-03.png，(858,1663,1583,1858) | 先功率归一化后按SNR混合，选最关键混合公式放大，未把原式改写成我们的新方法。 |
| 17 | Desktop/文献/02_JSAC2026_Multi_Domain_Contrastive_RF_Open_Set.pdf | p.5，Fig.2局部 | openrf200-05.png，(491,145,1301,458) | 高亮C拼接位置。原图C表示拼接，⊕是另一运算，未按任务书的近似说法错标。Gao等、JSAC 2026稿按给定稿件标注，不将稿件声称为已出版。 |

WiSig下载来源：[arXiv:2112.15363](https://arxiv.org/pdf/2112.15363)，独立目录C:/tmp/lit_wisig。Geirhos下载来源：[arXiv:2004.07780](https://arxiv.org/pdf/2004.07780)，独立目录C:/tmp/lit_geirhos；首次下载超时后curl续传完成，pdftoppm渲染第2页，未采用图片，论证依据仍保留在第8/12/13页正文。

## 本项目图：原路径与处理

| PPT页 | 原路径 | 处理 |
|---|---|---|
| 7 | reports/数据筛选_20261013/模型输入时频图示例_15ms.png | 裁取Matrice 100的2.4 GHz子图(0,865,610,1060)，保留0–15 ms刻度。已查看T0100_D00_S0000长图确认机型；T0011实际为Matrice 200，未误用。 |
| 8 | reports/结果图表/06_验证实验_挪动和遮挡信号.png | 第一行前三格(250,48,873,154)，100%/4%/0%原值不变；以PPT标签说明三种条件。 |
| 8 | reports/结果图表/07_时频图_Matrice300各文件片段.png | 裁取S0110换信道前/后，(0,190,495,370)、(0,375,495,554)，在图旁说明2456→2409 MHz。 |
| 11 | work/prep_lines_20261104/lines_data_2.4GHz_page10.png | 第13子图，(0,3864,1820,4186)，红线2426.88 MHz位置加橙框。该PSD是聚合图，稳定性证据来自出现率检验，未把被宽带覆盖的峰伪装成突出尖峰。 |
| 12 | R2任务书第12页给定0%/93.2%/0.9% | 重绘三个底噪探针比例，0.51为同来源平衡准确率，在正文解释，未混入比例轴比较。 |
| 13 | reports/中期检查_20261117/figs/图B_背景替换对照.png | 复用background.png，上面两张2.4 GHz子图，裁取(0,105,1800,682)，图旁解释两张图分别看什么。 |
| 14 | figs/图C_加入接收机A背景前后.png | 只保留虚警左子图(0,65,1285,960)，镜像原理为可编辑形状，频段来自R2第14页。 |
| 15 | work/side_L_wifimix_20261110/metrics_paired.csv | 重绘C1合并虚警与v3_drone_recall；保留J/La/Lb和标准差，去除无关类别。 |
| 16 | work/main_W_window_20261110/metrics_summary.csv | argmax/main；nonG_miss与v3_drone_recall，mean/std乘100；各点N来自同L/同band的v3_test_FA行。50t横轴标50，原数据不改。 |
| 17 | work/fusionB_20261110/B1b/图17_接收机A与B镜像背景对比.png | 裁取谱平坦度箱图(1810,590,2470,940)，配合AUC来源检验文字。 |
| 18 | work/fusionB_B2_20261112/metrics_summary.csv | 复用fusion_focus.png：2.4 GHz v3_test_FA、5.8 GHz v3_drone_recall，R/Z/F/F5，均值±标准差。橙箭头仅标趋势，不改柱值。 |
| 19 | work/side_Q2_band_detect_20261111/图_k6_g2_data_drone_01.png；图_k6_g2_raw_background_01.png | 正常Q0000裁(785,25,1530,285)，反例Q1026裁(785,550,1530,810)，沿用normal.png/merged.png。右图是城区背景宽带合并反例，未将其频宽冒充正文60–68 MHz那条录音。 |

## 没有放入或需要答辩前核对的内容

- 第7页找到正确Matrice 100长图，但全文件图按每列1 ms最大值聚合，不能从这张PNG可靠圈定一个精确10 ms空窗。采用同机型15 ms输入图说明短突发与间隙，橙框不标成“精确10 ms空窗”；14 ms与29%保留R2给定测量结论。
- 未加入Basak可选定位图，保留其方法及暗室叠加边界的文字依据，优先给正常/失败两张本项目证据图留可读空间。
- Soltani无同一图表直接给出97%→49%，采用实际正文证据。
- Open-RFNet图是时间/频率/纹理分支的拼接先例，不是本项目IQ手工特征网络的直接复现。
- 第18页“不支持把融合定为基线”保留待与导师讨论；第22页保留“据目前检索，答辩前再核对”；第23页保留结题时间待核对。

## 逐页检查与验收

R2共完成四次构建。第1、2轮分别导出24张1920×1080 PNG与总览并逐页打开检查，证据为r2_round1_*.png、r2_round2_*.png及contact_r2_round1/2.png。第3轮按COM实际BoundHeight排段落，修正文字与下一标题相挤；放大公式与WiFi两项图，处理引用行边界。第4轮补关键数字20–24 pt强调。最终再关闭写入器、复开主交付文件并逐页导出r2_*.png，检查修订页与整份总览。

| 检查项 | 结果 |
|---|---|
| 最终复开 | 24页 |
| 备注 | 24/24页有正文讲稿与补充细节 |
| 溢出 | 最终COM BoundHeight/BoundWidth检查为0，实际段高用于排布 |
| 重叠、边距与校徽 | 逐页图检，内容不压页眉/页脚/校徽；图文分栏或上下结构 |
| 结构与逻辑 | 1–9问题链对应页7/8/10/11/12/13/14/15/17–18；每个结果引出后续问题 |
| 小标题 | 内容稿有对应项才放，未硬凑；第10页按指定审核表例外 |
| 图片与说明 | 所有图等比缩放，图旁给出读图重点与推论，不以图片代替说明 |
| 文献出处 | 已按PDF实物核对图号、页码；纠正Soltani无对应图表、DroneRFa表4不含M100周期、Open-RFNet C拼接符号 |
| 字号 | 问题页叙述正文12–12.6 pt，小标题13.5 pt；图注/来源10 pt；主要数字20–24 pt；其余学校模板字体保留 |
| 颜色 | 蓝色方法/已完成；橙色问题；深蓝结果；灰色计划 |
| 动画/切换 | 主动画序列清除，EntryEffect=0 |
| 禁止修改项 | 模板原件、中期检查表、figs、work原有文件均只读，未改写 |
| PDF/包/DSH/heartbeat | 未导出PDF、未安装包、未调用DSH、未发送heartbeat |
| 残留词：请在此 | 0 |
| 残留词：这是此模版 | 0 |
| 残留词：单击添加 | 0 |
| 残留词：关键词 | 0 |
| 残留词：学校肇始 | 0 |
| 残留词：芃苇 | 0 |
| 残留词：PengV | 0 |
| 残留词：Page | 0 |

## 字数与最小字号（COM实测）

“叙述正文”不含小标题、图注、关键数字重复强调、参考文献对标表。第10页按指定审核表统计表内正文241字；第17页叙述正文218字，另保留R2明确要求的文献对标表，其单元格计入页面总字数。这样保留了R2的论证密度和文献表，没有为压低统计数而删除依据。总字数为去空白可编辑文本字符，不包含位图文字。

| 页 | 页面总字符 | 叙述正文字符 | 最小字号 pt（含图注/页码） |
|---|---:|---:|---:|
| 1 | 54 | — | 12.0 |
| 2 | 56 | — | 25.0 |
| 3 | 201 | — | 10.0 |
| 4 | 422 | — | 10.0 |
| 5 | 300 | — | 10.0 |
| 6 | 31 | — | 14.0 |
| 7 | 408 | 243 | 10.0 |
| 8 | 434 | 258 | 10.0 |
| 9 | 56 | — | 14.0 |
| 10 | 382 | 241 | 10.0 |
| 11 | 447 | 235 | 10.0 |
| 12 | 363 | 231 | 10.0 |
| 13 | 363 | 242 | 10.0 |
| 14 | 389 | 230 | 10.0 |
| 15 | 415 | 259 | 10.0 |
| 16 | 337 | — | 10.0 |
| 17 | 511 | 218 | 10.0 |
| 18 | 396 | 254 | 10.0 |
| 19 | 407 | 259 | 10.0 |
| 20 | 16 | — | 30.0 |
| 21 | 335 | — | 10.0 |
| 22 | 306 | — | 10.0 |
| 23 | 189 | — | 10.0 |
| 24 | 29 | — | 12.0 |

## 可编辑文字的逐数字出处

包含正文、表格、图注、关键数字、备注中的不同数字词元。实验事实均来自R2同页；文件编号、图号、页码为结构定位数，按实际PDF/图表标注。参考文献年份/图号见文献表，不能当作新实验结果。

| PPT页 | 数字词元 | R2出处页及行范围 |
|---|---|---|
| 1 | 1 | R2第1页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:75–78；图号/出处页另按原PDF核对 |
| 2 | 2 | R2第2页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:79–85；图号/出处页另按原PDF核对 |
| 3 | 0 | R2第3页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:86–99；图号/出处页另按原PDF核对 |
| 3 | 1% | R2第3页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:86–99；图号/出处页另按原PDF核对 |
| 3 | 3 | R2第3页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:86–99；图号/出处页另按原PDF核对 |
| 3 | 95% | R2第3页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:86–99；图号/出处页另按原PDF核对 |
| 3 | −10 | R2第3页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:86–99；图号/出处页另按原PDF核对 |
| 3 | −15 | R2第3页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:86–99；图号/出处页另按原PDF核对 |
| 3 | −5 | R2第3页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:86–99；图号/出处页另按原PDF核对 |
| 4 | 1 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 10 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 12 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 13 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 18 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 19 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 2 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 20 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 2020 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 2022 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 2024 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 21 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 3 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 33% | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 34 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 4 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 49% | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 5 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 50 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 6 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 80 | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 97% | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 4 | 99% | R2第4页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:100–117；图号/出处页另按原PDF核对 |
| 5 | 0 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 0% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 1 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 1.9% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 10 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 100 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 100% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 137 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 14 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 2 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 2.4 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 2024 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 24% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 29% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 3 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 300 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 34.8% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 4 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 4% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 5 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 5.8 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 57% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 6 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 7 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 8 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 9 | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 90% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 5 | 93% | R2第5页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:118–137；图号/出处页另按原PDF核对 |
| 6 | 6 | R2第6页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:138–142；图号/出处页另按原PDF核对 |
| 7 | 1 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 10 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 100 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 1153 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 12 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 14 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 15 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 18 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 2 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 20 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 2024 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 29% | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 300 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 34.8% | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 4 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 4.3 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 7 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 71.0% | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 85.4% | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 7 | 9 | R2第7页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:143–154；图号/出处页另按原PDF核对 |
| 8 | 0 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 0% | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 0110 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 06 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 1 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 100% | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 134 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 2 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 20 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 2020 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 2400 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 2409 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 2425 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 2456 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 3 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 300 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 31 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 34% | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 34.8% | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 4 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 4% | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 40 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 5.8 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 59% | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 70 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 8 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 9 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 9% | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 90 | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 8 | 96% | R2第8页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:155–167；图号/出处页另按原PDF核对 |
| 9 | 9 | R2第9页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:168–172；图号/出处页另按原PDF核对 |
| 10 | 0 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 00 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 000 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 1 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 1.56 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 10 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 112 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 120 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 122.88 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 137 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 144 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 15 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 1658 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 17 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 173 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 18 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 19 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 20 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 2020 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 25 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 3 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 37 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 3933 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 4 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 40 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 489 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 49% | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 5 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 5.8 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 50 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 6 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 61 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 73 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 890 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 9 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 97 | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 10 | 97% | R2第10页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:173–187；图号/出处页另按原PDF核对 |
| 11 | 10 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 11 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 13 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 15 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 18 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 2.4 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 2400 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 2400.00 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 2426.88 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 2440 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 2450 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 2457.60 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 32 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 4 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 47 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 5 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 5.8 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 5740 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 5740.00 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 5795.5 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 5800 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 5800.00 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 7 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 9 | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 90% | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 11 | 96% | R2第11页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:188–202；图号/出处页另按原PDF核对 |
| 12 | 0% | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 0.51 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 0.9% | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 1 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 100% | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 12 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 13.8 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 2 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 2.4 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 2020 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 3 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 5 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 5.8 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 6 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 8.8 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 89% | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 9 | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 93% | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 12 | 93.2% | R2第12页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:203–219；图号/出处页另按原PDF核对 |
| 13 | 0 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 0.4% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 0.6% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 100% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 13 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 18 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 2.4 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 20 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 2020 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 3 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 5.8 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 57% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 58% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 6 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 79% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 88% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 9 | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 90% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 92% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 13 | 98% | R2第13页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:220–234；图号/出处页另按原PDF核对 |
| 14 | 0 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 0% | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 0.23% | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 0.3 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 0.42% | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 1 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 1.5% | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 100% | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 1287 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 14 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 19 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 2 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 2342.6 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 2393.5 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 2405 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 2455 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 3 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 57 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 58% | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 6 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 61 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 62% | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 7 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 9 | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 14 | 95% | R2第14页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:235–252；图号/出处页另按原PDF核对 |
| 15 | 0% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 0.807 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 0.986 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 1.9% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 15 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 2 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 2.4 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 20% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 2024 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 21% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 24% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 3 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 5% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 53% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 62% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 7.3% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 77% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 8 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 81% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 88% | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | 9 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 15 | −5 | R2第15页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:253–271；图号/出处页另按原PDF核对 |
| 16 | 0 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 0.04 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 0.20 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 0.24% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 1.28% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 1.65% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 1.92% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 1.96% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 10 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 16 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 180 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 2.4 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 20 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 3 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 30 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 40 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 5.70% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 5.8 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 50 | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 7.22% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 81.8% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 85.9% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 86.0% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 92.4% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 93.1% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 16 | 95% | R2第16页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:272–284；图号/出处页另按原PDF核对 |
| 17 | 0 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 0.24 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 0.34 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 0.50 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 0.934 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 0.994 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 0.9958 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 10 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 12 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 17 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 18 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 19 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 2 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 2020 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 2021 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 2026 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 3 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 5 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 512 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 64 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 17 | 9 | R2第17页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:285–306；图号/出处页另按原PDF核对 |
| 18 | 0.4 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 1 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 1.55% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 10 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 18 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 19% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 2 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 2.4 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 2.5% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 21% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 24 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 3 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 5 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 5.5 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 5.8 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 52% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 85.4 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 86.2% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 89.2% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 89.5% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 9 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 94.9 | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 18 | 94.9% | R2第18页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:307–321；图号/出处页另按原PDF核对 |
| 19 | -2731 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 0000 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 01 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 1026 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 19 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 2 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 2.4 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 2022 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 21 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 2405 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 2455 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 2460 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 2475 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 3 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 30 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 4 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 5.8 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 6 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 60 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 68 | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 70% | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 80% | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 19 | 99% | R2第19页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:322–338；图号/出处页另按原PDF核对 |
| 20 | 20 | R2第20页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:339–342；图号/出处页另按原PDF核对 |
| 21 | 0% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 0.4% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 0.6% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 1.3% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 100 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 100% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 11 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 2 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 2.4 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 20 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 21 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 26 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 3 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 360 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 4.2% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 5.1% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 5.8 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 58% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 6.3% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 60 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 80 | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 85.4% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 85.7% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 86% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 21 | 95% | R2第21页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:343–360；图号/出处页另按原PDF核对 |
| 22 | 0% | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 0.934 | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 1 | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 1.9% | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 100% | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 2 | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 2.4 | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 22 | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 24% | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 3 | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 4 | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 5.8 | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 57% | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 22 | 90% | R2第22页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:361–381；图号/出处页另按原PDF核对 |
| 23 | 0 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 1 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 1% | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 2 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 2.4 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 23 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 2461 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 3 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 3.7% | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 4 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 5.8 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 80 | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 23 | 95% | R2第23页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:382–394；图号/出处页另按原PDF核对 |
| 24 | 24 | R2第24页，work\codex_tasks\PPT_TASK_R2_内容重写_逐页内容稿_20261117.md:395–409；图号/出处页另按原PDF核对 |

## 重绘图的CSV行与字段

行号包含表头，比例乘100显示。误差棒为3种子标准差，不是置信区间。C1先在每个种子合并七档k/n，再求三种子均值与样本标准差。原图裁切数字仍对应原图原单元格/坐标刻度，不另造数据。

| PPT页/图 | CSV文件:行 | 字段或计算 | 原始值/分子分母 |
|---|---|---|---|
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:444 | mean/std；N来自行447的v3_test_FA | L=10, band=2.4GHz, mean=0.02561592429433839, std=0.015168499973733351 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:2 | mean/std；N来自行5的v3_test_FA | L=20, band=2.4GHz, mean=0.05698529411764706, std=0.015297166339240107 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:886 | mean/std；N来自行889的v3_test_FA | L=30, band=2.4GHz, mean=0.01955555555555556, std=0.006012333003763858 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:1318 | mean/std；N来自行1321的v3_test_FA | L=40, band=2.4GHz, mean=0.0510992275698158, std=0.05934338786523523 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:1756 | mean/std；N来自行1759的v3_test_FA | L=50t, band=2.4GHz, mean=0.058491895701198025, std=0.028703892677119525 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:678 | mean/std；N来自行681的v3_test_FA | L=10, band=5.8GHz, mean=0.019159456118665017, std=0.007127665386075893 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:194 | mean/std；N来自行197的v3_test_FA | L=20, band=5.8GHz, mean=0.012755102040816327, std=0.002209248479041935 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:1110 | mean/std；N来自行1113的v3_test_FA | L=30, band=5.8GHz, mean=0.0024286581663630845, std=0.004206559338357929 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:1548 | mean/std；N来自行1551的v3_test_FA | L=40, band=5.8GHz, mean=0.0, std=0.0 |
| 16/window_miss.png | work\main_W_window_20261110\metrics_summary.csv:1988 | mean/std；N来自行1991的v3_test_FA | L=50t, band=5.8GHz, mean=0.0009578544061302681, std=0.0016590524976713384 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:446 | mean/std；N来自行447的v3_test_FA | L=10, band=2.4GHz, mean=0.8421052631578947, std=0.10257908389058333 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:4 | mean/std；N来自行5的v3_test_FA | L=20, band=2.4GHz, mean=0.8603174603174603, std=0.08536082323650003 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:888 | mean/std；N来自行889的v3_test_FA | L=30, band=2.4GHz, mean=0.9240506329113924, std=0.012658227848101278 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:1320 | mean/std；N来自行1321的v3_test_FA | L=40, band=2.4GHz, mean=0.9038461538461539, std=0.05087983290508829 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:1758 | mean/std；N来自行1759的v3_test_FA | L=50t, band=2.4GHz, mean=0.9294871794871794, std=0.09088106973562711 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:680 | mean/std；N来自行681的v3_test_FA | L=10, band=5.8GHz, mean=0.8183990442054959, std=0.1300899172168812 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:196 | mean/std；N来自行197的v3_test_FA | L=20, band=5.8GHz, mean=0.8589743589743589, std=0.03794525278704805 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:1112 | mean/std；N来自行1113的v3_test_FA | L=30, band=5.8GHz, mean=0.9311594202898551, std=0.025102185616940234 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:1550 | mean/std；N来自行1551的v3_test_FA | L=40, band=5.8GHz, mean=0.9206349206349206, std=0.027492869961410718 |
| 16/window_recall.png | work\main_W_window_20261110\metrics_summary.csv:1990 | mean/std；N来自行1991的v3_test_FA | L=50t, band=5.8GHz, mean=0.9367816091954023, std=0.00995431498602805 |
| 15/J | work\side_L_wifimix_20261110\metrics_paired.csv:43 | main/v3_drone_recall | J_mean=0.6190476190476191; J_std=0.11933299129658734 |
| 15/La | work\side_L_wifimix_20261110\metrics_paired.csv:43 | main/v3_drone_recall | La_mean=0.8761904761904762; La_std=0.0343385835758475 |
| 15/Lb | work\side_L_wifimix_20261110\metrics_paired.csv:43 | main/v3_drone_recall | Lb_mean=0.8063492063492065; Lb_std=0.09539669334169344 |
| 15/J | work\side_L_wifimix_20261110\metrics_paired.csv:362 | C1/-10 | 2026: 22/200; 2027: 17/200; 2028: 7/200 |
| 15/La | work\side_L_wifimix_20261110\metrics_paired.csv:362 | C1/-10 | 2026: 23/200; 2027: 83/200; 2028: 40/200 |
| 15/Lb | work\side_L_wifimix_20261110\metrics_paired.csv:362 | C1/-10 | 2026: 7/200; 2027: 7/200; 2028: 8/200 |
| 15/J | work\side_L_wifimix_20261110\metrics_paired.csv:364 | C1/-5 | 2026: 27/200; 2027: 27/200; 2028: 12/200 |
| 15/La | work\side_L_wifimix_20261110\metrics_paired.csv:364 | C1/-5 | 2026: 18/200; 2027: 124/200; 2028: 65/200 |
| 15/Lb | work\side_L_wifimix_20261110\metrics_paired.csv:364 | C1/-5 | 2026: 7/200; 2027: 7/200; 2028: 8/200 |
| 15/J | work\side_L_wifimix_20261110\metrics_paired.csv:366 | C1/0 | 2026: 32/200; 2027: 32/200; 2028: 11/200 |
| 15/La | work\side_L_wifimix_20261110\metrics_paired.csv:366 | C1/0 | 2026: 23/200; 2027: 152/200; 2028: 72/200 |
| 15/Lb | work\side_L_wifimix_20261110\metrics_paired.csv:366 | C1/0 | 2026: 6/200; 2027: 5/200; 2028: 8/200 |
| 15/J | work\side_L_wifimix_20261110\metrics_paired.csv:368 | C1/5 | 2026: 29/200; 2027: 30/200; 2028: 6/200 |
| 15/La | work\side_L_wifimix_20261110\metrics_paired.csv:368 | C1/5 | 2026: 45/200; 2027: 142/200; 2028: 66/200 |
| 15/Lb | work\side_L_wifimix_20261110\metrics_paired.csv:368 | C1/5 | 2026: 5/200; 2027: 0/200; 2028: 4/200 |
| 15/J | work\side_L_wifimix_20261110\metrics_paired.csv:370 | C1/10 | 2026: 8/200; 2027: 21/200; 2028: 13/200 |
| 15/La | work\side_L_wifimix_20261110\metrics_paired.csv:370 | C1/10 | 2026: 19/200; 2027: 74/200; 2028: 49/200 |
| 15/Lb | work\side_L_wifimix_20261110\metrics_paired.csv:370 | C1/10 | 2026: 3/200; 2027: 0/200; 2028: 3/200 |
| 15/J | work\side_L_wifimix_20261110\metrics_paired.csv:372 | C1/15 | 2026: 1/200; 2027: 8/200; 2028: 0/200 |
| 15/La | work\side_L_wifimix_20261110\metrics_paired.csv:372 | C1/15 | 2026: 3/200; 2027: 5/200; 2028: 5/200 |
| 15/Lb | work\side_L_wifimix_20261110\metrics_paired.csv:372 | C1/15 | 2026: 0/200; 2027: 0/200; 2028: 0/200 |
| 15/J | work\side_L_wifimix_20261110\metrics_paired.csv:374 | C1/20 | 2026: 0/200; 2027: 2/200; 2028: 0/200 |
| 15/La | work\side_L_wifimix_20261110\metrics_paired.csv:374 | C1/20 | 2026: 0/200; 2027: 0/200; 2028: 0/200 |
| 15/Lb | work\side_L_wifimix_20261110\metrics_paired.csv:374 | C1/20 | 2026: 0/200; 2027: 0/200; 2028: 0/200 |
| 18/R | work\fusionB_B2_20261112\metrics_summary.csv:2 | 2.4GHz/v3_test_FA rate_mean/rate_std | 0.041666666666666664 / 0.019094065395649333 |
| 18/Z | work\fusionB_B2_20261112\metrics_summary.csv:528 | 2.4GHz/v3_test_FA rate_mean/rate_std | 0.041666666666666664 / 0.02602082499332666 |
| 18/F | work\fusionB_B2_20261112\metrics_summary.csv:1054 | 2.4GHz/v3_test_FA rate_mean/rate_std | 0.025000000000000005 / 0.012499999999999999 |
| 18/F5 | work\fusionB_B2_20261112\metrics_summary.csv:1580 | 2.4GHz/v3_test_FA rate_mean/rate_std | 0.041666666666666664 / 0.028867513459481284 |
| 18/R | work\fusionB_B2_20261112\metrics_summary.csv:2235 | 5.8GHz/v3_drone_recall rate_mean/rate_std | 0.8538461538461539 / 0.09069097017347384 |
| 18/Z | work\fusionB_B2_20261112\metrics_summary.csv:2589 | 5.8GHz/v3_drone_recall rate_mean/rate_std | 0.9487179487179486 / 0.004441155916843323 |
| 18/F | work\fusionB_B2_20261112\metrics_summary.csv:2943 | 5.8GHz/v3_drone_recall rate_mean/rate_std | 0.8923076923076924 / 0.04070386632407065 |
| 18/F5 | work\fusionB_B2_20261112\metrics_summary.csv:3297 | 5.8GHz/v3_drone_recall rate_mean/rate_std | 0.8615384615384616 / 0.0076923076923077205 |

图像数字定位补充：第4页Soltani两个原文百分比对应III-D两种切分实验；WiSig曲线对应Fig.10坐标。第7页表4单元格保持原样、M100输入例刻度来自原图；第8页100/4/0为原06图第一行前三格；第11/17/19页刻度为所列原图对应轴；第13/14页曲线、柱与误差棒均保留原图位置和值。第12页0/93.2/0.9直接来自R2第12页。

## 模板原件与复现

复现：设置PYTHONIOENCODING=utf-8、KMP_DUPLICATE_LIB_OK=TRUE，使用D:/anaconda/python.exe -I依次运行r2_assets.py、r2_build.py 4、r2_verify.py、r2_report.py。r2_refs保存PDF渲染PNG和辅助文字；r2_assets/manifest.json记录裁切和数据行。PDF没有被Python打开。保存仍由本机WPS的PowerPoint.Application兼容COM完成，嵌入学校字体导致保存约需两分钟，等待其返回后才关闭并复开。
- 原模板大小：78282941字节，要求78282941，一致。
- 原模板修改时间：2026/11/17 19:58:26，要求2026/11/17 19:58:26，一致。
- 原模板SHA256：`4744a643bcbdbc2bbaf0300857a25211d3872c99d4a1f79406f3a295f3533d11`。
- 主交付物大小：65139570字节。
- 最终构建副本：r2_final_25156.pptx；最终审计：r2_final_audit.json。
