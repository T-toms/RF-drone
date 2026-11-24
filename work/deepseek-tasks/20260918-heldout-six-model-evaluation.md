# 任务：冻结六个模型的未使用源文件评价

请实际实现并完成评价，不停在启动。用户已要求“文件下载好了，进入下一步”。数据准备已由Codex验收，见work/heldout_prep_20260917/codex-acceptance-20260918.json。你只按下面已定接口实施，遇到重难点报告Codex，不擅自改变冻结方案。

## 输入、保护范围与记录修正

- 真实项目 C:/Users/Administrator/Desktop/数据集处理；复用.venv/Scripts/python.exe。读取work/heldout-eval-config-20260918.json及work/deepseek-plan.md。
- 新数据仍只读 F:/DroneRFa数据集/heldout_20260917，共12文件；索引dronerfa_baseline/outputs/heldout_preparation_20260917/heldout_window_index.csv，1820窗，其SHA256固定在config。保持CSV顺序，用完整全局行号作为噪声索引，不重新排序或限样。
- 原12个MAT、原代码、原索引、checkpoint、旧实验输出全部只读，不训练、不调参、不重选模型/门槛/噪声范围，不新增下载或安装环境。
- 允许新增dronerfa_baseline/evaluate_heldout_models.py及必要汇总/失败用例脚本；新报告/CSV/JSON/图仅位于dronerfa_baseline/outputs/heldout_evaluation_20260918/；过程日志可放work/heldout_eval_20260918/。不得编辑deepseek-context.md、deepseek-workflow.json、deepseek-plan.md和冻结config，禁止提交/推送/发布。
- 先小幅修正数据准备报告中的三项记账表述：download_records.json总体status仍running，实际12项均已完成/已有文件校验通过且进程结束，应标completed并注明完成时间依据；verification.json的scanned_samples_per_channel错误固定140M，应逐文件记录实际140M/150M/160M/180M；报告中161次请求/24次重试/23次429是最后r3运行数据，不称整轮。只修改这些记录及必要summary哈希，保留原文件SHA256、索引SHA256、信号统计和冻结规则。不要重新跑下载/全量扫描/去重/PSD。

## Codex已定的评价设计

1. 使用config列出的六个best_model.pt：2026 A第3轮/B第5轮，2027 A第2轮/B第5轮，2028 A第5轮/B第3轮。启动先核对文件SHA256、selection_policy=multi_noise及epoch；不使用clean_only权重。所有权重只读。
2. 原RF0 I/Q、10ms窗口、去均值/RMS归一化、复AWGN、Hamming STFT1024/hop512、log power/fftshift/样本标准化/256x256、ResNet18原样复用。frontend float32，模型CUDA AMP float16，logits先转float32再作drone-background差，差>0判1，等于0判0。默认batch_size=4与原推理相同；如内存问题只减批量并记录。
3. 条件固定clean/-10/-12/-15，完整1820窗，全局噪声seed=22026。严格复用DroneRFaWindowDataset(index, split='heldout', fixed_snr=..., apply_fixed_snr=..., seed=22026)，epoch保持0，故每窗RNG seed=22026+完整索引行号。绝不可随训练种子或模型变化。所有六模型共享同一组每档float32特征。
4. 优先每档一次生成约455MiB内存缓存，供六模型顺序评价，用完再处理下一档，避免巨型内存或反复读MAT。保存索引/行身份、条件、RNG约定、特征形状/dtype/完整特征SHA256。缓存若确需落盘，大文件仅F:/DroneRFa数据集/heldout_eval_20260918/feature_cache；不写C/D大缓存，按实际权限提示交用户批准，不替用户批准。使用现有CUDA，不改权限或另一Harness。
5. 主口径已定：每条件先计算每个源文件的背景误报率或无人机检出率，再在每类别4个文件内等权平均。Phantom检出率与Air2S检出率各自是4文件等权；无人机检出率是二者等权平均；背景特异度=1-4背景文件等权误报率。文件等权balanced_accuracy=(背景特异度+两机型等权无人机检出率)/2。每模型主综合分是四条件该BA算术平均。另完整保留全部窗口的tp/tn/fp/fn、pooled BA、两机型窗口检出率及背景窗口误报率。不得把主指标和pooled混名。
6. 比较全部三个训练种子的B-A配对差值，逐条件与四档均值均报告，列出三种子的描述性均值和范围；不按结果选最好种子，不把文件内窗口/同窗四条件当独立样本做显著性宣称。
7. Phantom新旧带宽差异保留为观察到的分布差异，当前证据不足以确认协议不匹配，不排除任何文件、不改输入RF1。MAT中心频率/采集会话元数据缺失，报告称“未使用源文件评价”，不宣称跨会话或现场绝对SNR；附加噪声档是模拟AWGN条件。

## 产物与实际验收

- evaluation_manifest.json：输入索引与12MAT已有哈希、六权重路径/hash/epoch、源代码hash、冻结配置、开始/结束时间、进程状态、运行命令、CUDA/AMP、每档共享特征hash、断点策略。输入原14文件哈希对照work/codex-preserved-inputs-20260917.json。
- window_predictions.csv：应43,680条=1820*4*6；保存training_seed,arm,epoch,condition,heldout_index及索引所有定位/类别字段，float32 logits、logit_margin、drone_score,prediction；每组合完整1820，不漏不重，无非有限logits。
- per_file_metrics.csv：应288条=12*4*6，保留文件窗数、tp/tn/fp/fn及检出/误报率。
- model_condition_metrics.csv：24条，明确文件等权主口径与pooled补充口径。
- paired_comparison.csv/summary.json：3种子×4条件及各四档均值的A/B差值；中文report.md清楚列结果、结论与解释限制，图可用于三种子逐条件对照。
- 有意义的必要验证：错误索引hash/权重epoch拒绝；不等长文件的file-equal与pooled差异用例；预测完整性/阈值一致/汇总可从CSV回算。真实IQ经现有接口复核固定少量窗的特征与同一噪声一致性，再从CSV独立抽查每模型各条件固定窗logit差与预测（统一批次和AMP，容差明确记录）。不重复全部数据准备测试或增加无关检查。
- 串行执行。进程长时仍在该桌面任务内跟进；只在43,680条及必要检查真正完成后报告完成。失败保留产物、错误与可恢复点，不隐瞒或跳过某模型/难文件。
- 完成列出实际改动、必要检查、全部结果与任何阻塞，等待Codex验收，不自行重训或转下一阶段。
