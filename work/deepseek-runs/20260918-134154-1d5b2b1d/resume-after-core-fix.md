用户再次明确“继续”。继续同一EVAL6任务，不创建新会话，不重复原下载或准备核查。

原工具调用已被Codex中断；当前会话idle且无完整预测产物。Codex已接管并修正evaluate_heldout_models.py的核心索引接口：verify_checkpoints(config, training_index_sha256, errors)，训练索引hash来自preserved_inputs原outputs/window_index.csv，不再把heldout hash作为训练hash。已真实核对六权重无errors，原检查点/索引/config均未改。不要覆盖或重复修正这些代码；对当前文件补有意义的拒绝用例即可。上一条排队索引反馈即使随后出现也以本条和现有代码为准。

启动慢的证据：Codex本机torch/CUDA正常识别，F盘第一个908MB MAT完整SHA256耗25.25秒且匹配。你的小规模运行会先verify_frozen_inputs全hash约12GB，长时没有输出不能直接判挂死。请在hash每文件开始/结束明确print(flush=True)，正式执行一次完整输入核验；失败用例用合成/临时小文件或mock，勿每用例再hash12GB。不要再用--max-windows4作为多类别冒烟，索引前4窗只属一个背景文件，file-equal汇总需每类4文件。先跑有意义的单元拒绝/不等长汇总测试，再直接全1820窗四档六模型正式评价，期间逐档/逐批显示进度并跟进到收尾。

请继续原任务文件work/deepseek-tasks/20260918-heldout-six-model-evaluation.md的全部授权与产物，冻结work/heldout-eval-config-20260918.json。先修三项数据准备记录表述，再补评价汇总、预测完整性/阈值/数值检查与少量真实IQ回读。最终须实际43680条预测、288逐文件统计、24模型条件统计、3种子配对差值以及中文报告。原数据、权重、旧代码/索引/实验只读；不训练、不调参、不安装、不额外下载、不扩大阶段。

现在额外授权唯一交接产物 C:/Users/Administrator/Desktop/数据集处理/work/deepseek-handoffs/20260918-134154-1d5b2b1d.json。结束前写UTF-8 JSON：schemaVersion=1，runId='20260918-134154-1d5b2b1d'，sessionId='session-e063be94-0447-4012-b6f6-bcc8bebced11'，kind=completed/needs_codex/needs_user，summary非空，changedFiles数组，checks数组（真实command/result/evidence），problem，attempts数组，suggestedNextStep，reportedAt。最多24000字符，无凭据。completed只表示交Codex验收，不自行派下一阶段。遇到确切技术难题记录原因和已尝试方案，交Codex；实际批准提示交用户，勿代批准。进度与最终报告均中文。

本项目已设原生两分钟heartbeat负责后续验收，你不编辑deepseek-context.md/workflow.json/plan.md/continuation.json或冻结config，不创建调度器。请实际持续执行至上述完成或真实阻塞，不停在脚本生成或后台启动。
