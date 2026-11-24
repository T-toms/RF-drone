# CASCADE12-DEV：锁定后的十二机型两阶段开发评价

状态：待正式派发；依赖 Stage1 对照/版本冻结、Stage2 contract 和正式训练验收。只有 Codex 签发模型锁定记录后才能打开测试 IQ。

## 锁定与输入

读取冻结开发划分与Stage2 contract。Stage1 的候选集合固定为原 seed2026、原 seed2027、单次 global_train_mean seed2027；仅按各自已冻结 epoch 的四条件 val pooled BA 均值选最高，同分优先原2026、原2027、最后归一化候选。记录随机种子也参与开发选择的偏差；不按测试结果改选。若候选因技术失败无有效结果，则记录失败后仅在已验收集合选择。Stage2 使用唯一seed2027依其预声明val指标选出的checkpoint。

测试前写不可覆盖 selection_lock.json：两模型/前端/映射/协议/代码/索引哈希、选型证据、Stage1 margin>0、Stage2 argmax、全部噪声档和种子；再次打开测试必须校验锁定身份。不得训练、调门限或筛选有利窗口。

development_test 20文件/2950窗；development_test_exposed 6文件/890窗（背景280，旧两机型610）。各自 clean/-10/-12/-15，noise seed 22026/32026，坐标为完整角色中的 full_split_position。与已验收contract共用去均值/RMS/AWGN/前端。每个窗口条件只产生一份输入/谱图；oracle Stage2 对真实无人机全部执行，cascade Stage2 对Stage1判为Drone的子集执行，共用输入并核对digest，不重新加噪。

## 输出与解释

- cascade 共15360预测；oracle 真实无人机共14240预测。主键(role,row_index,condition)唯一且全覆盖；输出Stage1两维logits/margin、oracle十二维logits、通过gate时的Stage2预测、最终13类标签和输入digest。
- 新十类开发测试的模型输出空间仍是全部12机型，预测为旧两类也算错；支持的10类各自召回、类内文件等权/类等权最终召回、pooled最终准确率、Stage1漏检。该角色无背景，FPR=N/A；不存在的真值类别recall=N/A，不补0参加平均。
- 已暴露角色单独给背景FPR和旧两类漏检/识别率，清楚标注既往暴露。
- 若补充合并13类结果，必须标为混合来源开发评价，不能叫13类全新最终测试、跨环境或跨距离泛化。
- 对每角色/条件分解互斥错误：背景误报；无人机被Stage1漏检；通过Stage1后Stage2错认。三者与最终错误数闭合。另报oracle与已检出无人机中的分类准确率，不能替代端到端结果。
- 四条件是同一批窗口的重复噪声评价，不作为四倍独立采集样本；只做描述性统计，避免窗口独立置信区间。

## 允许范围与验收

新增cascade评价/独立验证/必要负向测试，输出新 `dronerfa_baseline/outputs/cascade12_development_20260927/` 和中文报告。旧产物、原始MAT、冻结划分只读，不发布/改UI/重训。

核验锁定时间早于测试打开、完整样本角色与标签、oracle/cascade输入digest一致、实测推理门控正确、12/13类映射正确、误差分解和所有统计从逐窗结果复算、只读测试审计与模型重载一致。负向篡改映射、门限、噪声坐标、漏窗/重复窗、角色混合须被发现。完成本次评价后交付完整基线并停止扩展，结果好坏均原样报告。
