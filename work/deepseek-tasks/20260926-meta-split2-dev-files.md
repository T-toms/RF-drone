# META-SPLIT2：13 类开发用源文件级划分候选

Codex 已验收 META-HEADER1：官方归档只给 T0000 一个新文件日期簇且同时换频段；T0010/T0101 所有未下载候选都与现有文件同日、时间连续。停止把本归档当作可补齐 13 类跨批次 final test 的来源，不下载这些候选，不再扫描其他十类 293 个头。现有 104 文件仅做**同批次、源文件级开发划分**；不得命名 fresh final test 或 cross-session。

请在项目 `work/meta_split2/` 生成可复算的**候选** 104 行划分 CSV/JSON、核验脚本与方法说明，**不要直接改 G:\DroneRFa数据集\dataset_manifest.csv，不训练、不提取窗口、不下载**。输入固定为 G 正式 104 行清单（SHA256 `9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d`）、`work/meta_split1/source_table_104.csv` 的文件头时间、`work/sel12g/sel12g-mat-sha256-20260925.json`。

划分规则由 Codex 冻结如下：

- 对新增 10 类（每类 8 文件、origin=selected_80），按 MAT 文件头时间升序、同时间按文件名升序，前 5 个 train、第 6 个 val、最后 2 个 `development_test`。这是类内时间阻断的文件级留出，**仍属同一个文件时间簇**，不能称独立采集会话。
- 对旧 3 类 T0000/T0010/T0101，原 `development_original` 的 4 个文件全进 train；原 `evaluated_holdout` 的 4 个按时间升序前 2 个 val、后 2 个 `development_test_exposed`。旧文件已参与此前训练/选型/评价，因此旧三类测试仅描述性，不参与“未用于此前选型”的主指标，也不允许把此前 holdout 写成 fresh final。
- 期望总计 62 train、16 val、26 test（新十类 test 20 文件，旧三类 exposed test 6 文件）。每个源 MAT 恰好出现一次；任何将来生成的 1M 窗与其源文件必须同 split，禁止同一文件跨 split。不要将 `development_test` 改名为 `final_test`。

候选表至少记录 file_name、class_id、class_name、role、prior_use_role、former_partition、origin、file_time_proxy、campaign_proxy、MAT SHA256、G 绝对路径；说明角色与用途，并让机器检查每类计数、文件集合/大小/哈希、源文件零交叉、旧三类训练文件只来自 development_original、旧三类 test/val 历史已暴露、新十类此前未见于模型产物。可做只读检查，不重算 107 GB 全量哈希。给出将来训练/验证/报告的指标边界：13 类整体只能叫开发评估；新十类 file-level test 可单列主指标，但仍是同批次时间阻断；旧三类只作已暴露对照。不根据任何模型准确率更改此规则。

完成后写工作流交接，交给 Codex 验收并决定是否正式冻结该分割工件。若发现输入身份不符或时间字段不可信，停止并报告 needs_codex。
