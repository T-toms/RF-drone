# INFER-UI1：本地任意IQ信号分析界面

## 目标与冻结设计

按照 `work/arbitrary-signal-ui-spec.md` 和 `work/signal-analyzer-ui-config-20260920.json` 实现可运行、可测试的本地中文浏览器界面。复用已验收CASCADE3六个检查点，不训练、不调阈值。Codex负责最终浏览器验收。

项目虚拟环境没有Streamlit/Gradio/Flask/FastAPI/Plotly/Scipy，禁止为了本任务联网安装UI框架。使用Python标准库HTTP服务器、原生HTML/CSS/JavaScript，计算依赖只使用现有torch/numpy/h5py/matplotlib/Pillow。

## 允许新增或修改

建议新增：

- `dronerfa_baseline/signal_analyzer_core.py`
- `dronerfa_baseline/signal_analyzer_server.py`
- `dronerfa_baseline/launch_signal_analyzer.ps1`
- `dronerfa_baseline/signal_analyzer_web/index.html`
- `dronerfa_baseline/signal_analyzer_web/app.js`
- `dronerfa_baseline/signal_analyzer_web/styles.css`
- `dronerfa_baseline/test_signal_analyzer.py`
- `dronerfa_baseline/verify_signal_analyzer.py`
- `dronerfa_baseline/outputs/signal_analyzer_validation_20260920/`

可小范围复用或导入现有 `model.py`、`dronerfa_data.py`、`run_cascade3_evaluation.py` 的稳定函数，尽量不要复制出不一致的前端。不得修改冻结配置、索引、六权重和正式CASCADE3预测。

## 核心要求

1. 启动时核对 `cascade3-config-20260919.json` SHA256和六个权重哈希；项目 `.venv`/CUDA优先，CPU回退要显著提示速度与数值环境不同。
2. 模型常驻内存。每个窗口只生成一次float32 STFT特征，三套Stage1共享；每个seed仅在自己的Stage1 margin>0时运行对应Stage2。同一seed的Stage2必须用同一特征张量。
3. 三种子以最终三级标签多数决；1-1-1显示uncertain。保存所有每种子中间结果，禁止只保留集成标签。
4. 用户信号不添加噪声。采样率必须由用户确认100MHz；不匹配时允许预览但禁用分类。短于1,000,000点拒绝分类，不补零。
5. MAT v7.3支持默认RF0_I/RF0_Q和自定义键，形状[1,N]或[N]；NPY支持complex一维及[2,N]/[N,2]；NPZ支持iq或i/q键；RAW必须显式参数。HDF5/NPY/RAW流式切片，长文件不整文件载入内存。
6. 支持start sample、stride（默认1,000,000）、max windows/all、batch=4；每窗记录起止样点和相对时间。尾部不足一窗不分类并报告丢弃点数。
7. 本地路径必须在allow roots内且解析真实路径后仍在根内，防止目录穿越；API只监听127.0.0.1。大文件优先路径分析；浏览器上传按块写临时文件，失败/取消清理不完整文件。
8. 单GPU作业队列，进度包含读取窗、特征、Stage1、Stage2、导出；异常返回中文说明且服务器不退出。支持取消等待/运行任务，取消后不发布不完整正式结果。
9. 输出默认 `F:/DroneRFa数据集/analysis_outputs/<timestamp>_<safe_name>/`；测试和smoke必须改到项目scratch，不能污染F盘。产物包含配置/模型哈希、逐窗CSV、analysis JSON、中文summary、时间轴、波形预览和选中窗STFT。
10. 页面清楚标注：只支持当前三类和100MS/s/RF0协议；margin不是概率；未知设备可能被分到已知类；文件级结论为窗口多数描述，不是另一个校准模型。

## 界面要求

- 顶部显示服务、CUDA/CPU、六权重校验状态。
- 输入区支持允许根目录内的文件列表/路径、文件上传和格式参数；先“检查文件”，再“开始分析”。
- 结果区显示主导结论、是否出现无人机窗、三类/不确定窗数量与比例、模型一致率、总耗时。
- 时间轴能区分类别；选择窗口后显示I/Q波形、幅度、STFT和三个seed两阶段分数。
- 逐窗表可以排序/筛选，CSV/JSON/Markdown和图片可下载或打开。
- 页面离线可用，不引用CDN字体、脚本或图片。

## 验证与验收

1. 从现有 `window_predictions.csv` 独立计算7,720个dataset/condition/window的三seed多数决，报告四档heldout和old_test clean集成指标、1-1-1数量；这只是验证固定集成规则，不据此调规则。
2. 选择背景、Phantom、Air2S各至少一个已发布clean窗口，使用真实原始IQ走新核心路径；要求每seed离散决策与CASCADE3一致，数值比较按正式CUDA环境和实际batch构成记录，不用错误解释器静默复验。
3. 为MAT/NPY/NPZ/RAW构造小型夹具，验证格式探测、流式切片、I/Q顺序、起点/stride/尾部、短信号拒绝、错误采样率禁用、路径越界拒绝、上传中断清理、1-1-1不强制分类、文件级汇总规则。
4. 启动真实服务器，使用HTTP客户端完成health→inspect→analysis→poll→download流程；验证并发限制和取消。
5. 至少在三个真实文件上完成有界端到端smoke（每类一个，限定少量窗），产物写项目scratch；不要求全量重推24个大文件。
6. 生成中文使用说明、验证报告、界面截图或可审查的渲染证据。测试不得改CASCADE3正式输出；完成前后核对其预测CSV与六权重哈希。
7. 启动器不得依赖固定旧3080服务，不改防火墙、不绑定0.0.0.0、不自动提交/推送/发布、不恢复任何定时器。

完成后写completed/needs_codex/needs_user交接并停止实现任务。不要自行进入未知类、重采样、重新训练或噪声方式实验。
