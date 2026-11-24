[CmdletBinding()]
param(
    [string]$SourceRoot,
    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot
)

$ErrorActionPreference = 'Stop'
$utf8Bom = New-Object System.Text.UTF8Encoding($true)

if ([string]::IsNullOrWhiteSpace($SourceRoot)) { $SourceRoot = Split-Path -Parent $PSScriptRoot }

if (-not [IO.Path]::IsPathRooted($SourceRoot)) { throw 'SourceRoot 必须是绝对路径。' }
if (-not [IO.Path]::IsPathRooted($DestinationRoot)) { throw 'DestinationRoot 必须是绝对路径。' }
$SourceRoot = [IO.Path]::GetFullPath($SourceRoot).TrimEnd('\')
$DestinationRoot = [IO.Path]::GetFullPath($DestinationRoot).TrimEnd('\')
if (-not (Test-Path -LiteralPath $SourceRoot -PathType Container)) { throw "项目目录不存在：$SourceRoot" }
if (Test-Path -LiteralPath $DestinationRoot) { throw "目标目录必须是全新目录，当前已存在：$DestinationRoot" }

$items = New-Object 'System.Collections.Generic.List[object]'
function Add-PackageFile {
    param(
        [string]$SourceRelative,
        [string]$TargetDirectory,
        [string]$TargetName,
        [string]$Category,
        [string]$Description
    )
    $source = Join-Path $SourceRoot $SourceRelative
    $relativeTarget = Join-Path $TargetDirectory $TargetName
    $items.Add([pscustomobject]@{
        Source = $source
        SourceRelative = $SourceRelative
        TargetRelative = $relativeTarget
        Category = $Category
        Description = $Description
    }) | Out-Null
}

# 01 数据准备与迁移
Add-PackageFile 'dronerfa_baseline\outputs\window_index.csv' '01_数据准备与迁移' '数据准备_旧数据训练验证测试窗口索引.csv' '数据准备' '原始数据的冻结窗口索引'
Add-PackageFile 'dronerfa_baseline\outputs\dataset_summary.json' '01_数据准备与迁移' '数据准备_旧数据集统计摘要.json' '数据准备' '原始数据集统计摘要'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\report.md' '01_数据准备与迁移' '数据准备_新留出数据下载检查报告.md' '数据准备' '12个新留出文件的准备与检查报告'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\heldout_window_index.csv' '01_数据准备与迁移' '数据准备_新留出数据窗口索引.csv' '数据准备' '新留出数据冻结窗口索引'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\heldout_dataset_summary.json' '01_数据准备与迁移' '数据准备_新留出数据统计摘要.json' '数据准备' '新留出数据统计摘要'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\duplicate_check.json' '01_数据准备与迁移' '数据准备_新旧数据重复检查.json' '数据准备' '新旧文件与窗口重复检查'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\activity_band_check.json' '01_数据准备与迁移' '数据准备_RF0_RF1活动频带检查.json' '数据准备' 'RF0与RF1活动检查记录'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\activity_band_original.png' '01_数据准备与迁移' '数据准备_旧数据活动频带图.png' '数据准备' '旧数据活动频带可视化'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\activity_band_heldout.png' '01_数据准备与迁移' '数据准备_新留出数据活动频带图.png' '数据准备' '新留出数据活动频带可视化'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\download_records.json' '01_数据准备与迁移' '数据准备_新留出数据下载记录.json' '数据准备' '新留出文件下载记录'
Add-PackageFile 'dronerfa_baseline\outputs\heldout_preparation_20260917\verification.json' '01_数据准备与迁移' '数据准备_新留出数据验证结果.json' '数据准备' '新留出数据验证结果'
Add-PackageFile 'work\dataset-migration-20260920.json' '01_数据准备与迁移' '数据迁移_C盘数据合并到F盘校验证据.json' '数据迁移' 'C盘原始数据移动到F盘并建立兼容链接的证据'

# 02 第一阶段：EVAL6、传统基线与 FFT 比较
$evalFiles = @(
    @('report.md','第一阶段_噪声A与B留出评价报告.md','六个第一阶段模型在四种条件的评价报告'),
    @('evaluation_summary.json','第一阶段_噪声A与B留出评价汇总.json','留出评价汇总'),
    @('model_condition_metrics.csv','第一阶段_噪声A与B各模型各条件指标.csv','模型与条件级指标'),
    @('paired_comparison.csv','第一阶段_噪声A与B成对比较.csv','相同种子成对比较'),
    @('paired_seed_comparison.png','第一阶段_噪声A与B成对比较图.png','成对比较图'),
    @('per_file_metrics.csv','第一阶段_噪声A与B逐文件指标.csv','逐文件指标'),
    @('window_predictions.csv','第一阶段_噪声A与B正式逐窗口预测.csv','正式逐窗口预测'),
    @('evaluation_manifest.json','第一阶段_噪声A与B评价实验清单.json','留出评价实验清单'),
    @('independent_verification.json','第一阶段_噪声A与B独立验证.json','独立验证结果')
)
foreach ($x in $evalFiles) { Add-PackageFile ("dronerfa_baseline\outputs\heldout_evaluation_20260918\" + $x[0]) '02_第一阶段_无人机检测\01_噪声训练A_B对照' $x[1] '第一阶段' $x[2] }

$baselineFiles = @(
    @('report.md','第一阶段_能量阈值与PSD-SVM传统基线报告.md','传统基线报告'),
    @('comparison_metrics.csv','第一阶段_能量阈值与PSD-SVM对比指标.csv','传统基线对比指标'),
    @('experiment_manifest.json','第一阶段_能量阈值与PSD-SVM实验清单.json','传统基线实验清单'),
    @('verification.json','第一阶段_能量阈值与PSD-SVM验证结果.json','传统基线验证结果')
)
foreach ($x in $baselineFiles) { Add-PackageFile ("dronerfa_baseline\outputs\stage1_method_comparison_20260919\baselines\" + $x[0]) '02_第一阶段_无人机检测\02_传统基线' $x[1] '第一阶段' $x[2] }
Add-PackageFile 'dronerfa_baseline\outputs\stage1_method_comparison_20260919\baselines\energy_threshold\energy_distributions.png' '02_第一阶段_无人机检测\02_传统基线' '第一阶段_能量阈值各类功率分布图.png' '第一阶段' '能量分数分布图'
Add-PackageFile 'dronerfa_baseline\outputs\stage1_method_comparison_20260919\baselines\energy_threshold\selection.json' '02_第一阶段_无人机检测\02_传统基线' '第一阶段_能量阈值选择记录.json' '第一阶段' '能量阈值选择记录'
Add-PackageFile 'dronerfa_baseline\outputs\stage1_method_comparison_20260919\baselines\energy_threshold\metrics.csv' '02_第一阶段_无人机检测\02_传统基线' '第一阶段_能量阈值评价指标.csv' '第一阶段' '能量阈值评价指标'
Add-PackageFile 'dronerfa_baseline\outputs\stage1_method_comparison_20260919\baselines\psd_svm\psd_examples.png' '02_第一阶段_无人机检测\02_传统基线' '第一阶段_PSD特征示例图.png' '第一阶段' 'PSD特征示例'
Add-PackageFile 'dronerfa_baseline\outputs\stage1_method_comparison_20260919\baselines\psd_svm\selection.json' '02_第一阶段_无人机检测\02_传统基线' '第一阶段_PSD-SVM模型选择记录.json' '第一阶段' 'PSD与SVM模型选择记录'
Add-PackageFile 'dronerfa_baseline\outputs\stage1_method_comparison_20260919\baselines\psd_svm\metrics.csv' '02_第一阶段_无人机检测\02_传统基线' '第一阶段_PSD-SVM评价指标.csv' '第一阶段' 'PSD与SVM评价指标'
Add-PackageFile 'dronerfa_baseline\outputs\stage1_method_comparison_20260919\baselines\psd_svm\model.joblib' '02_第一阶段_无人机检测\04_正式模型' '第一阶段_PSD-SVM_线性C0.01_正式模型.joblib' '第一阶段模型' 'PSD与SVM正式模型'

$nfftFiles = @(
    @('report.md','第一阶段_NFFT四档比较报告.md','FFT点数比较报告'),
    @('nfft_summary.csv','第一阶段_NFFT四档比较汇总.csv','FFT点数比较汇总'),
    @('model_condition_metrics.csv','第一阶段_NFFT四档各模型各条件指标.csv','模型与条件级指标'),
    @('paired_nfft_comparison.csv','第一阶段_NFFT四档成对比较.csv','成对FFT点数比较'),
    @('per_file_metrics.csv','第一阶段_NFFT四档逐文件指标.csv','逐文件指标'),
    @('heldout_window_predictions.csv','第一阶段_NFFT四档留出集正式逐窗口预测.csv','留出集正式逐窗口预测'),
    @('test_clean_window_predictions.csv','第一阶段_NFFT四档旧测试集正式逐窗口预测.csv','旧测试集正式逐窗口预测'),
    @('model_input_examples.png','第一阶段_NFFT四档模型输入对比图.png','模型输入对比图'),
    @('native_stft_examples.png','第一阶段_NFFT四档原生STFT对比图.png','原生STFT对比图'),
    @('experiment_manifest.json','第一阶段_NFFT四档实验清单.json','FFT点数实验清单'),
    @('verification.json','第一阶段_NFFT四档验证结果.json','FFT点数实验验证结果')
)
foreach ($x in $nfftFiles) { Add-PackageFile ("dronerfa_baseline\outputs\stage1_method_comparison_20260919\nfft\" + $x[0]) '02_第一阶段_无人机检测\03_NFFT四档比较' $x[1] '第一阶段' $x[2] }

# 第一阶段正式模型
$noiseSources = @{
    '2026' = 'dronerfa_baseline\outputs\noise_ab_5epochs_20260917'
    '2027' = 'dronerfa_baseline\outputs\noise_ab_replicates_20260917\seed_2027'
    '2028' = 'dronerfa_baseline\outputs\noise_ab_replicates_20260917\seed_2028'
}
foreach ($seed in @('2026','2027','2028')) {
    Add-PackageFile ($noiseSources[$seed] + '\A_original\best_model.pt') '02_第一阶段_无人机检测\04_正式模型' ("第一阶段_噪声方案A_Seed${seed}_最佳模型.pt") '第一阶段模型' '噪声训练方案A正式模型'
    Add-PackageFile ($noiseSources[$seed] + '\B_plus_minus15\best_model.pt') '02_第一阶段_无人机检测\04_正式模型' ("第一阶段_噪声方案B含负15dB_Seed${seed}_最佳模型.pt") '第一阶段模型' '噪声训练方案B正式模型'
}
foreach ($nfft in @(512,1024,2048,4096)) {
    foreach ($seed in @(2026,2027,2028)) {
        Add-PackageFile ("dronerfa_baseline\outputs\stage1_method_comparison_20260919\nfft\nfft${nfft}\seed_${seed}\best_model.pt") '02_第一阶段_无人机检测\04_正式模型' ("第一阶段_NFFT${nfft}_Seed${seed}_最佳模型.pt") '第一阶段模型' 'FFT点数比较正式最佳模型'
    }
}

# 03 第二阶段
$stage2Files = @(
    @('report.md','第二阶段_Phantom与Air2S机型识别报告.md','机型识别报告'),
    @('nfft_summary.csv','第二阶段_NFFT512与1024比较汇总.csv','FFT点数比较汇总'),
    @('model_condition_metrics.csv','第二阶段_各模型各条件指标.csv','模型与条件级指标'),
    @('paired_nfft_comparison.csv','第二阶段_NFFT成对比较.csv','FFT点数成对比较'),
    @('per_file_metrics.csv','第二阶段_逐文件识别指标.csv','逐文件机型识别指标'),
    @('heldout_window_predictions.csv','第二阶段_留出集正式逐窗口预测.csv','留出集正式逐窗口预测'),
    @('test_clean_window_predictions.csv','第二阶段_旧测试集正式逐窗口预测.csv','旧测试集正式逐窗口预测'),
    @('validation_nfft_selection.json','第二阶段_仅用验证集选择NFFT记录.json','FFT点数选择记录'),
    @('selection_lock.json','第二阶段_模型选择锁定记录.json','模型选择锁定记录'),
    @('representative_phantom.png','第二阶段_Phantom代表性时频图.png','Phantom代表图'),
    @('representative_air2s.png','第二阶段_Air2S代表性时频图.png','Air2S代表图'),
    @('experiment_manifest.json','第二阶段_机型识别实验清单.json','机型识别实验清单'),
    @('verification.json','第二阶段_机型识别验证结果.json','机型识别验证结果')
)
foreach ($x in $stage2Files) { Add-PackageFile ("dronerfa_baseline\outputs\stage2_model_classification_20260919\" + $x[0]) '03_第二阶段_机型识别\01_报告与结果' $x[1] '第二阶段' $x[2] }
foreach ($nfft in @(512,1024)) {
    foreach ($seed in @(2026,2027,2028)) {
        Add-PackageFile ("dronerfa_baseline\outputs\stage2_model_classification_20260919\nfft${nfft}\seed_${seed}\best_model.pt") '03_第二阶段_机型识别\02_正式模型' ("第二阶段_NFFT${nfft}_Seed${seed}_Phantom与Air2S最佳模型.pt") '第二阶段模型' '机型识别正式最佳模型'
    }
}

# 04 串联三分类
$cascadeFiles = @(
    @('report.md','串联系统_背景_Phantom_Air2S三分类报告.md','两阶段串联三分类报告'),
    @('metrics_by_condition.csv','串联系统_各条件三分类指标.csv','各条件汇总指标'),
    @('cross_seed_summary.csv','串联系统_跨Seed结果汇总.csv','跨种子汇总'),
    @('per_file_metrics.csv','串联系统_逐文件三分类指标.csv','逐文件指标'),
    @('error_decomposition.csv','串联系统_错误来源分解.csv','错误分解'),
    @('confusion_matrices.csv','串联系统_混淆矩阵数值.csv','混淆矩阵数值'),
    @('window_predictions.csv','串联系统_正式逐窗口三分类预测.csv','正式逐窗口预测'),
    @('cascade_flow.png','串联系统_两阶段处理流程图.png','串联流程图'),
    @('recall_by_condition.png','串联系统_不同条件各类别召回率图.png','各条件召回率图'),
    @('confusion_heldout_clean.png','串联系统_留出集Clean混淆矩阵图.png','Clean混淆矩阵'),
    @('confusion_heldout_-10.png','串联系统_留出集负10dB混淆矩阵图.png','负10dB混淆矩阵'),
    @('confusion_heldout_-12.png','串联系统_留出集负12dB混淆矩阵图.png','负12dB混淆矩阵'),
    @('confusion_heldout_-15.png','串联系统_留出集负15dB混淆矩阵图.png','负15dB混淆矩阵'),
    @('confusion_old_test_clean.png','串联系统_旧测试集Clean混淆矩阵图.png','旧测试集混淆矩阵'),
    @('experiment_manifest.json','串联系统_三分类实验清单.json','串联实验清单'),
    @('verification.json','串联系统_三分类验证结果.json','串联实验验证结果')
)
foreach ($x in $cascadeFiles) { Add-PackageFile ("dronerfa_baseline\outputs\cascade3_evaluation_20260919\" + $x[0]) '04_两阶段串联_三分类' $x[1] '串联系统' $x[2] }

# 05 任意信号分析界面
$uiFiles = @(
    @('dronerfa_baseline\signal_analyzer_core.py','分析界面_模型推理核心程序.py','推理核心'),
    @('dronerfa_baseline\signal_analyzer_server.py','分析界面_本地HTTP服务器程序.py','本地服务器'),
    @('dronerfa_baseline\signal_analyzer_web\index.html','分析界面_网页主页面.html','网页结构'),
    @('dronerfa_baseline\signal_analyzer_web\app.js','分析界面_网页交互逻辑.js','网页交互'),
    @('dronerfa_baseline\signal_analyzer_web\styles.css','分析界面_网页样式.css','网页样式'),
    @('dronerfa_baseline\launch_signal_analyzer.ps1','分析界面_启动服务器脚本.ps1','服务器启动脚本'),
    @('dronerfa_baseline\open_signal_analyzer.ps1','分析界面_一键打开Chrome脚本.ps1','一键启动脚本'),
    @('dronerfa_baseline\signal_analyzer_README.md','分析界面_使用说明.md','界面使用说明'),
    @('dronerfa_baseline\test_signal_analyzer.py','分析界面_核心功能测试.py','核心测试'),
    @('dronerfa_baseline\test_open_signal_analyzer.py','分析界面_一键启动器测试.py','启动器测试'),
    @('dronerfa_baseline\verify_signal_analyzer.py','分析界面_独立验证程序.py','独立验证程序'),
    @('dronerfa_baseline\outputs\signal_analyzer_validation_20260920\verification_report.md','分析界面_完整验证报告.md','界面完整验证报告'),
    @('dronerfa_baseline\outputs\signal_analyzer_validation_20260920\verification.json','分析界面_完整验证结果.json','界面完整验证结果'),
    @('dronerfa_baseline\outputs\signal_analyzer_validation_20260920\real_iq_parity.json','分析界面_真实IQ推理一致性检查.json','真实IQ一致性检查'),
    @('dronerfa_baseline\outputs\signal_analyzer_validation_20260920\ui_screenshot.png','分析界面_验收截图.png','界面验收截图')
)
foreach ($x in $uiFiles) { Add-PackageFile $x[0] '05_任意信号分析界面' $x[1] '分析界面' $x[2] }

# 06 程序与复现实验材料
$programs = @(
    @('dronerfa_baseline\requirements.txt','复现环境_Python依赖版本.txt','Python依赖'),
    @('dronerfa_baseline\model.py','程序_ResNet18模型定义.py','模型定义'),
    @('dronerfa_baseline\dronerfa_data.py','程序_DroneRFa数据读取与STFT前端.py','数据与前端'),
    @('dronerfa_baseline\prepare_index.py','程序_旧数据窗口索引生成.py','旧数据索引'),
    @('dronerfa_baseline\train_resnet18.py','程序_ResNet18基线训练.py','基础训练'),
    @('dronerfa_baseline\run_noise_ab.py','程序_第一阶段噪声A_B训练实验.py','噪声训练实验'),
    @('dronerfa_baseline\run_noise_replicates.py','程序_第一阶段噪声A_B多Seed复现实验.py','噪声多种子实验'),
    @('dronerfa_baseline\evaluate_heldout_models.py','程序_第一阶段留出集模型评价.py','留出评价'),
    @('dronerfa_baseline\summarize_heldout_evaluation.py','程序_第一阶段留出集结果汇总.py','留出汇总'),
    @('dronerfa_baseline\run_classical_baselines.py','程序_第一阶段能量阈值与PSD-SVM基线.py','传统基线'),
    @('dronerfa_baseline\run_nfft_comparison.py','程序_第一阶段NFFT四档比较.py','NFFT比较'),
    @('dronerfa_baseline\run_stage2_classification.py','程序_第二阶段机型识别训练评价.py','第二阶段'),
    @('dronerfa_baseline\run_cascade3_evaluation.py','程序_两阶段串联三分类评价.py','串联评价'),
    @('dronerfa_baseline\verify_heldout_evaluation.py','验证程序_第一阶段留出评价.py','留出验证'),
    @('dronerfa_baseline\verify_nfft_comparison.py','验证程序_第一阶段NFFT比较.py','NFFT验证'),
    @('dronerfa_baseline\verify_stage2_classification.py','验证程序_第二阶段机型识别.py','第二阶段验证'),
    @('dronerfa_baseline\verify_cascade3_evaluation.py','验证程序_两阶段串联三分类.py','串联验证'),
    @('dronerfa_baseline\test_stage1_method_comparison.py','测试程序_第一阶段传统基线.py','第一阶段测试'),
    @('dronerfa_baseline\test_nfft_comparison.py','测试程序_第一阶段NFFT比较.py','NFFT测试'),
    @('dronerfa_baseline\test_stage2_classification.py','测试程序_第二阶段机型识别.py','第二阶段测试'),
    @('dronerfa_baseline\test_cascade3_evaluation.py','测试程序_两阶段串联三分类.py','串联测试')
)
foreach ($x in $programs) { Add-PackageFile $x[0] '06_程序与复现实验材料' $x[1] '复现实验' $x[2] }

# 07 论文与文档
Add-PackageFile 'dronerfa_baseline\DroneRFa项目四篇核心论文文献综述.docx' '07_参考文献与项目文档' '文献综述_DroneRFa项目四篇核心论文.docx' '文献' '可编辑文献综述'
Add-PackageFile 'dronerfa_baseline\DroneRFa项目四篇核心论文文献综述.md' '07_参考文献与项目文档' '文献综述_DroneRFa项目四篇核心论文.md' '文献' 'Markdown文献综述'
Add-PackageFile 'dronerfa_baseline\outputs\literature_review\docx_render_final\DroneRFa项目四篇核心论文文献综述.pdf' '07_参考文献与项目文档' '文献综述_DroneRFa项目四篇核心论文.pdf' '文献' 'PDF文献综述'
Add-PackageFile 'dronerfa_baseline\DroneRFa第一阶段实施说明.md' '07_参考文献与项目文档' '项目文档_第一阶段早期实施说明.md' '项目文档' '第一阶段早期实施说明'
$paperNames = @{
    '01_DroneRFa_Dataset_JEIT2024.pdf' = '参考论文_01_DroneRFa数据集_JEIT2024.pdf'
    '02_Robust_Low-Cost_Drone_Detection_Low_SNR_2024.pdf' = '参考论文_02_低成本低信噪比无人机检测_2024.pdf'
    '03_Deep_Residual_Learning_ResNet_CVPR2016.pdf' = '参考论文_03_ResNet深度残差学习_CVPR2016.pdf'
    '04_Combined_RF-Based_Drone_Detection_and_Classification_TCCN2022_preprint.pdf' = '参考论文_04_RF无人机检测与分类_TCCN2022预印本.pdf'
}
foreach ($paper in $paperNames.Keys) { Add-PackageFile ("dronerfa_baseline\参考论文\" + $paper) '07_参考文献与项目文档' $paperNames[$paper] '文献' '项目参考论文' }

# 08 验收与校验记录
$records = @(
    @('work\codex-preserved-inputs-20260917.json','验收记录_初始冻结输入哈希清单.json','初始输入哈希'),
    @('work\heldout-eval-config-20260918.json','验收记录_第一阶段留出评价冻结配置.json','留出评价配置'),
    @('work\base2-codex-acceptance-20260919.json','验收记录_第一阶段传统基线验收.json','传统基线验收'),
    @('work\base2-codex-independent-verification-20260919.json','验收记录_第一阶段传统基线独立复算.json','传统基线复算'),
    @('work\nfft4-codex-acceptance-20260919.json','验收记录_第一阶段NFFT四档验收.json','NFFT验收'),
    @('work\stage2-codex-verification.json','验收记录_第二阶段机型识别独立验证.json','第二阶段验证'),
    @('work\cascade3-config-20260919.json','验收记录_两阶段串联冻结配置.json','串联配置'),
    @('work\cascade3-codex-verification-final.json','验收记录_两阶段串联最终独立验证.json','串联最终验证'),
    @('work\infer-ui1-codex-acceptance-20260920.json','验收记录_任意信号分析界面验收.json','界面验收'),
    @('work\launcher1-codex-acceptance-20260920.json','验收记录_一键启动器验收.json','启动器验收'),
    @('work\signal-analyzer-ui-config-20260920.json','验收记录_任意信号分析界面冻结配置.json','界面配置'),
    @('work\deepseek-plan.md','项目记录_已完成阶段与后续计划.md','项目阶段计划')
)
foreach ($x in $records) { Add-PackageFile $x[0] '08_验收与校验记录' $x[1] '验收记录' $x[2] }

# 创建目标目录前完成全部验证。
$missing = @($items | Where-Object { -not (Test-Path -LiteralPath $_.Source -PathType Leaf) })
if ($missing.Count -gt 0) {
    $names = ($missing | ForEach-Object { $_.SourceRelative }) -join "`n"
    throw "以下必需源文件不存在，未创建成果包：`n$names"
}
$duplicateTargets = @($items | Group-Object TargetRelative | Where-Object Count -gt 1)
if ($duplicateTargets.Count -gt 0) { throw ('目标文件名重复：' + (($duplicateTargets | ForEach-Object Name) -join ', ')) }
$ambiguousNames = @('report.md','metrics.csv','verification.json','best_model.pt','window_predictions.csv')
$badNames = @($items | Where-Object { $ambiguousNames -contains (Split-Path -Leaf $_.TargetRelative) })
if ($badNames.Count -gt 0) { throw '存在未标注工作内容的目标文件名。' }

[IO.Directory]::CreateDirectory($DestinationRoot) | Out-Null
$overviewDir = Join-Path $DestinationRoot '00_总览与文件清单'
[IO.Directory]::CreateDirectory($overviewDir) | Out-Null
$overviewPath = Join-Path $overviewDir '00_近期工作成果总览.md'
$dataPath = Join-Path $overviewDir '00_数据集与分析输出位置说明.md'
$manifestPath = Join-Path $overviewDir '00_文件清单_来源_大小_SHA256.csv'

$overview = @"
# DroneRFa 项目近期工作成果总览

整理时间：$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')

本成果包整理了近期完成的数据准备、第一阶段无人机检测、第二阶段机型识别、两阶段串联三分类、任意 IQ 信号分析界面、正式程序与模型、参考文献及验收记录。

## 已完成工作与主要结果

- 数据：24 个 MAT 文件，约 23.025 GiB；当前实验类别为 Background、DJI Phantom 4 Pro、DJI Air 2S。
- 第一阶段传统 clean 基线：新留出集文件等权平衡准确率，能量阈值 98.30%，PSD+SVM 100.00%。
- 第一阶段 NFFT 比较：四条件文件等权平衡准确率跨 seed 均值分别为 512 点 99.04%、1024 点 98.07%、2048 点 97.40%、4096 点 97.98%。
- 第二阶段 Phantom 4 Pro 与 Air 2S 分类：验证集选择 NFFT=512，新留出集四条件文件等权平衡准确率 99.78%±0.09pp。
- 两阶段串联三分类：新留出集文件等权准确率为 clean 99.75%、-10 dB 99.98%、-12 dB 99.63%、-15 dB 95.14%。
- 已完成本地任意 IQ 信号分析界面，支持 MAT、NPY、NPZ 和显式参数 RAW；提供网页可视化、逐窗口分类及 CSV/JSON/PNG/ZIP 导出。

## 目录导航

- 01_数据准备与迁移：数据索引、重复检查、频带检查和 C/F 盘迁移证据。
- 02_第一阶段_无人机检测：传统基线、噪声训练对照、NFFT 四档比较和正式模型。
- 03_第二阶段_机型识别：Phantom 4 Pro/Air 2S 分类报告、预测和正式模型。
- 04_两阶段串联_三分类：Background/Phantom/Air2S 串联评价、图表和逐窗口预测。
- 05_任意信号分析界面：可运行界面程序、网页、启动器和验证材料。
- 06_程序与复现实验材料：训练、评价、验证程序和依赖。
- 07_参考文献与项目文档：论文、文献综述和项目说明。
- 08_验收与校验记录：冻结配置、独立复算和各阶段验收证据。

## 使用结果时必须保留的边界

- 人工 AWGN 的 clean/-10/-12/-15 dB 是相对于已有录音构造的实验条件，不代表现场绝对 SNR。
- 新留出集已参与部分描述性选型和比较，不能称为全新、完全未触碰的最终测试集。
- 不同源文件尚不能证明来自独立日期、环境或硬件会话，不能据此声称跨场景或跨设备泛化。
- 当前范围只有背景和两种已知无人机机型，尚不是开放集任意机型识别系统。

原始 MAT 数据没有重复复制到本成果包，具体位置见 00_数据集与分析输出位置说明.md。每个文件的来源、用途、大小和 SHA256 见 00_文件清单_来源_大小_SHA256.csv。
"@
$dataInfo = @"
# 数据集与分析输出位置说明

- 原始 12 个 MAT：F:\DroneRFa数据集\original
- 新留出 12 个 MAT：F:\DroneRFa数据集\heldout_20260917
- 任意信号分析输出：F:\DroneRFa数据集\analysis_outputs
- 项目程序原位置：$SourceRoot\dronerfa_baseline

两组数据共 24 个 MAT、约 23.025 GiB。为避免在 F 盘再次占用约 23GB，本成果包只保存索引、数据检查、迁移证据、报告、程序、模型和结果，不复制原始 MAT 数据本体。
"@
[IO.File]::WriteAllText($overviewPath, $overview, $utf8Bom)
[IO.File]::WriteAllText($dataPath, $dataInfo, $utf8Bom)

$manifest = New-Object 'System.Collections.Generic.List[object]'
$copyFailures = New-Object 'System.Collections.Generic.List[string]'
foreach ($item in $items) {
    $target = Join-Path $DestinationRoot $item.TargetRelative
    $targetDir = Split-Path -Parent $target
    [IO.Directory]::CreateDirectory($targetDir) | Out-Null
    [IO.File]::Copy($item.Source, $target, $false)
    $sourceHash = (Get-FileHash -LiteralPath $item.Source -Algorithm SHA256).Hash.ToLowerInvariant()
    $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($sourceHash -ne $targetHash) { $copyFailures.Add($item.TargetRelative) | Out-Null }
    $file = Get-Item -LiteralPath $target
    $manifest.Add([pscustomobject]@{
        RelativePath = $item.TargetRelative
        Category = $item.Category
        Description = $item.Description
        SourcePath = $item.Source
        Bytes = $file.Length
        SHA256 = $targetHash
        VerifiedAgainstSource = ($sourceHash -eq $targetHash)
    }) | Out-Null
}

foreach ($generated in @(
    @{Path=$overviewPath; Description='成果包总览'},
    @{Path=$dataPath; Description='数据与分析输出位置说明'}
)) {
    $file = Get-Item -LiteralPath $generated.Path
    $manifest.Add([pscustomobject]@{
        RelativePath = $file.FullName.Substring($DestinationRoot.Length + 1)
        Category = '总览'
        Description = $generated.Description
        SourcePath = 'generated'
        Bytes = $file.Length
        SHA256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        VerifiedAgainstSource = $true
    }) | Out-Null
}
$manifest | Sort-Object RelativePath | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding UTF8

$allFiles = @(Get-ChildItem -LiteralPath $DestinationRoot -Recurse -File)
$totalBytes = ($allFiles | Measure-Object Length -Sum).Sum
$expectedFileCount = $manifest.Count + 1 # 清单不对自身做递归哈希
if ($allFiles.Count -ne $expectedFileCount) { throw "成果包文件数异常：实际 $($allFiles.Count)，预期 $expectedFileCount" }
if ($copyFailures.Count -gt 0) { throw ('复制校验失败：' + ($copyFailures -join ', ')) }

[pscustomobject]@{
    DestinationRoot = $DestinationRoot
    FileCount = $allFiles.Count
    TotalBytes = [int64]$totalBytes
    TotalGiB = [math]::Round($totalBytes / 1GB, 3)
    CopiedAndVerified = $items.Count
    GeneratedFiles = 3
    VerificationFailures = $copyFailures.Count
    Manifest = $manifestPath
} | ConvertTo-Json -Depth 4
