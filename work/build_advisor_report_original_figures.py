from pathlib import Path
import hashlib
import json
from zipfile import ZipFile
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports' / 'DroneRFa进展汇报_导师版.docx'
SOURCE = ROOT / 'dronerfa_baseline/outputs/cascade3_evaluation_20260919'
doc = Document()
s = doc.sections[0]
s.page_width, s.page_height = Cm(21), Cm(29.7)
s.top_margin = s.bottom_margin = Cm(1.65)
s.left_margin = s.right_margin = Cm(1.9)
for name in ('Normal', 'Title', 'Caption'):
    style = doc.styles[name]
    style.font.name = 'Times New Roman'
    style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '宋体')
    style.font.color.rgb = RGBColor(0,0,0)
    style.font.size = Pt(11)
normal = doc.styles['Normal'].paragraph_format
normal.line_spacing = 1.15
normal.space_after = Pt(6)

p = doc.add_paragraph(style='Title')
p.alignment = 1
r = p.add_run('无人机射频检测与机型识别进展汇报')
r.font.size = Pt(17)
r.bold = True
p.paragraph_format.space_after = Pt(8)

def item(n, lead, body):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(.6)
    p.paragraph_format.first_line_indent = Cm(-.6)
    p.add_run(f'{n}. {lead}').bold = True
    p.add_run(body)
    return p

figures = []
def figure(name, caption, width=17.0):
    source = SOURCE / name
    p = doc.add_paragraph()
    p.alignment = 1
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(source), width=Cm(width))
    c = doc.add_paragraph(style='Caption')
    c.alignment = 1
    c.paragraph_format.line_spacing = 1
    c.paragraph_format.space_after = Pt(6)
    run = c.add_run(caption)
    run.font.size = Pt(9)
    figures.append({'source':str(source), 'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})

item(1, '完成两阶段识别实验。', '使用 DroneRFa 中的 Background、DJI Phantom 4 Pro 和 DJI Air 2S 信号，先通过 STFT+ResNet18 检测无人机，再识别两种机型。测试集包括 12 个原始文件、1,820 个信号片段。以下结果按文件和类别平衡计算；深度模型取三次重复实验的平均值。')
item(2, '完成检测阶段的传统方法对照。', '未额外加噪时，能量门限和 PSD+SVM 的平衡准确率分别为 98.30% 和 100.00%。PSD+SVM 直接从 I/Q 数据提取功率谱特征，不使用 STFT 图像。')
item(3, '完成检测阶段的 STFT 点数比较。', 'n_fft 为 512、1024、2048、4096 时，未额外加噪及 −10、−12、−15 dB 四种条件下的平均准确率分别为 99.04%、98.07%、97.40%、97.98%；其中 −15 dB 时分别为 96.62%、93.43%、91.58%、92.85%。512 点在本次测试中抗噪表现较好。')
item(4, '完成检测阶段和第二阶段的 SNR 测试。', '采用 n_fft=512 时，检测阶段在未额外加噪、−10、−12、−15 dB 下分别达到 99.87%、99.97%、99.69%、96.62%。第二阶段仅测试已知无人机信号，相同四种条件下分别为 99.84%、100.00%、99.97%、99.30%，平均为 99.78%；n_fft=1024 的平均结果为 99.46%。以上 SNR 为人工添加 AWGN 的实验设置。')
item(5, '完成两阶段连接后的三分类测试。', '未额外加噪、−10、−12、−15 dB 下，Background、Phantom 4 Pro、Air 2S 三分类的平均准确率分别为 99.75%、99.98%、99.63%、95.14%。强噪声下的下降主要来自检测阶段漏检 Air 2S，各类别变化见图 1。')
figure('recall_by_condition.png', '图 1  两阶段三分类在不同噪声条件下的各类别召回率（实验原图）', 16.7)

doc.add_page_break()
figure('confusion_heldout_-15.png', '图 2  −15 dB 下三次重复实验的三分类混淆矩阵（实验原图）', 17.2)
item(6, '分析强噪声下的分类错误。', '图 2 中，三次实验有 15.1%、12.9%、10.5% 的 Air 2S 信号被判为 Background，而被误判为 Phantom 4 Pro 的比例仅为 0.6%、0%、0%。第二阶段单独分类在 −15 dB 下仍为 99.30%，说明当前主要问题是无人机漏检。图中的 accuracy 按信号片段直接统计，与正文按文件和类别平衡计算的指标不同。')
item(7, '下一步完善测试和方法对照。', '当前仅测试两种机型，部分测试数据已用于参数比较，尚不能说明跨场景效果。后续先核实采集条件并设置独立测试集，再补充直接三分类和 Raw-IQ 对照，判断两阶段结构及 STFT 是否带来实际收益。')

doc.core_properties.title = '无人机射频检测与机型识别进展汇报'
for element in (doc.styles.element, doc.element):
    for border in element.xpath('.//w:pBdr'):
        border.getparent().remove(border)
doc.save(OUT)
with ZipFile(OUT) as z:
    media = [hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('word/media/')]
assert sorted(media) == sorted(x['sha256'] for x in figures)
assert not doc.tables
evidence = {'output': str(OUT), 'original_figures': figures, 'embedded_images_match_original_files': True, 'tables': 0, 'render_status': 'pending'}
(ROOT/'work/advisor_report_original_figures_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT)
