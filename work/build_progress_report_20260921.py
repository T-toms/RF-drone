from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
OUTPUT = ROOT / "reports" / "DroneRFa项目进展汇报_结果版_20260921.docx"
RESULT_FIGURE = ROOT / "reports" / "assets" / "progress_results.png"
CONFUSION_FIGURE = ROOT / "reports" / "assets" / "progress_confusion.png"
BLACK = RGBColor(0, 0, 0)


def set_font(run, size=11, bold=False):
    run.font.name = "SimSun"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = BLACK


def add_run(paragraph, text, size=11, bold=False):
    run = paragraph.add_run(text)
    set_font(run, size=size, bold=bold)


def add_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    add_run(p, text, size=13, bold=True)


def add_body(doc, text, after=5):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.line_spacing = 1.2
    p.paragraph_format.space_after = Pt(after)
    add_run(p, text)


def add_picture(doc, path, width_cm, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(path), width=Cm(width_cm))

    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(5)
    add_run(c, caption, size=9.5)


def build_report():
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(1.6)
    section.bottom_margin = Cm(1.6)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    normal = doc.styles["Normal"]
    normal.font.name = "SimSun"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(11)
    normal.font.color.rgb = BLACK

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(3)
    add_run(title, "DroneRFa 无人机射频识别项目进展汇报", size=18, bold=True)

    date = doc.add_paragraph()
    date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date.paragraph_format.space_after = Pt(8)
    add_run(date, "2026 年 9 月 21 日", size=10.5)

    add_heading(doc, "一、实验内容和主要结果")
    add_body(
        doc,
        "目前使用 DroneRFa 中的 Background、DJI Phantom 4 Pro 和 DJI Air 2S 三类信号。完整方法分为两个阶段：阶段 1 使用 STFT 时频图和 ResNet18 判断背景或无人机；阶段 2 在无人机信号中区分 Phantom 4 Pro 和 Air 2S。实验共使用 24 个原始 MAT 文件，测试数据包含其中 12 个文件、1,820 个信号片段。主要指标采用按原始文件等权的平衡准确率。",
    )
    add_body(
        doc,
        "阶段 1 先完成能量门限和 PSD+SVM 对照，结果分别为 98.30% 和 100.00%。随后比较 STFT 的 n_fft=512、1024、2048 和 4096，四种设置的平均结果分别为 99.04%、98.07%、97.40% 和 97.98%，因此后续采用 n_fft=512。阶段 2 的 Phantom 4 Pro / Air 2S 二分类结果为 99.78%±0.09 个百分点。",
    )
    add_body(
        doc,
        "将两个阶段连接后，完整三分类在正常条件、−10 dB、−12 dB 和 −15 dB AWGN 下的结果分别为 99.75%、99.98%、99.63% 和 95.14%。−15 dB 时性能下降明显，主要原因是 Air 2S 在阶段 1 被判为背景。",
    )

    add_picture(
        doc,
        RESULT_FIGURE,
        16.7,
        "图 1  检测阶段 n_fft 对比、不同 SNR 下的三分类结果及阶段 2 结果",
    )

    doc.add_page_break()

    add_heading(doc, "二、结果说明")
    add_picture(
        doc,
        CONFUSION_FIGURE,
        15.2,
        "图 2  正常条件与 −15 dB AWGN 下的三分类混淆矩阵",
    )
    add_body(
        doc,
        "正常条件下三类信号均能稳定识别。−15 dB 时，Background 和 Phantom 4 Pro 的识别率仍接近 99%，但 Air 2S 的识别率降至 87.0%，其中 12.8% 被判为 Background。这说明当前系统在强噪声下的主要问题不是两种无人机之间的混淆，而是阶段 1 对 Air 2S 的漏检。",
    )
    add_body(
        doc,
        "现有结果表明，三类信号在当前数据中具有较高可分性，n_fft=512 适合当前 STFT+ResNet18 设置，阶段 2 已较稳定。需要注意的是，PSD+SVM 达到 100.00%，可能说明 PSD 特征有效，也可能与不同录制文件的功率、设备或环境差异有关。下一步应先补齐采集条件信息，建立一组不参与参数选择的最终测试数据，再比较当前分阶段方法、直接三分类 ResNet18 和 Raw-IQ 模型。",
        after=0,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.core_properties.title = "DroneRFa 无人机射频识别项目进展汇报"
    doc.core_properties.author = "DroneRFa 项目组"
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_report()
