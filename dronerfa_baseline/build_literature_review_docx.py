from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from build_docx import (
    add_code_block,
    add_figure,
    add_inline,
    add_page_field,
    set_cell_margins,
    set_cell_shading,
    set_paragraph_shading,
    set_run_font,
)


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = PROJECT_DIR / "DroneRFa项目四篇核心论文文献综述.md"
OUTPUT = PROJECT_DIR / "DroneRFa项目四篇核心论文文献综述.docx"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
PILOT_DIR = PROJECT_DIR / "outputs" / "pilot_5epochs"

BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)
INK = RGBColor(24, 36, 48)
GRAY = RGBColor(85, 85, 85)
LIGHT_GRAY = "F2F4F7"
CALLOUT = "F4F6F9"

# standard_business_brief preset
CONTENT_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120
BODY_LINE_SPACING = 1.10
LIST_LINE_SPACING = 1.167

# Named table override: dense research comparison tables.
TABLE_FONT_PT = 8.5
WIDE_TABLE_FONT_PT = 7.2
TABLE_LINE_SPACING = 1.0


def set_table_geometry(table, widths: list[int]) -> None:
    if sum(widths) != CONTENT_WIDTH_DXA:
        raise ValueError(f"Table widths must sum to {CONTENT_WIDTH_DXA}, got {sum(widths)}")

    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    properties = table._tbl.tblPr

    table_width = properties.find(qn("w:tblW"))
    if table_width is None:
        table_width = OxmlElement("w:tblW")
        properties.append(table_width)
    table_width.set(qn("w:w"), str(CONTENT_WIDTH_DXA))
    table_width.set(qn("w:type"), "dxa")

    table_indent = properties.find(qn("w:tblInd"))
    if table_indent is None:
        table_indent = OxmlElement("w:tblInd")
        properties.append(table_indent)
    table_indent.set(qn("w:w"), str(TABLE_INDENT_DXA))
    table_indent.set(qn("w:type"), "dxa")

    layout = properties.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        properties.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        column = OxmlElement("w:gridCol")
        column.set(qn("w:w"), str(width))
        grid.append(column)

    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.width = Inches(widths[index] / 1440)
            properties = cell._tc.get_or_add_tcPr()
            cell_width = properties.find(qn("w:tcW"))
            if cell_width is None:
                cell_width = OxmlElement("w:tcW")
                properties.append(cell_width)
            cell_width.set(qn("w:w"), str(widths[index]))
            cell_width.set(qn("w:type"), "dxa")
            set_cell_margins(cell, top=80, start=120, bottom=80, end=120)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def column_widths(column_count: int) -> list[int]:
    patterns = {
        2: [2400, 6960],
        3: [1800, 3780, 3780],
        4: [1780, 2200, 2660, 2720],
        5: [1450, 1850, 2100, 2050, 1910],
        8: [960, 1460, 1180, 1180, 1420, 1080, 1040, 1040],
    }
    if column_count in patterns:
        return patterns[column_count]
    base = CONTENT_WIDTH_DXA // column_count
    widths = [base] * column_count
    widths[-1] += CONTENT_WIDTH_DXA - sum(widths)
    return widths


def create_numbering(document: Document, number_format: str) -> int:
    numbering = document.part.numbering_part.element
    abstract_ids = [
        int(element.get(qn("w:abstractNumId")))
        for element in numbering.findall(qn("w:abstractNum"))
    ]
    num_ids = [
        int(element.get(qn("w:numId")))
        for element in numbering.findall(qn("w:num"))
    ]
    abstract_id = max(abstract_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1

    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi_level = OxmlElement("w:multiLevelType")
    multi_level.set(qn("w:val"), "singleLevel")
    abstract.append(multi_level)

    level = OxmlElement("w:lvl")
    level.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    level.append(start)
    fmt = OxmlElement("w:numFmt")
    fmt.set(qn("w:val"), number_format)
    level.append(fmt)
    text = OxmlElement("w:lvlText")
    text.set(qn("w:val"), "%1." if number_format == "decimal" else "•")
    level.append(text)
    justification = OxmlElement("w:lvlJc")
    justification.set(qn("w:val"), "left")
    level.append(justification)

    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "720")
    tabs.append(tab)
    p_pr.append(tabs)
    indentation = OxmlElement("w:ind")
    indentation.set(qn("w:left"), "720")
    indentation.set(qn("w:hanging"), "360")
    p_pr.append(indentation)
    spacing = OxmlElement("w:spacing")
    spacing.set(qn("w:after"), "160")
    spacing.set(qn("w:line"), "280")
    spacing.set(qn("w:lineRule"), "auto")
    p_pr.append(spacing)
    level.append(p_pr)
    abstract.append(level)
    numbering.append(abstract)

    number = OxmlElement("w:num")
    number.set(qn("w:numId"), str(num_id))
    abstract_reference = OxmlElement("w:abstractNumId")
    abstract_reference.set(qn("w:val"), str(abstract_id))
    number.append(abstract_reference)
    numbering.append(number)
    return num_id


def apply_numbering(paragraph, num_id: int) -> None:
    properties = paragraph._p.get_or_add_pPr()
    num_properties = OxmlElement("w:numPr")
    level = OxmlElement("w:ilvl")
    level.set(qn("w:val"), "0")
    number = OxmlElement("w:numId")
    number.set(qn("w:val"), str(num_id))
    num_properties.extend((level, number))
    properties.append(num_properties)


def set_bottom_rule(paragraph, color: str = "B7C3CF") -> None:
    properties = paragraph._p.get_or_add_pPr()
    borders = properties.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        properties.append(borders)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "8")
    bottom.set(qn("w:color"), color)
    borders.append(bottom)


def configure_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = BODY_LINE_SPACING

    heading_tokens = {
        "Heading 1": (16, BLUE, 16, 8),
        "Heading 2": (13, BLUE, 12, 6),
        "Heading 3": (12, DARK_BLUE, 8, 4),
    }
    for name, (size, color, before, after) in heading_tokens.items():
        style = document.styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = color
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Number"):
        style = document.styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(11)
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.line_spacing = LIST_LINE_SPACING

    caption = document.styles["Caption"]
    caption.font.name = "Calibri"
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    caption.font.size = Pt(9)
    caption.font.italic = False
    caption.font.color.rgb = GRAY
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(8)

    code_style = document.styles.add_style("Code Block", 1)
    code_style.font.name = "Consolas"
    code_style._element.rPr.rFonts.set(qn("w:ascii"), "Consolas")
    code_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Consolas")
    code_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    code_style.font.size = Pt(8.5)
    code_style.paragraph_format.left_indent = Inches(0.18)
    code_style.paragraph_format.right_indent = Inches(0.18)
    code_style.paragraph_format.space_before = Pt(0)
    code_style.paragraph_format.space_after = Pt(0)
    code_style.paragraph_format.line_spacing = 1.05


def configure_page(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    header = section.header
    paragraph = header.paragraphs[0]
    paragraph.text = ""
    run = paragraph.add_run("DroneRFa 射频无人机检测研究  |  文献综述")
    set_run_font(run, size=9, color=GRAY)

    footer = section.footer
    footer.paragraphs[0].text = ""
    add_page_field(footer.paragraphs[0])


def add_masthead(document: Document) -> None:
    kicker = document.add_paragraph()
    kicker.paragraph_format.space_before = Pt(10)
    kicker.paragraph_format.space_after = Pt(5)
    run = kicker.add_run("研究综述与项目方法映射")
    set_run_font(run, size=10.5, bold=True, color=BLUE)

    title = document.add_paragraph()
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run("DroneRFa 项目四篇核心论文文献综述")
    set_run_font(run, size=23, bold=True, color=RGBColor(0, 0, 0))

    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(14)
    run = subtitle.add_run("无人机有无检测、低 SNR 鲁棒性与机型分类/多信号扩展")
    set_run_font(run, size=13, color=GRAY)

    metadata = [
        ("研究问题", "基于 2.4 GHz RF0 复数 IQ 的 Background/Drone 窗口级二分类"),
        ("文献范围", "DroneRFa（2024）、低 SNR CNN（2024）、ResNet（2016）、TCCN 联合检测（2022）"),
        ("项目状态", "数据与代码基线已完成；5 轮先导试验完成；正式泛化实验待完成"),
        ("编制日期", "2026 年 7 月"),
    ]
    for label, value in metadata:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(2)
        label_run = paragraph.add_run(f"{label}：")
        set_run_font(label_run, size=10.5, bold=True, color=INK)
        value_run = paragraph.add_run(value)
        set_run_font(value_run, size=10.5, color=INK)

    rule = document.add_paragraph()
    rule.paragraph_format.space_before = Pt(4)
    rule.paragraph_format.space_after = Pt(10)
    set_bottom_rule(rule)

    lead = document.add_paragraph()
    lead.paragraph_format.space_before = Pt(0)
    lead.paragraph_format.space_after = Pt(10)
    lead.paragraph_format.left_indent = Inches(0.18)
    lead.paragraph_format.right_indent = Inches(0.18)
    set_paragraph_shading(lead, CALLOUT)
    run = lead.add_run("核心结论  ")
    set_run_font(run, size=11, bold=True, color=DARK_BLUE)
    run = lead.add_run(
        "四篇论文分别承担数据基础、低 SNR 评估、模型骨架和后续定位扩展四个角色。"
        "当前已经实现前三者的第一阶段组合，尚未实现 YOLO 多信号时频定位。"
    )
    set_run_font(run, size=11, color=INK)


def add_table(document: Document, rows: list[list[str]]) -> None:
    column_count = len(rows[0])
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    set_table_geometry(table, column_widths(column_count))

    header_properties = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_properties.append(repeat)

    font_size = WIDE_TABLE_FONT_PT if column_count >= 7 else TABLE_FONT_PT
    for row_index, values in enumerate(rows):
        row_properties = table.rows[row_index]._tr.get_or_add_trPr()
        cannot_split = OxmlElement("w:cantSplit")
        row_properties.append(cannot_split)
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            if row_index == 0:
                set_cell_shading(cell, LIGHT_GRAY)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = TABLE_LINE_SPACING
            if column_count >= 7 or (column_index > 0 and len(value) < 18):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(paragraph, value, size=font_size)
            for run in paragraph.runs:
                run.bold = row_index == 0

    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def parse_markdown(document: Document, text: str) -> None:
    lines = text.splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []
    current_section = ""
    bullet_num_id: int | None = None
    decimal_num_id: int | None = None
    inserted_signal_figures = False
    inserted_result_figure = False

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                add_code_block(document, code_lines)
                code_lines = []
                in_code = False
                if current_section.startswith("8.2") and not inserted_signal_figures:
                    add_figure(
                        document,
                        FIGURE_DIR / "signal_overview.png",
                        "图 1  当前子集的 RF0 包络、PSD 与 STFT 对比",
                        width=6.2,
                    )
                    add_figure(
                        document,
                        FIGURE_DIR / "low_snr_stft.png",
                        "图 2  Phantom 4 Pro 在不同附加复 AWGN SNR 下的时频图",
                        width=6.2,
                    )
                    inserted_signal_figures = True
            else:
                in_code = True
            index += 1
            continue

        if in_code:
            code_lines.append(line)
            index += 1
            continue

        if not stripped or stripped == "---":
            index += 1
            continue

        if stripped == ">":
            index += 1
            continue

        if stripped.startswith("# "):
            index += 1
            continue

        if stripped.startswith("|"):
            table_rows: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                parts = [
                    cell.strip()
                    for cell in lines[index].strip().strip("|").split("|")
                ]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in parts):
                    table_rows.append(parts)
                index += 1
            add_table(document, table_rows)
            if current_section.startswith("9.") and not inserted_result_figure:
                add_figure(
                    document,
                    PILOT_DIR / "snr_robustness.png",
                    "图 3  先导试验中 balanced accuracy 与 F1 随附加 SNR 的变化",
                    width=5.9,
                )
                inserted_result_figure = True
            continue

        if stripped.startswith("### "):
            current_section = stripped[4:]
            paragraph = document.add_paragraph(style="Heading 2")
            add_inline(paragraph, current_section, size=13)
            bullet_num_id = None
            decimal_num_id = None
        elif stripped.startswith("## "):
            current_section = stripped[3:]
            paragraph = document.add_paragraph(style="Heading 1")
            add_inline(paragraph, current_section, size=16)
            bullet_num_id = None
            decimal_num_id = None
        elif stripped.startswith("- "):
            if bullet_num_id is None:
                bullet_num_id = create_numbering(document, "bullet")
            decimal_num_id = None
            paragraph = document.add_paragraph(style="List Bullet")
            apply_numbering(paragraph, bullet_num_id)
            add_inline(paragraph, stripped[2:])
        elif re.match(r"^\d+\.\s+", stripped):
            if decimal_num_id is None:
                decimal_num_id = create_numbering(document, "decimal")
            bullet_num_id = None
            paragraph = document.add_paragraph(style="List Number")
            apply_numbering(paragraph, decimal_num_id)
            add_inline(paragraph, re.sub(r"^\d+\.\s+", "", stripped))
        elif stripped.startswith("> "):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("> "):
                quote_lines.append(lines[index].strip()[2:])
                index += 1
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.right_indent = Inches(0.25)
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(8)
            set_paragraph_shading(paragraph, CALLOUT)
            add_inline(paragraph, "\n".join(quote_lines))
            continue
        else:
            bullet_num_id = None
            decimal_num_id = None
            paragraph = document.add_paragraph()
            plain = stripped.replace("**", "")
            if plain.endswith(("：", ":")):
                paragraph.paragraph_format.keep_with_next = True
            if stripped.startswith("**关键词：**"):
                set_paragraph_shading(paragraph, LIGHT_GRAY)
            add_inline(paragraph, stripped)

        index += 1


def main() -> None:
    document = Document()
    configure_styles(document)
    configure_page(document)
    document.core_properties.title = "DroneRFa 项目四篇核心论文文献综述"
    document.core_properties.subject = "无人机射频检测、低 SNR 与多信号扩展"
    document.core_properties.keywords = (
        "DroneRFa, RF, IQ, STFT, ResNet-18, low SNR, YOLO"
    )
    document.core_properties.author = "OpenAI Codex"

    add_masthead(document)
    source_text = SOURCE.read_text(encoding="utf-8")
    body_text = source_text.split("\n---\n", maxsplit=1)[-1]
    parse_markdown(document, body_text)
    document.save(OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
