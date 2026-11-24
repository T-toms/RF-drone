from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


PROJECT_DIR = Path(__file__).resolve().parent
SOURCE = PROJECT_DIR / "DroneRFa第一阶段实施说明.md"
OUTPUT = PROJECT_DIR / "DroneRFa第一阶段实施说明.docx"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"
PILOT_DIR = PROJECT_DIR / "outputs" / "pilot_5epochs"

BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)
INK = RGBColor(32, 55, 72)
GRAY = RGBColor(85, 85, 85)
LIGHT_GRAY = "F2F4F7"
BLUE_GRAY = "E8EEF5"
CALLOUT = "F4F6F9"


def set_run_font(
    run,
    size: float | None = None,
    bold: bool | None = None,
    italic: bool | None = None,
    color: RGBColor | None = None,
    monospace: bool = False,
) -> None:
    western = "Consolas" if monospace else "Calibri"
    east_asia = "Microsoft YaHei"
    run.font.name = western
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), western)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), western)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for edge, value in (
        ("top", top),
        ("start", start),
        ("bottom", bottom),
        ("end", end),
    ):
        element = margins.find(qn(f"w:{edge}"))
        if element is None:
            element = OxmlElement(f"w:{edge}")
            margins.append(element)
        element.set(qn("w:w"), str(value))
        element.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths: list[int]) -> None:
    if sum(widths) != 9360:
        raise ValueError(f"Table widths must sum to 9360 DXA, got {sum(widths)}")
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    properties = table._tbl.tblPr

    table_width = properties.find(qn("w:tblW"))
    if table_width is None:
        table_width = OxmlElement("w:tblW")
        properties.append(table_width)
    table_width.set(qn("w:w"), "9360")
    table_width.set(qn("w:type"), "dxa")

    table_indent = properties.find(qn("w:tblInd"))
    if table_indent is None:
        table_indent = OxmlElement("w:tblInd")
        properties.append(table_indent)
    table_indent.set(qn("w:w"), "120")
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
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER


def column_widths(column_count: int) -> list[int]:
    patterns = {
        2: [2700, 6660],
        3: [1900, 3730, 3730],
        4: [1600, 2500, 2630, 2630],
        5: [1360, 2000, 2000, 2000, 2000],
        8: [1000, 1500, 1100, 1100, 1400, 1100, 1080, 1080],
    }
    if column_count in patterns:
        return patterns[column_count]
    base = 9360 // column_count
    values = [base] * column_count
    values[-1] += 9360 - sum(values)
    return values


def add_hyperlink(paragraph, text: str, url: str):
    relationship_id = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run_element = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "2E74B5")
    properties.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    properties.append(underline)
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), "Calibri")
    fonts.set(qn("w:hAnsi"), "Calibri")
    fonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    properties.append(fonts)
    run_element.append(properties)
    text_element = OxmlElement("w:t")
    text_element.text = text
    run_element.append(text_element)
    hyperlink.append(run_element)
    paragraph._p.append(hyperlink)


INLINE_PATTERN = re.compile(
    r"(\*\*.+?\*\*|(?<!\*)\*[^*\n]+?\*(?!\*)|`.+?`|"
    r"\[[^\]]+\]\(https?://[^)]+\))"
)


def add_inline(paragraph, text: str, size: float = 11) -> None:
    cursor = 0
    for match in INLINE_PATTERN.finditer(text):
        if match.start() > cursor:
            run = paragraph.add_run(text[cursor : match.start()])
            set_run_font(run, size=size)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, size=size, bold=True)
        elif token.startswith("*"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, size=size, italic=True)
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, size=size - 0.5, monospace=True, color=DARK_BLUE)
        else:
            label, url = re.match(r"\[([^\]]+)\]\((https?://[^)]+)\)", token).groups()
            add_hyperlink(paragraph, label, url)
        cursor = match.end()
    if cursor < len(text):
        run = paragraph.add_run(text[cursor:])
        set_run_font(run, size=size)


def set_paragraph_shading(paragraph, fill: str) -> None:
    properties = paragraph._p.get_or_add_pPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def add_code_block(document: Document, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        paragraph = document.add_paragraph(style="Code Block")
        if index == 0:
            paragraph.paragraph_format.space_before = Pt(4)
        if index == len(lines) - 1:
            paragraph.paragraph_format.space_after = Pt(8)
        else:
            paragraph.paragraph_format.keep_with_next = True
        run = paragraph.add_run(line or " ")
        set_run_font(run, size=8.5, monospace=True, color=INK)
        set_paragraph_shading(paragraph, LIGHT_GRAY)


def add_table(document: Document, rows: list[list[str]]) -> None:
    column_count = len(rows[0])
    table = document.add_table(rows=len(rows), cols=column_count)
    table.style = "Table Grid"
    set_table_geometry(table, column_widths(column_count))

    header_properties = table.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_properties.append(repeat)

    font_size = 7.5 if column_count >= 8 else 9
    for row_index, values in enumerate(rows):
        row_properties = table.rows[row_index]._tr.get_or_add_trPr()
        cannot_split = OxmlElement("w:cantSplit")
        row_properties.append(cannot_split)
        for column_index, value in enumerate(values):
            cell = table.cell(row_index, column_index)
            if row_index == 0:
                set_cell_shading(cell, BLUE_GRAY)
            paragraph = cell.paragraphs[0]
            paragraph.paragraph_format.space_before = Pt(0)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.line_spacing = 1.1
            paragraph.paragraph_format.keep_with_next = row_index < len(rows) - 1
            if column_index > 0 and len(value) < 28:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(paragraph, value, size=font_size)
            for run in paragraph.runs:
                run.bold = row_index == 0

    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def add_figure(
    document: Document,
    image_path: Path,
    caption: str,
    width: float = 6.3,
) -> None:
    if not image_path.exists():
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(3)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(width))
    description = run._element.xpath(".//wp:docPr")
    if description:
        description[0].set("descr", caption)
    caption_paragraph = document.add_paragraph(style="Caption")
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.keep_with_next = False
    add_inline(caption_paragraph, caption, size=9)


def add_page_field(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_run_font(run, size=9, color=GRAY)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    display = OxmlElement("w:t")
    display.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instruction, separate, display, end))
    suffix = paragraph.add_run(" 页")
    set_run_font(suffix, size=9, color=GRAY)


def create_decimal_numbering(document: Document) -> int:
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
    number_format = OxmlElement("w:numFmt")
    number_format.set(qn("w:val"), "decimal")
    level.append(number_format)
    level_text = OxmlElement("w:lvlText")
    level_text.set(qn("w:val"), "%1.")
    level.append(level_text)
    justification = OxmlElement("w:lvlJc")
    justification.set(qn("w:val"), "left")
    level.append(justification)
    paragraph_properties = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "540")
    tabs.append(tab)
    paragraph_properties.append(tabs)
    indentation = OxmlElement("w:ind")
    indentation.set(qn("w:left"), "540")
    indentation.set(qn("w:hanging"), "270")
    paragraph_properties.append(indentation)
    level.append(paragraph_properties)
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
    number_properties = properties.find(qn("w:numPr"))
    if number_properties is None:
        number_properties = OxmlElement("w:numPr")
        properties.append(number_properties)
    level = OxmlElement("w:ilvl")
    level.set(qn("w:val"), "0")
    number = OxmlElement("w:numId")
    number.set(qn("w:val"), str(num_id))
    number_properties.append(level)
    number_properties.append(number)


def configure_styles(document: Document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    settings = {
        "Heading 1": (16, BLUE, 18, 10),
        "Heading 2": (13, BLUE, 14, 7),
        "Heading 3": (12, DARK_BLUE, 10, 5),
    }
    for name, (size, color, before, after) in settings.items():
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
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(11)
        style.paragraph_format.left_indent = Inches(0.375)
        style.paragraph_format.first_line_indent = Inches(-0.188)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.25

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
    section.different_first_page_header_footer = True

    first_header = section.first_page_header
    first_header.paragraphs[0].text = ""
    header = section.header
    paragraph = header.paragraphs[0]
    paragraph.text = ""
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = paragraph.add_run("DroneRFa 第一阶段实施说明")
    set_run_font(run, size=9, color=GRAY)

    first_footer = section.first_page_footer
    first_footer.paragraphs[0].text = ""
    footer = section.footer
    add_page_field(footer.paragraphs[0])


def add_cover(document: Document) -> None:
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(70)

    kicker = document.add_paragraph()
    kicker.alignment = WD_ALIGN_PARAGRAPH.CENTER
    kicker.paragraph_format.space_after = Pt(16)
    run = kicker.add_run("研究实施与复现指南")
    set_run_font(run, size=11, bold=True, color=BLUE)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(10)
    run = title.add_run("DroneRFa 第一阶段")
    set_run_font(run, size=29, bold=True, color=INK)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(30)
    run = subtitle.add_run("无人机有无检测与低 SNR 深度学习基线")
    set_run_font(run, size=15, color=DARK_BLUE)

    summary = document.add_paragraph()
    summary.alignment = WD_ALIGN_PARAGRAPH.CENTER
    summary.paragraph_format.left_indent = Inches(0.5)
    summary.paragraph_format.right_indent = Inches(0.5)
    summary.paragraph_format.space_after = Pt(34)
    run = summary.add_run(
        "基于 12 个 DroneRFa 实采文件、文件级无泄漏划分、"
        "STFT + ResNet-18，以及复 AWGN 分层评估"
    )
    set_run_font(run, size=11, color=GRAY)

    metrics = document.add_table(rows=2, cols=3)
    metrics.style = "Table Grid"
    set_table_geometry(metrics, [3120, 3120, 3120])
    header_properties = metrics.rows[0]._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    header_properties.append(repeat)
    labels = ("Clean 测试", "-12 dB 测试", "-15 dB 测试")
    values = ("1.0000", "0.9781", "0.7898")
    for column in range(3):
        set_cell_shading(metrics.cell(0, column), BLUE_GRAY)
        for row, value in enumerate((labels[column], values[column])):
            paragraph = metrics.cell(row, column).paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(value)
            set_run_font(
                run,
                size=9 if row == 0 else 17,
                bold=True,
                color=DARK_BLUE if row == 0 else INK,
            )

    note = document.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    note.paragraph_format.space_before = Pt(28)
    note.paragraph_format.space_after = Pt(0)
    run = note.add_run("版本：2026-07-24  |  状态：5 轮先导试验已完成")
    set_run_font(run, size=9.5, color=GRAY)
    document.add_page_break()

    lead = document.add_paragraph()
    lead.paragraph_format.space_after = Pt(12)
    set_paragraph_shading(lead, CALLOUT)
    run = lead.add_run("核心结论  ")
    set_run_font(run, size=11, bold=True, color=DARK_BLUE)
    run = lead.add_run(
        "当前近距离两机型子集在 clean 至 -12 dB 下表现稳定；"
        "-15 dB 时性能下降主要来自无人机漏检。"
        "该结果不能外推到远距离、其他机型或跨日期环境。"
    )
    set_run_font(run, size=11, color=INK)


def parse_markdown(document: Document, text: str) -> None:
    lines = text.splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []
    inserted_signal_figures = False
    inserted_result_figures = False
    current_numbering_id: int | None = None

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.startswith("```"):
            if in_code:
                add_code_block(document, code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(line)
            index += 1
            continue
        if not stripped:
            index += 1
            continue
        if stripped.startswith("# "):
            index += 1
            continue
        if stripped.startswith("|"):
            table_lines: list[list[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                parts = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in parts):
                    table_lines.append(parts)
                index += 1
            add_table(document, table_lines)
            continue
        if stripped.startswith("### "):
            paragraph = document.add_paragraph(style="Heading 3")
            add_inline(paragraph, stripped[4:], size=12)
        elif stripped.startswith("## "):
            paragraph = document.add_paragraph(style="Heading 2")
            add_inline(paragraph, stripped[3:], size=13)
        elif stripped.startswith("- "):
            current_numbering_id = None
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            next_is_bullet = (
                next_index < len(lines)
                and lines[next_index].strip().startswith("- ")
            )
            paragraph = document.add_paragraph(style="List Bullet")
            paragraph.paragraph_format.keep_with_next = next_is_bullet
            add_inline(paragraph, stripped[2:])
        elif re.match(r"^\d+\.\s+", stripped):
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            next_is_numbered = (
                next_index < len(lines)
                and re.match(r"^\d+\.\s+", lines[next_index].strip()) is not None
            )
            if current_numbering_id is None:
                current_numbering_id = create_decimal_numbering(document)
            paragraph = document.add_paragraph(style="List Number")
            apply_numbering(paragraph, current_numbering_id)
            paragraph.paragraph_format.keep_with_next = next_is_numbered
            add_inline(paragraph, re.sub(r"^\d+\.\s+", "", stripped))
            if not next_is_numbered:
                current_numbering_id = None
        elif stripped.startswith("> "):
            current_numbering_id = None
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.25)
            paragraph.paragraph_format.right_indent = Inches(0.25)
            paragraph.paragraph_format.space_before = Pt(4)
            paragraph.paragraph_format.space_after = Pt(8)
            set_paragraph_shading(paragraph, CALLOUT)
            add_inline(paragraph, stripped[2:])
        else:
            current_numbering_id = None
            paragraph = document.add_paragraph()
            if stripped.endswith(("：", ":")):
                paragraph.paragraph_format.keep_with_next = True
            add_inline(paragraph, stripped)

        if stripped.startswith("- `outputs/pilot_5epochs`") and not inserted_signal_figures:
            add_figure(
                document,
                FIGURE_DIR / "signal_overview.png",
                "图 1  RF0 背景、Phantom 4 Pro 与 Air 2S 的包络、PSD 和 STFT",
            )
            add_figure(
                document,
                FIGURE_DIR / "low_snr_stft.png",
                "图 2  Phantom 4 Pro 在不同附加复 AWGN SNR 下的时频图",
            )
            inserted_signal_figures = True

        if stripped.startswith("完整数值位于") and not inserted_result_figures:
            add_figure(
                document,
                PILOT_DIR / "training_curves.png",
                "图 3  5 轮先导试验的训练与验证曲线",
            )
            add_figure(
                document,
                PILOT_DIR / "snr_robustness.png",
                "图 4  测试集 balanced accuracy 与 F1 随附加 SNR 的变化",
                width=6.0,
            )
            inserted_result_figures = True

        index += 1


def main() -> None:
    document = Document()
    configure_styles(document)
    configure_page(document)
    document.core_properties.title = "DroneRFa 第一阶段：无人机有无检测与低 SNR 基线"
    document.core_properties.subject = "DroneRFa 深度学习检测基线实施与复现"
    document.core_properties.keywords = "DroneRFa, RF, IQ, STFT, ResNet-18, low SNR"
    document.core_properties.author = "OpenAI Codex"

    add_cover(document)
    parse_markdown(document, SOURCE.read_text(encoding="utf-8"))
    document.save(OUTPUT)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
