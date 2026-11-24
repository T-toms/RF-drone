"""Build a Chinese, source-auditable UAV RF literature review DOCX.

This script intentionally uses only the fresh DOI-verified source file in this
folder.  It does not read historical project instructions, experiment settings,
models, or result files.
"""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "references_metadata.csv"
OUTPUT_DOCX = ROOT / "无人机射频RF深度学习检测与识别_从零文献综述_2026-08.docx"

EXCLUDED_DOI = "10.1109/aero.2019.8741733"
MANUAL_REFERENCE = {
    "doi": "10.11999/JEIT230570",
    "track": "dataset",
    "evidence_role": "dataset_resource",
    "metadata_source": "Official JEIT",
    "title": "DroneRFa: A Large-scale Dataset of Drone Radio Frequency Signals for Detecting Low-altitude Drones",
    "authors": "Yu Ningning; Mao Shengjian; Zhou Chengwei; Sun Guowei; Shi Zhiguo; Chen Jiming",
    "year": "2024",
    "date": "2024-04-24",
    "venue": "Journal of Electronics & Information Technology",
    "volume": "46",
    "issue": "4",
    "pages_or_article": "1147–1156",
    "publisher": "Journal of Electronics & Information Technology",
    "reference_count": "",
    "cited_by_crossref": "",
    "url": "https://doi.org/10.11999/JEIT230570",
    "type": "journal-article",
}

BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
INK = "0B2545"
MUTED = "666666"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
CALLOUT = "F4F6F9"
GRID = "B8C4D3"
TITLE_DARK = "203748"


def normalize_doi(doi: str) -> str:
    return doi.strip().lower().replace("https://doi.org/", "")


def load_references() -> list[dict[str, str]]:
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cleaned = [row for row in rows if normalize_doi(row["doi"]) != EXCLUDED_DOI]
    cleaned.append(MANUAL_REFERENCE.copy())
    cleaned.sort(key=lambda row: (int(row.get("year") or 9999), row.get("title", "").casefold()))
    for index, row in enumerate(cleaned, start=1):
        row["key"] = f"R{index:02d}"
        row["doi"] = normalize_doi(row["doi"])
    return cleaned


def ensure_rfonts(rpr) -> OxmlElement:
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    return rfonts


def set_run_font(run, size: float | None = None, color: str | None = None, bold: bool | None = None,
                 italic: bool | None = None, underline: bool | None = None) -> None:
    run.font.name = "Calibri"
    rpr = run._element.get_or_add_rPr()
    rfonts = ensure_rfonts(rpr)
    rfonts.set(qn("w:ascii"), "Calibri")
    rfonts.set(qn("w:hAnsi"), "Calibri")
    # Named override: use a Chinese glyph fallback while preserving Calibri as the preset base font.
    rfonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if underline is not None:
        run.underline = underline


def set_style_font(style, size: float, color: str, bold: bool = False) -> None:
    style.font.name = "Calibri"
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor.from_string(color)
    style.font.bold = bold
    rpr = style.element.get_or_add_rPr()
    rfonts = ensure_rfonts(rpr)
    rfonts.set(qn("w:ascii"), "Calibri")
    rfonts.set(qn("w:hAnsi"), "Calibri")
    rfonts.set(qn("w:eastAsia"), "Microsoft YaHei")


def set_page_geometry(section) -> None:
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    set_style_font(normal, 11, "000000")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT

    for level, size, color, before, after in (
        (1, 16, BLUE, 18, 10),
        (2, 13, BLUE, 14, 7),
        (3, 12, DARK_BLUE, 10, 5),
    ):
        style = styles[f"Heading {level}"]
        set_style_font(style, size, color, bold=True)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.25
        style.paragraph_format.keep_with_next = True


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")


def set_cell_margins(cell, top: int = 80, start: int = 120, bottom: int = 80, end: int = 120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")
    cell.width = Inches(width_dxa / 1440)


def set_table_borders(table, color: str = GRID) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), "4")
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_table_geometry(table, widths_in: list[float], indent_dxa: int = 120) -> None:
    widths_dxa = [round(width * 1440) for width in widths_in]
    total_dxa = sum(widths_dxa)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total_dxa))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    grid = table._tbl.tblGrid
    for grid_col, width_dxa in zip(grid.gridCol_lst, widths_dxa):
        grid_col.set(qn("w:w"), str(width_dxa))
    for row in table.rows:
        for cell, width_dxa in zip(row.cells, widths_dxa):
            set_cell_width(cell, width_dxa)
            set_cell_margins(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    set_table_borders(table)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def fill_cell(cell, text: str, *, bold: bool = False, font_size: float = 8.3,
              color: str = "000000", align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(text)
    set_run_font(run, size=font_size, color=color, bold=bold)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths_in: list[float],
              caption: str | None = None) -> None:
    if caption:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(caption)
        set_run_font(r, size=8.5, color=MUTED, italic=True)

    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths_in)
    header = table.rows[0]
    repeat_header(header)
    prevent_row_split(header)
    for cell, header_text in zip(header.cells, headers):
        set_cell_shading(cell, LIGHT_BLUE)
        fill_cell(cell, header_text, bold=True, font_size=8.5, color=INK)
    for values in rows:
        row = table.add_row()
        prevent_row_split(row)
        for cell, value in zip(row.cells, values):
            fill_cell(cell, value)
    set_table_geometry(table, widths_in)


def add_body(doc: Document, text: str, *, color: str = "000000", italic: bool = False,
             size: float | None = None, align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
    p = doc.add_paragraph(style="Normal")
    p.alignment = align
    run = p.add_run(text)
    set_run_font(run, size=size, color=color, italic=italic)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_paragraph(style=f"Heading {level}")
    run = p.add_run(text)
    set_run_font(run)


def add_callout(doc: Document, text: str) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [6.5])
    prevent_row_split(table.rows[0])
    cell = table.cell(0, 0)
    set_cell_shading(cell, CALLOUT)
    fill_cell(cell, text, font_size=9.5, color=INK)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)


def add_page_field(paragraph) -> None:
    run = paragraph.add_run()
    set_run_font(run, size=8.5, color=MUTED)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = "PAGE"
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, end])


def configure_header_footer(section) -> None:
    section.different_first_page_header_footer = True
    first_header = section.first_page_header.paragraphs[0]
    first_header.text = ""
    first_footer = section.first_page_footer.paragraphs[0]
    first_footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = first_footer.add_run("检索与核验截止：2026-08-15")
    set_run_font(run, size=8.5, color=MUTED)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    run = header.add_run("无人机射频信号（RF）深度学习检测与识别｜从零文献综述")
    set_run_font(run, size=8.5, color=MUTED)

    footer = section.footer
    footer_table = footer.add_table(rows=1, cols=2, width=Inches(6.5))
    set_table_geometry(footer_table, [4.7, 1.8])
    prevent_row_split(footer_table.rows[0])
    for cell in footer_table.rows[0].cells:
        set_table_borders(footer_table, color="FFFFFF")
        set_cell_margins(cell, top=0, start=0, bottom=0, end=0)
    left = footer_table.cell(0, 0).paragraphs[0]
    left.paragraph_format.space_after = Pt(0)
    left_run = left.add_run("证据库：63 条核验条目（含数据资源与综述）")
    set_run_font(left_run, size=8.5, color=MUTED)
    right = footer_table.cell(0, 1).paragraphs[0]
    right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    right.paragraph_format.space_after = Pt(0)
    page_label = right.add_run("第 ")
    set_run_font(page_label, size=8.5, color=MUTED)
    add_page_field(right)
    page_suffix = right.add_run(" 页")
    set_run_font(page_suffix, size=8.5, color=MUTED)


def add_cover(doc: Document) -> None:
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(110)
    spacer.paragraph_format.space_after = Pt(0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    run = p.add_run("从零开始的领域地图")
    set_run_font(run, size=12, color="7A5A00", bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run("无人机射频信号（RF）\n深度学习检测与识别")
    set_run_font(run, size=30, color=TITLE_DARK, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(28)
    run = p.add_run("主流技术、权威文献、数据资源与可验证创新路线")
    set_run_font(run, size=14, color=DARK_BLUE)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("证据截止：2026 年 8 月 15 日｜首版文献证据库")
    set_run_font(run, size=10.5, color=MUTED, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run("范围：被动截获无人机及遥控器通信链路的 RF 信号；不以雷达、视觉或声学为主体")
    set_run_font(run, size=9.5, color=MUTED, italic=True)

    doc.add_page_break()


def short_author(row: dict[str, str]) -> str:
    people = [part.strip() for part in (row.get("authors") or "").split(";") if part.strip()]
    if not people:
        return "Unknown"
    return f"{people[0]} 等" if len(people) > 1 else people[0]


def ref_string(row: dict[str, str]) -> str:
    author = short_author(row)
    venue = row.get("venue") or row.get("publisher") or ""
    pieces = [f"[{row['key']}] {author}. ({row.get('year') or 'n.d.'}). {row.get('title') or ''}."]
    if venue:
        volume = row.get("volume") or ""
        issue = row.get("issue") or ""
        pages = row.get("pages_or_article") or ""
        issue_text = f"({issue})" if issue else ""
        tail = " ".join(part for part in [venue, f"{volume}{issue_text}".strip(), pages] if part)
        pieces.append(tail + ".")
    pieces.append(f"https://doi.org/{row['doi']}")
    return " ".join(pieces)


def build_document() -> None:
    references = load_references()
    by_doi = {row["doi"]: row for row in references}

    def cite(*dois: str) -> str:
        labels = []
        for doi in dois:
            key = normalize_doi(doi)
            labels.append(f"[{by_doi[key]['key']}]")
        return "、".join(labels)

    reviews = [row for row in references if row["track"] == "review"]
    datasets = [row for row in references if row["track"] == "dataset"]
    empirical = [row for row in references if row["track"] not in {"review", "dataset"}]

    doc = Document()
    section = doc.sections[0]
    set_page_geometry(section)
    configure_styles(doc)
    configure_header_footer(section)
    add_cover(doc)

    add_heading(doc, "0. 先给结论：新研究从什么开始", 1)
    add_callout(
        doc,
        "本报告不分析、不继承、也不引用本项目此前任何 DroneRFa 或 RFUAV 试验、脚本、参数或结果。"
        "它们仅作为历史文件存档。后续的研究问题、数据集选择、基线和成功标准都应从本报告重新定义。",
    )
    add_body(
        doc,
        "本次核验的领域核心不是泛称“无线电信号”，而是“无人机射频信号（RF）”："
        "接收端被动截获无人机与遥控器/地面站之间的控制、遥测或图传通信链路，并从原始 I/Q、频谱或时频表示中完成存在检测、机型/设备识别、飞行模式识别或频谱内多目标检测。",
    )
    add_body(
        doc,
        "将 63 条新核验条目按研究任务归纳后，可稳妥地把主流路线概括为："
        "SDR 被动接收 → I/Q、FFT/PSD、STFT 或小波等表示 → 1D-CNN、2D-CNN/ResNet 或端到端网络 →"
        "“有无无人机—类别/设备—模式”的单任务或分层判定。"
        "这一表述来自多篇实证研究与综述的共同技术结构，而不是从任何既有本地试验外推。"
        f" {cite('10.1016/j.future.2019.05.007', '10.1109/iciot48696.2020.9089657', '10.1109/access.2022.3172787', '10.3390/s22041453')}",
    )
    add_body(
        doc,
        "Transformer、开放集识别、跨域泛化、少样本学习、多机重叠检测、联邦学习和极轻量部署已经出现高质量论文，"
        "但在本证据库中应严谨称为“新兴/增长方向”，不应笼统称为已经取代 CNN 的主流。"
        f" {cite('10.3390/s23229098', '10.1109/TIFS.2024.3463535', '10.1109/TCCN.2025.3561305', '10.1109/TVT.2026.3696747')}",
    )

    add_heading(doc, "1. 范围、术语与物理边界", 1)
    add_heading(doc, "1.1 本报告纳入的“RF”", 2)
    scope_rows = [
        ["核心对象", "无人机、遥控器或地面站主动发射的通信 RF；接收端被动监听其控制、遥测、图传或协议/硬件特征。"],
        ["典型硬件", "SDR、天线和主机；文献中常见对 ISM 频段与 UAV—控制器链路的扫描、接收和分析。"],
        ["信号表示", "复数 I/Q、幅度/相位、包络、FFT/PSD、STFT 谱图、小波或散射系数。"],
        ["研究任务", "存在检测、型号/厂家/设备级识别、飞行模式分类、频谱内目标检测、少量定位/跟踪扩展。"],
        ["不作为主体", "雷达回波与微多普勒、视频图像、声学。这些可在系统边界中作为互补传感器出现，但不与通信 RF 的指标混表。"],
    ]
    add_table(doc, ["项目", "本报告定义"], scope_rows, [1.35, 5.15], "表 1  术语与纳入边界")
    add_body(
        doc,
        "被动通信 RF 的边界必须写在研究开端：它针对可接收的发射/通信链路；完全静默、自治或链路不可接收的无人机不能被该路线保证检出。"
        "这不是 RF 路线无效，而是它与雷达、光电或声学等传感路径的适用条件不同。"
        f" {cite('10.3390/s22041453', '10.3390/s24010125')}",
    )
    add_body(
        doc,
        "还应区分物理层 RF 与 Wi-Fi 的 MAC/流量/包字段分析：后者可作为相邻的数据链路路线讨论，"
        "但本报告的主统计和方法比较只纳入前者的 I/Q、频谱、时频图或发射端硬件/波形特征。"
        f" {cite('10.1016/j.iot.2024.101201')}",
    )

    add_heading(doc, "1.2 避免术语漂移", 2)
    add_body(
        doc,
        "正文统一使用“无人机射频信号（RF）”或“被动截获无人机通信 RF 链路”。“无线电信号”只在引用原文题名时保留，"
        "不会作为研究对象的泛化描述。这样可避免把通信 RF、雷达反射、GPS、一般无线网络流量混为同一类信号。",
    )

    add_heading(doc, "2. 文献证据库：如何快速收集而不把噪声当证据", 1)
    add_body(
        doc,
        f"本版从 64 条 DOI/数据资源种子开始，逐条以 Crossref 或 DataCite 回填题名、作者、年份、期刊/会议与 DOI；"
        f"其中 1 条题名显示为太阳能飞机姿态研究，已剔除；DroneRFa 的 DOI 不在这两个索引中，已改由《电子与信息学报》官网核验。"
        f"最终证据库共 {len(references)} 条：{len(empirical)} 条直接实证/方法论文、{len(datasets)} 条数据资源或数据集论文记录、{len(reviews)} 条综述。"
        "数据资源记录可能与其配套论文成对出现，因此不把该数字误说成不同数据集的数量。",
    )
    add_callout(
        doc,
        "透明性说明：这是一份“可审计的第一版领域地图”，不是声称已经完成 PRISMA 全库系统综述。"
        "全部条目已完成来源/DOI 核验；代表性开放全文与综述已做内容阅读；后续选定课题前，应对候选核心论文继续逐篇提取采集设置、划分单位、干扰条件和代码/数据可用性。",
    )
    workflow_rows = [
        ["检索式核心", "(drone OR UAV) AND (RF OR radio frequency OR I/Q OR RF fingerprint OR passive RF) AND (detect* OR classif* OR identif*) AND (deep learning OR CNN OR ResNet OR Transformer)。"],
        ["优先来源", "IEEE Xplore、ACM DL、Web of Science/Scopus、ScienceDirect、SpringerLink；中文补 CNKI/万方。Google Scholar 用于引文追踪，不作唯一题录源。"],
        ["纳入标准", "直接处理无人机/遥控器通信 RF；任务与检测、识别、模式、定位或多机频谱检测有关；有出版社 DOI/官方数据入口；优先真实采集和同行评议。"],
        ["关键审查字段", "链路方向、频段、接收机、样本原子单位、训练/测试划分、真实干扰、低 SNR、未知类、外部/跨域测试、延迟与代码/数据。"],
        ["降级处理", "无实验细节、只转述二手结果、未说明划分粒度、只在同一录制随机切窗测试的条目，不能支撑部署泛化结论。"],
    ]
    add_table(doc, ["步骤", "执行标准"], workflow_rows, [1.35, 5.15], "表 2  建立可复核文献库的最小流程")

    add_heading(doc, "3. 现在的主流做法：不是单一模型，而是四层流水线", 1)
    landscape_rows = [
        ["采集与预处理", "SDR 被动扫描/接收无人机—控制器链路；同步、分帧、带宽/频段选择、归一化。", cite('10.1145/2935620.2935632', '10.3390/s22041453')],
        ["1D 信号学习", "将原始或归一化 I/Q、多通道序列直接送入 1D-CNN/DNN；重点是低延迟和端到端特征学习。", cite('10.1109/iciot48696.2020.9089657', '10.3390/s23094202')],
        ["时频图学习", "FFT/PSD、STFT 或小波形成二维表示，再以 CNN/ResNet 识别；低 SNR、非平稳跳频或重叠信号常走此路。", cite('10.1109/COMSNETS51098.2021.9352891', '10.1109/TCCN.2021.3099114', '10.1109/JRFID.2024.3487303')],
        ["RF 指纹与层级任务", "利用发射机硬件/前导码/非标准波形与深度特征，完成机型、设备、厂家或运行模式的细粒度判定；也常先检测再分类。", cite('10.1109/TVT.2020.3042128', '10.3390/e23121678', '10.1109/JRFID.2022.3157653')],
        ["复杂环境与部署", "把 Wi-Fi、蓝牙、低 SNR、压缩采样、多机信号、低成本 SDR、实时性纳入实验；这类工作比“干净闭集分类”更接近系统问题。", cite('10.1109/ojcoms.2019.2955889', '10.1016/j.eswa.2021.115928', '10.3390/s22083072', '10.1038/s41598-026-48925-1')],
    ]
    add_table(doc, ["层", "主流实现", "代表来源"], landscape_rows, [1.2, 4.1, 1.2], "表 3  被动通信 RF 深度学习的主流技术图谱")
    add_body(
        doc,
        "两篇已阅读全文代表作能说明“主流”和“前沿”的边界：Glüge 等以实部/虚部连续傅里叶变换得到谱图，用 CNN 比较低 SNR 并做低成本 SDR 外场测试；"
        "Basak 等则将两阶段频谱检测+分类与 YOLO-lite 联合检测/分类进行比较，并显式处理同时出现的多信号。"
        "它们都属于通信 RF 证据，且说明输入表示、干扰条件和任务形式同样重要。"
        f" {cite('10.1109/JRFID.2024.3487303', '10.1109/TCCN.2021.3099114')}",
    )

    add_heading(doc, "4. 核心论文地图：可直接开始精读的锚点", 1)
    add_body(
        doc,
        "下表不是按宣称精度排名，而是按问题覆盖面选出的精读锚点。方法描述只概括论文明确题名、摘要或开放全文所支持的内容；"
        "性能数值、横向优劣和可部署性必须回到每篇论文的采集与划分细节核验。",
    )
    core_main = [
        ["奠基", "Nguyen 等，2016\n" + cite('10.1145/2935620.2935632'), "低成本被动 RF 检测可行性", "早期 SDR/RF 侦测锚点。"],
        ["数据与 DL", "Al-Sa’d 等，2019\n" + cite('10.1016/j.future.2019.05.007'), "RF 数据库 + 深度学习检测/识别", "开源数据与 DL 结合的代表性起点。"],
        ["公开数据", "Allahham 等，2019\n" + cite('10.1016/j.dib.2019.104313'), "DroneRF 数据集", "后续研究常用的早期公开基准资源。"],
        ["干扰", "Ezuma 等，2020\n" + cite('10.1109/ojcoms.2019.2955889'), "RF 指纹；Wi-Fi/蓝牙干扰", "把共频背景纳入识别问题。"],
        ["端到端 1D", "Allahham 等，2020\n" + cite('10.1109/iciot48696.2020.9089657'), "多通道 1D-CNN", "原始/序列型 RF 深度学习代表。"],
        ["RF 指纹", "Soltani 等，2020\n" + cite('10.1109/TVT.2020.3042128'), "非标准发射波形指纹", "细粒度发射机身份判定方向。"],
        ["残差网络", "Basak 等，2021\n" + cite('10.1109/COMSNETS51098.2021.9352891'), "RF 指纹 + 深度残差网络", "时频/特征图配合 ResNet 的代表。"],
        ["低 SNR", "Ozturk 等，2021\n" + cite('10.52953/QJGH3217'), "低 SNR CNN 分类", "低信噪比专门任务锚点。"],
        ["模式识别", "Yang 等，2021\n" + cite('10.3390/e23121678'), "检测 + 飞行模式；多通道 DNN", "从机型扩展到状态/模式。"],
        ["多机", "Sazdić-Jotić 等，2022\n" + cite('10.1016/j.eswa.2021.115928'), "单机与多机 RF 深度学习", "多源场景的早期实证。"],
        ["压缩采样", "Mo 等，2022\n" + cite('10.3390/s22083072'), "压缩感知 RF + DL", "降低采集/处理负担的路线。"],
        ["系统化 CNN", "Huynh-The 等，2022\n" + cite('10.1109/access.2022.3172787'), "RF-UAVNet", "面向 RF 监视的卷积网络代表。"],
        ["多信号联合", "Basak 等，2022\n" + cite('10.1109/TCCN.2021.3099114'), "两阶段 vs. YOLO-lite；频谱内多信号", "检测、定位频段、分类的联合问题。"],
        ["端到端", "Alam 等，2023\n" + cite('10.3390/s23094202'), "归一化原始 RF 的端到端网络", "检测、设备/厂家判定与未知 SNR 测试。"],
        ["多尺度时频", "Mandal & Satija，2023\n" + cite('10.1109/LSENS.2023.3289145'), "时频多尺度 CNN", "二维时频特征增强的代表。"],
        ["低 SNR 外场", "Glüge 等，2024\n" + cite('10.1109/JRFID.2024.3487303'), "谱图 CNN；低成本 SDR 实场", "低 SNR、公开数据与实场验证的高价值锚点。"],
    ]
    add_table(doc, ["主题", "论文", "核心方法/任务", "为什么精读"], core_main, [0.75, 1.55, 2.25, 1.95], "表 4  主流路线的核心精读锚点")

    add_heading(doc, "4.1 新兴方向的高价值论文", 2)
    core_emerging = [
        ["复值 IQ", "Prazak 等，2022\n" + cite('10.3390/drones6120374'), "复值 CNN + RF 指纹", "保留 I/Q 复数结构的比较对象。"],
        ["Transformer", "Yan 等，2023\n" + cite('10.3390/s23229098'), "SignalFormer 混合 Transformer", "新兴架构，不应替代主流统计。"],
        ["复杂电磁", "Zhang 等，2023\n" + cite('10.1109/JSEN.2023.3242985'), "复杂电磁环境深度分类", "真实背景与复杂干扰问题。"],
        ["开放集", "Yu 等，2024\n" + cite('10.1109/TIFS.2024.3463535'), "RF 信号语义开放集识别", "未知机型拒识的直接锚点。"],
        ["数据与指纹", "Zhao 等，2024\n" + cite('10.1109/MILCOM61039.2024.10773837'), "UAVSig 数据集 + 深度学习", "更近年的检测与指纹资源。"],
        ["多机谱图", "Yu 等，2025\n" + cite('10.1109/TCCN.2025.3537100'), "SMNet 多机检测/分类", "监控飞控信号的谱图多目标方向。"],
        ["真实干扰", "Rahman 等，2025\n" + cite('10.1016/j.aej.2025.02.031'), "级联学习 + 无线干扰", "干扰感知分类的近期工作。"],
        ["跨域", "论文，2026\n" + cite('10.1109/TCCN.2025.3561305'), "域不变对抗学习 + 流形正则", "跨环境/域外 RFFI 的直接方向。"],
        ["少样本", "论文，2026\n" + cite('10.1109/TVT.2026.3696747'), "脉冲少样本 RF 识别", "新型号低标注接入方向。"],
        ["边缘", "Zhu 等，2026\n" + cite('10.1109/TCCN.2025.3598245'), "固定分类器轻量网络", "算力/存储受限部署方向。"],
        ["实时系统", "Saber 等，2026\n" + cite('10.1038/s41598-026-48925-1'), "SDR 实时 RF 特征化", "从模型到 SDR 系统落地的近期证据。"],
    ]
    add_table(doc, ["方向", "论文", "核心方法/任务", "研究价值"], core_emerging, [0.75, 1.55, 2.25, 1.95], "表 5  新兴/增长方向的优先精读锚点")

    add_heading(doc, "5. 数据资源：必须与算法论文分开判断", 1)
    add_body(
        doc,
        "数据集选择不应预设为沿用任何项目历史数据。以下资源仅是可重新审查的候选；在正式实验前，应重新下载/取得合法访问、核验许可证、"
        "检查原始录制的 session、设备、地点、日期、接收机、标签和文件完整性。",
    )
    dataset_rows = [
        ["DroneRF", "早期公开的无人机 RF 时间序列；文献说明含 3 架无人机与 4 种运行模式。", "检测、分类、模式", "数据集论文\n" + cite('10.1016/j.dib.2019.104313')],
        ["Drone RC RF Dataset", "被动监听遥控器到无人机的控制 RF；DataPort 条目应以实际取得版本的说明为准。", "控制器/发射机识别", "官方数据入口\n" + cite('10.21227/pdg2-v660')],
        ["DroneDetect", "原始 I/Q；公开关联材料描述多机型/运行状态和 Wi-Fi、蓝牙等受控干扰条件。", "检测、分类、干扰", "官方数据入口\n" + cite('10.21227/5JJJ-1M32')],
        ["CardRF", "户外 UAV/UAS RF；标题明确包含蓝牙与 Wi-Fi 信号。", "干扰下检测/分类", "官方数据入口\n" + cite('10.21227/1xp7-ge95')],
        ["VTI_DroneSET_FFT", "以 FFT 表示发布的数据资源。", "频谱/时频分类", "Mendeley Data\n" + cite('10.17632/s6tgnnp5n2.3')],
        ["KU Leuven Drone RF", "官方数据资源入口；需逐项核验采集协议与许可。", "基准/外部测试候选", "官方数据入口\n" + cite('10.48804/HZRVNZ')],
        ["DroneRFa", "覆盖宽带 RF 检测与机型识别的数据集论文；官方入口要求手机/邮箱验证码。", "检测/数据覆盖讨论", "期刊官网核验\n" + cite('10.11999/JEIT230570')],
        ["UAVSig", "面向 RF 检测与指纹的论文/数据资源配对记录；官方提示 B205mini 原始 I/Q 有随机缺段。", "检测、RF 指纹", "论文与数据入口\n" + cite('10.1109/MILCOM61039.2024.10773837', '10.25346/S6/LVRRAE')],
        ["DroneRFb-Spectra", "无人机识别 RF 谱图数据资源。", "谱图识别", "IEEE DataPort DOI\n" + cite('10.21227/wv7h-sv64')],
    ]
    add_table(doc, ["候选资源", "信号/采集说明（仅已核验信息）", "适合任务", "来源"], dataset_rows, [1.15, 2.65, 1.25, 1.45], "表 6  可重新评估的无人机 RF 数据资源")
    add_body(
        doc,
        "值得特别记录的限制：公开数据不等于可直接泛化。近期稳健性研究指出，许多公开 RF 数据来自单一实验设置，"
        "训练和测试可能同源；只增加切片数量或人为加噪并不能替代不同地点、背景、接收机和时间的实测条件。"
        f" {cite('10.3390/s24227339')}",
    )
    add_body(
        doc,
        "补充准入提醒：有官方页面不等于匿名一键开放；例如 DroneRFa 需验证码，部分 DataPort 资源还需重新确认下载权限和版本。"
        "UAVSig 的随机缺段也必须在预处理和数据卡中显式记录，不能被模型当成设备特征。",
    )

    add_heading(doc, "6. 证据强度与常见评测陷阱", 1)
    risks = [
        ["相邻窗口泄漏", "同一原始录制切出的重叠/相邻窗口随机分到训练和测试，会把录制条件而非无人机特征带入结果。", "以原始录制 session 为最小分组单元；训练/验证/测试不共享 session。"],
        ["单场景泛化", "同一地点、同一 SDR、同一日期、同一控制器/固件的高分，不能自动推出跨场地或跨接收机有效。", "做 leave-one-site/day/receiver/controller/model-out；报告每个留出域。"],
        ["仅 AWGN", "真实 Wi-Fi、蓝牙、跳频、同频占用、频偏和设备伪影与白噪声不同。", "将真实背景录音与定义明确的 SNR 扰动分开报告。"],
        ["闭集强制归类", "未知型号会被迫归到已知类，accuracy 掩盖误报风险。", "设置未知设备/协议留出；报告 AUROC、OSCR、未知类 FPR 与校准。"],
        ["单机假设", "频谱中可能同时有多架无人机或非无人机发射源。", "采用多标签/时频目标检测；按机数、功率比、重叠比例分层评估。"],
        ["只报准确率", "类别不均衡、检测漏报与虚警、延迟和部署代价可能被隐藏。", "检测报 Pd、Pfa、PR-AUC；分类报 macro-F1/BA；另报延迟、内存、能耗。"],
    ]
    add_table(doc, ["风险", "为什么会误导", "从零实验的防护"], risks, [1.15, 2.85, 2.5], "表 7  无人机通信 RF 评测的关键防线")
    add_body(
        doc,
        "这张表不是对任何历史实验的批评，而是新项目的预注册式质量门槛。它与近年针对 RF UAV 检测器稳健性的专门讨论一致。"
        f" {cite('10.3390/s24227339', '10.1109/ojcoms.2019.2955889', '10.1109/TCCN.2021.3099114')}",
    )

    add_heading(doc, "7. 可做创新点：把问题、数据条件和评测绑定", 1)
    innovation_rows = [
        ["A. 跨域 + 开放集", "已知机型分类之外，判断“这是否为未见型号/协议/固件”，并在新地点、新接收机、新日期下保持可信。", "至少 3 个独立域；已知/未知设备划分；跨域留出。", "OSCR、AUROC、未知类 FPR、宏 F1、置信度校准。", cite('10.1109/TIFS.2024.3463535', '10.1109/TCCN.2025.3561305')],
        ["B. 多机重叠", "把谱图中的多个 RF 发射源视为多标签/目标检测，而不是先假定只存在一架无人机。", "真实或可追溯合成的多源混合；功率比、SNR、机数标签。", "mAP/多标签 F1、按功率比和重叠率分层。", cite('10.1109/TCCN.2021.3099114', '10.1109/TCCN.2025.3537100')],
        ["C. 真实干扰与低 SNR", "分离“真实无线背景影响”和“人为加噪影响”，同时做外场验证。", "Wi-Fi/蓝牙/拥塞背景、距离/方向变化、现场无机样本。", "Pd/Pfa、BA、宏 F1、SNR—性能曲线与置信区间。", cite('10.1109/ojcoms.2019.2955889', '10.1109/JRFID.2024.3487303')],
        ["D. 新型号少样本接入", "以少量新机型样本适配，而不是要求完整重新训练。", "按设备级留出；1/5/10-shot 支持集；多随机种子。", "每 shot 的宏 F1、开放集误报、适配时间。", cite('10.1109/TVT.2026.3696747')],
        ["E. 复数 I/Q 与接收机鲁棒", "检验相位/复数结构是否能跨接收机、同步误差和频偏带来可复现收益。", "同一设备由多 SDR/天线/通道录制；幅度谱与复数模型同协议对比。", "跨接收机留出、校准误差、带宽/频偏敏感性。", cite('10.3390/drones6120374', '10.1109/TCCN.2025.3561305')],
        ["F. 可信边缘系统", "同时报告模型、SDR 输入、吞吐、延迟和不确定性，而不是只给离线准确率。", "可复现实机软硬件清单；端侧日志；代表场景。", "端到端延迟、吞吐、内存、功耗、错误审计。", cite('10.1109/TCCN.2025.3598245', '10.1038/s41598-026-48925-1')],
    ]
    add_table(doc, ["候选命题", "研究问题", "最小数据条件", "必要指标", "锚点"], innovation_rows, [0.85, 1.7, 1.75, 1.2, 1.0], "表 8  可发表的 RF 创新命题及其可验证条件")
    add_body(
        doc,
        "若可开展新的多地点/多接收机采集，优先考虑 A（跨域+开放集），因为它把“未知目标”和“环境变化”同时纳入了真实部署问题。"
        "若短期只能使用公开数据，优先考虑 C 或 D，但必须把真实干扰、设备级留出和重复种子写入实验协议。"
        "这是基于当前文献空白的选题建议，不来自既有项目结果。",
    )

    add_heading(doc, "8. 从零实施路线：研究、数据与复现分三条线并行", 1)
    plan_rows = [
        ["第 0 阶段：锁定问题", "在 A–F 中只选 1 个主命题；明确“检测/分类/拒识/定位”边界、威胁模型和不适用范围。", "一页研究问题与预注册式指标清单。"],
        ["第 1 阶段：文献深读", "对表 4–5 的锚点逐篇提取信号链路、采样、接收机、数据拆分、干扰、指标、代码与局限。", "可筛选文献矩阵 + 每篇证据卡。"],
        ["第 2 阶段：数据审计/采集", "不复用结论；重新验证每个候选数据资源。若自采，建立 session、地点、日期、接收机、天线、设备、固件和环境日志。", "原始录制清单、数据卡、许可与标签规范。"],
        ["第 3 阶段：受控基线", "至少比较经典谱图 2D-CNN/ResNet、原始 I/Q 1D-CNN 和一个非深度学习或简单频谱基线；统一数据拆分。", "可复现训练脚本、种子、模型卡和计算成本。"],
        ["第 4 阶段：严苛评估", "in-domain 组隔离、跨域留出、真实干扰/低 SNR、未知类或多机条件按主命题选择。", "主指标、置信区间、错误案例和统计对比。"],
        ["第 5 阶段：交付", "发布 DOI/BibTeX、数据可得性、预处理、划分 manifest、训练配置、权重和部署日志。", "可审计论文附录与可复现实验包。"],
    ]
    add_table(doc, ["阶段", "行动", "完成判据"], plan_rows, [1.45, 3.55, 1.5], "表 9  新项目实施路线（不依赖历史试验）")

    add_heading(doc, "9. 快速收集、整理与阅读文献需要什么 skill", 1)
    skill_rows = [
        ["PDF", "抽取全文、表格、图注与实验设置；对关键论文核验采样率、切片、划分、干扰和结论边界。", "必需；已用于代表性论文的全文复核。"],
        ["Spreadsheets", "维护筛选矩阵：DOI、任务、链路、频段、数据集、划分单位、干扰、指标、代码与风险标签。", "建议在进入逐篇精读时使用。"],
        ["Documents", "生成并渲染校验中文 Word 综述、数据卡和研究计划。", "本报告使用。"],
        ["skill-installer", "从 GitHub 安装确有必要的工作流 skill；安装前查仓库、维护状态、许可证和权限。", "当前不必安装。"],
        ["可选 Zotero 集成", "用于个人文献库同步、RIS/BibTeX、PDF 附件和引用管理。", "只有需要与现有 Zotero 库同步时再启用。"],
    ]
    add_table(doc, ["能力", "在本项目中的用途", "结论"], skill_rows, [1.25, 3.75, 1.5], "表 10  文献工作流能力选择")
    add_body(
        doc,
        "因此，不建议为了“快”先从 GitHub 下载来源不明的批量爬取或付费墙绕过脚本。当前已具备 DOI 核验、PDF 阅读、表格管理和 Word 输出能力；"
        "若以后确需与个人 Zotero 库同步，再选择受信任的官方集成。",
    )

    add_heading(doc, "10. 结语：下一步该做什么", 1)
    add_callout(
        doc,
        "建议的下一动作不是立即训练模型，而是先选定一个可验证命题，并完成数据卡与分组拆分设计。"
        "这会决定应采什么 RF、需要哪些干扰/域、什么算未知类，以及哪些指标才能证明创新。",
    )
    add_body(
        doc,
        "本报告已给出可开始阅读的核心论文、数据资源候选和六条创新路线。下一轮工作应由研究问题驱动："
        "从表 8 选主命题 → 按表 9 建数据与评测协议 → 用表 4 的主流方法做可复现对照。"
        "全程不使用任何此前本地探索的结果作为证据或方向依据。",
    )

    doc.add_page_break()
    add_heading(doc, "附录 A. 核验参考文献（63 条）", 1)
    add_body(
        doc,
        "条目按“实证/方法—数据资源—综述”分组。62 条由 Crossref/DataCite 元数据回填；DroneRFa 条目由期刊官网单独核验。"
        "引用号与正文一致。出版社页和 DOI 是后续精读、导入 Zotero 或导出 BibTeX 的稳定入口。",
        color=MUTED,
        italic=True,
        size=9.5,
    )

    groups = [
        ("A. 直接实证与方法论文", empirical),
        ("B. 数据资源与数据集论文", datasets),
        ("C. 综述与领域地图", reviews),
    ]
    for heading, entries in groups:
        add_heading(doc, heading, 2)
        for row in entries:
            p = doc.add_paragraph(style="Normal")
            p.paragraph_format.left_indent = Inches(0.27)
            p.paragraph_format.first_line_indent = Inches(-0.27)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.08
            r = p.add_run(ref_string(row))
            set_run_font(r, size=9.2, color="000000")

    doc.core_properties.title = "无人机射频信号（RF）深度学习检测与识别：从零文献综述"
    doc.core_properties.subject = "被动通信 RF 的深度学习无人机检测与识别文献证据库"
    doc.core_properties.author = "Codex"
    doc.core_properties.comments = "Fresh literature synthesis; excludes historical local experiments."
    doc.save(OUTPUT_DOCX)


if __name__ == "__main__":
    build_document()
    print(OUTPUT_DOCX)
