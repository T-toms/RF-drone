"""Convert 个人研究报告_20261011.md to .docx (headings, paragraphs, bullets, tables, images, **bold**)."""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

HERE = Path(__file__).resolve().parent
SRC = HERE / "个人研究报告_20261011.md"
OUT = HERE / "个人研究报告_20261011.docx"
FONT = "宋体"
HEAD_FONT = "黑体"


def set_font(run, name=FONT, size=None, bold=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def add_inline(par, text, size=10.5):
    text = text.replace("`", "")
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part:
            continue
        bold = part.startswith("**") and part.endswith("**")
        set_font(par.add_run(part[2:-2] if bold else part), size=size, bold=bold)


def table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, r in enumerate(cells):
        for j, c in enumerate(r):
            p = t.cell(i, j).paragraphs[0]
            add_inline(p, f"**{c}**" if i == 0 and not c.startswith("**") else c, size=9)
    doc.add_paragraph()


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Cm(2.2)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    lines = SRC.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            table(doc, block)
            continue
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
        if m:
            from PIL import Image
            w, h = Image.open(HERE / m.group(1)).size
            if h / w * 13.5 > 10:
                doc.add_picture(str(HERE / m.group(1)), height=Cm(10))
            else:
                doc.add_picture(str(HERE / m.group(1)), width=Cm(13.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            set_font(p.add_run(line[2:]), HEAD_FONT, 16, True)
        elif line.startswith("## "):
            p = doc.add_paragraph()
            set_font(p.add_run(line[3:]), HEAD_FONT, 14, True)
        elif line.startswith("### "):
            p = doc.add_paragraph()
            set_font(p.add_run(line[4:]), HEAD_FONT, 12, True)
        elif line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, line[2:])
        elif line.strip():
            p = doc.add_paragraph()
            if re.fullmatch(r"\d{4} 年 .*日", line.strip()) or re.match(r"图 \d+　", line):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_inline(p, line, size=9)
            else:
                p.paragraph_format.first_line_indent = Pt(21)
                add_inline(p, line)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.25
        i += 1
    doc.save(OUT)
    print("written", OUT)


if __name__ == "__main__":
    sys.exit(main())
