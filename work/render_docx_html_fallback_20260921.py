from __future__ import annotations

import html
import zipfile
import base64
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(r"C:\Users\Administrator\Desktop\数据集处理")
DOCX = ROOT / "reports" / "DroneRFa项目进展汇报_结果版_20260921.docx"
OUT = ROOT / "work" / "render_progress_report_20260921" / "docx_fallback.html"
IMAGE_DATA = {}


def iter_block_items(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def has_page_break(paragraph: Paragraph) -> bool:
    return bool(paragraph._p.xpath('.//w:br[@w:type="page"]'))


def run_css(run) -> str:
    css = []
    if run.font.size:
        css.append(f"font-size:{run.font.size.pt:.2f}pt")
    if run.bold:
        css.append("font-weight:700")
    color = run.font.color.rgb
    if color:
        css.append(f"color:#{color}")
    return ";".join(css)


def paragraph_html(p: Paragraph, in_cell=False) -> str:
    blips = p._p.xpath(".//a:blip")
    if blips:
        relationship_id = blips[0].get(qn("r:embed"))
        source = IMAGE_DATA.get(relationship_id)
        if source:
            return f'<p class="figure"><img src="{source}" /></p>'
    text = p.text
    if not text and not p.runs:
        return '<p class="blank">&nbsp;</p>'
    style_name = p.style.name if p.style else ""
    prefix = "• " if style_name == "List Bullet" else ""
    fragments = []
    if prefix:
        fragments.append(html.escape(prefix))
    for run in p.runs:
        fragments.append(f'<span style="{run_css(run)}">{html.escape(run.text)}</span>')
    align = {1: "center", 2: "right", 3: "justify"}.get(p.alignment, "left")
    left = p.paragraph_format.left_indent.cm if p.paragraph_format.left_indent else 0
    before = p.paragraph_format.space_before.pt if p.paragraph_format.space_before else 0
    after = p.paragraph_format.space_after.pt if p.paragraph_format.space_after else 0
    size = "8.2pt" if in_cell else "9.5pt"
    return (
        f'<p style="text-align:{align};margin:{before:.1f}pt 0 {after:.1f}pt {left:.2f}cm;'
        f'font-size:{size};line-height:1.08">{"".join(fragments)}</p>'
    )


def cell_fill(cell) -> str:
    shd = cell._tc.get_or_add_tcPr().find(qn("w:shd"))
    if shd is not None:
        fill = shd.get(qn("w:fill"))
        if fill and fill != "auto":
            return f"#{fill}"
    return "transparent"


def table_html(table: Table) -> str:
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            body = "".join(paragraph_html(p, in_cell=True) for p in cell.paragraphs)
            cells.append(f'<td style="background:{cell_fill(cell)}">{body}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')
    return f'<table>{"".join(rows)}</table>'


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document(DOCX)
    for relationship_id, relationship in doc.part.rels.items():
        if "image" not in relationship.reltype:
            continue
        part = relationship.target_part
        payload = base64.b64encode(part.blob).decode("ascii")
        IMAGE_DATA[relationship_id] = f"data:{part.content_type};base64,{payload}"
    pages = [[]]
    for block in iter_block_items(doc):
        if isinstance(block, Paragraph) and has_page_break(block):
            pages.append([])
            continue
        pages[-1].append(paragraph_html(block) if isinstance(block, Paragraph) else table_html(block))

    with zipfile.ZipFile(DOCX) as zf:
        document_xml = zf.read("word/document.xml").decode("utf-8")
    explicit_breaks = document_xml.count('w:type="page"')

    page_html = []
    for index, blocks in enumerate(pages, 1):
        page_html.append(
            '<section class="page">'
            f'<div class="content" id="content-{index}">{"".join(blocks)}</div>'
            f'<footer>DroneRFa 项目阶段进度报告 · 2026年9月21日 · {index}</footer>'
            '</section>'
        )

    full = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>DOCX layout QA</title>
<style>
@page {{ size: A4; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: #dfe4e8; font-family: "Microsoft YaHei", sans-serif; color: #1f2933; }}
.page {{ position: relative; width: 210mm; height: 297mm; padding: 13.5mm 15.5mm 12.5mm; background: white; page-break-after: always; overflow: hidden; }}
.page:last-child {{ page-break-after: auto; }}
.content {{ height: 258mm; overflow: hidden; }}
.overflow::after {{ content: "内容超出本页"; position: absolute; right: 5mm; bottom: 7mm; color: #c00; font-weight: 700; }}
p {{ padding: 0; }} p.blank {{ margin: 0; height: 2pt; }}
.figure {{ margin: 3pt 0 2pt 0; text-align: center; }}
.figure img {{ max-width: 100%; max-height: 115mm; object-fit: contain; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; margin: 2.5pt 0; }}
td {{ border: 0.6pt solid #d6dce1; padding: 2.4mm 2.2mm; vertical-align: middle; overflow-wrap: anywhere; }}
td p {{ margin-top: 0 !important; margin-bottom: 0 !important; }}
footer {{ position: absolute; left: 15.5mm; right: 15.5mm; bottom: 5.5mm; text-align: center; font-size: 8pt; color: #5c6670; }}
</style></head><body>{''.join(page_html)}
<script>
for (const el of document.querySelectorAll('.content')) {{
  if (el.scrollHeight > el.clientHeight + 2) el.parentElement.classList.add('overflow');
}}
</script></body></html>'''
    OUT.write_text(full, encoding="utf-8")
    print(f"pages={len(pages)} explicit_breaks={explicit_breaks} html={OUT}")


if __name__ == "__main__":
    main()
