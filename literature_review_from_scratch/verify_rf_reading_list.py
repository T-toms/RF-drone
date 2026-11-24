"""Structural QA fallback for the journal-first RF reading-list DOCX."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


DOCX = Path(__file__).with_name("无人机通信链路RF_严格期刊与前沿阅读清单_2026-08.docx")


def all_text(document: Document) -> str:
    text = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                text.extend(paragraph.text for paragraph in cell.paragraphs)
    return "\n".join(text)


def main() -> None:
    doc = Document(str(DOCX))
    content = all_text(doc)
    table_checks = []
    for table in doc.tables:
        prop = table._tbl.tblPr
        table_checks.append({
            "geometry": prop.find(qn("w:tblW")) is not None and prop.find(qn("w:tblInd")) is not None and table._tbl.tblGrid is not None,
            "fixed": (lambda node: node is not None and node.get(qn("w:type")) == "fixed")(prop.find(qn("w:tblLayout"))),
            "rows_protected": all(row._tr.get_or_add_trPr().find(qn("w:cantSplit")) is not None for row in table.rows),
        })
    section = doc.sections[0]
    checks = {
        "exists": DOCX.exists(),
        "bytes": DOCX.stat().st_size,
        "paragraphs": len(doc.paragraphs),
        "tables": len(doc.tables),
        "reading_entries": len(set(re.findall(r"(?m)^\[(\d{1,2})\] ", content))),
        "strict_core_no_conference_guard": "A 类全部是期刊论文，无 conference/workshop" in content,
        "conference_last_guard": "会议/workshop 没有期刊 JIF" in content,
        "all_tables_fixed_geometry": all(item["geometry"] and item["fixed"] for item in table_checks),
        "all_rows_protected": all(item["rows_protected"] for item in table_checks),
        "letter_page": round(section.page_width / 914400, 2) == 8.5 and round(section.page_height / 914400, 2) == 11.0,
        "one_inch_margins": all(round(value / 914400, 2) == 1.0 for value in (section.top_margin, section.bottom_margin, section.left_margin, section.right_margin)),
        "no_placeholders": not re.search(r"(?i)TODO|TBD|lorem ipsum", content),
        "scope_guard": "不使用任何既往本地实验" in content,
    }
    print(checks)


if __name__ == "__main__":
    main()
