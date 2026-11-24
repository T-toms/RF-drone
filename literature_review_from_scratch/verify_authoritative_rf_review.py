"""Structural QA fallback when the DOCX renderer has no LibreOffice binary."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


ROOT = Path(__file__).resolve().parent
DOCX = ROOT / "无人机通信链路RF深度学习_高权威文献综述_2026-08.docx"
BIB = ROOT / "zotero_无人机通信链路RF高权威文献_34篇.bib"
MANIFEST = ROOT / "zotero_无人机通信链路RF高权威文献_34篇.csv"


def all_text(document: Document) -> str:
    chunks = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                chunks.extend(paragraph.text for paragraph in cell.paragraphs)
    return "\n".join(chunks)


def table_ok(table) -> tuple[bool, bool, bool]:
    properties = table._tbl.tblPr
    width = properties.find(qn("w:tblW"))
    indent = properties.find(qn("w:tblInd"))
    layout = properties.find(qn("w:tblLayout"))
    geometry_ok = width is not None and indent is not None and table._tbl.tblGrid is not None
    fixed_ok = layout is not None and layout.get(qn("w:type")) == "fixed"
    rows_ok = True
    for row in table.rows:
        row_properties = row._tr.get_or_add_trPr()
        rows_ok &= row_properties.find(qn("w:cantSplit")) is not None
        for cell in row.cells:
            cell_width = cell._tc.get_or_add_tcPr().find(qn("w:tcW"))
            rows_ok &= cell_width is not None
    return geometry_ok, fixed_ok, rows_ok


def main() -> None:
    document = Document(str(DOCX))
    text = all_text(document)
    refs = set(re.findall(r"(?m)^\[(H\d{2})\] ", text))
    codes = set(re.findall(r"\[?(H\d{2})\]?", text))
    geometries = [table_ok(table) for table in document.tables]
    section = document.sections[0]
    core_properties = document.core_properties
    bib_entries = re.findall(r"(?m)^@(?:article|inproceedings)\{", BIB.read_text(encoding="utf-8"))
    manifest_entries = MANIFEST.read_text(encoding="utf-8-sig").count("\n") - 1
    checks = {
        "docx_exists": DOCX.exists(),
        "docx_bytes": DOCX.stat().st_size,
        "paragraphs": len(document.paragraphs),
        "tables": len(document.tables),
        "reference_entries": len(refs),
        "all_reference_codes_present_in_tables_or_text": sorted(refs - codes) == [],
        "missing_bibliography_entries_for_codes": sorted(codes - refs),
        "fixed_table_geometry": all(geometry and fixed for geometry, fixed, _ in geometries),
        "all_table_rows_protected": all(rows for _, _, rows in geometries),
        "letter_page": round(section.page_width / 914400, 2) == 8.5 and round(section.page_height / 914400, 2) == 11.0,
        "one_inch_margins": all(round(value / 914400, 2) == 1.0 for value in (section.top_margin, section.bottom_margin, section.left_margin, section.right_margin)),
        "title": core_properties.title,
        "placeholder_hits": re.findall(r"(?i)TODO|TBD|lorem ipsum", text),
        "bibtex_entries": len(bib_entries),
        "manifest_entries": manifest_entries,
        "source_scope_guard_present": "不读取、不引用，也不反推任何本地既往实验" in text,
    }
    print(checks)


if __name__ == "__main__":
    main()
