"""Structural QA for the generated DOCX when a native renderer is unavailable."""

from pathlib import Path
import re

from docx import Document
from docx.oxml.ns import qn


DOCX_PATH = Path(__file__).with_name("无人机射频RF深度学习检测与识别_从零文献综述_2026-08.docx")


def main() -> None:
    document = Document(str(DOCX_PATH))
    document_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    reference_entries = set(re.findall(r"(?m)^\[(R\d{2})\] ", document_text))
    citations = set(re.findall(r"\[(R\d{2})\]", document_text))

    widths_ok = True
    fixed_layout_ok = True
    all_rows_protected = True
    table_shapes = []
    for table in document.tables:
        table_shapes.append((len(table.rows), len(table.columns)))
        properties = table._tbl.tblPr
        width = properties.find(qn("w:tblW"))
        layout = properties.find(qn("w:tblLayout"))
        widths_ok &= width is not None and table._tbl.tblGrid is not None
        fixed_layout_ok &= layout is not None and layout.get(qn("w:type")) == "fixed"
        for row in table.rows:
            row_properties = row._tr.get_or_add_trPr()
            all_rows_protected &= row_properties.find(qn("w:cantSplit")) is not None

    checks = {
        "exists": DOCX_PATH.exists(),
        "bytes": DOCX_PATH.stat().st_size,
        "paragraphs": len(document.paragraphs),
        "tables": len(document.tables),
        "table_shapes": table_shapes,
        "reference_entries": len(reference_entries),
        "unique_citations": len(citations),
        "missing_reference_for_citation": sorted(citations - reference_entries),
        "unreferenced_bibliography_entries": len(reference_entries - citations),
        "all_tables_have_fixed_geometry": widths_ok and fixed_layout_ok,
        "all_rows_protected_from_page_split": all_rows_protected,
        "placeholder_hits": re.findall(r"(?i)TODO|TBD|lorem ipsum", document_text),
    }
    print(checks)


if __name__ == "__main__":
    main()
