from pathlib import Path
from zipfile import ZipFile
import re

from docx import Document


path = Path(r"C:\Users\Administrator\Desktop\数据集处理\reports\DroneRFa项目进展汇报_结果版_20260921.docx")
with ZipFile(path) as archive:
    bad_member = archive.testzip()
    document_xml = archive.read("word/document.xml").decode("utf-8")

doc = Document(path)
text = "\n".join(p.text for p in doc.paragraphs)
text += "\n" + "\n".join(
    cell.text
    for table in doc.tables
    for row in table.rows
    for cell in row.cells
)

checks = {
    "exists": path.exists(),
    "bytes": path.stat().st_size,
    "zip_error": bad_member,
    "explicit_page_breaks": document_xml.count('w:type="page"'),
    "paragraphs": len(doc.paragraphs),
    "tables": len(doc.tables),
    "inline_images": len(doc.inline_shapes),
    "paragraph_borders": document_xml.count("<w:pBdr>"),
    "shading_elements": document_xml.count("<w:shd"),
    "document_colors": sorted(set(re.findall(r'<w:color w:val="([^"]+)"', document_xml))),
    "has_todo": "TODO" in text,
    "has_placeholder": "{{" in text or "}}" in text,
    "has_result_boundary": "可能与不同录制文件的功率、设备或环境差异有关" in text,
    "has_next_step": "建立一组不参与参数选择的最终测试数据" in text,
    "text_chars": len(text),
}
print(checks)

assert checks["exists"]
assert checks["bytes"] > 20_000
assert checks["zip_error"] is None
assert checks["explicit_page_breaks"] == 1
assert checks["tables"] == 0
assert checks["inline_images"] == 2
assert checks["paragraph_borders"] == 0
assert checks["shading_elements"] == 0
assert checks["document_colors"] in ([], ["000000"])
assert not checks["has_todo"]
assert not checks["has_placeholder"]
assert checks["has_result_boundary"]
assert checks["has_next_step"]
