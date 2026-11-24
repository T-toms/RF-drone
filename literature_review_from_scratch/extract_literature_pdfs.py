from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


SOURCE_DIR = Path(r"C:\Users\Administrator\Desktop\文献")
OUTPUT_DIR = Path(r"C:\Users\Administrator\Desktop\数据集处理\literature_review_from_scratch\tmp\pdfs\extracted")


def safe_stem(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", name)
    return value.strip("._")[:140]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    seen_hashes: dict[str, str] = {}

    for pdf_path in sorted(SOURCE_DIR.glob("*.pdf"), key=lambda p: p.name.lower()):
        file_hash = sha256(pdf_path)
        duplicate_of = seen_hashes.get(file_hash)
        if duplicate_of is None:
            seen_hashes[file_hash] = pdf_path.name

        reader = PdfReader(str(pdf_path))
        page_texts: list[str] = []
        extraction_errors: list[str] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text(extraction_mode="layout") or ""
            except Exception as exc:
                extraction_errors.append(f"page {page_number}: {type(exc).__name__}: {exc}")
                try:
                    text = page.extract_text() or ""
                except Exception as fallback_exc:
                    extraction_errors.append(
                        f"page {page_number} fallback: {type(fallback_exc).__name__}: {fallback_exc}"
                    )
                    text = ""
            page_texts.append(f"\n\n===== PAGE {page_number} =====\n\n{text}")

        metadata = reader.metadata or {}
        text_path = OUTPUT_DIR / f"{safe_stem(pdf_path.stem)}.txt"
        text_path.write_text("".join(page_texts), encoding="utf-8")
        records.append(
            {
                "filename": pdf_path.name,
                "path": str(pdf_path),
                "sha256": file_hash,
                "duplicate_of": duplicate_of,
                "size_bytes": pdf_path.stat().st_size,
                "pages": len(reader.pages),
                "text_chars": sum(len(text) for text in page_texts),
                "text_file": str(text_path),
                "metadata": {str(k): str(v) for k, v in metadata.items()},
                "extraction_errors": extraction_errors,
            }
        )

    inventory_path = OUTPUT_DIR / "inventory.json"
    inventory_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
