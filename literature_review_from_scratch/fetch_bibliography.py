"""Fetch reproducible bibliographic metadata for the curated UAV RF corpus.

The CSV seed list is deliberately curated before this script is run.  The
script retrieves metadata only from Crossref and, when needed, DataCite.  It
does not download publisher PDFs or access paywalled content.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
SEED_PATH = ROOT / "seed_dois.csv"
CSV_OUTPUT = ROOT / "references_metadata.csv"
JSON_OUTPUT = ROOT / "references_metadata.json"
REPORT_OUTPUT = ROOT / "metadata_fetch_report.json"
USER_AGENT = "Codex-UAV-RF-literature-review/1.0 (bibliographic metadata only)"


def get_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_error: Exception | None = None
    for delay_seconds in (0, 2, 6):
        if delay_seconds:
            time.sleep(delay_seconds)
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
    raise RuntimeError(f"Metadata request failed for {url}: {last_error}")


def iso_date(parts: Any) -> str:
    if not parts:
        return ""
    date_parts = parts[0] if isinstance(parts, list) else parts
    values = [str(value).zfill(2) for value in date_parts]
    if not values:
        return ""
    values[0] = str(date_parts[0])
    return "-".join(values)


def crossref_record(doi: str) -> dict[str, Any]:
    result = get_json(f"https://api.crossref.org/works/{quote(doi, safe='')}")
    return result["message"]


def datacite_record(doi: str) -> dict[str, Any]:
    result = get_json(f"https://api.datacite.org/dois/{quote(doi, safe='')}")
    return result["data"]["attributes"]


def crossref_to_row(seed: dict[str, str], message: dict[str, Any]) -> dict[str, Any]:
    authors = message.get("author", [])
    author_text = "; ".join(
        " ".join(part for part in (author.get("family", ""), author.get("given", "")) if part).strip()
        or author.get("name", "")
        for author in authors
    )
    issued = message.get("published-print") or message.get("published-online") or message.get("issued", {})
    return {
        **seed,
        "metadata_source": "Crossref",
        "title": " ".join(message.get("title", [])).strip(),
        "authors": author_text,
        "year": (issued.get("date-parts") or [[""]])[0][0],
        "date": iso_date(issued.get("date-parts")),
        "venue": " ".join(message.get("container-title", [])).strip(),
        "volume": message.get("volume", ""),
        "issue": message.get("issue", ""),
        "pages_or_article": message.get("page", message.get("article-number", "")),
        "publisher": message.get("publisher", ""),
        "reference_count": message.get("reference-count", ""),
        "cited_by_crossref": message.get("is-referenced-by-count", ""),
        "url": message.get("URL", f"https://doi.org/{seed['doi']}"),
        "type": message.get("type", ""),
    }


def datacite_to_row(seed: dict[str, str], attributes: dict[str, Any]) -> dict[str, Any]:
    author_text = "; ".join(
        creator.get("name", "") for creator in attributes.get("creators", []) if creator.get("name")
    )
    return {
        **seed,
        "metadata_source": "DataCite",
        "title": attributes.get("titles", [{}])[0].get("title", ""),
        "authors": author_text,
        "year": attributes.get("publicationYear", ""),
        "date": attributes.get("published", ""),
        "venue": attributes.get("container", {}).get("title", ""),
        "volume": "",
        "issue": "",
        "pages_or_article": "",
        "publisher": attributes.get("publisher", ""),
        "reference_count": "",
        "cited_by_crossref": "",
        "url": attributes.get("url", f"https://doi.org/{seed['doi']}"),
        "type": attributes.get("types", {}).get("resourceTypeGeneral", ""),
    }


def main() -> int:
    with SEED_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        seeds = list(csv.DictReader(handle))

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for index, seed in enumerate(seeds, start=1):
        doi = seed["doi"].strip()
        try:
            rows.append(crossref_to_row(seed, crossref_record(doi)))
        except RuntimeError as crossref_error:
            try:
                rows.append(datacite_to_row(seed, datacite_record(doi)))
            except RuntimeError as datacite_error:
                errors.append({"doi": doi, "crossref": str(crossref_error), "datacite": str(datacite_error)})
        print(f"[{index}/{len(seeds)}] {doi}")

    fieldnames = [
        "doi", "track", "evidence_role", "metadata_source", "title", "authors", "year", "date",
        "venue", "volume", "issue", "pages_or_article", "publisher", "reference_count",
        "cited_by_crossref", "url", "type",
    ]
    with CSV_OUTPUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with JSON_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
    report = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_count": len(seeds),
        "verified_count": len(rows),
        "failed_count": len(errors),
        "failures": errors,
    }
    with REPORT_OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({key: report[key] for key in ("seed_count", "verified_count", "failed_count")}, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
