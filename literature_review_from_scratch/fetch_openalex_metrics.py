"""Fetch current, open bibliometric signals for the RF-UAV literature pool.

The output intentionally labels OpenAlex citation counts as an open, time-stamped
proxy.  It does not label them as Web of Science, Scopus, or Google Scholar
counts, and it does not infer journal impact factors.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "references_metadata.csv"
OUTPUT = ROOT / "openalex_bibliometrics.json"
EXCLUDED = {"10.1109/aero.2019.8741733"}
MANUAL = {
    "doi": "10.11999/JEIT230570",
    "track": "dataset",
    "title": "DroneRFa: A Large-scale Dataset of Drone Radio Frequency Signals for Detecting Low-altitude Drones",
    "authors": "Yu Ningning; Mao Shengjian; Zhou Chengwei; Sun Guowei; Shi Zhiguo; Chen Jiming",
    "year": "2024",
}
USER_AGENT = "DroneRFLiteratureReview/1.0 (research metadata audit)"


def get_json(url: str, retries: int = 3) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(str(last_error))


def doi_url(doi: str) -> str:
    return "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="")


def clean_doi(value: str) -> str:
    return value.strip().lower().replace("https://doi.org/", "")


def load_rows() -> list[dict[str, str]]:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if clean_doi(row["doi"]) not in EXCLUDED]
    rows.append(MANUAL)
    return rows


def compact_work(raw: dict, fallback: dict[str, str]) -> dict:
    source = ((raw.get("primary_location") or {}).get("source") or {})
    authorships = raw.get("authorships") or []
    authors = [
        {
            "id": (authorship.get("author") or {}).get("id"),
            "display_name": (authorship.get("author") or {}).get("display_name"),
            "position": authorship.get("author_position"),
        }
        for authorship in authorships
    ]
    return {
        "doi": clean_doi(fallback["doi"]),
        "track": fallback.get("track"),
        "title": raw.get("display_name") or fallback.get("title"),
        "year": raw.get("publication_year") or fallback.get("year"),
        "cited_by_count_openalex": raw.get("cited_by_count"),
        "publication_date": raw.get("publication_date"),
        "type": raw.get("type"),
        "is_retracted": raw.get("is_retracted"),
        "source": {
            "id": source.get("id"),
            "display_name": source.get("display_name"),
            "issn_l": source.get("issn_l"),
            "type": source.get("type"),
            "is_in_doaj": source.get("is_in_doaj"),
        },
        "authors": authors,
        "openalex_id": raw.get("id"),
        "openalex_url": raw.get("id"),
    }


def compact_author(raw: dict) -> dict:
    summary = raw.get("summary_stats") or {}
    return {
        "id": raw.get("id"),
        "display_name": raw.get("display_name"),
        "works_count": raw.get("works_count"),
        "cited_by_count_openalex": raw.get("cited_by_count"),
        "h_index_openalex": summary.get("h_index"),
        "i10_index_openalex": summary.get("i10_index"),
        "last_known_institutions": [
            {
                "display_name": (inst.get("institution") or {}).get("display_name"),
                "country_code": (inst.get("institution") or {}).get("country_code"),
            }
            for inst in (raw.get("last_known_institutions") or [])
        ],
    }


def main() -> None:
    rows = load_rows()
    works: list[dict] = []
    failures: list[dict] = []
    for index, row in enumerate(rows, start=1):
        doi = clean_doi(row["doi"])
        print(f"[{index}/{len(rows)}] {doi}")
        try:
            works.append(compact_work(get_json(doi_url(doi)), row))
        except Exception as exc:  # Keep unresolved entries explicit rather than quietly omitting them.
            failures.append({"doi": doi, "error": str(exc)})
        time.sleep(0.12)

    # Author authority is only queried for the high-citation empirical/review candidate set.
    candidates = [work for work in works if work.get("track") not in {"dataset"}]
    candidates.sort(key=lambda work: work.get("cited_by_count_openalex") or -1, reverse=True)
    author_ids: list[str] = []
    for work in candidates[:35]:
        people = [person for person in work.get("authors", []) if person.get("id")]
        for person in (people[:1] + people[-1:]):
            if person["id"] not in author_ids:
                author_ids.append(person["id"])

    authors: dict[str, dict] = {}
    author_failures: list[dict] = []
    for index, author_id in enumerate(author_ids, start=1):
        print(f"[author {index}/{len(author_ids)}] {author_id}")
        try:
            authors[author_id] = compact_author(get_json(author_id))
        except Exception as exc:
            author_failures.append({"id": author_id, "error": str(exc)})
        time.sleep(0.12)

    payload = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenAlex API",
        "method_note": "Counts are time-stamped open bibliometric signals; do not equate them with Web of Science, Scopus, Google Scholar, Journal Impact Factor, or author identity verification.",
        "works": works,
        "work_failures": failures,
        "authors_for_top_35_non_dataset_candidates": authors,
        "author_failures": author_failures,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "works_found": len(works),
        "work_failures": len(failures),
        "authors_found": len(authors),
        "author_failures": len(author_failures),
        "output": str(OUTPUT),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
