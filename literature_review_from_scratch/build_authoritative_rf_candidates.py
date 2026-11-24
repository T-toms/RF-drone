"""Create a time-stamped, auditable shortlist of UAV physical-layer RF + DL papers.

OpenAlex is used only as an openly reproducible citation/author-metric proxy.
It is not Web of Science, Scopus, Google Scholar, JCR Journal Impact Factor,
or proof that an author is an authority in this narrow topic.
"""

from __future__ import annotations

import csv
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "references_metadata.csv"
OUT_CSV = ROOT / "authoritative_rf_dl_ranking.csv"
OUT_JSON = ROOT / "authoritative_rf_dl_openalex_raw.json"
OUT_MD = ROOT / "authoritative_rf_dl_ranking.md"
OUT_ZOTERO_CSV = ROOT / "zotero_authoritative_core_25.csv"
OUT_ZOTERO_DOIS = ROOT / "zotero_authoritative_core_25_dois.txt"
USER_AGENT = "DroneRFLiteratureReview/1.0 (public bibliometric audit)"

# Verified out of scope when checking the DOI against its publisher metadata.
EXCLUDED = {"10.1109/aero.2019.8741733"}

# Exact task scope: passively received drone/controller communication RF at
# the physical layer, with a neural/deep-learning model used by the paper.
# These are not assertions that every paper has a high JIF or a famous author.
CORE_DEEP_RF = {
    "10.1016/j.future.2019.05.007",
    "10.1109/iciot48696.2020.9089657",
    "10.1109/iciot48696.2020.9089489",
    "10.1109/tvt.2020.3042128",
    "10.1109/comsnets51098.2021.9352891",
    "10.3390/aerospace8030079",
    "10.3390/aerospace8070179",
    "10.52953/qjgh3217",
    "10.3390/e23121678",
    "10.1109/tr.2021.3119068",
    "10.1016/j.comnet.2021.108519",
    "10.1016/j.eswa.2021.115928",
    "10.1109/access.2022.3172787",
    "10.3390/s22083072",
    "10.1109/access.2022.3232036",
    "10.3390/drones6120374",
    "10.1109/tccn.2021.3099114",
    "10.3390/s23094202",
    "10.1109/jsen.2023.3242985",
    "10.1109/lsens.2023.3289145",
    "10.1016/j.iot.2023.100879",
    "10.3390/s23229098",
    "10.1109/tim.2023.3306822",
    "10.1109/tifs.2024.3463535",
    "10.1109/jrfid.2024.3487303",
    "10.1109/jsen.2023.3346209",
    "10.1109/tccn.2025.3537100",
    "10.1016/j.aej.2025.02.031",
    "10.1109/tccn.2025.3598245",
    "10.1109/tccn.2025.3561305",
    "10.1109/tvt.2026.3696747",
    "10.1038/s41598-026-48925-1",
}

# These stay in the literature map but do not enter the strict DL core list.
ADJACENT_NON_DL_OR_CONTEXT = {
    "10.1145/2935620.2935632",
    "10.1109/ojcoms.2019.2955889",
    "10.3390/s21061947",
    "10.1109/tvt.2021.3081049",
    "10.1016/j.jestch.2021.06.008",
    "10.1109/jrfid.2022.3157653",
    "10.1016/j.pmcj.2022.101569",
    "10.1109/tccn.2023.3327594",
    "10.3390/s23177650",
    "10.1016/j.fsidi.2025.301948",
}

# Transparent journal/conference quality proxy for tie-breaking, not JIF.
VENUE_TIER = {
    "IEEE Transactions on Information Forensics and Security": 4,
    "IEEE Transactions on Vehicular Technology": 4,
    "IEEE Transactions on Cognitive Communications and Networking": 4,
    "IEEE Transactions on Instrumentation and Measurement": 4,
    "IEEE Transactions on Reliability": 3,
    "IEEE Sensors Journal": 3,
    "Future Generation Computer Systems": 3,
    "Computer Networks": 3,
    "Expert Systems with Applications": 3,
    "IEEE Open Journal of the Communications Society": 3,
    "IEEE Access": 2,
    "IEEE Sensors Letters": 2,
    "IEEE Journal of Radio Frequency Identification": 2,
    "Internet of Things": 2,
    "Alexandria Engineering Journal": 2,
    "Sensors": 2,
    "Drones": 2,
    "Aerospace": 2,
    "Scientific Reports": 2,
}


def clean_doi(value: str) -> str:
    return value.strip().lower().replace("https://doi.org/", "")


def request_json(url: str, tries: int = 2) -> dict:
    error: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                return json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            error = exc
            if attempt + 1 < tries:
                time.sleep(1.25 * (attempt + 1))
    raise RuntimeError(str(error))


def fetch_work(doi: str) -> dict:
    # The DOI endpoint avoids search-based matching and returns OpenAlex's
    # current record for the exact identifier.
    url = "https://api.openalex.org/works/https://doi.org/" + urllib.parse.quote(doi, safe="/")
    return request_json(url)


def fetch_author(author_id: str) -> dict:
    # Work records use public OpenAlex IDs (https://openalex.org/A...), while
    # the API object endpoint is https://api.openalex.org/authors/A....
    short_id = author_id.rstrip("/").rsplit("/", 1)[-1]
    return request_json("https://api.openalex.org/authors/" + urllib.parse.quote(short_id, safe=""))


def load_input() -> list[dict[str, str]]:
    with INPUT.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {**row, "doi": clean_doi(row["doi"])}
            for row in csv.DictReader(handle)
            if clean_doi(row["doi"]) not in EXCLUDED and row["track"] != "dataset"
        ]


def primary_source(work: dict) -> dict:
    return ((work.get("primary_location") or {}).get("source") or {})


def author_summary(raw: dict | None) -> dict:
    if not raw:
        return {"name": "", "works": "", "cited_by": "", "h_index": ""}
    stats = raw.get("summary_stats") or {}
    return {
        "name": raw.get("display_name") or "",
        "works": raw.get("works_count", ""),
        "cited_by": raw.get("cited_by_count", ""),
        "h_index": stats.get("h_index", ""),
    }


def work_authors(work: dict) -> tuple[dict, dict]:
    authorships = work.get("authorships") or []
    if not authorships:
        return {}, {}
    return authorships[0].get("author") or {}, authorships[-1].get("author") or {}


def age_years(year: int | None, now_year: int) -> int:
    return max(1, now_year - int(year or now_year) + 1)


def authority_score(citations: int, age: int, venue_tier: int, author_h: int) -> float:
    # Interpretive sort aid only. It combines paper reach, age-normalized reach,
    # venue proxy, and author record; raw metrics remain in the output.
    return round(
        6 * math.log10(1 + citations)
        + 4 * math.log10(1 + citations / age)
        + 2.5 * venue_tier
        + min(8, math.log10(1 + author_h) * 3),
        2,
    )


def assign_tier(citations: int, age: int, venue_tier: int, is_core: bool) -> str:
    if not is_core:
        return "context"
    # S means established influence within this niche, not a claim of universal
    # top-tier/JIF status. New papers cannot be penalized solely for age.
    if venue_tier >= 3 and citations >= 35:
        return "S"
    if (venue_tier >= 3 and citations >= 12) or (citations >= 35 and age <= 4):
        return "A"
    return "B"


def csv_text(value: object) -> object:
    return "" if value is None else value


def main() -> None:
    retrieved = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = load_input()
    row_by_doi = {row["doi"]: row for row in rows}
    raw_works: dict[str, dict] = {}
    work_failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_work, doi): doi for doi in row_by_doi}
        for future in as_completed(futures):
            doi = futures[future]
            try:
                raw_works[doi] = future.result()
            except Exception as exc:
                work_failures[doi] = str(exc)

    # Query first and last author records for all exact-scope works. First/last
    # are proxies, not a statement of contribution or authority.
    author_ids: set[str] = set()
    for doi in CORE_DEEP_RF:
        work = raw_works.get(doi)
        if work:
            first, last = work_authors(work)
            author_ids.update(x.get("id") for x in (first, last) if x.get("id"))
    raw_authors: dict[str, dict] = {}
    author_failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_author, author_id): author_id for author_id in author_ids}
        for future in as_completed(futures):
            author_id = futures[future]
            try:
                raw_authors[author_id] = future.result()
            except Exception as exc:
                author_failures[author_id] = str(exc)

    now_year = datetime.now(timezone.utc).year
    output: list[dict] = []
    for doi, row in row_by_doi.items():
        work = raw_works.get(doi)
        if not work:
            continue
        source = primary_source(work)
        citations = int(work.get("cited_by_count") or 0)
        year = int(work.get("publication_year") or row.get("year") or now_year)
        age = age_years(year, now_year)
        first, last = work_authors(work)
        first_data = author_summary(raw_authors.get(first.get("id")))
        last_data = author_summary(raw_authors.get(last.get("id")))
        author_h_values = [v for v in (first_data["h_index"], last_data["h_index"]) if isinstance(v, int)]
        author_citation_values = [v for v in (first_data["cited_by"], last_data["cited_by"]) if isinstance(v, int)]
        max_h = max(author_h_values, default=0)
        max_author_citations = max(author_citation_values, default=0)
        venue = source.get("display_name") or row.get("venue") or ""
        venue_tier = VENUE_TIER.get(venue, 1)
        core = doi in CORE_DEEP_RF
        scope = "strict_core_rf_dl" if core else "context_or_non_dl_not_core"
        authority = authority_score(citations, age, venue_tier, max_h)
        output.append({
            "doi": doi,
            "scope_decision": scope,
            "track": row.get("track"),
            "evidence_role": row.get("evidence_role"),
            "title": work.get("display_name") or row.get("title"),
            "year": year,
            "venue": venue,
            "venue_type_openalex": source.get("type") or "",
            "publisher": source.get("host_organization_name") or row.get("publisher") or "",
            "venue_quality_proxy_1_to_4": venue_tier,
            "citation_source": "OpenAlex API",
            "citation_retrieved_at_utc": retrieved,
            "openalex_work_id": work.get("id"),
            "cited_by_count_openalex": citations,
            "citations_per_year_openalex": round(citations / age, 2),
            "fwci_openalex": work.get("fwci", ""),
            "citation_percentile_openalex": ((work.get("citation_normalized_percentile") or {}).get("value", "")),
            "first_author": first_data["name"] or first.get("display_name") or "",
            "first_author_works_openalex": first_data["works"],
            "first_author_cited_by_openalex": first_data["cited_by"],
            "first_author_h_index_openalex": first_data["h_index"],
            "last_author": last_data["name"] or last.get("display_name") or "",
            "last_author_works_openalex": last_data["works"],
            "last_author_cited_by_openalex": last_data["cited_by"],
            "last_author_h_index_openalex": last_data["h_index"],
            "max_first_last_author_h_index_openalex": max_h,
            "max_first_last_author_cited_by_openalex": max_author_citations,
            "raw_author_metric_note": "OpenAlex disambiguated author-record proxy; inspect the author record before claiming expertise.",
            "authority_sort_score": authority,
            "recommended_tier": assign_tier(citations, age, venue_tier, core),
            "zotero_recommendation": "sync_core" if core else "do_not_sync_as_core",
        })

    tier_rank = {"S": 0, "A": 1, "B": 2, "context": 3}
    output.sort(key=lambda x: (tier_rank[x["recommended_tier"]], -x["authority_sort_score"], -x["cited_by_count_openalex"]))
    strict_core = [x for x in output if x["scope_decision"] == "strict_core_rf_dl"]
    for index, item in enumerate(strict_core, 1):
        item["strict_core_rank"] = index
        item["zotero_recommendation"] = "sync_core_25" if index <= 25 else "watchlist_not_core"
    for item in output:
        item.setdefault("strict_core_rank", "")

    fields = list(output[0]) if output else []
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)

    payload = {
        "retrieved_at_utc": retrieved,
        "sources": {
            "bibliometrics": "OpenAlex API exact DOI/author-record endpoints",
            "bibliographic_seed": "Crossref/DataCite-derived references_metadata.csv",
        },
        "limitations": [
            "OpenAlex counts are time-stamped open bibliometric signals, not Web of Science, Scopus, Google Scholar, or JCR Journal Impact Factor.",
            "Venue quality proxy is a transparent, task-specific tie-breaker; it is not an impact factor, quartile, or official ranking.",
            "Author works/citation/h-index values are OpenAlex entity-record proxies and require identity checks before calling an author authoritative.",
            "A paper is in the core only if it is an exact-scope RF physical-layer + neural/deep-learning method; older adjacent RF/ML work and reviews stay contextual.",
        ],
        "excluded_invalid_doi": sorted(EXCLUDED),
        "strict_core_count": len([x for x in output if x["scope_decision"] == "strict_core_rf_dl"]),
        "works_found": len(raw_works),
        "work_failures": work_failures,
        "authors_found": len(raw_authors),
        "author_failures": author_failures,
        "ranked_records": output,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    selected = strict_core[:25]
    with OUT_ZOTERO_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        fields_for_sync = [
            "recommended_tier", "strict_core_rank", "doi", "title", "year", "venue",
            "cited_by_count_openalex", "citations_per_year_openalex", "fwci_openalex",
            "first_author", "first_author_h_index_openalex", "last_author",
            "last_author_h_index_openalex", "max_first_last_author_h_index_openalex",
            "citation_retrieved_at_utc",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields_for_sync, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected)
    OUT_ZOTERO_DOIS.write_text("\n".join(x["doi"] for x in selected) + "\n", encoding="utf-8")
    lines = [
        "# 高权威候选：无人机通信物理层 RF + 深度学习（公开计量审计）",
        "",
        f"抓取时间（UTC）：{retrieved}",
        "",
        "本清单以 OpenAlex 精确 DOI/作者记录为公开可复核的计量来源。OpenAlex 被引不是 Web of Science、Scopus、Google Scholar，也不是期刊影响因子；作者指标是消歧实体记录的代理，不能自动证明其在本细分方向的学术权威性。",
        "",
        "只保留被动截获无人机/遥控器通信链路物理层 RF、且论文明确使用神经网络或深度学习的实证方法论文。数据集、综述、纯传统 ML、相邻但非 DL 的 RF 指纹方法均不作为核心同步项。",
        "",
        "S：已形成较高公开引用影响且处于强相关期刊；A：高质量核心研究但影响尚在积累；B：较新或细分方向，需要配合引用成熟度阅读。分层不等同于 JCR 分区或影响因子。下表前 25 篇为建议同步 Zotero 的核心集，其余严格范围论文作为跟踪集。",
        "",
        "| 层级 | 排名 | 年份 | OpenAlex 被引 | 期刊/会议 | 首/末作者（最高 h 指数） | 论文 | DOI |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for item in selected:
        lines.append(
            f"| {item['recommended_tier']} | {item['strict_core_rank']} | {item['year']} | {item['cited_by_count_openalex']} | {item['venue']} | {item['first_author']} / {item['last_author']}（{item['max_first_last_author_h_index_openalex']}） | {item['title'].replace('|', '–')} | [{item['doi']}](https://doi.org/{item['doi']}) |"
        )
    lines += [
        "",
        "## 重要排除",
        "",
        "- `10.1109/AERO.2019.8741733` 的出版商/元数据题名为太阳能飞机能量最优姿态研究，和无人机 RF+DL 无关，已排除。",
        "- `10.1145/2935620.2935632`、`10.1109/OJCOMS.2019.2955889` 等可作为领域历史/传统 RF 基线，但不是严格 DL 核心项。",
        "",
        "## 文件说明",
        "",
        f"- 建议同步 Zotero 的 25 篇：`{OUT_ZOTERO_CSV.name}` / `{OUT_ZOTERO_DOIS.name}`",
        f"- 完整排序与指标：`{OUT_CSV.name}`",
        f"- 原始压缩元数据与失败日志：`{OUT_JSON.name}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "strict_core": len(strict_core),
        "zotero_core_25": len(selected),
        "S": sum(x["recommended_tier"] == "S" for x in selected),
        "A": sum(x["recommended_tier"] == "A" for x in selected),
        "B": sum(x["recommended_tier"] == "B" for x in selected),
        "works_found": len(raw_works),
        "work_failures": len(work_failures),
        "authors_found": len(raw_authors),
        "author_failures": len(author_failures),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
