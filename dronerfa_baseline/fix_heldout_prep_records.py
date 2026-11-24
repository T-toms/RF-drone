"""One-off record corrections for the accepted data-preparation stage.

Codex asked for three accounting fixes that do not touch the frozen index, the
recorded MAT hashes, the signal statistics or any evaluation rule:

  1. download_records.json still says status=running although all 12 files are
     finished and the downloading process has exited.
  2. verification.json hardcodes scanned_samples_per_channel=140000000 while the
     twelve files actually hold 140M/150M/160M/180M samples per channel.
  3. report.md presents the final r3 request/retry counts as if they covered the
     whole download.

Only those fields (plus derived summaries) are rewritten; every other byte of the
three files is preserved, and heldout_dataset_summary.json is not touched at all.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
PREP_DIR = PROJECT_DIR / "outputs" / "heldout_preparation_20260917"
DOWNLOAD_RECORDS = PREP_DIR / "download_records.json"
VERIFICATION = PREP_DIR / "verification.json"
REPORT = PREP_DIR / "report.md"

# Evidence captured while reviewing the records (no downloads were re-run).
DOWNLOADS_DIR = Path(r"F:\DroneRFa数据集\heldout_20260917\downloads")
DOWNLOAD_ROUND_LOG = DOWNLOADS_DIR / "download_heldout_20260918_r3.log"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fix_download_records() -> dict:
    data = load(DOWNLOAD_RECORDS)
    before_status = data["status"]
    files = data["files"]
    completed = sorted(
        entry["finished_at"] for entry in files.values() if entry.get("finished_at")
    )
    statuses: dict[str, int] = {}
    for entry in files.values():
        statuses[entry["status"]] = statuses.get(entry["status"], 0) + 1

    data["status"] = "completed"
    data["status_note"] = (
        "12/12 文件均已落位并通过校验：8 个本轮完成（completed），"
        "4 个已存在且校验通过（skipped_existing）；"
        "记账文件的 updated_at 是最后一次写入时间，不代表进程仍在运行。"
    )
    data["completed_at"] = max(completed) if completed else data.get("updated_at")
    data["completed_at_basis"] = (
        "逐文件 finished_at 的最大值；12 个文件全部具备 finished_at 或既有校验通过记录，"
        "且下载进程已结束"
    )
    data["file_status_counts"] = statuses
    data["client_stats_scope"] = (
        "仅最后一次 r3 运行的范围请求/重试/429 计数，不是整轮下载累计值"
    )
    DOWNLOAD_RECORDS.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "before_status": before_status,
        "after_status": data["status"],
        "completed_at": data["completed_at"],
        "file_status_counts": statuses,
        "downloads_dir_present": DOWNLOADS_DIR.is_dir(),
        "r3_log_present": DOWNLOAD_ROUND_LOG.is_file(),
    }


def fix_verification() -> dict:
    data = load(VERIFICATION)
    lengths = data["checks"]["lengths_consistent"]["detail"]
    finite = data["checks"]["all_samples_finite"]
    finite["scanned_samples_per_channel_by_file"] = dict(lengths)
    finite["scanned_samples_per_channel_total"] = sum(lengths.values())
    finite.pop("scanned_samples_per_channel", None)
    finite["note"] = (
        "逐文件、逐通道完整扫描全部存储样本，非抽样；"
        "各文件每通道样本数不同（140M/150M/160M/180M），因此按文件分别记录，"
        "不使用单一固定值。"
    )
    finite["scanned_samples_per_channel_note"] = (
        "原记录把每通道扫描长度统一写成 140,000,000，与实际长度不符，已按文件长度修正"
    )
    VERIFICATION.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "files": len(lengths),
        "total_samples": sum(lengths.values()),
        "distinct_lengths": sorted(set(lengths.values())),
    }


def fix_report() -> dict:
    text = REPORT.read_text(encoding="utf-8")
    old_line = (
        "- 限流：整轮共 161 次范围请求、24 次重试，其中 **23 次为 HTTP 429**，"
        "由退避重试自动化解。"
    )
    new_line = (
        "- 限流：最后一次 **r3** 运行共 161 次范围请求、24 次重试，其中 **23 次为 HTTP 429**，"
        "由退避重试自动化解；该计数只覆盖 r3 这一段，不是整轮下载的累计值"
        "（整轮请求数因多次断点续传而更大，各轮日志分别保留）。"
    )
    if old_line not in text:
        raise SystemExit("report.md rate-limit line not found; refusing to guess")
    text = text.replace(old_line, new_line)

    old_finite = "| 全样本非有限值 | **0**（逐文件、逐通道完整扫描，非抽样） |"
    new_finite = (
        "| 全样本非有限值 | **0**（逐文件、逐通道完整扫描，非抽样；"
        "每通道 140M/150M/160M/180M 按文件分别记录） |"
    )
    if old_finite in text:
        text = text.replace(old_finite, new_finite)

    old_section = "- **非有限值 0 个**（12 文件 × 4 通道全部）。"
    new_section = (
        "- **非有限值 0 个**（12 文件 × 4 通道全部）；"
        "每通道扫描长度按文件实际值记录（140M/150M/160M/180M），"
        "`verification.json` 的 `scanned_samples_per_channel_by_file` 保留逐文件数字。"
    )
    if old_section in text:
        text = text.replace(old_section, new_section)

    marker = "## 11. 失败与阻塞记录"
    note = (
        "### 记账修正（2026-09-18，评价前）\n\n"
        "本次只改记录表述，未重跑下载/扫描/去重/PSD，未改索引、MAT 哈希或信号统计：\n\n"
        "- `download_records.json`：总体 `status` 由 `running` 改为 `completed`，"
        "`completed_at` 取 12 个文件 `finished_at` 的最大值"
        "（2026-09-18T04:02:50Z，即北京时间 12:02:50）；8 个本轮完成、4 个已存在校验通过。\n"
        "- `verification.json`：删除固定值 `scanned_samples_per_channel=140000000`，"
        "改为按文件记录 `scanned_samples_per_channel_by_file`（140M/150M/160M/180M）。\n"
        "- 本报告第 4 节的 161 次请求/24 次重试/23 次 429 明确标注为最后一次 r3 运行的数据，"
        "不再称整轮。\n\n"
    )
    if marker in text and "### 记账修正" not in text:
        text = text.replace(marker, note + marker)
    REPORT.write_text(text, encoding="utf-8")
    return {"report_bytes": len(text.encode("utf-8"))}


def main() -> None:
    print(json.dumps(fix_download_records(), ensure_ascii=False), flush=True)
    print(json.dumps(fix_verification(), ensure_ascii=False), flush=True)
    print(json.dumps(fix_report(), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
