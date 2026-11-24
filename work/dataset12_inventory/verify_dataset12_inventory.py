# -*- coding: utf-8 -*-
"""独立校验《DroneRFa 本地数据范围与文件级划分清单（12 机型 + 背景）》。

不导入生成脚本，只读：
  - 生成的 Markdown 文档
  - G:\\DroneRFa数据集\\dataset_manifest.csv
  - work/meta_split2/split_candidate_104_20260926.csv

校验内容：
  C1  两个只读来源的 SHA256 与行数/唯一性
  C2  文档 104 个文件名与冻结划分表集合完全一致、无重复、无遗漏、无额外出现
  C3  每行角色与冻结划分表一致，角色总计 62/16/20/6
  C4  13 类各 8 个；旧三类 4/2/0/2，新十类 5/1/2/0
  C5  中文用途文案与角色一致
  C6  距离描述合规：D00 → 20～40 m；背景 D00 有专门说明；无 D 字段 → 未标距离
  C7  暴露标记与冻结划分表一致
  C8  第 3 节汇总表 13 行，计数与距离口径正确
  C9  第 5.2 / 5.3 节计数表正确
  C10 每个文件名对应 G 正式清单恰好一行，类别一致
  C11 必备声明存在（12 种无人机、13 类、104、背景不是无人机、S 段说明）
  C12 不含模型成绩字样（本轮不训练、不评价）
  C13 本地 S 码全部落在官方 S0000～S0111 段
  N1-N5 负向对照：篡改文件名/删行/重复行/改角色/给未知距离补 D00 均被检出

运行：.venv\\Scripts\\python.exe work\\dataset12_inventory\\verify_dataset12_inventory.py
"""

from __future__ import annotations

import csv
import hashlib
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = PROJECT_ROOT / "reports" / "DroneRFa_当前12机型与背景_距离及文件级划分_20260926.md"
MANIFEST_PATH = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
SPLIT_PATH = PROJECT_ROOT / "work" / "meta_split2" / "split_candidate_104_20260926.csv"

EXPECTED_MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"
EXPECTED_SPLIT_SHA256 = "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
EXPECTED_TOTAL_BYTES_TEXT = "107,450,183,727"
EXPECTED_ROLE_TOTALS = {
    "train": 62,
    "val": 16,
    "development_test": 20,
    "development_test_exposed": 6,
}
LEGACY_CLASSES = ("T0000", "T0010", "T0101")
LOCAL_S_CODES = ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111"]
ROLE_LABEL_CN = {
    "train": "训练：拟合模型参数",
    "val": "验证：选轮次、阈值与超参数",
    "development_test": "开发测试：新十类同批次文件级主指标",
    "development_test_exposed": "已暴露对照测试：只作描述性对照，不计入主指标",
}
FORBIDDEN_SCORE_TOKENS = ["准确率", "100.00%", "balanced accuracy", "召回率", "混淆矩阵", "epoch"]

FILE_ROW_RE = re.compile(r"^\|\s*`(?P<name>T\d{4}(?:_D\d{2})?_S\d{4}\.mat)`\s*\|\s*`(?P<role>[a-z_]+)`\s*\|(?P<rest>.+)\|\s*$")
MEMBER_TOKEN_RE = re.compile(r"`(T\d{4}(?:_D\d{2})?_S\d{4}\.mat)`")
SUMMARY_COUNTS_RE = re.compile(r"^(\d+) / (\d+) / (\d+) / (\d+)$")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def parse_name(file_name: str) -> tuple[str, str | None, str]:
    stem = file_name[: -len(".mat")]
    parts = stem.split("_")
    class_id = parts[0]
    d_code = None
    s_code = ""
    for part in parts[1:]:
        if part.startswith("D") and len(part) == 3 and part[1:].isdigit():
            d_code = part[1:]
        elif part.startswith("S") and part[1:].isdigit():
            s_code = part[1:]
    return class_id, d_code, s_code


def get_section(text: str, prefix: str) -> str:
    lines = text.splitlines()
    collected: list[str] = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            inside = line.startswith(prefix)
            continue
        if inside:
            collected.append(line)
    return "\n".join(collected)


def check_document(text: str, split_rows: list[dict], manifest_by_name: dict[str, dict]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    expected_by_name = {row["file_name"]: row for row in split_rows}
    if len(expected_by_name) != len(split_rows):
        checks.append(("C2a 冻结划分表文件名唯一", False, "冻结划分表存在重复文件名"))
        return checks

    lines = text.splitlines()
    file_rows: list[tuple[str, str, list[str]]] = []
    duplicates: list[str] = []
    seen: set[str] = set()
    for line in lines:
        match = FILE_ROW_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if name in seen:
            duplicates.append(name)
        seen.add(name)
        body = [cell.strip() for cell in match.group("rest").strip().strip("|").split("|")]
        file_rows.append((name, match.group("role"), body))

    # C2 集合完全一致 + 无重复 + 无额外出现
    names_in_doc = [name for name, _, _ in file_rows]
    missing = sorted(set(expected_by_name) - set(names_in_doc))
    extra = sorted(set(names_in_doc) - set(expected_by_name))
    stray: list[str] = []
    for name in expected_by_name:
        occurrences = MEMBER_TOKEN_RE.findall(text).count(name)
        if occurrences != 1:
            stray.append(f"{name}×{occurrences}")
    ok_c2 = (
        len(file_rows) == len(expected_by_name)
        and not duplicates
        and not missing
        and not extra
        and not stray
    )
    checks.append((
        "C2 文档 104 个文件名与冻结划分表集合一致（无重复/遗漏/额外出现）",
        ok_c2,
        f"文件行={len(file_rows)} 期望={len(expected_by_name)} 重复={duplicates or 0} 缺失={missing or 0} 额外={extra or 0} 出现次数异常={stray or 0}",
    ))

    # C3 角色一致 + 总计
    role_mismatch: list[str] = []
    role_totals = {role: 0 for role in EXPECTED_ROLE_TOTALS}
    unknown_roles: list[str] = []
    for name, role, _ in file_rows:
        expected_row = expected_by_name.get(name)
        if expected_row is None:
            continue
        if role not in role_totals:
            unknown_roles.append(f"{name}:{role}")
            continue
        role_totals[role] += 1
        if role != expected_row["role"]:
            role_mismatch.append(f"{name}: 文档={role} 冻结={expected_row['role']}")
    ok_c3 = not role_mismatch and not unknown_roles and role_totals == EXPECTED_ROLE_TOTALS
    checks.append((
        "C3 每行角色与冻结划分表一致；总计 62/16/20/6",
        ok_c3,
        f"总计={role_totals} 不一致={role_mismatch or 0} 未知角色={unknown_roles or 0}",
    ))

    # C4 逐类数量
    per_class: dict[str, dict[str, int]] = {}
    for name, role, _ in file_rows:
        class_id = parse_name(name)[0]
        per_class.setdefault(class_id, {key: 0 for key in EXPECTED_ROLE_TOTALS})
        per_class[class_id][role] += 1
    class_problems: list[str] = []
    if len(per_class) != 13:
        class_problems.append(f"类别数={len(per_class)}")
    for class_id, counts in sorted(per_class.items()):
        if sum(counts.values()) != 8:
            class_problems.append(f"{class_id} 文件数={sum(counts.values())}")
        if class_id in LEGACY_CLASSES:
            want = {"train": 4, "val": 2, "development_test": 0, "development_test_exposed": 2}
        else:
            want = {"train": 5, "val": 1, "development_test": 2, "development_test_exposed": 0}
        if counts != want:
            class_problems.append(f"{class_id} 角色={counts} 期望={want}")
    csv_classes = {row["class_id"] for row in split_rows}
    if set(per_class) != csv_classes:
        class_problems.append(f"类别集合不一致：{sorted(set(per_class) ^ csv_classes)}")
    checks.append((
        "C4 13 类各 8 个；旧三类 4/2/0/2，新十类 5/1/2/0",
        not class_problems,
        f"问题={class_problems or 0}",
    ))

    # C5 中文用途文案
    purpose_problems: list[str] = []
    for name, role, body in file_rows:
        if len(body) != 3:
            purpose_problems.append(f"{name}: 行结构应为 3 列（用途/距离/暴露），实际 {len(body)} 列")
            continue
        if body[0] != ROLE_LABEL_CN.get(role):
            purpose_problems.append(f"{name}: 用途='{body[0]}' 角色={role}")
    if len(file_rows) != 104:
        purpose_problems.append(f"参与检查的行数={len(file_rows)}，应为 104")
    checks.append((
        "C5 中文用途文案与角色一致（104 行全部实检）",
        not purpose_problems,
        f"检查行数={len(file_rows)} 问题={purpose_problems[:5] or 0}",
    ))

    # C6 距离描述
    distance_problems: list[str] = []
    for name, role, body in file_rows:
        if len(body) != 3:
            distance_problems.append(f"{name}: 行结构异常，无法校验距离列")
            continue
        cell = body[1]
        class_id, d_code, _ = parse_name(name)
        if d_code == "00" and class_id == "T0000":
            if "D00" not in cell or "20～40 m" not in cell or "背景不是无人机" not in cell:
                distance_problems.append(f"{name}: 背景距离说明不合规 '{cell}'")
        elif d_code == "00":
            if "D00" not in cell or "20～40 m" not in cell:
                distance_problems.append(f"{name}: D00 说明不合规 '{cell}'")
        else:
            if "文件名未标 D 字段" not in cell or "不能确认具体距离" not in cell:
                distance_problems.append(f"{name}: 未知距离说明不合规 '{cell}'")
            if "D00" in cell or "20～40" in cell:
                distance_problems.append(f"{name}: 未知距离被补写为 D00/20～40 '{cell}'")
    checks.append((
        "C6 距离描述严格按文件名与官方编码（缺失 D 字段标未知）",
        not distance_problems,
        f"检查行数={len(file_rows)} 问题={distance_problems[:5] or 0}",
    ))

    # C7 暴露标记
    exposure_problems: list[str] = []
    for name, role, body in file_rows:
        if len(body) != 3:
            exposure_problems.append(f"{name}: 行结构异常，无法校验暴露标记")
            continue
        cell = body[2]
        expected_row = expected_by_name.get(name)
        if expected_row is None:
            continue
        exposed = expected_row["exposed_to_prior_selection"] == "True"
        if exposed:
            if not cell.startswith("是"):
                exposure_problems.append(f"{name}: 应为已暴露 '{cell}'")
            former = expected_row["former_partition"]
            if former and former != "NR" and former not in cell:
                exposure_problems.append(f"{name}: 缺少原分区 {former} '{cell}'")
        else:
            if not cell.startswith("否"):
                exposure_problems.append(f"{name}: 应为未暴露 '{cell}'")
    checks.append((
        "C7 暴露标记与冻结划分表一致（旧三类含原分区）",
        not exposure_problems,
        f"检查行数={len(file_rows)} 问题={exposure_problems[:5] or 0}",
    ))

    # C8 第 3 节汇总表
    section3 = get_section(text, "## 3.")
    summary_rows: list[list[str]] = []
    for line in section3.splitlines():
        cells = split_cells(line)
        if len(cells) == 8 and cells[0].isdigit():
            summary_rows.append(cells)
    summary_problems: list[str] = []
    csv_by_class: dict[str, list[dict]] = {}
    for row in split_rows:
        csv_by_class.setdefault(row["class_id"], []).append(row)
    class_order = sorted(csv_by_class, key=lambda cid: int(cid[1:]))
    if len(summary_rows) != 13:
        summary_problems.append(f"汇总表行数={len(summary_rows)}")
    for cells, class_id in zip(summary_rows, class_order):
        index_cell, class_cell, name_cell, files_cell, distance_cell, counts_cell, test_cell, cluster_cell = cells
        if class_cell != f"`{class_id}`":
            summary_problems.append(f"第{index_cell}行类别码={class_cell} 期望={class_id}")
            continue
        files = csv_by_class[class_id]
        expected_name = files[0]["class_name"]
        if name_cell != expected_name:
            summary_problems.append(f"{class_id} 机型名={name_cell} 期望={expected_name}")
        prefix = files[0]["file_name"][: files[0]["file_name"].rindex("_S")]
        if f"{prefix}_S0000" not in files_cell or f"{prefix}_S0111" not in files_cell or "8 个" not in files_cell:
            summary_problems.append(f"{class_id} 文件列不合规 '{files_cell}'")
        d_code = parse_name(files[0]["file_name"])[1]
        if d_code == "00" and class_id == "T0000":
            if "D00" not in distance_cell or "背景" not in distance_cell:
                summary_problems.append(f"{class_id} 距离列不合规 '{distance_cell}'")
        elif d_code == "00":
            if "D00" not in distance_cell or "20～40 m" not in distance_cell:
                summary_problems.append(f"{class_id} 距离列不合规 '{distance_cell}'")
        else:
            if "未标距离" not in distance_cell or "20～40" in distance_cell or "D00" in distance_cell:
                summary_problems.append(f"{class_id} 距离列不合规 '{distance_cell}'")
        match = SUMMARY_COUNTS_RE.match(counts_cell)
        if not match:
            summary_problems.append(f"{class_id} 计数列格式不合规 '{counts_cell}'")
        else:
            want = [4, 2, 0, 2] if class_id in LEGACY_CLASSES else [5, 1, 2, 0]
            if [int(value) for value in match.groups()] != want:
                summary_problems.append(f"{class_id} 计数={match.groups()} 期望={want}")
        if class_id in LEGACY_CLASSES:
            if "development_test_exposed" not in test_cell:
                summary_problems.append(f"{class_id} 测试角色列不合规 '{test_cell}'")
        else:
            if "development_test`" not in test_cell:
                summary_problems.append(f"{class_id} 测试角色列不合规 '{test_cell}'")
        if files[0]["night_proxy"] not in cluster_cell or files[0]["campaign_proxy"] not in cluster_cell:
            summary_problems.append(f"{class_id} 时间簇列不合规 '{cluster_cell}'")
    checks.append((
        "C8 第 3 节 13 行汇总表计数与距离口径正确",
        not summary_problems,
        f"问题={summary_problems[:5] or 0}",
    ))

    # C9 第 5.2 / 5.3 节计数
    totals_problems: list[str] = []
    section5 = get_section(text, "## 5.")
    found_totals: dict[str, int] = {}
    for line in section5.splitlines():
        cells = split_cells(line)
        if len(cells) >= 3 and cells[0].startswith("`") and cells[0].strip("`") in EXPECTED_ROLE_TOTALS:
            if cells[1].isdigit():
                found_totals[cells[0].strip("`")] = int(cells[1])
    if found_totals != EXPECTED_ROLE_TOTALS:
        totals_problems.append(f"5.2 总计={found_totals}")
    per_class_rows: list[list[str]] = []
    for line in section5.splitlines():
        cells = split_cells(line)
        if len(cells) == 8 and cells[0].isdigit() and all(cell.isdigit() for cell in cells[3:8]):
            per_class_rows.append(cells)
    if len(per_class_rows) != 13:
        totals_problems.append(f"5.3 行数={len(per_class_rows)}")
    for cells, class_id in zip(per_class_rows, class_order):
        counts = [int(value) for value in cells[3:7]]
        want = [4, 2, 0, 2] if class_id in LEGACY_CLASSES else [5, 1, 2, 0]
        if cells[1] != f"`{class_id}`" or counts != want or cells[7] != "8":
            totals_problems.append(f"5.3 {cells[1]} 计数={counts} 期望={want} 合计={cells[7]}")
    if "| — | **合计** | **13 类** | **62** | **16** | **20** | **6** | **104** |" not in text:
        totals_problems.append("缺少 5.3 合计行 62/16/20/6/104")
    checks.append((
        "C9 第 5.2/5.3 节计数表正确（含 62/16/20/6/104 合计）",
        not totals_problems,
        f"问题={totals_problems[:5] or 0}",
    ))

    # C10 每个文件名对应 G 正式清单恰好一行
    manifest_problems: list[str] = []
    for name in sorted(expected_by_name):
        row = manifest_by_name.get(name)
        if row is None:
            manifest_problems.append(f"{name}: G 正式清单无此行")
            continue
        if row["class_id"] != expected_by_name[name]["class_id"]:
            manifest_problems.append(f"{name}: 清单 class_id 不一致")
        if row["class_name"] != expected_by_name[name]["class_name"]:
            manifest_problems.append(f"{name}: 清单 class_name 不一致")
    if len(manifest_by_name) != 104:
        manifest_problems.append(f"G 正式清单行数={len(manifest_by_name)}")
    if EXPECTED_TOTAL_BYTES_TEXT not in text:
        manifest_problems.append("文档未写明清单总字节 107,450,183,727")
    checks.append((
        "C10 104 个文件名均在 G 正式清单中恰好占一行且类别一致",
        not manifest_problems,
        f"问题={manifest_problems[:5] or 0}",
    ))

    # C11 必备声明
    required_phrases = [
        "12 种无人机",
        "背景不是无人机机型",
        "13 类",
        "104 个 MAT",
        "背景不是无人机",
        "S0000～S0111",
        "同批次",
        "development_test_exposed",
        "文件名未标距离",
    ]
    missing_phrases = [phrase for phrase in required_phrases if phrase not in text]
    checks.append((
        "C11 必备声明齐备（12 种无人机/13 类/104/背景不是无人机/S 段/同批次）",
        not missing_phrases,
        f"缺失={missing_phrases or 0}",
    ))

    # C12 不含模型成绩字样
    hits = [token for token in FORBIDDEN_SCORE_TOKENS if token in text]
    checks.append((
        "C12 文档不含模型成绩字样（本轮不训练、不评价）",
        not hits,
        f"命中={hits or 0}",
    ))

    # C13 本地 S 码全部落在官方 S0000～S0111 段
    s_codes = {parse_name(row["file_name"])[2] for row in split_rows}
    ok_c13 = s_codes == set(LOCAL_S_CODES) and "本地 104 个文件的 S 码全部落在这一段" in text
    checks.append((
        "C13 本地 S 码全部落在官方 S0000～S0111 段且文档说明一致",
        ok_c13,
        f"S 码集合={sorted(s_codes)}",
    ))

    return checks


def mutate_cases(text: str) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []

    # N1 改一个文件名
    cases.append(("N1 篡改文件名", text.replace("`T0001_D00_S0000.mat`", "`T0001_D00_S0099.mat`", 1)))

    # N2 删除一整行文件行
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("| `T0100_D00_S0101.mat`"))
    cases.append(("N2 删除一行文件行", "\n".join(lines[:index] + lines[index + 1:])))

    # N3 重复一行文件行
    cases.append(("N3 重复一行文件行", text.replace(lines[index], lines[index] + "\n" + lines[index], 1)))

    # N4 改一个角色
    cases.append((
        "N4 篡改角色",
        text.replace("| `T0001_D00_S0111.mat` | `development_test` |", "| `T0001_D00_S0111.mat` | `val` |", 1),
    ))

    # N5 给未知距离补 D00
    cases.append((
        "N5 给未知距离补写 D00",
        text.replace(
            "| `T1010_S0000.mat` | `train` | 训练：拟合模型参数 | 文件名未标 D 字段，当前清单不能确认具体距离 |",
            "| `T1010_S0000.mat` | `train` | 训练：拟合模型参数 | D00 → 20～40 m |",
            1,
        ),
    ))
    return cases


def main() -> int:
    manifest_sha = sha256_of(MANIFEST_PATH)
    split_sha = sha256_of(SPLIT_PATH)
    with SPLIT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        split_rows = list(csv.DictReader(handle))
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    manifest_by_name = {row["file_name"]: row for row in manifest_rows}
    text = DOC_PATH.read_text(encoding="utf-8")

    checks: list[tuple[str, bool, str]] = []
    checks.append((
        "C1a G 正式清单 SHA256 与预期一致",
        manifest_sha == EXPECTED_MANIFEST_SHA256,
        manifest_sha,
    ))
    checks.append((
        "C1b 冻结划分表 SHA256 与预期一致",
        split_sha == EXPECTED_SPLIT_SHA256,
        split_sha,
    ))
    checks.append((
        "C1c 冻结划分表 104 行且文件名唯一",
        len(split_rows) == 104 and len({row["file_name"] for row in split_rows}) == 104,
        f"行数={len(split_rows)} 唯一={len({row['file_name'] for row in split_rows})}",
    ))

    checks.extend(check_document(text, split_rows, manifest_by_name))

    negative_problems: list[str] = []
    for label, mutated in mutate_cases(text):
        mutated_checks = check_document(mutated, split_rows, manifest_by_name)
        failed = [name for name, ok, _ in mutated_checks if not ok]
        if not failed:
            negative_problems.append(f"{label}: 未被检出")
    checks.append((
        "N1-N5 负向对照：5 种篡改均被检出",
        not negative_problems,
        f"问题={negative_problems or 0}",
    ))

    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name} :: {detail}")
    print(f"\n{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
