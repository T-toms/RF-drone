# -*- coding: utf-8 -*-
"""生成《DroneRFa 本地数据范围与文件级划分清单（12 机型 + 背景）》。

只读输入（不改写任何来源）：
  1. G:\\DroneRFa数据集\\dataset_manifest.csv                （正式清单，104 行）
  2. work/meta_split2/split_candidate_104_20260926.csv       （META-SPLIT2 冻结划分）
  3. dronerfa_baseline/参考论文/01_DroneRFa_Dataset_JEIT2024.pdf 的官方表 3（人工核对后写在下方常量里）

104 个文件名、角色、距离描述、时间代理全部由冻结 CSV 派生，避免手工抄写错误。
运行：.venv\\Scripts\\python.exe work\\dataset12_inventory\\generate_dataset12_inventory.py
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(r"G:\DroneRFa数据集\dataset_manifest.csv")
SPLIT_PATH = PROJECT_ROOT / "work" / "meta_split2" / "split_candidate_104_20260926.csv"
FREEZE_PATH = PROJECT_ROOT / "work" / "meta_split2" / "CODEX-FREEZE-20260926.json"
OUT_PATH = PROJECT_ROOT / "reports" / "DroneRFa_当前12机型与背景_距离及文件级划分_20260926.md"

EXPECTED_MANIFEST_SHA256 = "9765771d84c6dfc6b5744fd73ad548b1eab5eb9e2a1ae35952d54c0cbe15644d"
EXPECTED_SPLIT_SHA256 = "d33486b7b5eafe349b392f55b84b54c8a5637e19c082d2fc7fc57c141bba4ed9"
EXPECTED_TOTAL_FILES = 104
EXPECTED_TOTAL_BYTES = 107_450_183_727
EXPECTED_ROLE_TOTALS = {
    "train": 62,
    "val": 16,
    "development_test": 20,
    "development_test_exposed": 6,
}

OFFICIAL_PAGE = "https://jeit.ac.cn/cn/article/doi/10.11999/JEIT230570?viewType=HTML"
LOCAL_PAPER = r"dronerfa_baseline\参考论文\01_DroneRFa_Dataset_JEIT2024.pdf"

# 官方论文表 3《二进制编码与标签对应关系》（DOI 10.11999/JEIT230570）逐条核对结果。
# 本地 13 类只用到前 13 条；其余条目本地没有对应文件，单列出来避免误读。
OFFICIAL_T_TABLE = [
    ("0000", "背景(含蓝牙、WiFi)"),
    ("0001", "DJI Phantom 3"),
    ("0010", "DJI Phantom 4 Pro"),
    ("0011", "DJI MATRICE 200"),
    ("0100", "DJI MATRICE 100"),
    ("0101", "DJI Air 2S"),
    ("0110", "DJI Mini 3 Pro"),
    ("0111", "DJI Inspire 2"),
    ("1000", "DJI Mavic Pro"),
    ("1001", "DJI Mini 2"),
    ("1010", "DJI Mavic 3"),
    ("1011", "DJI MATRICE 300"),
    ("1100", "DJI Phantom 4 Pro RTK"),
    ("1101", "DJI MATRICE 30T"),
    ("1110", "DJI AVATA"),
    ("1111", "DJI通信模块自组机"),
    ("10000", "DJI MATRICE 600 Pro"),
    ("10001", "VBar 飞控器"),
    ("10010", "FrSky X20 飞控器"),
    ("10011", "Futaba T6IZ 飞控器"),
    ("10100", "Taranis Plus 飞控器"),
    ("10101", "RadioLink AT9S 飞控器"),
    ("10110", "Futaba T14SG 飞控器"),
    ("10111", "云卓 T12 飞控器"),
    ("11000", "云卓 T10 飞控器"),
]
OFFICIAL_D_TABLE = [
    ("00", "20～40 m"),
    ("01", "40～80 m"),
    ("10", "80～150 m"),
]
OFFICIAL_S_TABLE = [
    ("0000～0111", "设置初始通信在 915 MHz 或 2.4 GHz"),
    ("1000～1111", "若设备支持则切换至 2.4 GHz 或 5.8 GHz"),
]
LOCAL_CLASS_COUNT = 13
LOCAL_S_CODES = ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111"]

ROLE_LABEL_CN = {
    "train": "训练：拟合模型参数",
    "val": "验证：选轮次、阈值与超参数",
    "development_test": "开发测试：新十类同批次文件级主指标",
    "development_test_exposed": "已暴露对照测试：只作描述性对照，不计入主指标",
}

LEGACY_CLASSES = ("T0000", "T0010", "T0101")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_rows() -> list[dict]:
    with SPLIT_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_TOTAL_FILES:
        raise SystemExit(f"冻结划分表行数异常：{len(rows)}")
    return rows


def read_manifest_rows() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_TOTAL_FILES:
        raise SystemExit(f"G 正式清单行数异常：{len(rows)}")
    return rows


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


def distance_cell(class_id: str, d_code: str | None) -> str:
    if d_code == "00" and class_id == "T0000":
        return "文件名含 D00（场景编码）；背景不是无人机，不代表距接收机 20～40 m"
    if d_code == "00":
        return "D00 → 20～40 m"
    return "文件名未标 D 字段，当前清单不能确认具体距离"


def distance_cell_summary(class_id: str, d_code: str | None) -> str:
    if d_code == "00" and class_id == "T0000":
        return "D00（背景场景编码，无无人机距离含义）"
    if d_code == "00":
        return "D00 → 20～40 m"
    return "文件名未标距离（不能确认）"


def exposure_cell(row: dict) -> str:
    if row["exposed_to_prior_selection"] == "True":
        former = row["former_partition"] or "NR"
        return f"是（旧三类；原分区 `{former}`）"
    return "否（新十类，此前未进入本项目数据/模型产物）"


def main() -> int:
    manifest_sha = sha256_of(MANIFEST_PATH)
    split_sha = sha256_of(SPLIT_PATH)
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise SystemExit(f"G 正式清单哈希不符：{manifest_sha}")
    if split_sha != EXPECTED_SPLIT_SHA256:
        raise SystemExit(f"冻结划分表哈希不符：{split_sha}")

    rows = read_split_rows()
    manifest_rows = read_manifest_rows()
    manifest_by_name = {row["file_name"]: row for row in manifest_rows}
    if len(manifest_by_name) != EXPECTED_TOTAL_FILES:
        raise SystemExit("G 正式清单存在重复文件名")

    by_class: dict[str, list[dict]] = {}
    for row in rows:
        by_class.setdefault(row["class_id"], []).append(row)
    if len(by_class) != LOCAL_CLASS_COUNT:
        raise SystemExit(f"类别数异常：{len(by_class)}")

    class_order = sorted(by_class, key=lambda cid: int(cid[1:]))
    total_bytes = 0
    for class_id in class_order:
        files = sorted(by_class[class_id], key=lambda r: parse_name(r["file_name"])[2])
        by_class[class_id] = files
        if len(files) != 8:
            raise SystemExit(f"{class_id} 文件数异常：{len(files)}")
        names = {row["class_name"] for row in files}
        if len(names) != 1:
            raise SystemExit(f"{class_id} 机型名称不唯一：{names}")
        d_codes = {parse_name(row["file_name"])[1] for row in files}
        if len(d_codes) != 1:
            raise SystemExit(f"{class_id} 距离码不唯一：{d_codes}")
        s_codes = sorted(parse_name(row["file_name"])[2] for row in files)
        if s_codes != LOCAL_S_CODES:
            raise SystemExit(f"{class_id} S 码异常：{s_codes}")
        nights = {row["night_proxy"] for row in files}
        campaigns = {row["campaign_proxy"] for row in files}
        if len(nights) != 1 or len(campaigns) != 1:
            raise SystemExit(f"{class_id} 时间代理跨簇：{nights} {campaigns}")
        for row in files:
            manifest_row = manifest_by_name.get(row["file_name"])
            if manifest_row is None:
                raise SystemExit(f"G 正式清单缺少文件：{row['file_name']}")
            if manifest_row["class_id"] != row["class_id"]:
                raise SystemExit(f"清单类别不一致：{row['file_name']}")
            if manifest_row["class_name"] != row["class_name"]:
                raise SystemExit(f"清单机型名不一致：{row['file_name']}")
            total_bytes += int(manifest_row["size_bytes"])

    if total_bytes != EXPECTED_TOTAL_BYTES:
        raise SystemExit(f"总字节异常：{total_bytes}")

    role_totals = {role: 0 for role in EXPECTED_ROLE_TOTALS}
    for row in rows:
        role_totals[row["role"]] += 1
    if role_totals != EXPECTED_ROLE_TOTALS:
        raise SystemExit(f"角色总计异常：{role_totals}")

    # ---------------- 文档正文 ----------------
    out: list[str] = []
    add = out.append

    add("# DroneRFa 本地数据范围与文件级划分清单（12 种机型 + 背景）")
    add("")
    add("编制日期：2026-09-26  ")
    add("适用范围：只读核对“本地有哪些机型、文件名对应什么距离、104 个 MAT 怎样按文件划分”。本轮**不训练、不推理、不做评价**，文档中不包含任何模型成绩。  ")
    add("唯一事实来源：G 盘正式清单 `G:\\DroneRFa数据集\\dataset_manifest.csv` 与 META-SPLIT2 冻结划分表 `work/meta_split2/split_candidate_104_20260926.csv`（两者 SHA256 见第 1 节）。")
    add("")
    add("> **通俗摘要（先看这三段）**")
    add(">")
    add("> 1. **本地现在有 12 种无人机**，另外还有 1 类 **Background（背景）**。背景不是无人机机型，所以本地一共是 **13 类**，每类 8 个 MAT 文件，合计 **104 个 MAT**（约 100.07 GiB）。")
    add("> 2. **距离看文件名里的 D 码**。其中 9 种无人机的本地文件名带 `D00`，按官方论文表 3 表示 **20～40 m** 档；`T1010`（DJI Mavic 3）、`T1011`（DJI Matrice 300）、`T1100`（DJI Phantom 4 Pro RTK）这 3 种的本地文件名**没有 D 字段**，所以**当前清单不能确认它们的具体距离**，不能补写成 D00 或 20～40 m。")
    add("> 3. 104 个文件已按**源文件**冻结划分为 **62 train / 16 val / 20 开发测试 / 6 已暴露对照测试**。同一个 MAT 派生出的所有窗口都继承该文件的角色，不会跨组。这是**同批次文件级开发划分**：它**不是**独立最终测试，也**不能**用来声称跨采集会话、跨接收机或跨环境泛化。")
    add("")
    add("---")
    add("")
    add("## 1. 只读来源与身份核对")
    add("")
    add("| 项目 | 值 |")
    add("| --- | --- |")
    add("| 数据根目录 | `G:\\DroneRFa数据集\\original` |")
    add("| 正式清单 | `G:\\DroneRFa数据集\\dataset_manifest.csv` |")
    add(f"| 正式清单 SHA256（本次实测） | `{manifest_sha}` |")
    add(f"| 正式清单 SHA256（预期） | `{EXPECTED_MANIFEST_SHA256}` |")
    add("| 清单行数 | 104（文件名唯一，无重复） |")
    add(f"| 清单总字节 | {total_bytes:,} 字节 ≈ 100.07 GiB |")
    add("| 冻结划分表 | `work/meta_split2/split_candidate_104_20260926.csv` |")
    add(f"| 冻结划分表 SHA256（本次实测） | `{split_sha}` |")
    add(f"| 冻结划分表 SHA256（预期） | `{EXPECTED_SPLIT_SHA256}` |")
    add("| 冻结决策文件 | `work/meta_split2/CODEX-FREEZE-20260926.json` |")
    add("| 距离编码出处 | 官方论文表 3（DOI `10.11999/JEIT230570`），官方页面 " + OFFICIAL_PAGE + " |")
    add("")
    add("本轮只读取上述文件，**没有修改** G 盘数据、正式清单、冻结划分表或任何旧实验产物；也没有重新哈希 100 GiB 的 MAT 内容。")
    add("")
    add("> 核对说明：官方页面本次直接抓取返回 HTTP 403，无法在线取表；因此表 3 的数值改用仓库内官方论文 PDF 副本 `" + LOCAL_PAPER + "` 逐条核对（该 PDF 首页即 DOI `10.11999/JEIT230570`，《电子与信息学报》2024 年第 46 卷第 4 期）。复现命令见第 8 节。")
    add("")
    add("---")
    add("")
    add("## 2. 文件名编码规则：T = 机型、D = 距离、S = 信号片段")
    add("")
    add("官方论文把文件名写成“前缀字符 + 二进制编码”，不同标签之间用 `_` 分隔，例如 `T0010_D01_S1110.mat`。本地 104 个文件全部形如 `T####[_D##]_S####.mat`。")
    add("")
    add("### 2.1 本地 13 类的 T 码（官方表 3 逐条核对）")
    add("")
    add("| 类别码 | 官方表 3 含义 | 本地是否有 8 个文件 |")
    add("| --- | --- | --- |")
    for code, meaning in OFFICIAL_T_TABLE[:LOCAL_CLASS_COUNT]:
        add(f"| `T{code}` | {meaning} | 有（8 个 MAT） |")
    add("")
    add("官方表 3 还列有下面这些条目，**本地目前没有对应文件**，本清单不把它们算进 12 种无人机：")
    add("")
    add("| 类别码 | 官方表 3 含义 |")
    add("| --- | --- |")
    for code, meaning in OFFICIAL_T_TABLE[LOCAL_CLASS_COUNT:]:
        add(f"| `T{code}` | {meaning} |")
    add("")
    add("### 2.2 D 码（距离）")
    add("")
    add("| 距离码 | 官方表 3 含义 | 本地情况 |")
    add("| --- | --- | --- |")
    add(f"| `D00` | {OFFICIAL_D_TABLE[0][1]} | 10 个类别的文件名带 `D00`（9 种无人机 + 背景） |")
    add(f"| `D01` | {OFFICIAL_D_TABLE[1][1]} | 本地 104 个文件中没有此编码 |")
    add(f"| `D10` | {OFFICIAL_D_TABLE[2][1]} | 本地 104 个文件中没有此编码 |")
    add("")
    add("因此：")
    add("")
    add("- **9 种无人机的本地文件**（`T0001`/`T0010`/`T0011`/`T0100`/`T0101`/`T0110`/`T0111`/`T1000`/`T1001`）文件名带 `D00`，可按论文表 3 解释为 **20～40 m** 档。")
    add("- **`T1010` DJI Mavic 3、`T1011` DJI Matrice 300、`T1100` DJI Phantom 4 Pro RTK** 的本地文件名**没有 D 字段**（文件名形如 `T####_S####.mat`，中间没有 `_D##_`），所以**文件名未标距离、当前清单不能确认具体距离**；文档中一律按“未标距离”记录，不推断、不补写。")
    add(f"- **`T0000` Background** 的文件名同样带 `D00`，但含义只是“该文件采用近距离场景编码”。**背景本身没有“无人机距接收机 20～40 m”的物理含义。**")
    add("")
    add("### 2.3 S 码（信号片段）")
    add("")
    add("| S 码段 | 官方表 3 含义 | 本地情况 |")
    add("| --- | --- | --- |")
    add(f"| `S{OFFICIAL_S_TABLE[0][0]}` | {OFFICIAL_S_TABLE[0][1]} | 本地 104 个文件的 S 码全部落在这一段 |")
    add(f"| `S{OFFICIAL_S_TABLE[1][0]}` | {OFFICIAL_S_TABLE[1][1]} | 本地 104 个文件中没有此段文件 |")
    add("")
    add("本地 104 个文件的 S 码全部落在官方 `S0000～S0111` 段（即上表 `S` + `0000～0111` 这一段），没有 `S1000～S1111` 段文件。本地每类固定使用同样的 8 个 S 码：`S0000`、`S0001`、`S0010`、`S0011`、`S0100`、`S0101`、`S0110`、`S0111`；本节已按冻结划分表逐文件核对该集合，13 类完全一致。S 码本身只表示“信号片段编号”，**不表示距离，也不表示划分角色**（划分角色只按冻结划分表读取）。")
    add("")
    add("---")
    add("")
    add("## 3. 13 类汇总表")
    add("")
    add("| # | 类别码 | 机型名称 | 8 个文件（首/末，此处省略 `.mat`） | 文件名距离码与范围 | train / val / dev_test / 已暴露 | 测试角色 | 文件时间簇 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for index, class_id in enumerate(class_order, start=1):
        files = by_class[class_id]
        class_name = files[0]["class_name"]
        first = files[0]["file_name"]
        prefix = first[: first.rindex("_S")]
        files_cell = f"`{prefix}_S0000` … `{prefix}_S0111`（8 个，完整名见 4.{index}）"
        d_code = parse_name(first)[1]
        d_summary = distance_cell_summary(class_id, d_code)
        counts = {role: 0 for role in EXPECTED_ROLE_TOTALS}
        for row in files:
            counts[row["role"]] += 1
        counts_cell = " / ".join(str(counts[role]) for role in ("train", "val", "development_test", "development_test_exposed"))
        if class_id in LEGACY_CLASSES:
            test_role = "`development_test_exposed` ×2：已暴露描述性对照"
        else:
            test_role = "`development_test` ×2：同批次文件级开发主指标"
        night = files[0]["night_proxy"]
        campaign = files[0]["campaign_proxy"]
        add(
            f"| {index} | `{class_id}` | {class_name} | {files_cell} | {d_summary} | {counts_cell} | {test_role} | {night} / {campaign} |"
        )
    add("")
    add("读表要点：")
    add("")
    add("- **12 种无人机 + 1 类背景 = 13 类**；每类 8 个 MAT，13 × 8 = **104 个 MAT**。")
    add("- 背景 `T0000` 只是“无无人机”的参照类，**不计入无人机机型数量**。")
    add("- 第 6 列是**文件数**，不是窗口数；每个 MAT 派生出的 1M 采样窗全部继承该文件的角色。")
    add("- “文件时间簇”来自冻结划分表的文件时间代理（夜 / 批）；它只说明批次先后，不构成采集会话证据。")
    add("")
    add("---")
    add("")
    add("## 4. 逐类逐文件清单（104 个完整文件名）")
    add("")
    add("下面 13 个小节把 104 个 MAT **逐个列全**：完整文件名、角色、中文用途、按文件名可确认的距离信息、此前是否已暴露。角色字符串与冻结划分表一致，可直接与 `work/meta_split2/split_candidate_104_20260926.csv` 对行。")
    add("")
    role_order = ("train", "val", "development_test", "development_test_exposed")
    for index, class_id in enumerate(class_order, start=1):
        files = by_class[class_id]
        class_name = files[0]["class_name"]
        d_code = parse_name(files[0]["file_name"])[1]
        official_meaning = OFFICIAL_T_TABLE[int(class_id[1:], 2)][1]
        tag = "旧三类" if class_id in LEGACY_CLASSES else "新十类"
        counts = {role: 0 for role in role_order}
        for row in files:
            counts[row["role"]] += 1
        times = sorted(row["file_time_proxy"] for row in files)
        add(f"### 4.{index} `{class_id}` — {class_name}（{tag}）")
        add("")
        add(f"- 官方表 3 对应条目：`T{class_id[1:]}` = {official_meaning}。")
        if class_id == "T0000":
            add("- 距离：文件名含 `D00`（近距离场景编码）。**背景不是无人机机型**，该编码不表示“背景距接收机 20～40 m”。")
        elif d_code == "00":
            add("- 距离：文件名含 `D00` → **20～40 m**（官方表 3）。")
        else:
            add("- 距离：本地文件名**没有 D 字段** → **未标距离，当前清单不能确认具体距离**。")
        add(f"- 文件时间代理：{times[0].replace('T', ' ')} ～ {times[-1].replace('T', ' ')}（夜 {files[0]['night_proxy']}，批 {files[0]['campaign_proxy']}）。")
        if class_id in LEGACY_CLASSES:
            add(
                f"- 划分（{tag}）：{counts['train']} train / {counts['val']} val / {counts['development_test']} development_test / {counts['development_test_exposed']} development_test_exposed；"
                "8 个文件此前**均已暴露**（4 个 `development_original`、4 个 `evaluated_holdout`），其中 2 个 `development_test_exposed` **只能作已暴露描述性对照**。"
            )
        else:
            add(
                f"- 划分（{tag}）：{counts['train']} train / {counts['val']} val / {counts['development_test']} development_test / {counts['development_test_exposed']} development_test_exposed；"
                "8 个文件此前**均未进入**本项目任何数据或模型产物，其中 2 个 `development_test` 是同批次文件级开发主指标的来源。"
            )
        add("")
        add("| 文件名 | 角色 | 中文用途 | 距离（按文件名） | 此前是否已暴露 |")
        add("| --- | --- | --- | --- | --- |")
        for row in sorted(files, key=lambda r: parse_name(r["file_name"])[2]):
            name = row["file_name"]
            file_class, file_d, _ = parse_name(name)
            add(
                f"| `{name}` | `{row['role']}` | {ROLE_LABEL_CN[row['role']]} | {distance_cell(file_class, file_d)} | {exposure_cell(row)} |"
            )
        add("")
    add("---")
    add("")
    add("## 5. 划分规则与数量核对")
    add("")
    add("### 5.1 规则")
    add("")
    add(f"1. **最小分组键是源文件**：`file:<MAT 文件名>`。一个 MAT 的所有 1M 采样窗只能进同一个角色，**绝不跨 train/val/测试**。")
    add("2. 旧三类（`T0000`、`T0010`、`T0101`）按 **4 train / 2 val / 0 development_test / 2 development_test_exposed** 划分。")
    add("3. 新十类按 **5 train / 1 val / 2 development_test / 0 development_test_exposed** 划分。")
    add("4. 角色只从冻结划分表读取；`work/meta_split1/source_table_104.csv` 的同名列不能用于判断既往用途。")
    add("5. 角色与文件名的 S 码没有绑定关系：例如旧三类的 `S0010`、`S0011` 在本轮属于 train，而在更早的旧索引里它们曾是 val / test。**必须以冻结划分表为准。**")
    add("")
    add("### 5.2 总计")
    add("")
    add("| 角色 | 文件数 | 占比 | 用途 |")
    add("| --- | --- | --- | --- |")
    add("| `train` | 62 | 59.6% | 拟合模型参数 |")
    add("| `val` | 16 | 15.4% | 选轮次、阈值与超参数 |")
    add("| `development_test` | 20 | 19.2% | 新十类同批次文件级开发主指标 |")
    add("| `development_test_exposed` | 6 | 5.8% | 旧三类已暴露描述性对照 |")
    add("| **合计** | **104** | **100%** | 13 类 × 每类 8 个 MAT |")
    add("")
    add("### 5.3 逐类数量（与第 3 节汇总表一致）")
    add("")
    add("| # | 类别码 | 机型名称 | train | val | development_test | development_test_exposed | 合计 |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for index, class_id in enumerate(class_order, start=1):
        files = by_class[class_id]
        counts = {role: 0 for role in role_order}
        for row in files:
            counts[row["role"]] += 1
        add(
            f"| {index} | `{class_id}` | {files[0]['class_name']} | {counts['train']} | {counts['val']} | {counts['development_test']} | {counts['development_test_exposed']} | 8 |"
        )
    add(
        f"| — | **合计** | **13 类** | **{EXPECTED_ROLE_TOTALS['train']}** | **{EXPECTED_ROLE_TOTALS['val']}** | **{EXPECTED_ROLE_TOTALS['development_test']}** | **{EXPECTED_ROLE_TOTALS['development_test_exposed']}** | **104** |"
    )
    add("")
    add("---")
    add("")
    add("## 6. 两套测试文件的指标口径")
    add("")
    add("| 指标口径 | 可用文件 | 允许的表述 | 禁止的表述 |")
    add("| --- | --- | --- | --- |")
    add("| 新十类开发主指标 | 20 个 `development_test` 文件（新十类各 2 个） | 同批次、源文件级留出的**开发评估** | 全新 final test；跨采集会话泛化；开放集结果 |")
    add("| 旧三类对照 | 6 个 `development_test_exposed` 文件（背景/Phantom 4 Pro/Air 2S 各 2 个） | **已暴露**描述性对照 | 未见数据；独立测试；泛化证明 |")
    add("| 13 类整体 | 104 个文件 | 开发用途的分层评估 | 跨采集会话、跨接收机、跨环境鲁棒性 |")
    add("")
    add("为什么这样分：")
    add("")
    add("- 新十类的 80 个文件是 2026-09-25 才补齐的，此前**未出现在本项目的任何数据或模型产物**中，所以它们的 20 个测试文件可以单列为开发主指标。")
    add("- 旧三类的 24 个文件此前已用于开发与评价（12 个 `development_original` + 12 个 `evaluated_holdout`），因此它们的 6 个测试文件只能是**已暴露对照**，不能计入“未见数据”的主指标。")
    add("- 两类测试文件都与**同类训练文件处于同一文件时间簇**（10 类在夜 `N20221127`，3 类在夜 `N20230227`）。时间代理不是采集会话证据，所以本划分只能称“同批次文件级留出”。")
    add("")
    add("---")
    add("")
    add("## 7. 必须保留的边界（不能说的话）")
    add("")
    add("1. **不能称独立最终测试**：104 个文件里，旧 24 个已被使用，新 80 个与训练同批次；13 类整体不存在“完全未参与选型的 final test”。")
    add("2. **不能称跨采集会话**：文件时间的“夜/批”只是文件时间代理（RAR 时间可能继承原文件时间），不是独立证明的采集会话 ID。")
    add("3. **不能给 3 个类别补距离**：`T1010`、`T1011`、`T1100` 的本地文件名没有 D 字段，本清单只写“未标距离”。")
    add("4. **背景不是无人机**：`T0000` 的 `D00` 只是近距离场景编码。")
    add("5. **不标注室内/室外**：官方论文说明户外采集距离 20～150 m、室内场地约 2 m，但本地文件名只保留 T/D/S 编码，无法确认每一类属于哪种场景，因此本清单不据场景推断距离。")
    add("6. **本地覆盖不完整**：官方表 3 共 25 个条目，本地只有 13 个；论文还说明每类“不少于 12 个片段”，而本地每类只有 8 个文件。")
    add("7. **不允许窗口级随机划分**：同一源文件的相邻 10 ms 片段高度相关，随机划分会把同源近重复同时放进训练和测试。")
    add("8. **本文档不含模型结果**：本轮没有训练、推理或评价；第 3～5 节只有数据事实与划分计数，不含任何模型指标数值。")
    add("")
    add("---")
    add("")
    add("## 8. 复现与核对命令")
    add("")
    add("```powershell")
    add("# 生成本清单（只读 G 盘清单与冻结划分表，自动核对两个 SHA256）")
    add(r".\.venv\Scripts\python.exe work\dataset12_inventory\generate_dataset12_inventory.py")
    add("")
    add("# 独立校验：本清单 104 个文件名与冻结划分表完全一致、13 类各 8 个、62/16/20/6、距离描述合规")
    add(r".\.venv\Scripts\python.exe work\dataset12_inventory\verify_dataset12_inventory.py")
    add("")
    add("# 官方表 3 的本地 PDF 复核（官方页面直接抓取会返回 403，故用仓库内论文副本）")
    add(r'pdftotext -layout -enc UTF-8 "dronerfa_baseline\参考论文\01_DroneRFa_Dataset_JEIT2024.pdf" tmp\paper.txt')
    add("```")
    add("")
    add("---")
    add("")
    add("*本清单的 104 行文件级事实全部由冻结划分表程序化派生；如需修改划分，请先按 META-SPLIT2 的流程重新冻结，再重新生成本文档。*")
    add("")

    OUT_PATH.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"written: {OUT_PATH}")
    print(f"lines: {len(out)}")
    print(f"classes: {len(class_order)}  files: {len(rows)}  bytes: {total_bytes:,}")
    print(f"role totals: {role_totals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
