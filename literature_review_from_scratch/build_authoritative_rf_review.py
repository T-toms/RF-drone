"""Build the independently screened UAV communication-RF literature review.

This builder deliberately reads only DOI metadata and public bibliometric audit
files in this directory.  It does not read or use any historical project
instructions, experimental configurations, models, or results.
"""

from __future__ import annotations

import csv
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

import build_literature_review as shared


ROOT = Path(__file__).resolve().parent
METADATA = ROOT / "references_metadata.csv"
OPENALEX = ROOT / "authoritative_rf_dl_ranking.csv"
OUTPUT_DOCX = ROOT / "无人机通信链路RF深度学习_高权威文献综述_2026-08.docx"
OUTPUT_BIB = ROOT / "zotero_无人机通信链路RF高权威文献_34篇.bib"
OUTPUT_MANIFEST = ROOT / "zotero_无人机通信链路RF高权威文献_34篇.csv"

# Tier labels are evidence labels, never a claim that every item has a high JIF.
# The DOI order is also the intended reading order inside each tier.
SOURCE_PLAN = [
    # Flagship evidence: venue / impact / author / task evidence jointly strong.
    ("10.1016/j.future.2019.05.007", "F", ["input:rf", "model:dnn", "role:dataset-anchor"],
     "早期公开数据与深度学习基准锚点；字段内高被引。", "数据代际与随机切分不能替代跨地点泛化。"),
    ("10.1109/tvt.2020.3042128", "F", ["input:iq", "task:rffi", "evidence:channel"],
     "TVT Q1；非标准波形的设备级 RF 指纹与神经网络。", "设备级 RFFI 不等于机型识别，受信道和接收机漂移影响。"),
    ("10.1109/tccn.2021.3099114", "F", ["input:stft", "task:multi-drone", "model:drnn"],
     "TCCN Q1；联合检测/分类并覆盖多信号情形。", "需区分真实同时发射和离线拼接的多机评测。"),
    ("10.1109/tvt.2021.3081049", "F", ["input:preamble", "task:rffi", "method:federated"],
     "TVT Q1；专有波形前导特征与联邦学习的代表。", "联邦学习尚不是领域主流，前导提取依赖协议与接收条件。"),
    ("10.1109/jsen.2023.3242985", "F", ["input:stft", "risk:interference", "model:cnn"],
     "复杂电磁环境下的深度分类；强通信与 RF 作者参与。", "背景混合增强不等同于真实现场联合采集。"),
    ("10.1109/tim.2023.3306822", "F", ["input:rf", "task:rffi", "risk:channel"],
     "TIM Q1；非标准波形、未知信道与信道扰动问题更接近部署。", "较新，引用仍在积累；须拆分实测和仿真增强证据。"),
    ("10.1109/tifs.2024.3463535", "F", ["task:open-set", "input:rf", "model:semantics"],
     "TIFS Q1；面向未知类别拒识的高等级前沿证据。", "开放集划分必须完整留出未知机型/设备，不能只留出窗口。"),
    # Mainstream direct DL implementations: important, but not all high-IF flagship papers.
    ("10.1109/iciot48696.2020.9089657", "M", ["input:raw-rf", "model:1d-cnn", "task:three-level"],
     "多通道 1D-CNN 的早期高被引主流路线。", "IEEE 会议而非有 JIF 期刊。"),
    ("10.1109/iciot48696.2020.9089489", "M", ["input:energy", "model:cnn", "role:early-baseline"],
     "能量瞬态与 CNN 的早期高被引基线。", "会议论文；不能用会议属性包装为高 IF 证据。"),
    ("10.1016/j.comnet.2021.108519", "M", ["input:raw-rf", "model:1d-cnn", "task:classification"],
     "原始 RF 13 层 1D-CNN，适合对照最常见时域路线。", "主要依赖单一数据资源时，不能外推部署性能。"),
    ("10.1016/j.eswa.2021.115928", "M", ["input:stft", "task:multi-drone", "model:dl"],
     "单机/多机 RF 检测识别的高影响应用期刊证据。", "作者影响力与外部测试深度仍需逐篇核验。"),
    ("10.1109/access.2022.3172787", "M", ["input:raw-rf", "model:lightweight-cnn", "role:baseline"],
     "RF-UAVNet 是常见高性能/轻量 CNN 比较基线。", "IEEE Access Q2；同源数据随机窗口切分可能乐观。"),
    ("10.3390/s22083072", "M", ["input:compressed-rf", "model:dl", "risk:front-end"],
     "压缩感知 RF 与 DL，适合采样率与前端成本研究。", "压缩感知不是最主流输入，需比较相同采样预算的基线。"),
    ("10.1109/lsens.2023.3289145", "M", ["input:time-frequency", "model:multiscale-cnn"],
     "时频多尺度 CNN，代表双域/多尺度特征融合。", "Sensors Letters 为短文；证据深度与独立测试须补。"),
    ("10.3390/s23094202", "M", ["input:stft", "model:dl", "task:identification"],
     "端到端 RF 检测识别的补充主流实现。", "Sensors Q2；必须复查切分是否按会话/设备隔离。"),
    # Data, reviews, and non-DL comparative evidence.
    ("10.1109/ojcoms.2019.2955889", "C", ["risk:interference", "baseline:ml", "task:rffi"],
     "Wi-Fi/蓝牙干扰下 RF 检测的高被引工程基准。", "以特征工程/ML 为主，不应写成深度学习主流论文。"),
    ("10.1016/j.pmcj.2022.101569", "C", ["input:wavelet", "baseline:ml", "task:identification"],
     "小波/多域 RF 特征的高影响可解释基线。", "ML 为主；用于 DL 对照而非 DL 创新主证据。"),
    ("10.3390/s21061947", "C", ["task:hierarchical", "baseline:ml", "input:rf"],
     "层次化检测/识别基线，适合定义任务级联。", "传统学习为主；Sensors 不能单独担保结论强度。"),
    ("10.1016/j.dib.2019.104313", "C", ["resource:dataset", "dataset:dronerf", "role:provenance"],
     "DroneRF 数据出处；只要使用该数据就应引用。", "数据论文，不是算法权威性证据。"),
    ("10.1109/access.2019.2942944", "C", ["role:review", "scope:multi-sensor"],
     "高被引反无人机 ML 综述，用于系统边界和历史脉络。", "跨雷达、视觉、声学；不能替代 RF 专项统计。"),
    ("10.3390/s22041453", "C", ["role:survey", "platform:sdr", "scope:counter-uav"],
     "SDR 方案与反无人机系统综述，用于链路/平台背景。", "非 RF-DL 专项算法证据。"),
    ("10.1109/jsen.2022.3171293", "C", ["role:review", "scope:multi-sensor"],
     "无人机探测技术综述，用于物理边界和任务定义。", "不是通信物理层 RF-DL 专项综述。"),
    ("10.1016/j.iot.2024.101201", "C", ["role:systematic-review", "scope:rf-wifi"],
     "RF/Wi-Fi UAV 监测系统系统综述，范围最相近。", "RF 物理层与 Wi-Fi 流量分析必须分开计数。"),
    ("10.3390/s24227339", "C", ["role:robustness", "risk:split", "risk:synthetic-noise"],
     "直接讨论深度 RF 检测器鲁棒性，约束评测协议。", "较新且为专门评测研究；不是核心模型论文。"),
    ("10.1109/tr.2021.3119068", "C", ["model:dl", "task:monitoring", "role:reliability"],
     "可靠性视角的 UAV 监测深度学习证据。", "任务/数据条件需逐篇核验，不替代跨域测试。"),
    ("10.3390/e23121678", "C", ["task:mode", "model:multichannel-dnn", "input:rf"],
     "工作模式分类的多通道 DNN 补充证据。", "特征工程与网络共同作用，需消融判断贡献。"),
    # Research radar: technically viable ideas without mature citation consensus.
    ("10.1109/tccn.2025.3537100", "R", ["task:multi-drone", "input:spectrogram", "model:smnet"],
     "SMNet：多机控制信号的谱图检测/分类。", "新作，不能称为已被广泛复现的主流。"),
    ("10.1109/tccn.2025.3561305", "R", ["task:cross-domain", "task:rffi", "method:domain-adversarial"],
     "域不变对抗学习与流形正则的跨域 RFFI。", "需要多地点/多接收机真实数据才能验证。"),
    ("10.1109/tccn.2025.3598245", "R", ["model:mobile-net", "deployment:edge", "method:fixed-classifier"],
     "MobileNetV3-Small 与固定分类器的轻量化方向。", "论文很新，不能从作者声望推导已成熟效果。"),
    ("10.1109/tvt.2026.3696747", "R", ["task:few-shot", "model:spiking", "input:rf"],
     "少样本脉冲网络，针对新设备样本稀缺。", "2026 新作，尚无引用成熟度，作为研究雷达。"),
    ("10.1109/jrfid.2024.3487303", "R", ["risk:low-snr", "deployment:low-cost", "model:cnn"],
     "低成本 SDR、低 SNR 场景下的 CNN 工程参考。", "JRFID Q2；需报告真实环境而非只加 AWGN。"),
    ("10.1109/jsen.2023.3346209", "R", ["model:hybrid", "deployment:lightweight", "input:rf"],
     "轻量混合网络，用于效率-精度对照。", "并非已形成稳定主流，端侧指标应实测。"),
    ("10.3390/s23229098", "R", ["model:transformer", "input:rf", "task:identification"],
     "Transformer-CNN 混合识别方向。", "新且引用有限；Transformer 不是目前领域主流。"),
    ("10.3390/drones6120374", "R", ["model:complex-cnn", "input:complex-iq", "task:rffi"],
     "复值 CNN 保留 I/Q 相位结构。", "需用接收机/信道隔离实验检验真实增益。"),
]

TIER_NAMES = {
    "F": "F. 旗舰证据：高权威、直接相关且较成熟",
    "M": "M. 主流深度学习实现：重要但不等同高 IF 旗舰",
    "C": "C. 数据、综述和比较证据：定义边界与评测基线",
    "R": "R. 前沿研究雷达：有价值但尚未形成成熟共识",
}


def norm_doi(value: str) -> str:
    return value.strip().lower().replace("https://doi.org/", "")


def load_sources() -> list[dict[str, str]]:
    with METADATA.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = {norm_doi(row["doi"]): row for row in csv.DictReader(handle)}
    with OPENALEX.open("r", encoding="utf-8-sig", newline="") as handle:
        metrics = {norm_doi(row["doi"]): row for row in csv.DictReader(handle)}

    selected: list[dict[str, str]] = []
    missing: list[str] = []
    for index, (doi, tier, tags, rationale, caveat) in enumerate(SOURCE_PLAN, start=1):
        doi = norm_doi(doi)
        if doi not in metadata:
            missing.append(doi)
            continue
        row = dict(metadata[doi])
        row.update({
            "doi": doi,
            "key": f"H{index:02d}",
            "tier": tier,
            "tier_name": TIER_NAMES[tier],
            "tags": "; ".join(["scope:passive-rf", f"tier:{tier.lower()}", *tags]),
            "rationale": rationale,
            "caveat": caveat,
        })
        row.update(metrics.get(doi, {}))
        selected.append(row)
    if missing:
        raise RuntimeError(f"Selected DOI(s) not found in metadata: {missing}")
    if len({row["doi"] for row in selected}) != len(selected):
        raise RuntimeError("Duplicate DOI in SOURCE_PLAN")
    return selected


def author_short(row: dict[str, str]) -> str:
    authors = [name.strip() for name in row.get("authors", "").split(";") if name.strip()]
    return f"{authors[0]} 等" if len(authors) > 1 else (authors[0] if authors else "作者待核")


def cite(by_doi: dict[str, dict[str, str]], *dois: str) -> str:
    return "、".join(f"[{by_doi[norm_doi(doi)]['key']}]" for doi in dois)


def fmt_count(row: dict[str, str]) -> str:
    count = row.get("cited_by_count_openalex", "")
    annual = row.get("citations_per_year_openalex", "")
    if not count:
        return "未列入 DL 计量表"
    return f"{count}（约 {annual}/年）"


def ref_string(row: dict[str, str]) -> str:
    venue = row.get("venue") or row.get("publisher") or ""
    volume = row.get("volume") or ""
    issue = f"({row['issue']})" if row.get("issue") else ""
    pages = row.get("pages_or_article") or ""
    tail = " ".join(part for part in [venue, f"{volume}{issue}".strip(), pages] if part)
    return f"[{row['key']}] {row.get('authors') or '作者待核'}. ({row.get('year') or 'n.d.'}). {row.get('title') or ''}. {tail}. https://doi.org/{row['doi']}"


def bib_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def citation_key(row: dict[str, str]) -> str:
    first = (row.get("authors") or "source").split(";")[0].strip().lower()
    compact = "".join(ch for ch in first if ch.isalnum())[:18] or "source"
    return f"{compact}_{row.get('year')}_{row['key'].lower()}"


def write_bibtex(sources: list[dict[str, str]]) -> None:
    entries: list[str] = []
    for row in sources:
        is_proceedings = row.get("type") == "proceedings-article"
        entry_type = "inproceedings" if is_proceedings else "article"
        venue_key = "booktitle" if is_proceedings else "journal"
        authors = " and ".join(name.strip() for name in row.get("authors", "").split(";") if name.strip())
        fields = [
            ("author", authors),
            ("title", row.get("title", "")),
            (venue_key, row.get("venue") or row.get("publisher", "")),
            ("year", row.get("year", "")),
            ("volume", row.get("volume", "")),
            ("number", row.get("issue", "")),
            ("pages", row.get("pages_or_article", "")),
            ("doi", row["doi"]),
            ("url", f"https://doi.org/{row['doi']}"),
            ("keywords", row["tags"]),
            ("note", f"{row['tier_name']} | {row['rationale']}"),
        ]
        body = ",\n".join(
            f"  {name} = {{{bib_escape(value)}}}" for name, value in fields if value
        )
        entries.append(f"@{entry_type}{{{citation_key(row)},\n{body}\n}}")
    OUTPUT_BIB.write_text("% Curated independently; no local experimental evidence used.\n\n" + "\n\n".join(entries) + "\n", encoding="utf-8")


def write_manifest(sources: list[dict[str, str]]) -> None:
    columns = [
        "key", "tier", "doi", "title", "year", "venue", "authors", "tags",
        "cited_by_count_openalex", "citations_per_year_openalex", "fwci_openalex",
        "first_author_h_index_openalex", "last_author_h_index_openalex", "rationale", "caveat",
    ]
    with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row.get(column, "") for column in columns} for row in sources)


def setup_header_footer(doc: Document) -> None:
    section = doc.sections[0]
    section.different_first_page_header_footer = True
    section.first_page_header.paragraphs[0].text = ""
    first_footer = section.first_page_footer.paragraphs[0]
    first_footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = first_footer.add_run("独立检索与核验截止：2026-08-15｜不使用任何既往本地实验结论")
    shared.set_run_font(run, size=8.5, color=shared.MUTED)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    run = header.add_run("无人机通信链路 RF 深度学习｜高权威文献综述")
    shared.set_run_font(run, size=8.5, color=shared.MUTED)

    footer = section.footer
    table = footer.add_table(rows=1, cols=2, width=Inches(6.5))
    shared.set_table_geometry(table, [4.6, 1.9])
    shared.prevent_row_split(table.rows[0])
    shared.set_table_borders(table, color="FFFFFF")
    for cell in table.rows[0].cells:
        shared.set_cell_margins(cell, top=0, start=0, bottom=0, end=0)
    left = table.cell(0, 0).paragraphs[0]
    left.paragraph_format.space_after = Pt(0)
    run = left.add_run("34 条分层条目｜DOI + OpenAlex 审计快照")
    shared.set_run_font(run, size=8.5, color=shared.MUTED)
    right = table.cell(0, 1).paragraphs[0]
    right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    right.paragraph_format.space_after = Pt(0)
    run = right.add_run("第 ")
    shared.set_run_font(run, size=8.5, color=shared.MUTED)
    shared.add_page_field(right)
    run = right.add_run(" 页")
    shared.set_run_font(run, size=8.5, color=shared.MUTED)


def add_cover(doc: Document) -> None:
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_before = Pt(108)
    spacer.paragraph_format.space_after = Pt(0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    run = p.add_run("独立检索｜来源、被引、作者与方法证据分层")
    shared.set_run_font(run, size=11.5, color="7A5A00", bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run("无人机通信链路射频信号（RF）\n深度学习检测与识别")
    shared.set_run_font(run, size=29, color=shared.TITLE_DARK, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(28)
    run = p.add_run("高权威文献综述：主流做法、证据边界与可实现创新路线")
    shared.set_run_font(run, size=14, color=shared.DARK_BLUE)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("证据快照：2026 年 8 月 15 日｜34 条可同步 Zotero 的精选条目")
    shared.set_run_font(run, size=10.5, color=shared.MUTED, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run("范围：被动截获无人机/遥控器通信链路的物理层 RF；雷达、视觉、声学及 Wi-Fi 流量分析不并入核心统计")
    shared.set_run_font(run, size=9.5, color=shared.MUTED, italic=True)
    doc.add_page_break()


def add_ref_paragraph(doc: Document, row: dict[str, str]) -> None:
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.left_indent = Inches(0.27)
    p.paragraph_format.first_line_indent = Inches(-0.27)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.08
    run = p.add_run(ref_string(row))
    shared.set_run_font(run, size=9.2, color="000000")


def add_source_table(doc: Document, sources: list[dict[str, str]], tier: str, widths: list[float], caption: str) -> None:
    rows = []
    for row in sources:
        if row["tier"] != tier:
            continue
        rows.append([
            row["key"],
            f"{author_short(row)} ({row.get('year')})\n{row.get('venue') or ''}",
            fmt_count(row),
            f"{row['rationale']}\n限制：{row['caveat']}",
        ])
    shared.add_table(doc, ["编号", "论文 / 来源", "OpenAlex 被引*", "为何保留；不能越界的结论"], rows, widths, caption)
    shared.add_body(doc, "* 被引为 OpenAlex 公开快照（UTC 2026-08-15 15:19:50），不是 WoS、Scopus、Google Scholar 或期刊影响因子；近年论文必须按发表时长阅读。", color=shared.MUTED, italic=True, size=8.7)


def build_document() -> None:
    sources = load_sources()
    by_doi = {row["doi"]: row for row in sources}
    write_bibtex(sources)
    write_manifest(sources)

    doc = Document()
    section = doc.sections[0]
    shared.set_page_geometry(section)
    shared.configure_styles(doc)  # compact_reference_guide preset, exact values in shared builder
    setup_header_footer(doc)      # editorial_cover first-page pattern, adapted for this report
    add_cover(doc)

    shared.add_heading(doc, "执行摘要", 1)
    shared.add_callout(
        doc,
        "结论先行：该窄领域中，真正同时具备强期刊/会议来源、较高公开被引、可核验的相关作者履历、且直接处理通信物理层 RF 的旗舰论文数量有限。不能把所有论文都称为“高权威核心”。本综述把 34 条文献分为旗舰证据、主流深度学习、数据/综述/基线和前沿雷达四层；它们都值得进入 Zotero，但承担不同的论证角色。",
    )
    shared.add_body(
        doc,
        "目前最有共识的技术主线是：SDR 被动截获控制器/无人机通信链路的原始 RF（I/Q、幅度或能量瞬态），或者生成频谱/时频谱图；再以 1D-CNN、2D-CNN、ResNet 或较轻量的卷积网络完成三级任务——有无目标、机型/控制器或设备识别、飞行/工作模式识别。该判断由早期公开数据与 CNN 基准、TVT/TCCN 物理层研究和多篇后续实现共同支撑 "
        + cite(by_doi, "10.1016/j.future.2019.05.007", "10.1109/iciot48696.2020.9089657", "10.1109/tccn.2021.3099114", "10.1016/j.comnet.2021.108519") + "。",
    )
    shared.add_body(
        doc,
        "开放集未知机型、跨地点/跨接收机泛化、真实多机重叠、低 SNR 干扰、少样本扩展和端侧能耗/时延是可做且有明确论文锚点的创新方向；但 Transformer、YOLO、联邦学习、复值网络和少样本网络目前仍应称为“前沿分支”，而非领域主流 "
        + cite(by_doi, "10.1109/tifs.2024.3463535", "10.1109/tccn.2025.3537100", "10.1109/tccn.2025.3561305", "10.3390/s23229098", "10.1109/tvt.2026.3696747") + "。",
    )
    shared.add_body(
        doc,
        "本结论完全从公开文献重新出发。它不读取、不引用，也不反推任何本地既往实验、实施说明、配置、模型或结果。",
        color=shared.DARK_BLUE,
        italic=True,
    )

    shared.add_heading(doc, "1. 范围：这里的“无人机 RF”具体是什么", 1)
    scope_rows = [
        ["纳入", "被动接收无人机、遥控器或地面站通信链路的物理层 RF；原始 I/Q、幅度、能量瞬态、PSD、FFT、STFT、时频图等表示。"],
        ["主任务", "存在检测、机型/控制器/设备识别、飞行/工作模式识别；可包括单机或真实同时多机。"],
        ["相邻但分开", "Wi-Fi MAC/流量/协议层分析可作系统背景，不能与物理层 RF 的模型结果混为同一统计总体。"],
        ["不纳入核心", "雷达、视觉、声学、遥感图像等；它们可以是多模态系统的另一路，不是本报告的主证据。"],
        ["物理限制", "RF 方法观测的是可接收的活动通信链路，不保证发现静默、自主或不在接收范围内的飞行器。"],
    ]
    shared.add_table(doc, ["边界", "本综述的处理方式"], scope_rows, [1.3, 5.2], "表 1  术语与纳入边界")
    shared.add_body(doc, "把范围固定下来很重要：若混入雷达/视觉，文献数量会显著增加，却不能回答“通信链路 RF 深度学习的主流与空白”。系统综述也强调 RF 与 Wi-Fi 路线需区分 " + cite(by_doi, "10.1016/j.iot.2024.101201", "10.1109/jsen.2022.3171293") + "。")

    shared.add_heading(doc, "2. “权威”如何判定：不把影响因子当作单篇论文质量", 1)
    shared.add_body(doc, "用户所需的“权威”由四个可分开的信号组成：期刊/会议来源、单篇被引、作者在通信/RF 主题上的可核验履历、以及方法证据与可复现性。影响因子只能用于来源层级，不能替代单篇研究评价；作者声望也不能替代独立测试。")
    score_rows = [
        ["来源权威 25", "JCR/索引/同行评审、期刊定位；保留指标年份与来源。", "TIFS、TVT、TCCN、TIM 属 IEEE 2026-01 Title List 所列的 Q1；JSEN/Access 等分层处理。"],
        ["单篇影响 25", "OpenAlex 精确 DOI 命中的被引、年均被引和 FWCI。", "同一来源、同一抓取日比较；不将它写成 Scopus/WoS。"],
        ["作者主题权威 15", "院系/实验室主页、专业角色、RF/通信持续产出；OpenAlex h-index 仅作辅助。", "优先查 Chowdhury、Pollin、Dobre、Chen 等相关团队的官方履历。"],
        ["方法证据 25", "原始 RF、真实干扰、设备/会话/地点隔离、外部或跨域测试。", "没有严格切分，准确率不进入关键结论。"],
        ["复现透明度 10", "数据、代码、采集参数、数据划分、失败条件公开。", "数据或代码公开是加分，不等于结论必然可部署。"],
    ]
    shared.add_table(doc, ["维度", "怎么核验", "使用规则"], score_rows, [1.15, 2.75, 2.6], "表 2  文献的 100 分制筛选框架")
    shared.add_body(doc, "本次审计中，53 条非数据集候选以 DOI 精确匹配 OpenAlex；对严格“被动通信物理层 RF + 神经/深度学习”范围，选出 34 条供分层阅读与 Zotero 同步。JIF/Q 分区采用 IEEE 2026-01 Title List（其表中标注为 JIF 与分区快照）；Elsevier/MDPI 期刊页仅作对应期刊的来源信号。", color=shared.MUTED, italic=True, size=9.3)

    shared.add_heading(doc, "3. 分层证据库：哪些论文能承担什么结论", 1)
    shared.add_heading(doc, "3.1 F 层：旗舰证据（7 篇）", 2)
    shared.add_body(doc, "F 层不是“唯一正确”的论文，而是当前最适合承担领域主结论的交集：强相关、较成熟的引用信号、较强来源或主题团队、且有直接 RF 问题。其中文献年份不同，故同时给出年化被引，而不机械比较总被引。")
    add_source_table(doc, sources, "F", [0.45, 1.7, 1.0, 3.35], "表 3  F 层旗舰证据（OpenAlex 快照）")

    shared.add_heading(doc, "3.2 M 层：主流深度学习实现（8 篇）", 2)
    shared.add_body(doc, "这些工作共同说明了 1D/2D CNN、时频卷积、原始 RF/能量/压缩感知等工程路线为什么成为主流。不过，会议论文、Q2 期刊和同源数据集论文不能因有高引用就被包装为“高 IF 权威”。")
    add_source_table(doc, sources, "M", [0.45, 1.7, 1.0, 3.35], "表 4  M 层主流深度学习实现")

    shared.add_heading(doc, "3.3 C 层：数据、综述与比较证据（11 篇）", 2)
    shared.add_body(doc, "C 层往往不是深度学习模型主证据，却决定研究是否有清晰范围、合理基线、数据出处与可信评测。特别是干扰、传统 ML、数据论文、系统综述和鲁棒性论文应被保留，但不能把它们的结论偷换为“DL 主流”。")
    add_source_table(doc, sources, "C", [0.45, 1.7, 1.0, 3.35], "表 5  C 层范围/基线/数据证据")

    shared.add_heading(doc, "3.4 R 层：研究雷达（8 篇）", 2)
    shared.add_body(doc, "R 层的价值在于提出可测试的未来问题，而不是提供已稳定的共识。对 2025-2026 年论文，低总被引是时间窗口造成的正常现象；应先看来源、作者主题能力、问题设置和 FWCI，再要求独立复现。")
    add_source_table(doc, sources, "R", [0.45, 1.7, 1.0, 3.35], "表 6  R 层前沿研究雷达")

    shared.add_heading(doc, "4. 主流技术图谱：从信号到任务，而不是“再堆一个 CNN”", 1)
    pipeline_rows = [
        ["采集", "SDR 被动监听目标频段；记录接收机、天线、时间、地点、背景、距离和设备/控制器状态。", "可审计的 recording/session 元数据；禁止把切窗当独立采集。"],
        ["表示", "原始 I/Q 或幅度、能量瞬态、功率谱；最常见的是 STFT/时频图。", "优先做原始表示、简单基线与时频图的同预算对照。"],
        ["模型", "1D-CNN（原始时域）、2D-CNN/ResNet（时频图）、多尺度 CNN；轻量卷积用于端侧。", "模型复杂度必须与跨域、低 SNR、速度/功耗收益一起报告。"],
        ["任务", "存在检测 → 机型/控制器/设备识别 → 模式分类；必要时做层次化或多机目标检测。", "明确标签层级，不能把设备指纹、机型与模式准确率混成一项。"],
        ["评测", "训练/测试按 recording、session、device、site、date、receiver 分组；外部域、未知类和真实干扰独立测试。", "随机窗口切分和只添加 AWGN 不能作为部署结论的主结果。"],
    ]
    shared.add_table(doc, ["层", "当前常见实现", "从零研究的硬约束"], pipeline_rows, [0.75, 3.25, 2.5], "表 7  被动通信 RF 深度学习的主流流水线")
    shared.add_body(doc, "从技术谱系看，最稳定的共识不是某一种新网络，而是“合适的物理层表示 + 卷积模型 + 清楚任务层级”的闭环。原始 RF/1D-CNN、时频图/2D-CNN 与多尺度路线均有早期高影响文献支持 " + cite(by_doi, "10.1109/iciot48696.2020.9089657", "10.1109/iciot48696.2020.9089489", "10.1016/j.comnet.2021.108519", "10.1109/lsens.2023.3289145") + "。")
    shared.add_body(doc, "传统特征工程 + kNN/SVM/RF 仍应保留为可解释、低成本、抗分布漂移的比较基线，尤其适用于检验某个深度模型到底从表示还是从切分中获益 " + cite(by_doi, "10.1109/ojcoms.2019.2955889", "10.1016/j.pmcj.2022.101569", "10.3390/s21061947") + "。")

    shared.add_heading(doc, "5. 可以做什么创新：以可实现性和评测条件筛选", 1)
    shared.add_body(doc, "下表的“可实现性”基于现有模型/数据采集成本与已有论文锚点判断，并不预言性能。每个方向都只有在相应的数据隔离与测试条件满足时，才可能形成可信创新。")
    innovation_rows = [
        ["开放集 + 跨域", "未知机型拒识，同时跨地点/接收机/日期泛化。", "至少 5 类设备；完整未知设备留出；2+ 场地或接收机。", "AUROC/OSCR、已知类准确率、未知类误接纳率、每域置信区间。", "高：TIFS 开放集 + TCCN 跨域直接锚点。"],
        ["真实多机重叠", "检测同时出现的多个控制链路，而非拼接后的多标签分类。", "同步接收、时标一致的 2-3 个真实发射源与背景。", "事件级 precision/recall、mAP、每源识别率、重叠比例曲线。", "中高：SMNet/TCCN 路线；采集成本是瓶颈。"],
        ["低 SNR 与真实干扰", "把 Wi-Fi/蓝牙等真实背景和低成本 SDR 限制纳入训练与测试。", "独立背景录制；不同干扰物、距离和接收机；不只 AWGN。", "SNR/INR 曲线、背景留出测试、检测延迟、误警率。", "高：已有干扰与低 SNR 锚点。"],
        ["复值 I/Q 与域稳健", "保留相位结构，同时控制接收机/信道扰动。", "原始复 I/Q、校准记录、多接收机或多天采集。", "跨接收机/跨日准确率、校准前后差异、消融。", "中：模型可实现，数据纪律要求高。"],
        ["少样本新设备", "在少量新设备样本下扩展类别并保留未知拒识。", "训练设备与新设备按物理设备留出；每类 1/5/10-shot。", "shot 曲线、校准误差、旧类遗忘、未知类拒识。", "中：方法现成，难点是避免设备泄漏。"],
        ["端侧协同", "在 SDR/嵌入式环境报告模型、采样、延迟、功耗的整体预算。", "目标硬件与实时数据流；至少一个现场背景集。", "端到端延迟、吞吐、功耗、模型大小、低 SNR 准确率。", "高：轻量网络现成；需要真实硬件测量。"],
    ]
    shared.add_table(doc, ["创新命题", "研究问题", "最低数据条件", "必要指标", "可实现性依据"], innovation_rows, [0.85, 1.35, 1.55, 1.4, 1.35], "表 8  可发表创新点及其最低验证条件")
    shared.add_body(doc, "对应论文锚点是开放集 " + cite(by_doi, "10.1109/tifs.2024.3463535") + "，多机 " + cite(by_doi, "10.1109/tccn.2021.3099114", "10.1109/tccn.2025.3537100") + "，跨域 " + cite(by_doi, "10.1109/tccn.2025.3561305") + "，低 SNR " + cite(by_doi, "10.1109/jrfid.2024.3487303") + "，复值 I/Q " + cite(by_doi, "10.3390/drones6120374") + "，少样本 " + cite(by_doi, "10.1109/tvt.2026.3696747") + " 和轻量化 " + cite(by_doi, "10.1109/tccn.2025.3598245", "10.1109/jsen.2023.3346209") + "。")

    shared.add_heading(doc, "6. 从零实验的评测协议：先守住证据，再比较模型", 1)
    protocol_rows = [
        ["数据单位", "把连续录制（recording）/会话（session）/物理设备（device）/地点（site）/日期（date）/接收机（receiver）作为分组单位。"],
        ["主划分", "训练、验证、测试之间不共享同一录制的重叠窗口；优先 device/site/date/receiver 留出。"],
        ["开放集", "未知类必须是未见过的完整机型/设备，而不是从同一设备随机截出的窗口。"],
        ["低 SNR", "报告实测背景与干扰来源；若使用人工加噪，只作为补充消融，不替代现场测试。"],
        ["统计", "报 macro-F1、PR-AUC/ROC-AUC、每类结果、置信区间或重复采样；多机还报事件级指标。"],
        ["复现", "发布切分清单、随机种子、预处理参数、接收机/采样设置和失败案例。"],
    ]
    shared.add_table(doc, ["检查点", "最低要求"], protocol_rows, [1.25, 5.25], "表 9  从零研究的评测硬门槛")
    shared.add_body(doc, "鲁棒性综述已明确提示：若从同一采集池切窗再随机划分，或只做模拟噪声，部署性能容易被高估。因此本项目应把独立地点、日期、信道、背景与接收机测试设为主结果，而不是附录 " + cite(by_doi, "10.3390/s24227339") + "。")

    shared.add_heading(doc, "7. 建议阅读与实施顺序", 1)
    reading_rows = [
        ["第 1 轮：定义主线", "F 层中的 H01-H04、H07；写出任务定义、信号表示和模型基线的证据卡。", "形成一页范围声明与三类任务标签。"],
        ["第 2 轮：选基线", "M 层 8 篇；逐篇记录输入、切窗、切分、模型、指标、真实干扰。", "确定 1D-CNN、时频 CNN、传统 ML 三类可比基线。"],
        ["第 3 轮：审计风险", "C 层中的 H16-H24；优先数据出处、干扰、系统综述、鲁棒性。", "写出数据卡、分组切分方案、外部测试矩阵。"],
        ["第 4 轮：立题", "R 层按目标选择一条；先写最小数据条件和失败判据。", "只保留一个可在现有采集能力内完成的创新命题。"],
    ]
    shared.add_table(doc, ["阶段", "阅读/行动", "完成判据"], reading_rows, [1.25, 3.65, 1.6], "表 10  先读什么、再决定做什么")
    shared.add_callout(doc, "最值得优先实施的组合是：以 1D-CNN 与时频 CNN 为扎实基线，做“设备/会话/地点/接收机隔离的开放集跨域 RF 识别”，并把真实低 SNR/干扰和端侧成本作为次级维度。它的模型复杂度可控、创新问题明确、评价标准可审计；但是否开题取决于能否获得跨地点/跨设备/跨接收机的数据，而不是旧实验精度。")

    doc.add_page_break()
    shared.add_heading(doc, "附录 A. Zotero 同步条目（34 篇，按分层）", 1)
    shared.add_body(doc, "以下 34 条与随附 BibTeX 清单一一对应。文献条目含 DOI、层级与主题标签；同步时应保留标签，而不要把层级误读为期刊分区。", color=shared.MUTED, italic=True, size=9.5)
    for tier in ("F", "M", "C", "R"):
        shared.add_heading(doc, TIER_NAMES[tier], 2)
        for row in sources:
            if row["tier"] == tier:
                add_ref_paragraph(doc, row)

    shared.add_heading(doc, "附录 B. 计量和来源说明", 1)
    shared.add_body(doc, "论文被引与作者指标：OpenAlex 精确 DOI 审计，抓取时间 UTC 2026-08-15 15:19:50，原始快照与派生 CSV 已与本综述同目录保存。OpenAlex 是开放、可复核的代理库；其数值不能冠名为 Scopus、Web of Science 或 Google Scholar。")
    shared.add_body(doc, "期刊指标：IEEE 期刊的 JIF/Q 分区使用 IEEE Title List, January 2026（表中指向当期 JCR 快照）。例如 TIFS 8.0/Q1、TVT 7.1/Q1、TCCN 7.0/Q1、TIM 5.9/Q1；IEEE Sensors Journal 4.5/Q2、IEEE Access 3.6/Q2、OJCOMS 6.1/Q1 但其索引为 ESCI。FGCS 和 ESWA 的出版社页面分别显示 JIF 6.1 与 7.5；所有这类来源指标均只作为来源层级，不能替代单篇评价。")
    shared.add_body(doc, "作者信号：优先以院系/实验室官方主页、RF/通信研究线和专业角色核验。OpenAlex h-index 只作为审计辅助；应避免使用学校排名或单一 Google Scholar 数字推定论文质量。")
    shared.add_body(doc, "核验入口：IEEE Title List: https://open.ieee.org/wp-content/uploads/IEEE-Title-List-January-2026.pdf；OpenAlex: https://openalex.org；DORA 研究评价原则: https://sfdora.org/read/。", color=shared.MUTED, italic=True, size=9.0)

    doc.core_properties.title = "无人机通信链路 RF 深度学习检测与识别：高权威文献综述"
    doc.core_properties.subject = "被动通信物理层 RF、深度学习、文献权威性和可验证创新路线"
    doc.core_properties.author = "Codex"
    doc.core_properties.comments = "Independent literature synthesis; excludes all historic local experiments."
    doc.save(OUTPUT_DOCX)
    print(OUTPUT_DOCX)
    print(OUTPUT_BIB)
    print(OUTPUT_MANIFEST)


if __name__ == "__main__":
    build_document()
