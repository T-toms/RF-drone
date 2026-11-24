"""Build a journal-first reading list for UAV communication-link RF research.

Only DOI metadata and bibliometric audit files in this directory are read.
No historical local experiment, configuration, model, or result is consulted.
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
OUTPUT = ROOT / "无人机通信链路RF_严格期刊与前沿阅读清单_2026-08.docx"

# All citation counts below are only used as a dated OpenAlex proxy, never as
# Scopus, Web of Science, Google Scholar, or JIF counts.
PLAN = [
    # 1. Orientation: read first to establish the physical/assessment boundary.
    (1, "P0 范围与评测", "10.1016/j.iot.2024.101201", "RF/Wi-Fi 系统综述",
     "先划清通信物理层 RF 与 Wi-Fi 流量分析的边界。", "写一页纳入/排除标准。", "综述背景；不作 RF-DL 实证主结论"),
    (2, "P0 范围与评测", "10.3390/s24227339", "鲁棒性与切分规则",
     "先理解随机窗口切分、同源录制与合成噪声为何会夸大性能。", "写出 session/device/site/date/receiver 隔离方案。", "评测规则；不作算法权威性证明"),
    # 2. Strict periodical core: all journals; no conference/workshop record.
    (3, "A 严格期刊核心", "10.1016/j.future.2019.05.007", "FGCS 数据与 DNN 锚点",
     "理解检测-机型-模式三类任务和公开数据谱系。", "记录数据组成、标签层级、切分与可复现材料。", "主结论：RF+DNN 路线已成形；不是当前前沿"),
    (4, "A 严格期刊核心", "10.1109/tvt.2020.3042128", "TVT 非标准波形 RFFI",
     "区分设备级 RF 指纹与机型识别，并关注未见信道。", "记录 I/Q 表示、设备留出和信道假设。", "主结论：设备级 RFFI 可行；不可替代机型检测"),
    (5, "A 严格期刊核心", "10.1109/ojcoms.2019.2955889", "OJCOMS 干扰基线",
     "掌握 Wi-Fi/蓝牙干扰下的传统 RF 基线。", "写出干扰定义、特征和传统 ML 比较项。", "强工程基线；非 DL 主流证据"),
    (6, "A 严格期刊核心", "10.1109/tvt.2021.3081049", "TVT 专有波形与联邦",
     "理解专有协议前导特征、协作训练和隐私边界。", "记录前导可提取条件与联邦学习的额外假设。", "可作可行路线；联邦不是主流共识"),
    (7, "A 严格期刊核心", "10.1109/tccn.2021.3099114", "TCCN 联合检测-分类",
     "掌握时频表示、检测-分类任务及多信号问题。", "记录多机是否真实同时发射、评价单位和模型结构。", "主结论：联合检测/分类是成熟问题设置之一"),
    (8, "A 严格期刊核心", "10.1016/j.eswa.2021.115928", "ESWA 单机/多机 DL",
     "补足单机与多机的应用性深度学习证据。", "比较数据来源、重叠信号构造和外部测试。", "主结论：单/多机任务重要；不证明跨域鲁棒"),
    (9, "A 严格期刊核心", "10.1109/tifs.2024.3463535", "TIFS 开放集识别",
     "以未知机型拒识重新定义研究目标。", "记录未知类如何完整留出、OSCR/AUROC 与开放集协议。", "主结论：开放集是高价值前沿；尚非已解决共识"),
    # 3. Current frontier: periodical papers, deliberately separated from mature evidence.
    (10, "B 当前前沿", "10.1109/tim.2023.3306822", "TIM 未知信道与实测",
     "把未知信道和真实 USRP 采集带入泛化问题。", "分列实测、信道增强和跨域结果。", "前沿支撑；只可限定性外推"),
    (11, "B 当前前沿", "10.1109/jsen.2023.3242985", "复杂电磁背景",
     "研究真实背景、混合增强和谱图处理的边界。", "记录背景是否独立采集、增强是否泄漏。", "问题重要；不能称真实部署已解决"),
    (12, "B 当前前沿", "10.1109/jrfid.2024.3487303", "低成本低 SNR",
     "将低成本 SDR 与低 SNR 纳入工程约束。", "记录实测干扰、SNR 曲线、时延和误警率。", "工程参考；JRFID Q2，非旗舰主结论"),
    (13, "B 当前前沿", "10.1109/tccn.2025.3537100", "SMNet 多无人机",
     "关注谱图上真实多源/多机检测问题。", "核查是否同步真实发射、事件级指标和重叠比例。", "正在兴起；引用窗口仍短"),
    (14, "B 当前前沿", "10.1109/tccn.2025.3561305", "跨域 RFFI",
     "直接学习地点、接收机和信道域偏移。", "记录域定义、源/目标隔离、跨域消融。", "可作前沿立项依据；尚无充分独立复现"),
    (15, "B 当前前沿", "10.1109/tccn.2025.3598245", "轻量固定分类器",
     "理解 MobileNetV3-Small 与固定分类器的端侧取舍。", "记录参数量、时延、功耗、低 SNR 实测。", "观察名单；新文低被引不能作主结论"),
    (16, "B 当前前沿", "10.1109/tvt.2026.3696747", "少样本脉冲网络",
     "检验少样本扩类是否能避免设备泄漏。", "记录 shot 协议、物理设备留出、旧类遗忘。", "研究雷达；2026 新文，尚无引用成熟度"),
    # 4. Baseline/data: read after establishing a research question.
    (17, "C 基线与数据", "10.1016/j.comnet.2021.108519", "原始 RF 1D-CNN",
     "建立端到端时域 CNN 的低风险基线。", "记录输入窗、采样、同源切分和计算量。", "方法基线；不把单数据集精度写成泛化结论"),
    (18, "C 基线与数据", "10.1109/access.2022.3172787", "RF-UAVNet 轻量 CNN",
     "建立参数量/准确率对照网络。", "记录模型大小、参数量和 DroneRF 依赖。", "工程基线；IEEE Access Q2"),
    (19, "C 基线与数据", "10.3390/s21061947", "层级传统 ML",
     "保证深度学习与可解释层级基线公平比较。", "记录检测、类型、模式各层的误差传播。", "传统 ML 基线；非 DL 主结论"),
    (20, "C 基线与数据", "10.1016/j.dib.2019.104313", "DroneRF 数据论文",
     "追溯大量实验的共同数据源与限制。", "制作数据卡：设备、模式、采集会话和许可。", "数据出处；不作算法权威性证据"),
    (21, "C 基线与数据", "10.3390/s22083072", "压缩采样 RF + DL",
     "在采样率/前端成本受限时加入表示对照。", "比较相同采样预算的原始 RF 与压缩采样。", "方法变体；非当前主流输入"),
    (22, "C 基线与数据", "10.1109/lsens.2023.3289145", "时频多尺度 CNN",
     "形成时频卷积的多尺度对照。", "记录 STFT 参数、多尺度设计与消融。", "方法基线；短文且非高 IF 核心"),
    # 5. Historical conference/workshop material is deliberately last and optional.
    (23, "D 历史会议（可选）", "10.1109/iciot48696.2020.9089489", "早期 CNN 会议基线",
     "只追溯早期 CNN 思路和对照实现。", "复现时记录其会议属性与限制。", "不可作高 IF 或权威主结论"),
    (24, "D 历史会议（可选）", "10.1109/iciot48696.2020.9089657", "多通道 1D-CNN 会议基线",
     "只用于重建原始 RF 多通道对照。", "记录训练窗、通道定义与评测方式。", "不可作高 IF 或权威主结论"),
    (25, "D 历史会议（可选）", "10.1145/2935620.2935632", "DroNet workshop 起点",
     "了解深度学习之前的低成本 RF 检测起点。", "只写历史定位，不用作当前技术判据。", "不可作高 IF 或权威主结论"),
]

TIER_TEXT = {
    "P0 范围与评测": "先读。用于固定问题边界与评测纪律，不承担模型优劣结论。",
    "A 严格期刊核心": "无会议/workshop。优先用于领域主结论；旧文只说明奠基价值，不称当前前沿。",
    "B 当前前沿": "近年期刊工作。问题与作者/来源可以强，但因引用窗口短，不能被写成成熟共识。",
    "C 基线与数据": "用于可复现的模型、数据和公平对照；不进入“高 IF 权威主结论”。",
    "D 历史会议（可选）": "明确不是高 IF 期刊证据；只在复现早期网络或写技术史时阅读。",
}


def norm(value: str) -> str:
    return value.strip().lower().replace("https://doi.org/", "")


def load_rows() -> list[dict[str, str]]:
    with METADATA.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = {norm(row["doi"]): row for row in csv.DictReader(handle)}
    with OPENALEX.open("r", encoding="utf-8-sig", newline="") as handle:
        metrics = {norm(row["doi"]): row for row in csv.DictReader(handle)}
    rows: list[dict[str, str]] = []
    missing: list[str] = []
    for order, category, doi, focus, why, output, conclusion in PLAN:
        doi = norm(doi)
        if doi not in metadata:
            missing.append(doi)
            continue
        row = dict(metadata[doi])
        row.update(metrics.get(doi, {}))
        row.update({
            "order": str(order), "category": category, "doi": doi, "focus": focus,
            "why": why, "reading_output": output, "conclusion": conclusion,
        })
        rows.append(row)
    if missing:
        raise RuntimeError(f"DOI missing from metadata: {missing}")
    return rows


def author_short(row: dict[str, str]) -> str:
    people = [name.strip() for name in row.get("authors", "").split(";") if name.strip()]
    return f"{people[0]} 等" if len(people) > 1 else (people[0] if people else "作者待核")


def metric(row: dict[str, str]) -> str:
    count = row.get("cited_by_count_openalex", "")
    yearly = row.get("citations_per_year_openalex", "")
    return f"{count}（约 {yearly}/年）" if count else "不纳入方法审计"


def ref(row: dict[str, str]) -> str:
    venue = row.get("venue") or row.get("publisher") or ""
    vol = row.get("volume") or ""
    issue = f"({row['issue']})" if row.get("issue") else ""
    pages = row.get("pages_or_article") or ""
    tail = " ".join(part for part in [venue, f"{vol}{issue}".strip(), pages] if part)
    return f"[{row['order']}] {row.get('authors') or '作者待核'}. ({row.get('year') or 'n.d.'}). {row.get('title') or ''}. {tail}. https://doi.org/{row['doi']}"


def setup_header_footer(doc: Document) -> None:
    section = doc.sections[0]
    section.different_first_page_header_footer = True
    section.first_page_header.paragraphs[0].text = ""
    fp = section.first_page_footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run("检索与指标快照：2026-08-15｜不使用既往本地实验结论")
    shared.set_run_font(run, size=8.5, color=shared.MUTED)
    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.paragraph_format.space_after = Pt(0)
    run = header.add_run("无人机通信链路 RF｜严格期刊与前沿阅读清单")
    shared.set_run_font(run, size=8.5, color=shared.MUTED)
    footer = section.footer
    table = footer.add_table(rows=1, cols=2, width=Inches(6.5))
    shared.set_table_geometry(table, [4.7, 1.8])
    shared.prevent_row_split(table.rows[0])
    shared.set_table_borders(table, color="FFFFFF")
    for cell in table.rows[0].cells:
        shared.set_cell_margins(cell, top=0, start=0, bottom=0, end=0)
    left = table.cell(0, 0).paragraphs[0]
    left.paragraph_format.space_after = Pt(0)
    run = left.add_run("25 篇｜先期刊、后前沿、最后才读会议")
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
    spacer.paragraph_format.space_before = Pt(110)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(18)
    run = p.add_run("研究阅读路线｜先证据，后模型，最后才是历史会议")
    shared.set_run_font(run, size=11.5, color="7A5A00", bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run("无人机通信链路 RF\n严格期刊与前沿阅读清单")
    shared.set_run_font(run, size=29, color=shared.TITLE_DARK, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(28)
    run = p.add_run("25 篇：分类、阅读顺序、每篇的笔记产出与证据边界")
    shared.set_run_font(run, size=14, color=shared.DARK_BLUE)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("范围：被动截获无人机/遥控器通信链路的物理层 RF")
    shared.set_run_font(run, size=10.5, color=shared.MUTED, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run("会议论文被刻意置于最后的历史复现层；不承担高 IF 权威结论")
    shared.set_run_font(run, size=9.5, color=shared.MUTED, italic=True)
    doc.add_page_break()


def add_plan_table(doc: Document, rows: list[dict[str, str]], category: str, caption: str) -> None:
    selected = [row for row in rows if row["category"] == category]
    values = []
    for row in selected:
        title = f"{author_short(row)} ({row.get('year')})\n{row.get('venue') or ''}\nDOI: {row['doi']}"
        action = f"读什么：{row['why']}\n笔记产出：{row['reading_output']}"
        values.append([row["order"], title, metric(row), action, row["conclusion"]])
    shared.add_table(doc, ["序", "论文", "OpenAlex 被引*", "本篇要完成什么", "可承担的结论"], values, [0.35, 1.75, 0.9, 2.25, 1.25], caption)
    shared.add_body(doc, "* OpenAlex 截止 2026-08-15 的 DOI 精确命中快照；不是 JIF，也不是 Scopus/WoS/Google Scholar 引用。", color=shared.MUTED, italic=True, size=8.6)


def add_reference(doc: Document, row: dict[str, str]) -> None:
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.left_indent = Inches(0.27)
    p.paragraph_format.first_line_indent = Inches(-0.27)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.08
    run = p.add_run(ref(row))
    shared.set_run_font(run, size=9.1, color="000000")


def build() -> None:
    rows = load_rows()
    doc = Document()
    shared.set_page_geometry(doc.sections[0])
    shared.configure_styles(doc)  # compact_reference_guide: fixed 1 in / 9360 DXA system
    setup_header_footer(doc)      # editorial_cover pattern adapted to a research reference guide
    add_cover(doc)

    shared.add_heading(doc, "怎么使用这份清单", 1)
    shared.add_callout(doc, "不要按年份把文献一股脑地读完。先用 P0 定义研究边界和评测纪律；再精读 A 类期刊证据；随后才把 B 类前沿变成候选创新。C 类用于公平基线和数据选择。会议论文在 D 类，只在需要复现历史网络时才打开。")
    shared.add_body(doc, "本清单完全由公开文献重新整理，不使用任何既往本地实验、配置、模型或结果。", color=shared.DARK_BLUE, italic=True)
    shared.add_body(doc, "A 类全部是期刊论文，无 conference/workshop。它们也并非都“最新”：2019-2022 年的工作承担成熟主线和数据/任务定义，2023-2026 年的期刊工作承担当前问题定义。新论文不因引用少而无价值，但也不能被写成已被充分复现的共识。")
    rules = [
        ["每篇固定记录", "问题/标签层级；信号表示与接收设置；切分单位；模型；指标；真实/合成干扰；失败条件。"],
        ["主结论门槛", "至少由两篇独立期刊论文共同支撑；不以单篇高准确率或作者名气替代验证。"],
        ["会议处理", "会议/workshop 没有期刊 JIF，不计入“高影响因子权威核心”；仅保留为历史/复现对照。"],
        ["创新选择", "只从 B 类中选一个问题，并先检查最低数据条件是否满足；缺跨域数据就不立跨域题。"],
    ]
    shared.add_table(doc, ["规则", "执行方式"], rules, [1.25, 5.25], "表 1  阅读时必须遵守的四条规则")

    shared.add_heading(doc, "一周阅读顺序", 1)
    schedule = [
        ["第 1 天", "1-2", "范围、Wi-Fi/RF 边界、随机切分和合成噪声的风险。", "一页研究范围 + 评测硬门槛。"],
        ["第 2 天", "3-5", "FGCS 数据/DNN、TVT RFFI、干扰传统基线。", "任务树、数据卡和传统基线。"],
        ["第 3 天", "6-9", "专有波形、联合检测/分类、多机、开放集。", "当前研究问题的候选清单。"],
        ["第 4 天", "10-12", "未知信道、复杂背景、低成本低 SNR。", "现实部署风险矩阵。"],
        ["第 5 天", "13-16", "多机、跨域、轻量化、少样本。", "只保留一个可执行创新命题。"],
        ["第 6 天", "17-22", "1D-CNN、时频 CNN、层级 ML、数据和压缩采样基线。", "实验基线和数据采集最小方案。"],
        ["第 7 天（可选）", "23-25", "历史会议/Workshop。", "只记录可复现的旧基线，不纳入主结论。"],
    ]
    shared.add_table(doc, ["时间", "阅读序号", "目的", "当日产出"], schedule, [1.05, 0.75, 2.95, 1.75], "表 2  7 天阅读节奏")
    shared.add_body(doc, "如果只读 6 篇：优先 1、2、4、7、9、14。它们依次解决范围、评测、物理层 RFFI、联合任务、开放集和跨域，足以决定下一步该不该采集新数据。", color=shared.DARK_BLUE, italic=True)

    for category, caption in [
        ("P0 范围与评测", "表 3  P0：范围与评测（必须先读）"),
        ("A 严格期刊核心", "表 4  A：严格期刊核心（无会议论文）"),
        ("B 当前前沿", "表 5  B：当前前沿（期刊新作，不等于成熟共识）"),
        ("C 基线与数据", "表 6  C：模型/数据基线（用于公平对照）"),
        ("D 历史会议（可选）", "表 7  D：历史会议/Workshop（最后、可选）"),
    ]:
        shared.add_heading(doc, category, 1)
        shared.add_body(doc, TIER_TEXT[category])
        add_plan_table(doc, rows, category, caption)

    shared.add_heading(doc, "读完后的立题检查", 1)
    checks = [
        ["若选开放集", "是否有完整未知设备/机型留出，而非同一设备的随机窗口？"],
        ["若选跨域", "是否有至少两个地点、日期、接收机或信道域？没有，就降为后续采集计划。"],
        ["若选多机", "是否有同步真实同时发射记录？没有，就不把离线叠加称为多机部署验证。"],
        ["若选低 SNR/干扰", "是否有独立现场背景和干扰记录？仅 AWGN 只能作为补充消融。"],
        ["若选轻量化", "是否有端侧硬件、端到端延迟与功耗测量？没有，就仅是网络复杂度比较。"],
    ]
    shared.add_table(doc, ["拟研究问题", "开题前必须回答"], checks, [1.55, 4.95], "表 8  从阅读到立题的真实性检查")
    shared.add_callout(doc, "最稳妥的起点不是立即复现会议 CNN，而是先以 A 类期刊建立问题和可信评测，再用 C 类建立低风险基线。若跨地点/跨接收机数据可得，优先选择“开放集 + 跨域”；若只有单场景数据，优先做严格会话/设备隔离与低 SNR 评测，而不要声称跨域泛化。")

    doc.add_page_break()
    shared.add_heading(doc, "附录：按阅读顺序列出的 DOI 参考文献", 1)
    shared.add_body(doc, "A 类期刊和 B 类前沿是 Zotero 中用于精读的主目录；C/D 只作基线与历史标记。", color=shared.MUTED, italic=True, size=9.2)
    current = None
    for row in rows:
        if row["category"] != current:
            current = row["category"]
            shared.add_heading(doc, current, 2)
        add_reference(doc, row)

    shared.add_heading(doc, "指标与权威性口径", 1)
    shared.add_body(doc, "期刊级信号与单篇质量必须分开。IEEE 期刊的 JIF/Q 分区使用 IEEE Title List, January 2026 快照：TIFS 8.0/Q1、TVT 7.1/Q1、TCCN 7.0/Q1、TIM 5.9/Q1；IEEE Sensors Journal 4.5/Q2、IEEE Access 3.6/Q2。FGCS 与 ESWA 的出版社页分别显示 JIF 6.1 和 7.5。JIF 仅说明来源层级，不替代论文实验质量。")
    shared.add_body(doc, "OpenAlex 引用为公开可复核代理；作者权威性以官方院系/实验室主页、通信/RF 研究线和专业角色为准，不能只凭 h-index 或学校排名。核验入口：IEEE Title List: https://open.ieee.org/wp-content/uploads/IEEE-Title-List-January-2026.pdf；OpenAlex: https://openalex.org；DORA: https://sfdora.org/read/。", color=shared.MUTED, italic=True, size=9.0)

    doc.core_properties.title = "无人机通信链路 RF：严格期刊与前沿阅读清单"
    doc.core_properties.subject = "分类、阅读顺序与期刊权威性分层"
    doc.core_properties.author = "Codex"
    doc.core_properties.comments = "Fresh literature reading list; excludes historical local experiments."
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
