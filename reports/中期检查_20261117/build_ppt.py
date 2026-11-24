# -*- coding: utf-8 -*-
"""用学校模板(芃苇)搭中期答辩 PPT 模板稿。模板原件不动，输出 中期答辩_模板稿.pptx。
用法: anaconda python -I build_ppt.py
"""
import os
import win32com.client as win32

D = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(D, '模板原件.pptx')
OUT = os.path.join(D, '中期答辩_模板稿.pptx')
FIG = os.path.join(D, 'figs')


def rgb(r, g, b):
    return r + 256 * g + 65536 * b


BLUE, DBLUE, ORANGE = rgb(0, 121, 186), rgb(0, 90, 140), rgb(240, 163, 10)
GRAY, LGRAY, WHITE, TXT = rgb(150, 150, 150), rgb(242, 245, 248), rgb(255, 255, 255), rgb(64, 64, 64)
FONT_CN, FONT_EN = '微软雅黑', 'Arial'
X0, X1, Y0, Y1 = 35, 925, 78, 498           # 内容区

app = win32.DispatchEx('PowerPoint.Application')
pres = app.Presentations.Open(TPL, False, False, False)
pres.SaveAs(OUT)

# ---------- 保留封面1、目录5、过渡页6、尾页44，其余删掉 ----------
keep = {1: 'cover', 5: 'toc', 6: 'trans', 44: 'end'}
ids = {name: pres.Slides(i).SlideID for i, name in keep.items()}
for i in range(pres.Slides.Count, 0, -1):
    if i not in keep:
        pres.Slides(i).Delete()
S = {name: pres.Slides.FindBySlideID(sid) for name, sid in ids.items()}
for s in S.values():                          # 去掉页面外的“提示条”
    for k in range(s.Shapes.Count, 0, -1):
        sh = s.Shapes(k)
        if sh.Top < -5 or sh.Top >= 539:
            sh.Delete()

layout = None
for lay in pres.SlideMaster.CustomLayouts:
    if lay.Name == '仅标题':
        layout = lay


def shape_by_name(slide, name):
    for sh in slide.Shapes:
        if sh.Name == name:
            return sh
    raise KeyError(name)


def set_notes(slide, text):
    slide.NotesPage.Shapes.Placeholders(2).TextFrame.TextRange.Text = text


def end_index():
    return S['end'].SlideIndex


def new_slide(title, notes=''):
    s = pres.Slides.AddSlide(end_index(), layout)
    s.Shapes.Title.TextFrame.TextRange.Text = title
    if notes:
        set_notes(s, notes)
    return s


def style(tr, size, color=TXT, bold=False, cn=FONT_CN):
    tr.Font.NameFarEast = cn
    tr.Font.Name = FONT_EN
    tr.Font.Size = size
    tr.Font.Color.RGB = color
    tr.Font.Bold = -1 if bold else 0


def text(slide, l, t, w, h, paras, size=14, color=TXT, align=1, space=4, anchor=1):
    """paras: list of str 或 (str, dict(size,color,bold,bullet))"""
    tb = slide.Shapes.AddTextbox(1, l, t, w, h)
    tf = tb.TextFrame
    tf.WordWrap = -1
    tf.AutoSize = 0
    tf.VerticalAnchor = anchor
    tf.MarginLeft = tf.MarginRight = 4
    items = [p if isinstance(p, tuple) else (p, {}) for p in paras]
    tf.TextRange.Text = '\r'.join(s for s, _ in items)
    for i, (s, o) in enumerate(items, 1):
        pr = tf.TextRange.Paragraphs(i)
        style(pr, o.get('size', size), o.get('color', color), o.get('bold', False))
        pr.ParagraphFormat.Alignment = o.get('align', align)
        pr.ParagraphFormat.SpaceAfter = space
        if o.get('bullet'):
            pr.ParagraphFormat.Bullet.Visible = -1
            pr.ParagraphFormat.Bullet.Character = 8226
            pr.ParagraphFormat.Bullet.Font.Color.RGB = BLUE
            pr.IndentLevel = 1
    return tb


def box(slide, l, t, w, h, fill=LGRAY, line=None, radius=True):
    sh = slide.Shapes.AddShape(5 if radius else 1, l, t, w, h)
    if radius:
        sh.Adjustments.SetItem(1, 0.06)
    sh.Fill.ForeColor.RGB = fill
    if line is None:
        sh.Line.Visible = 0
    else:
        sh.Line.ForeColor.RGB = line
        sh.Line.Weight = 1.25
    sh.Shadow.Visible = 0
    sh.TextFrame.TextRange.Text = ''
    return sh


def card(slide, l, t, w, h, head, body, color=BLUE, body_size=12.5):
    box(slide, l, t, w, h, fill=WHITE, line=color)
    hd = box(slide, l, t, w, 30, fill=color)
    tr = hd.TextFrame.TextRange
    tr.Text = head
    style(tr, 14, WHITE, True)
    hd.TextFrame.MarginLeft = 8
    tr.ParagraphFormat.Alignment = 2
    paras = [(b, {'bullet': True}) if isinstance(b, str) else b for b in body]
    text(slide, l + 6, t + 36, w - 12, h - 42, paras, size=body_size, space=3)


def stat(slide, l, t, w, h, num, label, color=BLUE):
    box(slide, l, t, w, h, fill=LGRAY)
    text(slide, l, t + 6, w, h * 0.55, [(num, {'size': 26, 'bold': True, 'color': color, 'align': 2})], anchor=3)
    text(slide, l, t + h * 0.58, w, h * 0.38, [(label, {'size': 12, 'align': 2})])


def picture(slide, name, l, t, w, h):
    p = os.path.join(FIG, name)
    pic = slide.Shapes.AddPicture(p, 0, -1, l, t)
    r = min(w / pic.Width, h / pic.Height)
    pic.Width, pic.Height = pic.Width * r, pic.Height * r
    pic.Left = l + (w - pic.Width) / 2
    pic.Top = t + (h - pic.Height) / 2
    return pic


def placeholder(slide, l, t, w, h, label):
    sh = box(slide, l, t, w, h, fill=LGRAY, line=GRAY, radius=False)
    sh.Line.DashStyle = 4
    tr = sh.TextFrame.TextRange
    tr.Text = label
    style(tr, 14, GRAY)
    tr.ParagraphFormat.Alignment = 2


def banner(slide, msg, t=462):
    b = box(slide, X0, t, X1 - X0, 34, fill=rgb(232, 242, 250))
    text(slide, X0 + 10, t + 2, X1 - X0 - 20, 30,
         [(msg, {'size': 14, 'bold': True, 'color': DBLUE})], anchor=3)


def transition(no_cn, part, title, lines, notes=''):
    rng = S['trans'].Duplicate()
    s = rng.Item(1)
    s.MoveTo(end_index() - 1)
    shape_by_name(s, '文本占位符 3').TextFrame.TextRange.Text = no_cn
    shape_by_name(s, '文本占位符 11').TextFrame.TextRange.Text = part
    shape_by_name(s, '文本占位符 12').TextFrame.TextRange.Text = title
    shape_by_name(s, '文本占位符 24').TextFrame.TextRange.Text = '\r'.join(lines)
    if notes:
        set_notes(s, notes)
    return s


# ================= 封面、目录 =================
cv = S['cover']
shape_by_name(cv, '标题 1').TextFrame.TextRange.Text = '基于深度学习的无人机\r射频信号检测识别'
shape_by_name(cv, '文本占位符 85').TextFrame.TextRange.Text = '大学生创新训练计划（SRTP）中期检查答辩'
shape_by_name(cv, '文本占位符 87').TextFrame.TextRange.Text = '指导教师：xxx　　汇报人：xxx'
set_notes(cv, '【开场，约20秒】各位老师好，我汇报的项目是……。下面分四部分汇报：研究背景与目标、技术路线、已完成的工作与遇到的难题、创新点和下一步计划。')
shape_by_name(S['toc'], '文本占位符 4').TextFrame.TextRange.Text = \
    '研究背景与目标\r研究方案与技术路线\r已完成工作与关键问题\r创新点、问题与计划'

# ================= 壹 研究背景与目标 =================
transition('壹', '第一部分', '研究背景与目标', ['低空安全需求与射频检测', '两阶段目标与考核指标', '文献调研：主流做法与不足'])

s = new_slide('研究背景与意义', '【约40秒】讲清为什么做、为什么用射频、真实场景难在哪。第三张卡片是后面所有“难题”的伏笔：实验室分数高不等于实测可用。')
cw = (X1 - X0 - 40) / 3
card(s, X0, 92, cw, 300, '低空安全需求', [
    '消费级无人机数量快速增长，“黑飞”威胁机场、铁路、重要设施',
    '需要在复杂电磁环境中及时、可靠地发现无人机'])
card(s, X0 + cw + 20, 92, cw, 300, '射频检测的优势', [
    '无人机与遥控器持续通信（图传、飞控），被动接收即可发现',
    '不受光照、遮挡影响，成本低',
    '信号带有型号特征，可进一步识别机型'])
card(s, X0 + 2 * (cw + 20), 92, cw, 300, '实测场景的难点', [
    '城区WiFi/蓝牙与无人机同频',
    '不同接收机底噪不同',
    '背景样本少、信噪比低',
    '模型容易“走捷径”：实验室高分≠实测可用'], color=ORANGE)
banner(s, '目标：在真实采集数据上，建立可信的“检测 + 识别”射频方法', t=420)

s = new_slide('研究目标与考核指标', '【约40秒】两阶段：先判断有没有无人机，再判断是哪种。指标是和导师定的；评测原则是从文献里总结出来的，后面每个实验都遵守。')
steps = ['原始IQ信号', '时频图\n+ IQ手工特征', '第一阶段\n有无无人机', '第二阶段\n识别型号']
bw = 150
for i, st in enumerate(steps):
    x = X0 + 10 + i * (bw + 40)
    b = box(s, x, 100, bw, 70, fill=BLUE if i >= 2 else DBLUE)
    tr = b.TextFrame.TextRange
    tr.Text = st.replace('\n', '\r')
    style(tr, 15, WHITE, True)
    tr.ParagraphFormat.Alignment = 2
    if i < 3:
        a = s.Shapes.AddShape(33, x + bw + 8, 125, 24, 20)   # 右箭头
        a.Fill.ForeColor.RGB = ORANGE
        a.Line.Visible = 0
cw = (X1 - X0 - 40) / 3
card(s, X0, 200, cw, 200, '第一阶段（不加噪声）', ['各机型漏检 ≤ 1%', '背景虚警 ≤ 1%'])
card(s, X0 + cw + 20, 200, cw, 200, '低信噪比', ['不加噪 / 0 / −5 / −10 / −15 dB', '平均准确率 ≥ 95%'])
card(s, X0 + 2 * (cw + 20), 200, cw, 200, '第二阶段', ['识别具体型号', '同型号、同图传带宽比较'])
banner(s, '评测原则：按录音分组划分 · 分来源统计 · 每组3个随机种子 · 报告95%置信上界', t=420)

s = new_slide('文献调研：主流做法与本项目对标', '【约60秒】这一页体现“查文献、对标主流”。每一行讲：主流怎么做 → 有什么不足 → 我们怎么做。重点讲第三、四行：随机划分泄漏、换接收机掉分，这是我们设计对照实验的依据。')
rows = [
    ('方面', '主流做法（代表文献）', '存在的不足', '本项目做法'),
    ('输入表示', '时频图 + CNN（DroneRFa 2024、Ozturk 2020、Glüge 2024）', '短窗口可能抓不到突发信号', '时频图 + ResNet18 作基线；比较 10–50 ms 窗长'),
    ('手工特征与融合', '统计/高阶特征 + 分类器（Ezuma 2020）；深浅特征拼接（Yang 2021、Han 2021）', '特征可能带接收机指纹；低信噪比下失效', '融合前先做来源检验，再在嵌入层拼接'),
    ('数据划分', '多数随机切分窗口（如 DroneRFa 随机 6:2:2）', '同一录音相邻片段泄漏，成绩偏高', '按录音分组划分，训练/测试零泄漏'),
    ('泛化评测', '多报告随机测试集总体准确率', '换接收机 99%→<33%（WiSig）；捷径难发现（Geirhos 2020）', '背景替换、白噪声/陷波噪声对照；分接收机统计'),
]
tbl_sh = s.Shapes.AddTable(len(rows), 4, X0, 84, X1 - X0, 330)
tbl = tbl_sh.Table
widths = [110, 300, 230, 250]
for j, wdt in enumerate(widths, 1):
    tbl.Columns(j).Width = wdt
for i, row in enumerate(rows, 1):
    for j, val in enumerate(row, 1):
        c = tbl.Cell(i, j)
        tr = c.Shape.TextFrame.TextRange
        tr.Text = val
        hdr = (i == 1)
        style(tr, 12.5 if not hdr else 14, WHITE if hdr else (DBLUE if j in (1, 4) else TXT), hdr or j == 1)
        c.Shape.Fill.ForeColor.RGB = BLUE if hdr else (WHITE if i % 2 else LGRAY)
        if j == 4 and not hdr:
            c.Shape.Fill.ForeColor.RGB = rgb(232, 242, 250)
banner(s, '调研量：高权威文献34篇（精读10篇）· 特征融合文献19篇（全文12篇）· 公开WiFi/蓝牙数据集21个', t=430)

# ================= 贰 研究方案与技术路线 =================
transition('贰', '第二部分', '研究方案与技术路线', ['总体技术路线与进度', '工作量总览'])

s = new_slide('总体技术路线', '【约40秒】按顺序讲一遍流程，指出已完成到哪一步：前六步完成，第七步进行中。')
flow = [('文献调研\n确定路线', 2), ('公开数据集\n基线', 2), ('实测数据\n审核预处理', 2), ('第一阶段\n检测训练', 2), ('捷径排查\n与修正', 2),
        ('时频图+IQ\n特征融合', 2), ('按信号频带截取\n公开WiFi/蓝牙叠加', 1), ('低信噪比\n训练', 0), ('第二阶段\n型号识别', 0)]
colors = {2: BLUE, 1: ORANGE, 0: GRAY}
bw, bh, gap = 150, 72, 35
for i, (lab, st) in enumerate(flow):
    row, col = divmod(i, 5)
    if row == 1:
        col = 4 - col                      # 蛇形排列
    x = X0 + 20 + col * (bw + gap)
    y = 110 + row * 150
    b = box(s, x, y, bw, bh, fill=colors[st])
    tr = b.TextFrame.TextRange
    tr.Text = lab.replace('\n', '\r')
    style(tr, 14, WHITE, True)
    tr.ParagraphFormat.Alignment = 2
    if i < len(flow) - 1:
        if i == 4:
            a = s.Shapes.AddShape(36, x + bw / 2 - 10, y + bh + 14, 20, 50)     # 下箭头
        elif row == 0:
            a = s.Shapes.AddShape(33, x + bw + 6, y + bh / 2 - 9, 22, 18)
        else:
            a = s.Shapes.AddShape(34, x - gap + 7, y + bh / 2 - 9, 22, 18)       # 左箭头
        a.Fill.ForeColor.RGB = rgb(190, 205, 220)
        a.Line.Visible = 0
for k, (lab, c) in enumerate([('已完成', BLUE), ('进行中', ORANGE), ('计划', GRAY)]):
    box(s, X0 + 20 + k * 110, 420, 18, 18, fill=c)
    text(s, X0 + 42 + k * 110, 416, 80, 26, [(lab, {'size': 13})])

s = new_slide('工作量总览', '【约40秒】用数字说明工作量。可以强调：每个结论都配了对照实验，所以训练次数多。数字依据：数据=DroneRFa用到约107 GB + 实验室/城区/v3约255 GB；文件=3933+890+1658；训练次数=11月H/J/L/M/W/B2b合计。')
stats = [('约360 GB', '处理的射频数据'), ('6400+', '原始录音文件'), ('60+', '模型训练（11月）'),
         ('26', '实验任务书'), ('20', '实验报告'), ('100+', '分析图')]
sw = (X1 - X0 - 50) / 6
for i, (n, lab) in enumerate(stats):
    stat(s, X0 + i * (sw + 10), 88, sw, 95, n, lab, color=BLUE if i % 2 == 0 else ORANGE)
tl = [('9月', '文献调研\nDroneRFa基线'), ('10月', '基线错误诊断\n转实测数据、全面审核'),
      ('11月上旬', '截频切片、第一轮训练\n捷径排查与修正'), ('11月中旬', '特征融合实验\n频带截取、公开数据调研')]
line = s.Shapes.AddLine(X0 + 30, 270, X1 - 30, 270)
line.Line.ForeColor.RGB = BLUE
line.Line.Weight = 3
tw = (X1 - X0 - 60) / 4
for i, (m, desc) in enumerate(tl):
    cx = X0 + 30 + tw * i + tw / 2
    dot = s.Shapes.AddShape(9, cx - 9, 261, 18, 18)
    dot.Fill.ForeColor.RGB = ORANGE
    dot.Line.ForeColor.RGB = WHITE
    text(s, cx - tw / 2, 222, tw, 30, [(m, {'size': 16, 'bold': True, 'color': DBLUE, 'align': 2})])
    text(s, cx - tw / 2 + 5, 292, tw - 10, 120, [(d, {'size': 13, 'align': 2}) for d in desc.split('\n')])

# ================= 叁 已完成工作与关键问题 =================
transition('叁', '第三部分', '已完成工作与关键问题', ['公开数据集基线', '实测数据中的五个难题与解决', '时频图 + IQ 特征融合'])

s = new_slide('公开数据集基线：DroneRFa', '【约40秒】先在公开数据上把流程跑通。重点讲发现：M100 每 14 ms 才发一次突发，10 ms 窗口抓不到——这说明不能只看分数，要回到信号本身找原因。')
picture(s, '图A_DroneRFa两步识别_测试集_不加噪.png', X0, 80, 530, 375)
text(s, X0 + 545, 84, X1 - X0 - 545, 370, [
    ('数据与方法', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('12种机型 + 背景，按文件划分', {'bullet': True}),
    ('10 ms 时频图 + ResNet18，两步识别', {'bullet': True}),
    ('结果', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('不加噪声 85.4%，−15 dB 51.1%', {'bullet': True}),
    ('Matrice 300 仅 34.8%', {'bullet': True}),
    ('发现', {'bold': True, 'color': ORANGE, 'size': 15}),
    ('Matrice 100 每 14 ms 才发 1–2 ms 突发，10 ms 窗约 29% 抓不到信号', {'bullet': True}),
    ('→ 引出窗口筛选与窗长比较', {'bold': True}),
], size=13.5)

s = new_slide('难题一：实测数据“不干净”', '【约50秒】转到实验室实测数据后，先做了全面审核。每张卡片讲“发现了什么→怎么处理”。这些问题不处理，后面训练结果都不可信。')
cw = (X1 - X0 - 30) / 2
cards = [
    ('重复与损坏文件', ['查出 97 个重复、15 个全零、25 个前段为零的文件', '→ 逐一核验、标记并剔除']),
    ('训练/测试划分泄漏', ['原始划分的训练和测试来自同一段录音', '→ 按录音分组重新划分，训练/测试零泄漏']),
    ('采样率未记录', ['文件中没有采样率，无法换算频率和时长', '→ 同步序列匹配推断为 122.88 MS/s，并经确认']),
    ('接收机本振泄漏谱线', ['接收机固有的窄带谱线会被模型当作标签线索', '→ 盘点谱线，正弦对消/陷波；所有样本用同一陷波表']),
]
for i, (h, b) in enumerate(cards):
    r_, c_ = divmod(i, 2)
    card(s, X0 + c_ * (cw + 30), 86 + r_ * 175, cw, 160, h, b, color=BLUE if i % 2 == 0 else DBLUE, body_size=13.5)

s = new_slide('难题二：模型在“走捷径”——把WiFi当背景', '【约60秒】第一轮虚警看起来低于1%，但我们追问模型依据什么。对照设计：同一批无人机样本、同样功率比，只换垫底的背景。结论：2.4G 模型部分把“有 WiFi”当成背景；5.8G 没有这种现象。这个对照是后面修正的出发点。')
picture(s, '图B_背景替换对照.png', X0, 76, 520, 385)
text(s, X0 + 535, 84, X1 - X0 - 535, 370, [
    ('问题', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('第一轮虚警 <1%，模型到底依据什么？', {'bullet': True}),
    ('对照设计', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('同一批无人机样本、同样功率比，只换“垫底”背景：白噪声 vs 城区背景', {'bullet': True}),
    ('发现', {'bold': True, 'color': ORANGE, 'size': 15}),
    ('2.4 GHz、0 dB：检出率 90% → 57%', {'bullet': True}),
    ('城区无人机加白噪声后反而更易检出', {'bullet': True}),
    ('→ 模型把“出现WiFi”当成“是背景”', {'bold': True}),
    ('5.8 GHz 无此现象', {'bullet': True}),
], size=13.5)

s = new_slide('难题三：缺少同接收机背景，且不能补录', '【约50秒】实验室接收机A只录了无人机，没有背景，又不能补录。办法：同一录音里有一段空闲频段，把它镜像到工作频带，构造出同一接收机的背景，并检验镜像泄漏。加入训练后 A 背景虚警从 100% 降到 0%。')
picture(s, '图C_加入接收机A背景前后.png', X0, 74, X1 - X0, 255)
cw = (X1 - X0 - 40) / 3
card(s, X0, 336, cw, 122, '问题', ['接收机A只有无人机、没有背景', '模型把“A的干净底噪”当成无人机：虚警100%'], color=ORANGE, body_size=12.5)
card(s, X0 + cw + 20, 336, cw, 122, '解决', ['同录音空闲频段 2342–2393 MHz 共轭镜像到工作频带', '镜像泄漏检验，不合格的块剔除'], body_size=12.5)
card(s, X0 + 2 * (cw + 20), 336, cw, 122, '结果', ['A背景虚警 100% → 0%', '无人机召回、城区背景虚警不变'], color=DBLUE, body_size=12.5)

s = new_slide('难题四：WiFi干扰怎么加才不引入新捷径', '【约50秒】实验室无人机没有 WiFi，城区背景有 WiFi。两种叠加方案对比：只叠到无人机上会学出新捷径，城区叠加集虚警升到 24%；对称叠加到无人机和背景上，虚警 1.9%，城区无人机召回从 62% 升到 81%。结论：干扰必须对称叠加。')
picture(s, '图D_叠加WiFi训练前后.png', X0, 74, X1 - X0, 262)
cw = (X1 - X0 - 40) / 3
card(s, X0, 342, cw, 116, 'La：只叠加到无人机', ['新捷径：“A底噪 + WiFi = 无人机”', '城区叠加集虚警 7.3% → 24%'], color=ORANGE, body_size=12.5)
card(s, X0 + cw + 20, 342, cw, 116, 'Lb：对称叠加', ['城区背景同时叠到无人机和背景上', '虚警 1.9%，城区无人机召回 62% → 81%'], body_size=12.5)
card(s, X0 + 2 * (cw + 20), 342, cw, 116, '结论', ['叠加干扰必须对称', 'Lb 作为 2.4 GHz 基线'], color=DBLUE, body_size=12.5)

s = new_slide('窗长比较：10–50 ms', '【约30秒】【结论待复核后补】讲时间分辨率与样本数量的取舍。')
placeholder(s, X0, 80, 560, 375, '【插图：窗长比较结果（任务W）】')
text(s, X0 + 575, 84, X1 - X0 - 575, 370, [
    ('设计', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('10 / 20 / 30 / 40 / 50 ms × 两个频段 × 3个种子', {'bullet': True}),
    ('其他设置不变，一次只改窗长', {'bullet': True}),
    ('取舍', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('窗越长越能覆盖突发周期，但样本越少、时间分辨率越低', {'bullet': True}),
    ('结论', {'bold': True, 'color': ORANGE, 'size': 15}),
    ('【待补】', {'bullet': True}),
], size=13.5)

s = new_slide('时频图 + IQ手工特征融合', '【约60秒】导师建议第一阶段加入 IQ 特征。我们先查文献定融合方式（嵌入层拼接），并在融合前检查特征是否带接收机指纹。四组对照：R 只用时频图、Z 特征置零、F 10个特征、F5 5个特征。结论：2.4G 看不出稳定提升；5.8G 加特征反而变差，推测突发类特征携带了来源信息——说明手工特征必须先检验来源。')
picture(s, '图E_时频图加IQ特征对比.png', X0, 74, 560, 330)
text(s, X0 + 572, 78, X1 - X0 - 572, 380, [
    ('设计（对标文献）', {'bold': True, 'color': DBLUE, 'size': 14.5}),
    ('ResNet18 512维 + 特征64维，嵌入层拼接', {'bullet': True}),
    ('融合前剔除带接收机指纹的特征', {'bullet': True}),
    ('R / Z / F / F5 四组 × 3种子', {'bullet': True}),
    ('结果', {'bold': True, 'color': DBLUE, 'size': 14.5}),
    ('2.4 GHz：F 虚警最低（2.5%），但三种子方向不一致', {'bullet': True}),
    ('5.8 GHz：加特征后城区无人机召回下降 5.6–8.7 个百分点', {'bullet': True}),
], size=13)
banner(s, '启示：手工特征可能携带“录音来源”信息，必须先检验来源，再决定是否融合', t=420)

s = new_slide('难题五：按信号所在频带截取（进行中）', '【约50秒】固定频带会截掉部分信号，所以改为自动找信号所在频带。实验室和城区 5.8G 正常；城区 2.4G 的 WiFi 和图传交错，被并成一大块。思路有文献旁证：同一发射机的频点同开同关。')
placeholder(s, X0, 80, 520, 375, '【插图：频带检测示例——正常 vs 城区2.4G被合并】')
text(s, X0 + 535, 84, X1 - X0 - 535, 370, [
    ('问题', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('固定频带截不全：如城区 Mavic 2 的信号 99% 在 2460–2475 MHz', {'bullet': True}),
    ('进展', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('自动检测信号频带：实验室 2.4/5.8 GHz、城区 5.8 GHz 正常', {'bullet': True}),
    ('难点', {'bold': True, 'color': ORANGE, 'size': 15}),
    ('城区 2.4 GHz：WiFi 几乎一直在，又与图传交错 → 合并成 60–68 MHz 一块', {'bullet': True}),
    ('思路', {'bold': True, 'color': DBLUE, 'size': 15}),
    ('同一发射机的频点“同开同关”→ 按开关时刻相关性切分', {'bullet': True}),
    ('公开WiFi/蓝牙数据：调研21个，选定NIST，只下载需要的录音', {'bullet': True}),
], size=13)

# ================= 肆 创新点、问题与计划 =================
transition('肆', '第四部分', '创新点、问题与计划', ['创新点', '存在问题', '下一阶段计划'])

s = new_slide('创新点', '【约50秒】四条创新点都对应前面的难题。措辞用“据目前检索”，不说“首次”。')
cw = (X1 - X0 - 30) / 2
inno = [
    ('① 面向实测数据的“捷径”检验流程', ['背景替换、白噪声/陷波噪声、分接收机统计、同型号对比', '不只看随机测试集总准确率']),
    ('② 不补录数据构造同接收机背景', ['利用同录音空闲频段镜像，并做泄漏检验', '接收机A背景虚警 100% → 0%']),
    ('③ 对称干扰叠加', ['干扰同时叠加到背景和无人机，避免引入新捷径', '城区叠加集虚警 24%（不对称）vs 1.9%（对称）']),
    ('④ 先检验来源、再融合的IQ特征；按信号频带截取', ['融合前剔除带接收机指纹的特征', '按频带截取：进行中']),
]
for i, (h, b) in enumerate(inno):
    r_, c_ = divmod(i, 2)
    card(s, X0 + c_ * (cw + 30), 84 + r_ * 172, cw, 158, h, b, color=BLUE if i in (0, 3) else DBLUE, body_size=13.5)
text(s, X0, 432, X1 - X0, 40, [('注：据目前检索，相关文献多在随机划分下报告总体准确率，较少系统检验模型是否依赖来源线索。【答辩前再核对】', {'size': 11, 'color': GRAY})])

s = new_slide('存在问题', '【约40秒】如实说明局限，并说明各自的应对办法。')
text(s, X0 + 10, 86, X1 - X0 - 20, 400, [
    ('背景样本少：测试背景仅 80 片，“虚警 ≤1%”统计上还无法验证（0/80 的 95% 上界为 3.7%）→ 同时报告置信上界', {'bullet': True}),
    ('5.8 GHz 没有实验室接收机的背景，接收机与标签混在一起', {'bullet': True}),
    ('实验室接收机 2.4 GHz 只录到约 2461 MHz，高于此的图传信号缺失', {'bullet': True}),
    ('城区 2.4 GHz 的频带切分尚未解决', {'bullet': True}),
    ('手工特征在低信噪比下可能失效，需在加噪阶段复核', {'bullet': True}),
    ('本地仅一块 8 GB 显卡，长窗口和 IQ 输入训练受限', {'bullet': True}),
], size=15, space=12)

s = new_slide('下一阶段计划', '【约30秒】按时间顺序讲。【结题时间待核对】')
plan = [('2026.11—12', '公开WiFi/蓝牙对称叠加训练\n城区2.4G频带切分\n按信号频带截取后重训'),
        ('2027.01', '第一阶段达标\n漏检、虚警均 ≤1%'),
        ('2027.02—03', '低信噪比训练\n平均准确率 ≥95%'),
        ('2027.03—04', '第二阶段型号识别\n泛化测试'),
        ('2027.04—05', '消融与对照补全\n论文与结题报告')]
line = s.Shapes.AddLine(X0 + 20, 200, X1 - 20, 200)
line.Line.ForeColor.RGB = BLUE
line.Line.Weight = 3
pw = (X1 - X0 - 40) / 5
for i, (when, what) in enumerate(plan):
    cx = X0 + 20 + pw * i + pw / 2
    dot = s.Shapes.AddShape(9, cx - 10, 190, 20, 20)
    dot.Fill.ForeColor.RGB = ORANGE if i == 0 else BLUE
    dot.Line.ForeColor.RGB = WHITE
    text(s, cx - pw / 2, 150, pw, 30, [(when, {'size': 15, 'bold': True, 'color': DBLUE, 'align': 2})])
    box(s, cx - pw / 2 + 6, 225, pw - 12, 180, fill=LGRAY)
    text(s, cx - pw / 2 + 10, 232, pw - 20, 170, [(w, {'size': 13, 'align': 2}) for w in what.split('\n')], space=6)

# ================= 尾页 =================
e = S['end']
for sh in e.Shapes:
    if sh.HasTextFrame and sh.TextFrame.HasText:
        t = sh.TextFrame.TextRange.Text
        if t.startswith('答辩学生'):
            sh.TextFrame.TextRange.Text = '汇报人: xxx'
set_notes(e, '谢谢各位老师，请批评指正。')

S['trans'].Delete()
pres.Save()
print('slides:', pres.Slides.Count)
pres.Close()
try:
    app.Quit()
except Exception:
    pass
