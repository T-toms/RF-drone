# -*- coding: utf-8 -*-
"""把 中期检查表_一二三项_原稿.md 的第一、二、三节填进中期检查表副本（原件不动）。
用法: anaconda python -I fill_form.py
"""
import os, re, sys
import win32com.client as win32

D = os.path.dirname(os.path.abspath(__file__))
SRC_DOC = os.path.join(D, '中期检查表_原件.doc')
OUT_DOC = os.path.join(D, '中期检查表_已填.doc')
MD = os.path.join(D, '中期检查表_一二三项_原稿.md')

text = open(MD, encoding='utf-8').read()
text = re.sub(r'【[^】]*】', '', text)          # 去掉【核对】等内部标记
secs = {}
for m in re.finditer(r'^## (一|二|三)\s*$(.*?)(?=^## |\Z)', text, re.M | re.S):
    secs[m.group(1)] = m.group(2).strip('\n')


def md_paras(block):
    """-> list of (plain_text, [(a,b) bold spans], kind)"""
    out = []
    for raw in block.split('\n'):
        line = raw.rstrip()
        if not line.strip():
            continue
        kind = 'p'
        if line.startswith('### '):
            line, kind = line[4:], 'h'
        elif re.match(r'^\s+- ', line):          # 二级条目
            line = '　　' + re.sub(r'^\s+- ', '', line)
        elif line.startswith('- '):
            line = '· ' + line[2:]
        spans, plain, pos = [], '', 0
        for part in re.split(r'(\*\*.*?\*\*)', line):
            if part.startswith('**') and part.endswith('**'):
                s = part[2:-2]
                spans.append((len(plain), len(plain) + len(s)))
                plain += s
            else:
                plain += part
        out.append((plain, spans, kind))
    return out


def fill_cell(doc, cell, paras):
    # 删掉原表为留空而放的空段落（保留第一段标题）
    ps = cell.Range.Paragraphs
    for i in range(ps.Count, 1, -1):
        p = ps(i).Range
        if p.Text.strip('\r\x07 \t　') == '':
            if i == ps.Count:          # 单元格最后一段不能删，清空即可
                doc.Range(p.Start, p.End - 1).Text = ''
                prev = ps(i - 1).Range
                doc.Range(prev.End - 1, prev.End).Delete()
            else:
                p.Delete()
    for plain, spans, kind in paras:
        end = cell.Range.End - 1
        r = doc.Range(end, end)
        r.InsertAfter('\r' + plain)
        start = r.Start + 1
        body = doc.Range(start, r.End)
        body.Font.Name = '宋体'
        body.Font.NameAscii = 'Times New Roman'
        body.Font.NameOther = 'Times New Roman'
        body.Font.Size = 10.5
        body.Font.Bold = (kind == 'h')
        pf = body.ParagraphFormat
        pf.Alignment = 3 if kind == 'p' else 0   # 两端对齐
        pf.LineSpacingRule = 0                  # 单倍
        pf.SpaceBefore = 3 if kind == 'h' else 0
        pf.SpaceAfter = 0
        pf.FirstLineIndent = 0
        pf.LeftIndent = 0
        for a, b in spans:
            doc.Range(start + a, start + b).Font.Bold = True


def schedule_rows(block):
    rows = []
    for line in block.split('\n'):
        if line.startswith('|') and '---' not in line and '起止时间' not in line:
            cells = [c.strip() for c in line.strip('|').split('|')]
            rows.append(cells)
    return rows


word = win32.DispatchEx('Word.Application')
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(SRC_DOC, False, True)
    doc.SaveAs2(OUT_DOC, 0)                    # wdFormatDocument (.doc)
    t = doc.Tables(1)
    fill_cell(doc, t.Cell(11, 1), md_paras(secs['一']))
    fill_cell(doc, t.Cell(12, 1), md_paras(secs['二']))
    rows = schedule_rows(secs['三'])
    n_slots = 5                                # 原表第15–19行
    while len(rows) > n_slots:
        t.Rows.Add(t.Rows(15 + n_slots))
        n_slots += 1
    for i, (when, what) in enumerate(rows):
        for col, val in ((1, when), (2, what)):
            c = t.Cell(15 + i, col)
            c.Range.Text = val
            c.Range.Font.Name = '宋体'
            c.Range.Font.NameAscii = 'Times New Roman'
            c.Range.Font.Size = 10.5
            c.Range.ParagraphFormat.Alignment = 1 if col == 1 else 0
            c.VerticalAlignment = 1
    doc.Save()
    print('pages:', doc.ComputeStatistics(2))
    doc.Close(False)
finally:
    try:
        word.Quit()
    except Exception:
        pass
print('ok', OUT_DOC)
