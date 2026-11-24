# -*- coding: utf-8 -*-
"""Apply MID-R4b to the accepted R4 package without resaving it in WPS."""
from copy import deepcopy
from pathlib import Path
import hashlib
import posixpath
from zipfile import ZipFile

from lxml import etree as E
from PIL import Image, ImageDraw, ImageFont

B = Path(__file__).resolve().parent
A = B / 'r4_assets'
OUT = B / 'r4_work.pptx'
BASE_HASH = '0cbcfb9b45f9ed12463211c3ce29c15030a74e7bf5c64ab8e254cd8d6d833ae5'
NS = {
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}
PT = 12700


def xml(node, path):
    return node.find(path, NS)


def label(node):
    return ''.join(node.xpath('.//a:t/text()', namespaces=NS))


def replace_text(node, text):
    runs = node.findall('.//a:t', NS)
    runs[0].text = text
    for run in runs[1:]:
        run.text = None


def transform(node):
    for path in ['p:spPr/a:xfrm', 'p:xfrm']:
        xf = xml(node, path)
        if xf is not None:
            return xf


def box(node):
    xf = transform(node)
    if xf is None:
        return None
    off, ext = xml(xf, 'a:off'), xml(xf, 'a:ext')
    return tuple(int(element.get(key)) / PT for element, key in
                 [(off, 'x'), (off, 'y'), (ext, 'cx'), (ext, 'cy')])


def set_box(node, *, x=None, y=None, w=None, h=None):
    xf = transform(node)
    for path, key, value in [('a:off', 'x', x), ('a:off', 'y', y),
                             ('a:ext', 'cx', w), ('a:ext', 'cy', h)]:
        if value is not None:
            xml(xf, path).set(key, str(round(value * PT)))


def text_shape(tree, prefix):
    return next(sp for sp in tree if label(sp).startswith(prefix))


def crop_assets():
    for src, dest, box_ in [
        ('r2_refs/gluge200-03.png', 'gluge_mix3.png', (850, 1645, 1575, 1860)),
        ('r2_refs/drone200-07.png', 'drone_table4_r4b.png', (133, 218, 796, 739)),
        ('r4_assets/channel_before.png', 'channel_before_r4b.png', (0, 19, 495, 180)),
        ('r4_assets/channel_after.png', 'channel_after_r4b.png', (0, 19, 495, 179)),
        ('normal.png', 'normal_r4b.png', (0, 29, 745, 260)),
        ('merged.png', 'merged_r4b.png', (0, 29, 745, 260)),
    ]:
        with Image.open(B / src) as im:
            im.crop(box_).save(A / dest)

    # The recording ID is printed inside the plot. Cover only its annotation
    # box and leave the visible time-frequency evidence around it intact.
    with Image.open(A / 'm100_window.png') as im:
        im = im.convert('RGB')
        ImageDraw.Draw(im).rectangle((101, 20, 195, 33), fill='#363D65')
        im.save(A / 'm100_window_r4b.png')

    with Image.open(A / 'line2426.png') as im:
        im = im.convert('RGB')
        draw = ImageDraw.Draw(im)
        draw.rectangle((750, 279, 1070, 321), fill='white')
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 24)
        draw.text((910, 284), '绝对频率（MHz）', font=font, anchor='mt', fill='#404040')
        im.save(A / 'line2426_r4b.png')

    with Image.open(B / 'r2_refs/soltani200-05.png') as im:
        im = im.crop((133, 150, 841, 387)).convert('RGB')
        ImageDraw.Draw(im).rectangle((393, 201, im.width, im.height), fill='white')
        im.save(A / 'soltani49_r4b.png')


PICTURES = {
    18: 'drone_table4_r4b.png',
    19: 'm100_window_r4b.png',
    21: 'channel_before_r4b.png',
    22: 'channel_after_r4b.png',
    24: 'soltani49_r4b.png',
    26: 'window_miss.png',
    27: 'window_recall.png',
    28: 'line2426_r4b.png',
    29: 'source_probe.png',
    30: 'background_r4b.png',
    31: 'background_fa_r4b.png',
    32: 'gluge_mix3.png',
    33: 'wifi_focus.png',
    34: 'fusion.png',
    35: 'normal_r4b.png',
    36: 'merged_r4b.png',
}

# Vertical boundaries are in points. The bounds were measured in WPS, then
# adjusted for the final paragraph positions and line spacing.
ARROW_GAPS = {
    4: {27: [(200.7, 209.8), (281.5, 290.6)]},
    5: {27: [(148.8, 162.8), (180.7, 194.7), (266.4, 279), (314.8, 330)],
        487: [(148.8, 162.8), (180.7, 194.7), (266.4, 275.4), (293.4, 307.4)]},
    8: {27: [(246.5, 255.8), (324.4, 333.8), (385.2, 411.7)]},
    10: {27: [(183.7, 194.5), (248.2, 258.9), (348.4, 356), (391.7, 420.4)]},
    11: {27: [(182.9, 193.6), (247.3, 258.1), (347.6, 355.1), (390.9, 419.5)]},
    12: {27: [(164.9, 177.3), (213.1, 241.8), (295.5, 322.5), (376.2, 387)]},
    13: {27: [(164.9, 193.6), (247.3, 258.1), (329.7, 338.8), (392.5, 403.3)]},
    14: {27: [(193.4, 205.9), (269.3, 281.9), (345.2, 355), (417.4, 428)]},
    15: {27: [(182.9, 193.6), (247.3, 258.1), (347.6, 355.1), (390.9, 403.3)]},
    16: {27: [(182.9, 193.6), (247.3, 258.1), (329.7, 338.8), (392.5, 403.3)]},
    17: {492: [(165.8, 178.2), (214, 226.4), (262.2, 274.6)]},
}


def footer(tree, page):
    refs = [sp for sp in tree if box(sp) and abs(box(sp)[1] - 497) < .1 and label(sp)]
    for ref in refs:
        set_box(ref, x=35, y=462, w=890, h=16)
        for p in ref.findall('p:txBody/a:p/a:pPr', NS):
            p.set('algn', 'r')
        body = xml(ref, 'p:txBody/a:bodyPr')
        body.set('tIns', '0')
        body.set('bIns', '0')
    if page == 10:
        set_box(text_shape(tree, '2426.88 MHz固定'), y=400)
        set_box(text_shape(tree, '489块 / 173组'), y=427)
    if page == 11:
        set_box(text_shape(tree, '90%  vs'), y=425)
        set_box(text_shape(tree, '同一信号、0 dB'), x=757, y=435, w=168, h=16)
    if page == 20:
        for prefix in ('跨设备／带宽／状态 →', '留出未知型号 →', '检测 → 型号 →'):
            set_box(text_shape(tree, prefix), h=16)


def table_parameters(tree):
    frame = next(sp for sp in tree if xml(sp, './/a:tbl') is not None)
    table = xml(frame, './/a:tbl')
    header, sample, band, stft, output = table.findall('a:tr', NS)
    for cell in stft.findall('a:tc', NS)[1:]:
        replace_text(cell, '1024点汉宁窗 / 帧移512')
    index = table.index(stft)
    for values in [('陷波（MHz）', '2426.88 / 2440 / 2450', '5795.5 / 5800'),
                   ('片段', '20 ms', '20 ms')]:
        row = deepcopy(band)
        for cell, value in zip(row.findall('a:tc', NS), values):
            replace_text(cell, value)
        table.insert(index, row)
        index += 1
    heights = [19, 18, 18, 18, 20, 18, 18]
    for row, height in zip(table.findall('a:tr', NS), heights):
        row.set('h', str(round(height * PT)))
    set_box(frame, h=sum(heights))


def picture_layout(tree, relationships):
    media = {r.get('Id'): posixpath.basename(r.get('Target')) for r in relationships}
    fit = {
        18: (38, 355, 128), 19: (200, 368, 231), 21: (528, 414, 120), 22: (751, 414, 120),
        24: (250, 328, 178), 26: (580, 318, 150), 27: (758, 318, 150),
        30: (600, 311, 325), 31: (600, 263, 325), 32: (600, 131, 300),
        33: (600, 263, 325), 35: (600, 146, 283), 36: (600, 330, 283),
    }
    for pic in tree.findall('p:pic', NS):
        blip = xml(pic, './/a:blip')
        name = media[blip.get(f'{{{NS["r"]}}}embed')]
        for number, (x, y, w) in fit.items():
            if name == f'image{number}.png':
                with Image.open(A / PICTURES[number]) as im:
                    set_box(pic, x=x, y=y, w=w, h=w * im.height / im.width)


def narrative(tree, page):
    if page == 5:
        for prefix, y in [('改进方法　先画', 278), ('效果与启示　窗长', 329)]:
            set_box(text_shape(tree, prefix), y=y)
    if page == 14:
        for sp in tree:
            if label(sp).startswith(('研究背景　', '现象　', '分析与验证　', '改进方法　', '效果与启示　')):
                for line in sp.findall('.//a:lnSpc/a:spcPct', NS):
                    line.set('val', '120000')
        set_box(text_shape(tree, '改进方法　'), y=354)
        set_box(text_shape(tree, '效果与启示　'), y=427)
    for x, gaps in ARROW_GAPS.get(page, {}).items():
        arrows = [sh for sh in tree.findall('p:cxnSp', NS)
                  if round(box(sh)[0]) == x and box(sh)[2] < 1 and box(sh)[3] < 24]
        arrows.sort(key=lambda sh: box(sh)[1])
        assert len(arrows) == len(gaps), (page, x)
        for arrow, (end, start) in zip(arrows, gaps):
            height = min(8, start - end - 2)
            set_box(arrow, y=(start + end - height) / 2, h=height)
    if page == 17:
        workload = text_shape(tree, '约360 GB')
        methods = deepcopy(workload)
        nv = xml(methods, 'p:nvSpPr/p:cNvPr')
        next_id = max(int(pr.get('id')) for pr in tree.findall('.//p:cNvPr', NS)) + 1
        nv.set('id', str(next_id))
        nv.set('name', '对照方法')
        replace_text(methods, '对照方法：背景替换、置零、打乱、平移')
        set_box(methods, y=332, h=24)
        tree.insert(tree.index(workload), methods)
        set_box(workload, y=391)


def change_page(page, data, rels):
    root = E.fromstring(data)
    tree = xml(root, 'p:cSld/p:spTree')
    footer(tree, page)
    narrative(tree, page)
    picture_layout(tree, E.fromstring(rels))
    if page == 8:
        table_parameters(tree)
        set_box(text_shape(tree, '窗长比较：'), y=432)
    if page in (11, 12):
        for sp in list(tree):
            if xml(sp, 'p:nvSpPr/p:cNvPr') is not None and xml(sp, 'p:nvSpPr/p:cNvPr').get('name') == 'r4_native_label':
                tree.remove(sp)
    if page == 12:
        set_box(text_shape(tree, '100% → 0%'), y=433)
    if page == 13:
        set_box(text_shape(tree, 'Glüge原文式'), y=226)
        set_box(text_shape(tree, '三种叠加做法'), y=408)
        set_box(text_shape(tree, '24%  vs'), y=424)
    return E.tostring(root, encoding='utf-8', xml_declaration=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path, help='accepted R4 work package before the R4b patch')
    args = parser.parse_args()
    source = args.baseline
    assert hashlib.sha256(source.read_bytes()).hexdigest() == BASE_HASH, source
    crop_assets()
    temporary = A / 'r4b_pending.pptx'
    changed_pages = set(range(4, 21))
    with ZipFile(source) as old:
        with ZipFile(temporary, 'w') as new:
            changes = {f'ppt/media/image{i}.png': (A / name).read_bytes() for i, name in PICTURES.items()}
            for page in changed_pages:
                slide = f'slide{page}.xml'
                key = f'ppt/slides/{slide}'
                changes[key] = change_page(page, old.read(key), old.read(f'ppt/slides/_rels/{slide}.rels'))
            for info in old.infolist():
                new.writestr(info, changes.pop(info.filename, None) or old.read(info.filename))
            assert not changes, changes.keys()
    temporary.replace(OUT)


if __name__ == '__main__':
    main()
