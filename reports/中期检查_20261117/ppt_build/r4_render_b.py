# -*- coding: utf-8 -*-
"""Export the R4b work package through WPS in read-only mode."""
from pathlib import Path
import argparse
import json

from PIL import Image, ImageDraw
import win32com.client as win32

B = Path(__file__).resolve().parent


def render(destination, audit):
    destination.mkdir(parents=True, exist_ok=True)
    app = win32.DispatchEx('PowerPoint.Application')
    deck = app.Presentations.Open(str(B / 'r4_work.pptx'), True, False, False)
    data = []
    try:
        assert deck.Slides.Count == 21
        for slide in deck.Slides:
            i = slide.SlideIndex
            slide.Export(str(destination / f'r4_{i:02d}.png'), 'PNG', 1920, 1080)
            shapes = []
            for sh in slide.Shapes:
                rect = [float(v) for v in (sh.Left, sh.Top, sh.Width, sh.Height)]
                if sh.HasTextFrame and sh.TextFrame.HasText:
                    tr = sh.TextFrame.TextRange
                    shapes.append({'text': tr.Text, 'rect': rect, 'bounds': [tr.BoundLeft, tr.BoundTop,
                                                                            tr.BoundWidth, tr.BoundHeight]})
                elif sh.Type == 9:
                    shapes.append({'connector': rect})
            data.append({'page': i, 'shapes': shapes})
            print(f'Exported {i:02d}', flush=True)
    finally:
        deck.Close()
    audit.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def contact(destination, output):
    sheet = Image.new('RGB', (1920, 1740), '#DFE5EA')
    draw = ImageDraw.Draw(sheet)
    for i in range(21):
        x, y = i % 4 * 480, i // 4 * 290
        with Image.open(destination / f'r4_{i + 1:02d}.png') as slide:
            assert slide.size == (1920, 1080)
            sheet.paste(slide.resize((480, 270), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 8, y + 272), str(i + 1), fill='black')
    sheet.save(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, default=B.parent / 'ppt_preview')
    parser.add_argument('--audit', type=Path, default=B / 'r4_assets/audit_r4b.json')
    parser.add_argument('--contact', type=Path, default=B.parent / 'contact_r4.png')
    args = parser.parse_args()
    render(args.preview, args.audit)
    contact(args.preview, args.contact)
