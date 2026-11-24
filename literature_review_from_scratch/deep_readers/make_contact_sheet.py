from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

src = Path(__file__).parent / "_rendered"
files = sorted(src.glob("*.png"))
thumbs = []
for path in files:
    im = Image.open(path).convert("RGB")
    im.thumbnail((420, 560))
    canvas = Image.new("RGB", (440, 600), "white")
    x = (440 - im.width) // 2
    canvas.paste(im, (x, 25))
    ImageDraw.Draw(canvas).text((10, 5), path.stem, fill="black")
    thumbs.append(canvas)

cols = 2
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 440, rows * 600), "#dddddd")
for i, im in enumerate(thumbs):
    sheet.paste(im, ((i % cols) * 440, (i // cols) * 600))
sheet.save(src / "contact_sheet.png")
