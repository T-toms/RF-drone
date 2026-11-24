from pathlib import Path
from PIL import Image


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "_rendered"
OUT = ROOT / "_evidence"
OUT.mkdir(parents=True, exist_ok=True)

# Coordinates were chosen after visual inspection of the rendered pages.
CROPS = {
    "01_dronerfa": ("dronerfa_p8.png", (120, 180, 900, 850)),
    "02_openrf": ("openrf_p14.png", (180, 70, 850, 820)),
    "03_lowsnr": ("lowsnr_p8.png", (70, 80, 940, 900)),
    "04_tccn": ("tccn_p9.png", (60, 50, 930, 650)),
    "05_tvt20": ("tvt20_p13.png", (70, 60, 520, 350)),
    "06_tvt21": ("tvt21_p10.png", (50, 50, 520, 700)),
    "07_drffr2": ("drffr2_p9.png", (70, 180, 950, 900)),
    "08_llm": ("llm_p6.png", (50, 50, 950, 680)),
    "09_sdrlite": ("sdrlite_p12.png", (50, 200, 520, 900)),
    "10_resnet": ("resnet_p6.png", (50, 60, 520, 650)),
}

for stem, (filename, box) in CROPS.items():
    with Image.open(SRC / filename) as im:
        crop = im.crop(box)
        crop.save(OUT / f"{stem}_key_evidence.png", optimize=True)
        print(f"{stem}: {crop.size}")
