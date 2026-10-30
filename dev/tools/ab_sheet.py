"""Side-by-side sheet of an ab_run.py result: one row per variant (the reference first), frames cropped to the union of
the figures, on a checkerboard.
usage: python ab_sheet.py <ab_dir> <out.png> [reference.png] [frames, e.g. 1,3,5,7]"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

d, out = Path(sys.argv[1]), sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] else None
pick = [int(x) - 1 for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else list(range(8))
runs = json.load(open(d / "ab.json"))
rows = [(r["variant"].split(":")[0], [Image.open(r["frames"][i]).convert("RGBA") for i in pick]) for r in runs]
box = None
for _, ims in rows:
    for im in ims:
        b = im.getchannel("A").point(lambda v: 255 if v > 40 else 0).getbbox()
        if b:
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
m = 16
box = (box[0] - m, box[1] - m, box[2] + m, box[3] + m)
cw, ch = box[2] - box[0], box[3] - box[1]
label = 120
cols = len(pick) + (1 if ref else 0)
sheet = Image.new("RGBA", (label + cols * cw, len(rows) * ch), (255, 255, 255, 255))
checker = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
cd = ImageDraw.Draw(checker)
for y in range(0, ch, 16):
    for x in range(0, cw, 16):
        if (x // 16 + y // 16) % 2:
            cd.rectangle((x, y, x + 15, y + 15), fill=(231, 229, 228, 255))
draw = ImageDraw.Draw(sheet)
refim = Image.open(ref).convert("RGBA") if ref else None
for r, (name, ims) in enumerate(rows):
    draw.text((8, r * ch + 8), name, fill=(0, 0, 0, 255))
    cells = ([refim.resize(ims[0].size).crop(box)] if refim else []) + [im.crop(box) for im in ims]
    for c, cell in enumerate(cells):
        tile = checker.copy()
        tile.alpha_composite(cell)
        sheet.paste(tile, (label + c * cw, r * ch))
scale = min(1.0, 2400 / sheet.width)
sheet.convert("RGB").resize((int(sheet.width * scale), int(sheet.height * scale)), Image.Resampling.LANCZOS).save(out)
print(sheet.size, scale)
