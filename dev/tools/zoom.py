"""Zoomed contact sheet of a Phase 0 run's frames, cropped to the union of the figures, on a checkerboard.
usage: python zoom.py <session frames dir or preview dir> <out.png> [reference.png]"""
import sys
from pathlib import Path

from PIL import Image

d, out = Path(sys.argv[1]), sys.argv[2]
fs = sorted(d.glob("*.png"), key=lambda p: int(p.name[1:p.name.index("_")]))
latest = {}
for f in fs:  # keep the newest file per frame index
    i = int(f.name[1:f.name.index("_")])
    if i not in latest or f.stat().st_mtime > latest[i].stat().st_mtime:
        latest[i] = f
ims = [Image.open(latest[i]).convert("RGBA") for i in sorted(latest)]
if len(sys.argv) > 3:
    ims.insert(0, Image.open(sys.argv[3]).convert("RGBA").resize(ims[0].size))
box = None
for im in ims:
    b = im.getchannel("A").point(lambda v: 255 if v > 40 else 0).getbbox()
    if b:
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
pad = 10
box = (max(0, box[0] - pad), max(0, box[1] - pad), box[2] + pad, box[3] + pad)
crops = [im.crop(box) for im in ims]
w, h = crops[0].size
sheet = Image.new("RGBA", (w * len(crops), h))
for y in range(0, h, 16):
    for x in range(0, sheet.width, 16):
        c = 205 if (x // 16 + y // 16) % 2 else 240
        sheet.paste((c, c, c, 255), (x, y, x + 16, y + 16))
for i, c in enumerate(crops):
    sheet.alpha_composite(c, (i * w, 0))
scale = min(1.0, 2400 / sheet.width)
sheet.convert("RGB").resize((int(sheet.width * scale), int(h * scale))).save(out)
print(box, len(crops))
