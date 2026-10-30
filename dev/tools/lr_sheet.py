"""Left/right check: each generated frame of an ab_run.py result next to its target stick figure (near limbs red, far
limbs blue), in the character's own facing. usage: python lr_sheet.py <character.png> <animation> <ab_dir> <out.png>"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, r"D:\Game\ainm\animgen\backend")
from animgen.server import make_engine  # noqa: E402
from animgen.session import Session  # noqa: E402
from animgen.skeleton import (LANK, LELB, LHIP, LKNE, LSHO, LWRI, NECK, NOSE, RANK, RELB, RHIP, RKNE,  # noqa: E402
                              RSHO, RWRI, fit_frame, measure)
from animgen.templates import load_template  # noqa: E402

NEAR = [(NECK, RSHO), (RSHO, RELB), (RELB, RWRI), (NECK, RHIP), (RHIP, RKNE), (RKNE, RANK)]
FAR = [(NECK, LSHO), (LSHO, LELB), (LELB, LWRI), (NECK, LHIP), (LHIP, LKNE), (LKNE, LANK)]

char, anim, ab, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), sys.argv[4]
s = Session(make_engine("diffusers", Path(r"C:\AnimGen\models")), Path(out).parent / "lr_work")
ch = s.set_character(char, None)
tpl = load_template(anim)
props = measure(ch.skeleton, tpl.rest, ch.placement.figure_height, ch.placement.anchor[1])
size = ch.placement.image.width
targets = [fit_frame(f.skeleton, tpl.rest, props, ch.placement.anchor, f.grounded) for f in tpl.frames]
if ch.placement.mirrored:
    targets = [t.mirrored(size) for t in targets]
runs = json.load(open(ab / "ab.json"))
frames0 = [Image.open(p).convert("RGBA") for p in runs[0]["frames"]]
box = None
for im in frames0:
    b = im.getchannel("A").getbbox()
    box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
box = (box[0] - 30, box[1] - 30, box[2] + 30, box[3] + 30)
cw, chh = box[2] - box[0], box[3] - box[1]
sheet = Image.new("RGB", (cw * 8, (chh + 20) * (1 + len(runs))), "white")
d = ImageDraw.Draw(sheet)
for c, t in enumerate(targets):          # first row: the targets
    def q(p):
        return p[0] - box[0] + c * cw, p[1] - box[1]
    for bones, col, w in ((FAR, (60, 90, 230), 5), (NEAR, (220, 40, 40), 7)):
        for a, b in bones:
            if t.points[a] is not None and t.points[b] is not None:
                d.line([q(t.points[a]), q(t.points[b])], fill=col, width=w)
    if t.points[NOSE] is not None:
        x, y = q(t.points[NOSE])
        d.ellipse((x - 6, y - 6, x + 6, y + 6), fill="black")
for r, run in enumerate(runs, start=1):
    d.text((4, r * (chh + 20)), run["variant"], fill="black")
    for c, p in enumerate(run["frames"]):
        im = Image.open(p).convert("RGBA").crop(box)
        tile = Image.new("RGBA", im.size, (235, 235, 235, 255))
        tile.alpha_composite(im)
        sheet.paste(tile.convert("RGB"), (c * cw, r * (chh + 20) + 20))
k = min(1.0, 2400 / sheet.width)
sheet.resize((int(sheet.width * k), int(sheet.height * k))).save(out)
print(sheet.size, k)
