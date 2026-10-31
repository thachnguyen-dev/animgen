"""Stick figures of every motion template, near limbs (the character's right side, facing right) red, far limbs
blue, to check left/right in the templates themselves. usage: python template_sheet.py <out.png> [animation ...]"""
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, r"C:\AnimGen\project\animgen\backend")
from animgen.skeleton import (LANK, LELB, LHIP, LKNE, LSHO, LWRI, NECK, NOSE, RANK, RELB, RHIP, RKNE,  # noqa: E402
                              RSHO, RWRI)
from animgen.templates import load_template  # noqa: E402

NEAR = [(NECK, RSHO), (RSHO, RELB), (RELB, RWRI), (NECK, RHIP), (RHIP, RKNE), (RKNE, RANK)]
FAR = [(NECK, LSHO), (LSHO, LELB), (LELB, LWRI), (NECK, LHIP), (LHIP, LKNE), (LKNE, LANK)]
CELL = 180

out = sys.argv[1]
anims = sys.argv[2:] or ["idle", "walk", "run", "a1a", "a1b", "a2", "a5", "a6"]
sheet = Image.new("RGB", (CELL * 8 + 60, CELL * len(anims)), "white")
d = ImageDraw.Draw(sheet)
for r, anim in enumerate(anims):
    tpl = load_template(anim)
    d.text((4, r * CELL + 4), anim, fill="black")
    pts_all = [p for f in tpl.frames for p in f.skeleton.points if p is not None]
    xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
    k = (CELL - 20) / max(max(xs) - min(xs), max(ys) - min(ys))
    for c, f in enumerate(tpl.frames):
        ox, oy = 60 + c * CELL + 10, r * CELL + 10

        def q(p):
            return ox + (p[0] - min(xs)) * k, oy + (p[1] - min(ys)) * k
        sk = f.skeleton.points
        for bones, col, w in ((FAR, (60, 90, 230), 3), (NEAR, (220, 40, 40), 4)):
            for a, b in bones:
                if sk[a] is not None and sk[b] is not None:
                    d.line([q(sk[a]), q(sk[b])], fill=col, width=w)
        if sk[NOSE] is not None:
            x, y = q(sk[NOSE])
            d.ellipse((x - 3, y - 3, x + 3, y + 3), fill="black")
        d.text((ox, oy + CELL - 30), str(c + 1), fill="grey")
sheet.save(out)
print(sheet.size)
