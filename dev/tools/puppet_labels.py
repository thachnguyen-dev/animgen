"""Which body part the puppet cut gives each pixel of a character (no GPU).
usage: python puppet_labels.py <character.png> <out.png>"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, r"C:\AnimGen\project\animgen\backend")
from animgen import puppet  # noqa: E402
from animgen.server import make_engine  # noqa: E402
from animgen.session import Session  # noqa: E402

COLOURS = {"head": (255, 220, 0), "torso": (120, 120, 120), "uarm": (0, 160, 255), "larm": (0, 60, 200),
           "uleg": (0, 200, 80), "lleg": (0, 110, 40)}

char, out = sys.argv[1], sys.argv[2]
s = Session(make_engine("diffusers", __import__("pathlib").Path(r"C:\AnimGen\models")), __import__("pathlib").Path(out).parent / "labels_work")
ch = s.set_character(char, None)
img, sk = ch.placement.image, ch.skeleton
names, label, alpha = puppet.cut_labels(np.array(img.convert("RGBA")), sk)
vis = np.array(img.convert("RGBA")).astype(np.float32)
for i, n in enumerate(names):
    m = (label == i) & alpha
    vis[m, :3] = vis[m, :3] * 0.35 + np.array(COLOURS[n]) * 0.65
b = Image.fromarray(vis.astype(np.uint8), "RGBA").getbbox()
Image.fromarray(vis.astype(np.uint8), "RGBA").crop(b).save(out)
print(names)
