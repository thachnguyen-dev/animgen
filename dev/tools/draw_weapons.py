"""Temporary weapon art in a bold-outline cartoon style (not the owner's art). Drawn at 4x and reduced.

Writes test_sword.png and test_bow_1..3.png into <out_dir> and prints the grip and tip points (image pixels).
Scale: for a character about 1150 px tall (test_hero.png).
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

S = 4
INK = (30, 24, 22, 255)


def finish(img: Image.Image, path: Path) -> None:
    img.resize((img.width // S, img.height // S), Image.Resampling.LANCZOS).save(path)


def poly(d, pts, fill, w=6):
    pts = [(x * S, y * S) for x, y in pts]
    d.polygon(pts, fill=fill)
    d.line(pts + [pts[0]], fill=INK, width=w * S, joint="curve")


def sword(out: Path):
    W, H = 560, 70
    img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cy = H / 2
    poly(d, [(14, cy - 8), (26, cy - 8), (26, cy + 8), (14, cy + 8)], (200, 160, 60, 255))           # pommel
    poly(d, [(26, cy - 6), (112, cy - 6), (112, cy + 6), (26, cy + 6)], (110, 64, 36, 255))          # grip
    for x in range(36, 110, 14):                                                                     # grip wrap
        d.line([(x * S, (cy - 6) * S), ((x + 8) * S, (cy + 6) * S)], fill=INK, width=2 * S)
    poly(d, [(112, cy - 26), (126, cy - 26), (126, cy + 26), (112, cy + 26)], (200, 160, 60, 255))   # guard
    poly(d, [(126, cy - 11), (500, cy - 11), (548, cy), (500, cy + 11), (126, cy + 11)], (205, 212, 222, 255))
    d.line([(132 * S, cy * S), (505 * S, cy * S)], fill=(150, 158, 172, 255), width=3 * S)           # fuller
    finish(img, out / "test_sword.png")
    print("sword grip", (70, int(cy)), "tip", (548, int(cy)))


def bow(out: Path, stage: int):
    W, H = 320, 820
    img = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    gx, gy = 210, H / 2                      # grip; the archer is on the left, the target on the right
    bend = [0, 0, 18, 34][stage]             # limbs bend more as the bow is drawn
    pull = [0, 0, 70, 125][stage]            # string pulled towards the archer
    tips = []
    for sgn in (-1, 1):                      # upper and lower limb
        pts = []
        for i in range(41):
            t = i / 40
            y = gy + sgn * t * (370 - bend * 0.6)
            x = gx - (45 + bend) * t ** 2 + 18 * max(0.0, t - 0.88) / 0.12   # limbs sweep back, tips recurve
            pts.append((x, y))
        tips.append(pts[-1])
        outer = [(x * S, y * S) for x, y in pts]
        d.line(outer, fill=INK, width=17 * S, joint="curve")
        d.line(outer, fill=(150, 96, 48, 255), width=10 * S, joint="curve")
    (ux, uy), (lx, ly) = tips
    nock = (min(ux, lx) - pull, gy)
    d.line([(ux * S, uy * S), (nock[0] * S, nock[1] * S), (lx * S, ly * S)], fill=(70, 62, 58, 255), width=3 * S)
    poly(d, [(gx - 9, gy - 34), (gx + 9, gy - 34), (gx + 9, gy + 34), (gx - 9, gy + 34)], (90, 52, 30, 255), w=4)
    finish(img, out / f"test_bow_{stage}.png")
    print(f"bow {stage} grip", (gx, int(gy)), "tip", (int(ux), int(uy)))


if __name__ == "__main__":
    o = Path(sys.argv[1])
    sword(o)
    for st in (1, 2, 3):
        bow(o, st)
