"""Synthetic test images (§0.2: the cloud build has no owner art; it tests with images it makes itself)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from animgen.authoring import Pose, solve
from animgen.skeleton import Skeleton


def character(path: Path, size=(600, 900), height=760, facing="right", with_alpha=True) -> Path:
    """A simple side-view figure with real transparency: coloured limbs, a head facing ``facing``."""
    w, h = size
    pts, _ = solve(Pose())
    foot = h - 60
    cx = w / 2
    sk = Skeleton([None if p is None else (cx + p[0] * height, foot + p[1] * height) for p in pts])
    if facing == "left":
        sk = sk.mirrored(w)
    canvas = np.zeros((h, w, 4), np.uint8)

    def seg(a, b, colour, width):
        pa, pb = sk.points[a], sk.points[b]
        cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), colour + (255,), width, cv2.LINE_AA)
        cv2.circle(canvas, (int(pb[0]), int(pb[1])), width // 2, colour + (255,), -1, cv2.LINE_AA)

    seg(5, 6, (60, 60, 150), 30)
    seg(6, 7, (230, 190, 160), 26)
    seg(11, 12, (40, 40, 40), 42)
    seg(12, 13, (40, 40, 40), 40)
    pel = sk.pelvis()
    nk = sk.points[1]
    cv2.line(canvas, (int(nk[0]), int(nk[1])), (int(pel[0]), int(pel[1])), (80, 80, 200, 255), 90, cv2.LINE_AA)
    seg(8, 9, (50, 50, 50), 44)
    seg(9, 10, (50, 50, 50), 42)
    seg(2, 3, (80, 80, 200), 32)
    seg(3, 4, (240, 200, 170), 28)
    face = [sk.points[i] for i in (0, 14, 16)]
    hx, hy = np.mean([p[0] for p in face]), np.mean([p[1] for p in face])
    cv2.circle(canvas, (int(hx), int(hy) - 10), 78, (200, 120, 40, 255), -1, cv2.LINE_AA)  # hair
    nose = sk.points[0]
    cv2.circle(canvas, (int(nose[0]), int(nose[1])), 22, (240, 200, 170, 255), -1, cv2.LINE_AA)
    img = Image.fromarray(canvas, "RGBA")
    if not with_alpha:
        img = Image.fromarray(canvas[..., :3], "RGB")
    img.save(path)
    return path


def checkerboard_character(path: Path, size=(600, 900), block=16) -> Path:
    """The common mistake: a checkerboard drawn into the pixels instead of transparency."""
    char = character(path.with_name("_tmp_char.png"), size)
    with Image.open(char) as c:
        c = c.convert("RGBA")
    ys, xs = np.mgrid[0:size[1], 0:size[0]]
    board = np.where(((xs // block + ys // block) % 2)[..., None] == 0, 255, 204).astype(np.uint8)
    bg = Image.fromarray(np.repeat(board, 3, axis=2), "RGB").convert("RGBA")
    bg.alpha_composite(c)
    bg.convert("RGB").save(path)
    char.unlink()
    return path


def two_figures(path: Path) -> Path:
    a = character(path.with_name("_a.png"), (400, 900), 700)
    with Image.open(a) as im:
        im = im.convert("RGBA")
    out = Image.new("RGBA", (900, 900), (0, 0, 0, 0))
    out.alpha_composite(im, (0, 0))
    out.alpha_composite(im, (480, 0))
    out.save(path)
    a.unlink()
    return path


def opaque(path: Path) -> Path:
    Image.new("RGB", (300, 300), (40, 120, 200)).save(path)
    return path
