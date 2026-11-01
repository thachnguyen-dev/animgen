"""One picture of an animation's 8 poses for the animation list (owner, 2026-11-01): the same skeleton drawing as the
contact sheets in docs/templates (authoring.contact_sheet), drawn from the template as it is now, so the built-in
poses and the owner's own poses show their current frames. Four frames a row; the frame number is the only text.
All frames share one scale and one ground line, fitted to the animation's whole range of movement, so the motion
between frames can be compared."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .skeleton import Skeleton, draw_openpose
from .templates import MotionTemplate

COLUMNS = 4
MARGIN = 12
GROUND = (90, 90, 90)
NUMBER = (170, 170, 170)


def draw(tpl: MotionTemplate, cell: int = 160) -> Image.Image:
    """The template's frames in reading order, a ground line under each, numbered from 1."""
    rows = (len(tpl.frames) + COLUMNS - 1) // COLUMNS
    pts = [p for fr in tpl.frames for p in fr.skeleton.points if p is not None]
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    y0, y1 = min(p[1] for p in pts), max(p[1] for p in pts)
    scale = (cell - 2 * MARGIN) / max(x1 - x0, y1 - y0, 1e-6)
    ox = cell / 2 - (x0 + x1) / 2 * scale                # centred across the animation's width
    oy = cell - MARGIN - y1 * scale                      # the lowest point of any frame rests on the ground line
    ground = int(round(cell - MARGIN))
    sheet = np.zeros((cell * rows, cell * COLUMNS, 3), np.uint8)
    for i, fr in enumerate(tpl.frames):
        moved = [None if p is None else (p[0] * scale + ox, p[1] * scale + oy) for p in fr.skeleton.points]
        img = draw_openpose(Skeleton(moved), (cell, cell), stick_width=3)
        cv2.line(img, (0, ground), (cell, ground), GROUND, 1)
        cv2.putText(img, str(i + 1), (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, NUMBER, 1, cv2.LINE_AA)
        r, c = divmod(i, COLUMNS)
        sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = img
    for c in range(1, COLUMNS):                          # a thin line between the frames
        sheet[:, c * cell - 1] = GROUND
    for r in range(1, rows):
        sheet[r * cell - 1, :] = GROUND
    return Image.fromarray(sheet)
