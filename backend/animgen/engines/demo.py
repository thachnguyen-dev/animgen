"""Demo engine: CPU only, no models, deterministic.

It reads proportions from the silhouette and draws a flat figure on the fitted skeleton, using the reference's
own colours. It exists so the full app can be built and tested without a GPU (§0.2). Its frames are clearly not
AI output; the UI shows a banner while it is in use.
"""

from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

from .. import config
from ..authoring import Pose, solve
from ..imaging import alpha_array, mask_bounds, opaque_mask
from ..skeleton import (LANK, LEAR, LELB, LEYE, LHIP, LKNE, LSHO, LWRI, NECK, NOSE, RANK, REAR, RELB, REYE, RHIP,
                        RKNE, RSHO, RWRI, Skeleton)
from .base import Engine, FrameRequest


class ProportionalPoseReader:
    """Places the standard rest skeleton inside the figure's bounding box (feet on the foot line).

    It cannot tell which way the figure faces, so the session treats its facing as unknown."""

    detects_facing = False

    def read(self, image: Image.Image) -> Skeleton | None:
        b = mask_bounds(opaque_mask(alpha_array(image)))
        if b is None:
            return None
        pts, _ = solve(Pose())
        h = float(b.height)
        cx = (b.left + b.right) / 2.0
        foot = b.bottom - 1
        return Skeleton([None if p is None else (cx + p[0] * h, foot + p[1] * h) for p in pts])


def _band_colour(rgba: np.ndarray, top: float, bottom: float, b) -> tuple[int, int, int]:
    y0 = int(b.top + top * b.height)
    y1 = max(y0 + 1, int(b.top + bottom * b.height))
    region = rgba[y0:y1]
    m = region[..., 3] > config.ALPHA_OPAQUE
    if not m.any():
        return (128, 128, 128)
    c = np.median(region[..., :3][m], axis=0)
    return tuple(int(v) for v in c)


class DemoDrawer:
    def __init__(self) -> None:
        # frame index (0-based) -> number of attempts that should fail; used by tests of the retry logic
        self.fail_plan: dict[int, int] = {}
        self._attempts: dict[int, int] = {}

    def draw(self, req: FrameRequest) -> Image.Image:
        size = req.reference.size
        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        canvas[:] = req.background
        fi = int(req.extra.get("frame_index", -1))
        n = self._attempts.get(fi, 0)
        self._attempts[fi] = n + 1
        if n < self.fail_plan.get(fi, 0):
            return Image.fromarray(canvas)  # an empty frame: fails the technical check

        ref = np.asarray(req.reference.convert("RGBA"))
        b = mask_bounds(ref[..., 3] > config.ALPHA_OPAQUE)
        if b is None:
            return Image.fromarray(canvas)
        head_c = _band_colour(ref, 0.0, 0.16, b)
        body_c = _band_colour(ref, 0.20, 0.48, b)
        leg_c = _band_colour(ref, 0.55, 0.92, b)
        skin_c = tuple(int(v) for v in np.clip(np.array(head_c) * 0.5 + np.array([230, 200, 180]) * 0.5, 0, 255))
        far = lambda c: tuple(int(v * 0.72) for v in c)  # noqa: E731

        s = req.target_skeleton
        H = float(b.height)
        arm_w, leg_w, body_w = max(2, int(0.045 * H)), max(3, int(0.06 * H)), max(4, int(0.12 * H))

        def line(a, c, colour, width):
            pa, pb = s.points[a], s.points[c]
            if pa is None or pb is None:
                return
            cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), colour, width, cv2.LINE_AA)
            cv2.circle(canvas, (int(pb[0]), int(pb[1])), width // 2, colour, -1, cv2.LINE_AA)

        # far limbs first, then body, then near limbs
        line(LSHO, LELB, far(body_c), arm_w)
        line(LELB, LWRI, far(skin_c), arm_w)
        line(LHIP, LKNE, far(leg_c), leg_w)
        line(LKNE, LANK, far(leg_c), leg_w)
        pel = s.pelvis()
        if pel is not None and s.points[NECK] is not None:
            nk = s.points[NECK]
            cv2.line(canvas, (int(nk[0]), int(nk[1])), (int(pel[0]), int(pel[1])), body_c, body_w, cv2.LINE_AA)
        line(RHIP, RKNE, leg_c, leg_w)
        line(RKNE, RANK, leg_c, leg_w)
        line(RSHO, RELB, body_c, arm_w)
        line(RELB, RWRI, skin_c, arm_w)
        for wri in (LWRI, RWRI):
            p = s.points[wri]
            if p is not None:
                r = int(arm_w * (0.75 if req.hands_gripping else 0.6))
                cv2.circle(canvas, (int(p[0]), int(p[1])), r, skin_c if wri == RWRI else far(skin_c), -1,
                           cv2.LINE_AA)
        # head: centred a little above and in front of the neck
        face = [p for p in (s.points[NOSE], s.points[REYE], s.points[LEYE], s.points[REAR], s.points[LEAR])
                if p is not None]
        if face:
            hx = float(np.mean([p[0] for p in face]))
            hy = float(np.mean([p[1] for p in face]))
            cv2.circle(canvas, (int(hx), int(hy)), int(0.095 * H), head_c, -1, cv2.LINE_AA)
            nose = s.points[NOSE]
            if nose is not None:
                cv2.circle(canvas, (int(nose[0]), int(nose[1])), max(2, int(0.02 * H)), skin_c, -1, cv2.LINE_AA)
        return Image.fromarray(canvas)


class KeyRemover:
    """Removes the flat background by colour distance, with a soft edge."""

    def remove(self, image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        d = np.sqrt(((rgb - np.array(background, np.float32)) ** 2).sum(axis=-1))
        alpha = np.clip((d - 20.0) / 30.0, 0.0, 1.0)
        out = np.concatenate([rgb, alpha[..., None] * 255.0], axis=-1).astype(np.uint8)
        return Image.fromarray(out, mode="RGBA")


class DemoEngine(Engine):
    name = "demo"
    is_demo = True

    def __init__(self) -> None:
        super().__init__()
        self.pose = ProportionalPoseReader()
        self.drawer = DemoDrawer()
        self.remover = KeyRemover()
