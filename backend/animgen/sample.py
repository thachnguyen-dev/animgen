"""The sample character: a figure made of simple shapes whose body parts are known exactly (owner, 2026-10-31).

Used by the pose editor (poses.py), where the owner moves its joints to make a new animation, and by
dev/tools/shape_character.py, which renders every animation with it. Each part is its own flat-coloured shape with
a dark outline; the rig is written straight from those shapes, so no AI drawing is needed and frames take seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import parts as parts_mod
from .authoring import Pose, solve
from .authoring34 import Pose3, project, solve3
from .skeleton import (LANK, LELB, LEYE, LHIP, LKNE, LSHO, LWRI, NECK, NOSE, RANK, REAR, RELB, REYE, RHIP,  # noqa
                       RKNE, RSHO, RWRI)

TORSO_PAD_34 = 0.03        # the torso block reaches this far past the shoulder and hip joints
# The character's own pose (the bind pose): arms and legs a little apart so every part shows.
BIND = Pose(spine=2, head=0, r_sh=22, r_el=25, l_sh=-22, l_el=25, r_hip=14, r_knee=4, l_hip=-12, l_knee=4)
FIGURE_PX = 900            # figure height in the drawn image
OUTLINE = (30, 30, 40)
SKIN = (250, 208, 170)
# Part widths in template units (figure height 1.0) and colours; far limbs darker.
WIDTH = {"arm_upper": 0.06, "arm_lower": 0.052, "leg_upper": 0.08, "leg_lower": 0.064, "torso": 0.14}
COLOUR = {"torso": (60, 110, 205), "arm_upper": (225, 75, 60), "arm_lower": (240, 150, 60),
          "far_arm_upper": (150, 50, 40), "far_arm_lower": (165, 100, 40), "leg_upper": (60, 160, 80),
          "leg_lower": (110, 200, 90), "far_leg_upper": (40, 105, 55), "far_leg_lower": (75, 135, 60)}
LIMBS = {"far_arm_upper": (LSHO, LELB), "far_arm_lower": (LELB, LWRI), "far_leg_upper": (LHIP, LKNE),
         "far_leg_lower": (LKNE, LANK), "leg_upper": (RHIP, RKNE), "leg_lower": (RKNE, RANK),
         "arm_upper": (RSHO, RELB), "arm_lower": (RELB, RWRI)}


def joints(size: tuple[int, int], view: str = "side") -> tuple[list, float]:
    """The bind pose in image pixels, feet on a line near the bottom, and pixels per template unit."""
    if view == "34":
        # The same body in 3D, turned 45 degrees toward the viewer (authoring34): the near arm at the back edge of
        # the body, the far arm at its front edge, both eyes and the near ear showing.
        pts = solve3(Pose3(base=BIND, chest=0.0, hips=0.0, r_out=14.0, l_out=14.0))[0]
    else:
        pts, _ = solve(BIND)
    u = FIGURE_PX
    w, h = size
    return [None if p is None else (w * 0.45 + p[0] * u, h - 40 + p[1] * u) for p in pts], u


def _capsule(img: np.ndarray, a, b, width: float, colour, line: int) -> None:
    pa, pb = tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b)
    cv2.line(img, pa, pb, (*OUTLINE, 255), int(round(width + 2 * line)), cv2.LINE_AA)
    cv2.line(img, pa, pb, (*colour, 255), int(round(width)), cv2.LINE_AA)


def _circle(img: np.ndarray, c, r: float, colour, line: int) -> None:
    pc = tuple(int(round(v)) for v in c)
    cv2.circle(img, pc, int(round(r + line)), (*OUTLINE, 255), -1, cv2.LINE_AA)
    cv2.circle(img, pc, int(round(r)), (*colour, 255), -1, cv2.LINE_AA)


def draw_layers(P: list, u: float, size: tuple[int, int], view: str = "side") -> dict[str, np.ndarray]:
    """Each part as a canvas-sized RGBA layer, drawn whole (nothing hidden)."""
    w, h = size
    line = max(2, int(round(0.006 * u)))
    out: dict[str, np.ndarray] = {}

    def blank() -> np.ndarray:
        return np.zeros((h, w, 4), np.uint8)

    for name, (a, b) in LIMBS.items():
        img = blank()
        kind = name.replace("far_", "")
        _capsule(img, P[a], P[b], WIDTH[kind] * u, COLOUR[name], line)
        if kind == "arm_lower":            # the fist
            d = np.subtract(P[b], P[a]) / (np.linalg.norm(np.subtract(P[b], P[a])) + 1e-6)
            fist = SKIN if not name.startswith("far") else tuple(int(c * 0.8) for c in SKIN)
            _circle(img, np.add(P[b], d * 0.025 * u), 0.036 * u, fist, line)
        if kind == "leg_lower":            # the foot points forward (in 3/4: forward and toward the viewer)
            dark = (70, 70, 80) if not name.startswith("far") else (50, 50, 58)
            ankle = P[b]
            fx, fy, _ = project((0.07, 0.0, 0.0)) if view == "34" else (0.07, 0.0, 0.0)
            _capsule(img, (ankle[0] - 0.01 * u, ankle[1] + 0.012 * u),
                     (ankle[0] + fx * u, ankle[1] + (0.02 + fy) * u), 0.045 * u, dark, line)
        out[name] = img
    # Torso: a capsule from the neck to the middle of the hips (side), or a block over both shoulders and hips (3/4).
    img = blank()
    hip = ((P[RHIP][0] + P[LHIP][0]) / 2, (P[RHIP][1] + P[LHIP][1]) / 2)
    if view == "34":
        # A block over both shoulders and both hips; the chest faces partly toward the viewer.
        pad = TORSO_PAD_34 * u
        sl, sr = sorted((P[RSHO], P[LSHO]))
        hl, hr = sorted((P[RHIP], P[LHIP]))
        quad = np.array([(sl[0] - pad, sl[1] - 0.01 * u), (sr[0] + pad, sr[1] - 0.01 * u),
                         (hr[0] + pad, hr[1] + 0.02 * u), (hl[0] - pad, hl[1] + 0.02 * u)], np.int32)
        cv2.polylines(img, [quad], True, (*OUTLINE, 255), int(2 * line + 0.05 * u), cv2.LINE_AA)
        cv2.polylines(img, [quad], True, (*COLOUR["torso"], 255), int(0.05 * u), cv2.LINE_AA)
        cv2.fillPoly(img, [quad], (*COLOUR["torso"], 255), cv2.LINE_AA)
        mid = ((sl[0] + sr[0]) / 2 + 0.02 * u, sl[1] + 0.005 * u)     # a V collar, toward the front of the chest
        for end in ((mid[0] - 0.035 * u, mid[1] - 0.01 * u), (mid[0] + 0.035 * u, mid[1] - 0.01 * u)):
            cv2.line(img, (int(mid[0]), int(mid[1] + 0.05 * u)), (int(end[0]), int(end[1])), (*OUTLINE, 255), line,
                     cv2.LINE_AA)
    else:
        _capsule(img, (P[NECK][0], P[NECK][1] + 0.02 * u), hip, WIDTH["torso"] * u, COLOUR["torso"], line)
    cv2.line(img, (int(hip[0] - 0.07 * u), int(hip[1] - 0.01 * u)), (int(hip[0] + 0.07 * u), int(hip[1] - 0.01 * u)),
             (*OUTLINE, 255), line, cv2.LINE_AA)                             # a belt line
    out["torso"] = img
    # Head: a circle with hair at the back, an eye and a mouth facing forward.
    img = blank()
    c = ((P[NOSE][0] + P[REAR][0]) / 2 - 0.005 * u, (P[NOSE][1] + P[REAR][1]) / 2 - 0.015 * u)
    r = 0.088 * u
    _capsule(img, P[NECK], (P[NECK][0], P[NECK][1] - 0.04 * u), 0.05 * u, SKIN, line)   # neck
    _circle(img, c, r, SKIN, line)
    pil = Image.fromarray(img, "RGBA")
    d = ImageDraw.Draw(pil)
    box = (c[0] - r, c[1] - r, c[0] + r, c[1] + r)
    d.pieslice(box, 150, 300, fill=(90, 60, 45, 255), outline=(*OUTLINE, 255), width=line)  # hair, back and top
    for e, s in ((P[REYE], 1.0), (P[LEYE], 0.8)):
        if e is None:
            continue
        d.ellipse((e[0] - 0.016 * s * u, e[1] - 0.016 * u, e[0] + 0.016 * s * u, e[1] + 0.016 * u),
                  fill=(255, 255, 255, 255), outline=(*OUTLINE, 255), width=max(1, line // 2))
        d.ellipse((e[0] - 0.002 * u, e[1] - 0.008 * u, e[0] + 0.012 * s * u, e[1] + 0.008 * u), fill=(*OUTLINE, 255))
    m = (P[NOSE][0] - 0.015 * u, P[NOSE][1] + 0.03 * u)
    d.line((m[0] - 0.02 * u, m[1], m[0] + 0.005 * u, m[1]), fill=(*OUTLINE, 255), width=line)
    if view == "34":                       # the near ear and a nose on the front of the face
        ear = P[REAR]
        d.ellipse((ear[0] - 0.014 * u, ear[1] - 0.022 * u, ear[0] + 0.014 * u, ear[1] + 0.022 * u), fill=SKIN + (255,),
                  outline=(*OUTLINE, 255), width=line)
        n = P[NOSE]
        d.line((n[0] - 0.005 * u, n[1] - 0.03 * u, n[0] + 0.012 * u, n[1] - 0.005 * u, n[0] - 0.006 * u, n[1]),
               fill=(*OUTLINE, 255), width=max(1, line // 2))
    out["head"] = np.array(pil)
    # The open hand (frames that open the near hand): a palm with four fingers and a thumb past the wrist.
    img = blank()
    a, b = np.array(P[RELB]), np.array(P[RWRI])
    dvec = (b - a) / (np.linalg.norm(b - a) + 1e-6)
    side = np.array([-dvec[1], dvec[0]])
    palm = b + dvec * 0.03 * u
    for k in (-1.5, -0.5, 0.5, 1.5):
        base = palm + side * k * 0.012 * u
        _capsule(img, base, base + dvec * 0.06 * u, 0.012 * u, SKIN, max(1, line // 2))
    _capsule(img, palm - side * 0.025 * u, palm - side * 0.05 * u + dvec * 0.02 * u, 0.014 * u, SKIN, max(1, line // 2))
    _circle(img, palm, 0.03 * u, SKIN, line)
    out["open_hand"] = img
    return out


def composite(layers: dict[str, np.ndarray]) -> Image.Image:
    h, w = next(iter(layers.values())).shape[:2]
    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for name in parts_mod.NAMES:
        if name in layers and name != "open_hand":
            out.alpha_composite(Image.fromarray(layers[name], "RGBA"))
    return out


SIZE = (700, 1100)        # the drawn image


def make(session, view: str, image_path: Path):
    """Draw the sample character for ``view`` ("side" or "34"), open it in ``session`` with its skeleton confirmed
    and its angle set, and write its rig (the shapes as body-part layers) into the session's rig folder. Returns
    the session's character."""
    from .session import LAYERS_DIR, LAYERS_INFO, PSD_FILE

    P, u = joints(SIZE, view)
    image = composite(draw_layers(P, u, SIZE, view))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)

    ch = session.set_character(str(image_path), "right")
    raw = image.getchannel("A").getbbox()
    can = ch.placement.image.getchannel("A").getbbox()
    k = (can[3] - can[1]) / (raw[3] - raw[1])

    def to_canvas(p):
        return None if p is None else [can[0] + (p[0] - raw[0]) * k, can[1] + (p[1] - raw[1]) * k]
    pts = [to_canvas(p) for p in P]
    ch = session.confirm_skeleton(pts)
    ch = session.set_angle(view)

    canvas = ch.placement.image.size
    layers = draw_layers([None if p is None else tuple(p) for p in pts], u * k, canvas, view)
    folder = session.rig_dir / ch.key
    folder.mkdir(parents=True, exist_ok=True)
    guide = composite(layers)
    guide.save(folder / "rig.png")
    (folder / "hand.png").unlink(missing_ok=True)
    with open(folder / "rig.json", "w", encoding="utf-8") as f:
        json.dump({"seed": 0, "skeleton": ch.skeleton.to_json(), "still": []}, f)
    parts_mod.save_layers(folder / LAYERS_DIR, layers)
    parts_mod.write_psd(folder / PSD_FILE, guide, layers, False)
    with open(folder / LAYERS_INFO, "w", encoding="utf-8") as f:
        json.dump({"bind": ch.skeleton.to_json(), "psd_time": (folder / PSD_FILE).stat().st_mtime_ns}, f)
    session._load_rig()
    if session.rig is None:
        raise RuntimeError("the sample rig did not load")
    return ch
