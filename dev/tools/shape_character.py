"""A test character made of simple shapes, for checking and tuning the motion templates (owner, 2026-10-31).

Every body part is its own flat-coloured shape with a dark outline, so the parts are known exactly: the script draws
the character image, confirms its skeleton and writes its rig (the body-part layers and their Photoshop file) straight
into the app's data folder. No AI drawing is needed: opening the image in the app finds the rig at once, and every
animation is made in seconds. Then it renders all the animations into sheets and GIFs.

usage: python shape_character.py [--data C:/AnimGen] [--out D:/Game/ainm/pose_test]
(run with the app closed: it writes into the app's data folder)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "backend"))
from animgen import parts as parts_mod  # noqa: E402
from animgen.authoring import Pose, solve  # noqa: E402
from animgen.engines.demo import DemoEngine  # noqa: E402
from animgen.session import LAYERS_DIR, LAYERS_INFO, PSD_FILE, Session  # noqa: E402
from animgen.skeleton import (LANK, LELB, LHIP, LKNE, LSHO, LWRI, NECK, NOSE, RANK, REAR, RELB, REYE, RHIP,  # noqa
                              RKNE, RSHO, RWRI)
from animgen.tasks import TaskRunner  # noqa: E402
from animgen.templates import ANIMATIONS  # noqa: E402

IMAGE = HERE.parents[2] / "dev" / "test_art" / "test_shapes.png"
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


def joints(size: tuple[int, int]) -> tuple[list, float]:
    """The bind pose in image pixels, feet on a line near the bottom, and pixels per template unit."""
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


def draw_layers(P: list, u: float, size: tuple[int, int]) -> dict[str, np.ndarray]:
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
        if kind == "leg_lower":            # the foot points forward
            dark = (70, 70, 80) if not name.startswith("far") else (50, 50, 58)
            ankle = P[b]
            _capsule(img, (ankle[0] - 0.01 * u, ankle[1] + 0.012 * u), (ankle[0] + 0.07 * u, ankle[1] + 0.02 * u),
                     0.045 * u, dark, line)
        out[name] = img
    # Torso: a capsule from the neck to the middle of the hips.
    img = blank()
    hip = ((P[RHIP][0] + P[LHIP][0]) / 2, (P[RHIP][1] + P[LHIP][1]) / 2)
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
    e = P[REYE]
    d.ellipse((e[0] - 0.016 * u, e[1] - 0.016 * u, e[0] + 0.016 * u, e[1] + 0.016 * u), fill=(255, 255, 255, 255),
              outline=(*OUTLINE, 255), width=max(1, line // 2))
    d.ellipse((e[0] - 0.002 * u, e[1] - 0.008 * u, e[0] + 0.012 * u, e[1] + 0.008 * u), fill=(*OUTLINE, 255))
    m = (P[NOSE][0] - 0.015 * u, P[NOSE][1] + 0.03 * u)
    d.line((m[0] - 0.02 * u, m[1], m[0] + 0.005 * u, m[1]), fill=(*OUTLINE, 255), width=line)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/AnimGen")
    ap.add_argument("--out", default="D:/Game/ainm/pose_test")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)

    # 1. The character image (what the owner opens in the app).
    size = (700, 1100)
    P, u = joints(size)
    image = composite(draw_layers(P, u, size))
    IMAGE.parent.mkdir(parents=True, exist_ok=True)
    image.save(IMAGE)

    # 2. Its skeleton on the app's canvas, confirmed as the owner would.
    s = Session(DemoEngine(), data, require_rig=True)
    ch = s.set_character(str(IMAGE), "right")
    raw = image.getchannel("A").getbbox()
    can = ch.placement.image.getchannel("A").getbbox()
    k = (can[3] - can[1]) / (raw[3] - raw[1])

    def to_canvas(p):
        return None if p is None else [can[0] + (p[0] - raw[0]) * k, can[1] + (p[1] - raw[1]) * k]
    pts = [to_canvas(p) for p in P]
    ch = s.confirm_skeleton(pts)

    # 3. The rig: the same shapes drawn on the canvas are the body-part layers (bind pose = the image's pose).
    canvas = ch.placement.image.size
    layers = draw_layers([None if p is None else tuple(p) for p in pts], u * k, canvas)
    folder = s.rig_dir / ch.key
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
    s._load_rig()
    assert s.rig is not None, "the rig did not load"
    print("character", IMAGE, "rig", folder)

    # 4. Every animation, as a sheet (one row each) and a GIF.
    out.mkdir(parents=True, exist_ok=True)
    runner = TaskRunner()
    rows = []
    for tid, _anim, _variant, name_vi, ready in ANIMATIONS:
        if not ready:
            continue
        t = runner.wait(runner.start("t", lambda task: s.generate(task, tid, None, seed=1)).id, 600)
        if t.state != "completed":
            print(tid, t.state, t.error)
            continue
        frames = [Image.open(f["preview"]).convert("RGBA") for f in s.state()["frames"] if f["preview"]]
        rows.append((tid, name_vi, frames))
    box = None
    for _, _, frames in rows:
        for im in frames:
            b = im.getchannel("A").getbbox()
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    box = (max(0, box[0] - 12), max(0, box[1] - 12), box[2] + 12, box[3] + 12)
    cw, chh = box[2] - box[0], box[3] - box[1]
    label = 34
    for tid, name_vi, frames in rows:
        tiles = []
        for im in frames:
            bg = Image.new("RGBA", im.size, (215, 215, 215, 255))
            bg.alpha_composite(im)
            tiles.append(bg.convert("RGB").crop(box))
        strip = Image.new("RGB", (cw * 8, chh + label), (255, 255, 255))
        d = ImageDraw.Draw(strip)
        d.text((8, 8), f"{tid} - {name_vi}", fill=(0, 0, 0))
        for i, t in enumerate(tiles):
            strip.paste(t, (i * cw, label))
            d.text((i * cw + 6, label + 4), str(i + 1), fill=(60, 60, 60))
        strip.save(out / f"{tid}.png")
        tiles[0].save(out / f"{tid}.gif", save_all=True, append_images=tiles[1:], duration=110, loop=0)
    sheet = Image.new("RGB", (cw * 8, (chh + label) * len(rows)), (255, 255, 255))
    for r, (tid, _, _) in enumerate(rows):
        sheet.paste(Image.open(out / f"{tid}.png"), (0, r * (chh + label)))
    sheet.save(out / "all.png")
    print("saved", out, len(rows), "animations")


if __name__ == "__main__":
    main()
