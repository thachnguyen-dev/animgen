"""The three-quarter view pose set (owner, 2026-10-31): every template again, for characters drawn turned about 45
degrees toward the viewer.

The side-view poses (authoring.py) are lifted into 3D and projected, rather than drawn again by hand: a joint angle
in a side pose is a swing in the body's own forward-down plane, so the same motion seen from 45 degrees is that
swing turned and foreshortened. What a side view cannot show is added in 3D, from common 3/4 animation practice:

- the body has width: the shoulders and hips are apart, so the far arm and leg show beside the body;
- the hips turn with the legs and the shoulders against them in walks and runs (pelvis about 5-10 degrees, chest a
  little less, the other way: hips and shoulders counter-rotate, nearly parallel at the passing pose);
- the chest turns with the striking arm: a slash or punch winds the torso back and swings it through, a two-handed
  or spear stance turns the near shoulder back (the body follows the arm that leads);
- the arms hang a little away from the body, so they do not vanish into the torso;
- what is in front changes with the motion: the far arm reaching forward comes in front of the body, the far leg
  stepping forward comes in front of the near one, the near arm swung far back goes behind the body. Each frame
  says so ("front": meshes drawn over the body, "back": meshes drawn behind it).

The view: the character faces right and toward the viewer. Its near side (the r_ limbs) is the one toward the
viewer, which in a 3/4 front view is on the left of the body in the picture; the far arm shows at the front edge.
The camera looks down a little (PITCH), so nearer things sit slightly lower.

``python -m animgen.authoring34`` writes ``template_data/view34/*.json`` and contact sheets into
``docs/templates/view34/``. Every pose is Proposed until the owner approves it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import authoring as side
from .authoring import (ANKLE_H, FOREARM, HEAD_EAR, HEAD_EYE, HEAD_NOSE, SHIN, SHO_DROP, SPINE, THIGH, UPPER_ARM,
                        Pose)
from .skeleton import (LANK, LELB, LEYE, LHIP, LKNE, LSHO, LWRI, N, NECK, NOSE, RANK, REAR, RELB, REYE, RHIP, RKNE,
                       RSHO, RWRI, Skeleton)

VIEW = "34"
YAW = 45.0            # degrees the character is turned toward the viewer from a side view
PITCH = 12.0          # degrees the camera looks down
SHO_HALF = 0.085      # half the shoulder width, template units (figure height 1.0)
HIP_HALF = 0.055      # half the hip width
EYE_HALF = 0.03       # half the distance between the eyes
EAR_HALF = 0.07
ARM_OUT = 8.0         # degrees the arms hang away from the body
LEG_OUT = 3.0         # degrees the legs stand apart
# How much the torso turns with the arms and legs.
CHEST_PER_REACH = 80.0     # degrees per template unit the near wrist leads the far one (front-back)
CHEST_MAX = 28.0
HIPS_PER_STRIDE = 22.0     # degrees per template unit the near ankle leads the far one
HIPS_MAX = 12.0
COUNTER = 0.6              # in a cycle, the chest turns against the hips by this share of the hip turn
DEPTH_MARGIN = 0.02        # template units: a limb this far nearer than the body is drawn in front of it


@dataclass
class Pose3:
    """A side pose (authoring.Pose) with what a 3/4 view adds. Turns are degrees about the vertical axis, positive
    bringing the near side forward."""
    base: Pose
    chest: float | None = None       # the shoulders' turn; None: from the arms (CHEST_PER_REACH)
    hips: float | None = None        # the pelvis' turn; None: from the legs (HIPS_PER_STRIDE)
    head: float | None = None        # the head's turn; None: halfway back from the chest's to straight ahead
    r_out: float = ARM_OUT           # arm swing out to its own side (negative: across the body)
    l_out: float = ARM_OUT
    r_fore_out: float = 0.0          # the forearm's extra swing out (negative: across the body)
    l_fore_out: float = 0.0
    leg_out: float = LEG_OUT


def _rot_y(v: tuple, deg: float) -> tuple:
    """Turn a body-frame vector (forward, down, side) about the vertical; positive brings the near side forward."""
    a = math.radians(deg)
    f, d, s = v
    return (f * math.cos(a) + s * math.sin(a), d, -f * math.sin(a) + s * math.cos(a))


def _add(a: tuple, b: tuple) -> tuple:
    return tuple(x + y for x, y in zip(a, b))


def _dir(swing: float, out: float, side_sign: int) -> tuple:
    """A limb direction: ``swing`` from hanging down toward forward (as in a side pose), ``out`` away from the body
    to the limb's own side."""
    a, b = math.radians(swing), math.radians(out)
    return (math.sin(a) * math.cos(b), math.cos(a) * math.cos(b), side_sign * math.sin(b))


def _scale(v: tuple, k: float) -> tuple:
    return tuple(x * k for x in v)


def project(p: tuple) -> tuple[float, float, float]:
    """Image x, image y and depth (toward the viewer) of a body-space point (forward, down, near side)."""
    t, c = math.radians(YAW), math.radians(PITCH)
    f, d, s = p
    x = f * math.cos(t) - s * math.sin(t)
    depth = f * math.sin(t) + s * math.cos(t)
    y = d * math.cos(c) + depth * math.sin(c)
    return x, y, depth


def _side_points(p: Pose) -> list:
    pts, _ = side.solve(p)
    return pts


def turns(p3: Pose3) -> tuple[float, float]:
    """The chest's and the hips' turn, as set or worked out from the side pose."""
    pts = _side_points(p3.base)
    hips = p3.hips
    if hips is None:
        stride = pts[RANK][0] - pts[LANK][0]
        hips = max(-HIPS_MAX, min(HIPS_MAX, stride * HIPS_PER_STRIDE))
    chest = p3.chest
    if chest is None:
        lead = (pts[RWRI][0] - pts[RSHO][0]) - (pts[LWRI][0] - pts[LSHO][0])
        chest = max(-CHEST_MAX, min(CHEST_MAX, lead * CHEST_PER_REACH))
        if abs(lead) < 0.12:                         # arms swinging in a cycle: shoulders against the hips
            chest -= hips * COUNTER
    return chest, hips


def solve3(p3: Pose3) -> tuple[list, bool, list, list]:
    """Image points (18, template units, the lowest ankle on the ground) of a 3/4 pose, whether it is grounded, and
    which meshes are drawn in front of / behind the body in this frame."""
    p = p3.base
    chest, hips = turns(p3)
    pelvis = (p.dx, 0.0, 0.0)
    s = math.radians(p.spine)
    neck = _add(pelvis, _rot_y((math.sin(s) * SPINE, -math.cos(s) * SPINE, 0.0), (chest + hips) / 2))
    pts3: list = [None] * N
    pts3[NECK] = neck
    head_turn = p3.head if p3.head is not None else chest * 0.5
    tilt = p.spine * 0.5 + p.head

    def head_point(v2: tuple, lateral: float) -> tuple:
        x, y = side._rot(v2, tilt)
        return _add(neck, _rot_y((x, y, lateral), head_turn))
    pts3[NOSE] = head_point(HEAD_NOSE, 0.0)
    pts3[REYE] = head_point(HEAD_EYE, EYE_HALF)
    pts3[LEYE] = head_point(HEAD_EYE, -EYE_HALF)
    pts3[REAR] = head_point(HEAD_EAR, EAR_HALF)

    for sho, elb, wri, sh, el, out, fout, sign in (
            (RSHO, RELB, RWRI, p.r_sh, p.r_el, p3.r_out, p3.r_fore_out, 1),
            (LSHO, LELB, LWRI, p.l_sh, p.l_el, p3.l_out, p3.l_fore_out, -1)):
        pts3[sho] = _add(neck, _rot_y((0.0, SHO_DROP, sign * SHO_HALF), chest))
        pts3[elb] = _add(pts3[sho], _scale(_rot_y(_dir(sh, out, sign), chest), UPPER_ARM))
        pts3[wri] = _add(pts3[elb], _scale(_rot_y(_dir(sh + el, out + fout, sign), chest), FOREARM))
    for hip, kne, ank, hp, kn, sign in ((RHIP, RKNE, RANK, p.r_hip, p.r_knee, 1),
                                        (LHIP, LKNE, LANK, p.l_hip, p.l_knee, -1)):
        pts3[hip] = _add(pelvis, _rot_y((0.0, 0.0, sign * HIP_HALF), hips))
        pts3[kne] = _add(pts3[hip], _scale(_rot_y(_dir(hp, p3.leg_out, sign), hips), THIGH))
        pts3[ank] = _add(pts3[kne], _scale(_rot_y(_dir(hp - kn, p3.leg_out, sign), hips), SHIN))

    # On the ground in 3D: the lowest ankle (or, lying down, the lowest joint) at ankle height, then lifted.
    low = max(q[1] for q in pts3 if q is not None) if p.floor else max(pts3[RANK][1], pts3[LANK][1])
    dy = -ANKLE_H - low - p.lift
    pts3 = [None if q is None else (q[0], q[1] + dy, q[2]) for q in pts3]
    proj = [None if q is None else project(q) for q in pts3]
    pts = [None if q is None else (round(q[0], 4), round(q[1], 4)) for q in proj]
    depth = [None if q is None else q[2] for q in proj]

    front, back = [], []
    body = depth[NECK]
    if (depth[LELB] + depth[LWRI]) / 2 > body + DEPTH_MARGIN:
        front.append("arm_far")
    if (depth[LKNE] + depth[LANK]) / 2 > (depth[RKNE] + depth[RANK]) / 2:
        front.append("leg_far")
    if (depth[RELB] + depth[RWRI]) / 2 < body - DEPTH_MARGIN:
        back.append("arm")
    return pts, p.lift <= 0.0, front, back


def weapon_angle(p3: Pose3, wd: dict) -> float:
    """A side pose's weapon angle (image degrees in the body's forward-down plane) seen in the 3/4 view."""
    chest, _ = turns(p3)
    a = math.radians(wd["angle"])
    v = _rot_y((math.cos(a), math.sin(a), 0.0), chest)
    x, y, _ = project(v)
    return round(math.degrees(math.atan2(y, x)), 1)


def lift(poses: list[Pose], **extra) -> list[Pose3]:
    return [Pose3(base=p, **extra) for p in poses]


# Per template: anything set by hand on top of the automatic turns (frame index -> Pose3 fields).
TWEAKS: dict[str, dict[int, dict]] = {
    # The kick turns the hips into the kick and the chest a little against it.
    "a1b": {2: dict(hips=-10.0, chest=8.0), 3: dict(hips=-14.0, chest=10.0), 4: dict(hips=-14.0, chest=10.0),
            5: dict(hips=-8.0, chest=6.0)},
    # Two hands on one grip: the near shoulder back, the chest turned toward profile so both arms reach the hilt.
    "a3": {i: dict(chest=-18.0, l_out=-6.0, r_out=4.0) for i in range(8)},
    # The bow: the body side-on to the shot, as an archer stands.
    "a5": {i: dict(chest=-30.0, hips=-10.0) for i in range(8)},
}


def poses34(tid: str) -> list[Pose3]:
    out = lift(side.TEMPLATES[tid]["fn"]())
    for i, kw in TWEAKS.get(tid, {}).items():
        for k, v in kw.items():
            setattr(out[i], k, v)
    return out


def rest_points() -> list:
    return solve3(Pose3(base=Pose()))[0]


def build(tid: str) -> dict:
    spec = side.TEMPLATES[tid]
    data = side.build(tid)                         # names, grip, kinds, prompt; points replaced below
    data["view"] = VIEW
    data["rest"] = [None if q is None else [q[0], q[1]] for q in rest_points()]
    for fr, p3 in zip(data["frames"], poses34(tid)):
        pts, grounded, front, back = solve3(p3)
        fr["points"] = [None if q is None else [q[0], q[1]] for q in pts]
        fr["contact"] = "ground" if grounded else "air"
        if front:
            fr["front"] = front
        if back:
            fr["back"] = back
        if spec["grip"] != "never":
            for key in ("weapon", "weapon2"):
                if fr.get(key):
                    fr[key] = {**fr[key], "angle": weapon_angle(p3, fr[key])}
    return data


def write_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tid in side.TEMPLATES:
        path = out_dir / f"{tid}.json"
        with open(path, "w", encoding="utf-8", newline="\r\n") as f:
            json.dump(build(tid), f, ensure_ascii=False, indent=1)
            f.write("\n")
        paths.append(path)
    return paths


def contact_sheet(tid: str, cell: int = 256):
    """The 8 poses side by side, as authoring.contact_sheet draws them."""
    import cv2
    import numpy as np  # noqa: F401
    from PIL import Image, ImageDraw

    from .skeleton import draw_openpose
    data = build(tid)
    sheet = Image.new("RGB", (cell * 8, cell + 24), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    scale = cell * 0.62
    ox, oy = cell * 0.5, cell * 0.88
    for i, fr in enumerate(data["frames"]):
        sk = Skeleton([None if q is None else (q[0] * scale + ox, q[1] * scale + oy) for q in fr["points"]])
        img = draw_openpose(sk, (cell, cell), stick_width=3)
        cv2.line(img, (0, int(oy)), (cell, int(oy)), (90, 90, 90), 1)
        for wd in (fr.get("weapon"), fr.get("weapon2")):
            if not wd:
                continue
            el = sk.points[RELB if wd["hand"] == "r" else LELB]
            wr = sk.points[RWRI if wd["hand"] == "r" else LWRI]
            hx, hy = wr[0] + (wr[0] - el[0]) * 0.3, wr[1] + (wr[1] - el[1]) * 0.3
            a = math.radians(wd["angle"])
            col = (255, 255, 255) if wd["layer"] == "front" else (150, 150, 150)
            length = 0.3 * scale
            cv2.line(img, (int(hx), int(hy)), (int(hx + math.cos(a) * length), int(hy + math.sin(a) * length)),
                     col, 2)
        sheet.paste(Image.fromarray(img), (i * cell, 24))
        extra = " ".join(fr.get("front", []) + [f"-{m}" for m in fr.get("back", [])])
        d.text((i * cell + 6, 5), f"{tid} 3/4 {i + 1}/8 {extra}", fill=(230, 230, 230))
    return sheet


if __name__ == "__main__":
    import sys
    here = Path(__file__).parent
    for p in write_all(here / "template_data" / "view34"):
        print("wrote", p)
    docs = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent.parent / "docs" / "templates" / "view34"
    docs.mkdir(parents=True, exist_ok=True)
    for tid in side.TEMPLATES:
        contact_sheet(tid).save(docs / f"{tid}.png")
        print("drew", docs / f"{tid}.png")
