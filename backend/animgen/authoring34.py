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
- what is in front changes with the motion: the far leg stepping forward comes in front of the near one, the near
  arm swung far back goes behind the body. Each frame says so ("front": meshes drawn over the near leg or the
  body, "back": meshes drawn behind it). The far arm always stays behind the body (owner, 2026-10-31: it looked
  wrong in front of it);
- a kick swings across the picture rather than toward the viewer, so the leg keeps its length and reach, and the
  bow is drawn level across the picture, not back into it (owner, 2026-10-31); the far arm always stays behind
  the body except on a two-handed grip, where it is drawn over the head and body;
- two hands on one weapon stay together: the far hand is put on the handle next to the near one in 3D and its arm
  solved to reach it (owner, 2026-10-31: in a 3/4 view each arm swinging from its own shoulder pulls them apart),
  and the spear's near (right) hand leads, its arm reaching forward, the far hand behind it on the shaft.

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
    r_leg_out: float | None = None   # one leg's own swing out (negative: across the body); None: leg_out
    l_leg_out: float | None = None
    joined: bool = False             # both hands on one handle: the far hand goes next to the near one
    far_behind: bool = False         # joined: the far hand on the handle behind the near one (toward the body)
    near_elbow_forward: bool = False # the near elbow bent forward of the shoulder and the hand (a two-handed grip)
    far_arm_front: bool = False      # the far arm drawn over the head and the body (a two-handed grip)
    near_draw: float = 0.0           # the near hand pulled this far back toward the body, the elbow bent back
    arm_turn: float = 0.0            # the arms' swing turned this much more than the chest (negative: across the
                                     # picture rather than toward the viewer)


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

    arms = chest + p3.arm_turn
    for sho, elb, wri, sh, el, out, fout, sign in (
            (RSHO, RELB, RWRI, p.r_sh, p.r_el, p3.r_out, p3.r_fore_out, 1),
            (LSHO, LELB, LWRI, p.l_sh, p.l_el, p3.l_out, p3.l_fore_out, -1)):
        pts3[sho] = _add(neck, _rot_y((0.0, SHO_DROP, sign * SHO_HALF), chest))
        pts3[elb] = _add(pts3[sho], _scale(_rot_y(_dir(sh, out, sign), arms), UPPER_ARM))
        pts3[wri] = _add(pts3[elb], _scale(_rot_y(_dir(sh + el, out + fout, sign), arms), FOREARM))
    if p3.near_elbow_forward:
        # The elbow out in front of both the shoulder and the hand (in the picture): the hands, with the weapon,
        # move the least distance forward or back that allows it.
        pole = _rot_y(NEAR_POLE_FORWARD, arms)
        best = None
        for off in sorted((k * 0.01 for k in range(-10, 11)), key=abs):
            target = _add(pts3[RWRI], _rot_y((off, 0.0, 0.0), arms))
            elbow, wrist = _reach3(pts3[RSHO], target, pole)
            ex, sx, wx = project(elbow)[0], project(pts3[RSHO])[0], project(wrist)[0]
            if ex > sx + ELBOW_LEAD and ex > wx + ELBOW_LEAD:
                best = (elbow, wrist)
                break
        pts3[RELB], pts3[RWRI] = best or _reach3(pts3[RSHO], pts3[RWRI], pole)
    if p3.near_draw:
        # Drawn back: the shoulder stays, the elbow goes back past it and the hand comes back toward the body.
        target = _add(pts3[RWRI], _rot_y((-p3.near_draw, 0.0, 0.0), chest))
        pts3[RELB], pts3[RWRI] = _reach3(pts3[RSHO], target, _rot_y(NEAR_POLE_BACK, chest))
    if p3.joined:
        # The handle runs from the near hand to the far one as in the side pose, in the body's forward-down plane.
        sp = _side_points(p)
        handle = (sp[LWRI][0] - sp[RWRI][0], sp[LWRI][1] - sp[RWRI][1], 0.0)
        if p3.far_behind:
            handle = _scale(handle, -1.0)
        target = _add(pts3[RWRI], _rot_y(handle, arms))
        pts3[LELB], pts3[LWRI] = _reach3(pts3[LSHO], target, _rot_y(FAR_POLE, arms))
    for hip, kne, ank, hp, kn, out, sign in (
            (RHIP, RKNE, RANK, p.r_hip, p.r_knee, p3.r_leg_out, 1),
            (LHIP, LKNE, LANK, p.l_hip, p.l_knee, p3.l_leg_out, -1)):
        out = p3.leg_out if out is None else out
        pts3[hip] = _add(pelvis, _rot_y((0.0, 0.0, sign * HIP_HALF), hips))
        pts3[kne] = _add(pts3[hip], _scale(_rot_y(_dir(hp, out, sign), hips), THIGH))
        pts3[ank] = _add(pts3[kne], _scale(_rot_y(_dir(hp - kn, out, sign), hips), SHIN))

    # On the ground in 3D: the lowest ankle (or, lying down, the lowest joint) at ankle height, then lifted.
    low = max(q[1] for q in pts3 if q is not None) if p.floor else max(pts3[RANK][1], pts3[LANK][1])
    dy = -ANKLE_H - low - p.lift
    pts3 = [None if q is None else (q[0], q[1] + dy, q[2]) for q in pts3]
    proj = [None if q is None else project(q) for q in pts3]
    pts = [None if q is None else (round(q[0], 4), round(q[1], 4)) for q in proj]
    depth = [None if q is None else q[2] for q in proj]

    front, back = [], []
    body = depth[NECK]
    if p3.far_arm_front:
        front.append("arm_far")
    if (depth[LKNE] + depth[LANK]) / 2 > (depth[RKNE] + depth[RANK]) / 2:
        front.append("leg_far")
    if (depth[RELB] + depth[RWRI]) / 2 < body - DEPTH_MARGIN and not p3.joined:     # a grip stays in front
        back.append("arm")
    return pts, p.lift <= 0.0, front, back


FAR_POLE = (-0.2, 1.0, -0.6)   # the far elbow of a two-handed grip bends down, out and a little back
NEAR_POLE_FORWARD = (1.0, 0.25, 0.3)   # the near elbow of a two-handed grip points forward (owner, 2026-10-31)
NEAR_POLE_BACK = (-1.0, 0.4, 0.0)      # a drawn-back arm: the elbow bends back and a little down
ELBOW_LEAD = 0.01     # template units: how far the elbow is at least in front of the shoulder and the hand


def _reach3(shoulder: tuple, target: tuple, pole: tuple) -> tuple[tuple, tuple]:
    """Elbow and wrist of an arm from ``shoulder`` reaching ``target`` (as near as it can), the elbow bent toward
    ``pole``."""
    v = tuple(t - s for t, s in zip(target, shoulder))
    d = math.sqrt(sum(c * c for c in v)) or 1e-6
    u = tuple(c / d for c in v)
    d = min(d, UPPER_ARM + FOREARM - 1e-4)
    a = (UPPER_ARM ** 2 - FOREARM ** 2 + d ** 2) / (2 * d)
    h = math.sqrt(max(UPPER_ARM ** 2 - a ** 2, 0.0))
    along = sum(pc * uc for pc, uc in zip(pole, u))
    n = tuple(pc - along * uc for pc, uc in zip(pole, u))
    nl = math.sqrt(sum(c * c for c in n)) or 1e-6
    n = tuple(c / nl for c in n)
    elbow = tuple(s + a * uc + h * nc for s, uc, nc in zip(shoulder, u, n))
    wrist = tuple(s + d * uc for s, uc in zip(shoulder, u))
    return elbow, wrist


def weapon_angle(p3: Pose3, wd: dict) -> float:
    """A side pose's weapon angle (image degrees in the body's forward-down plane) seen in the 3/4 view."""
    chest, _ = turns(p3)
    a = math.radians(wd["angle"])
    v = _rot_y((math.cos(a), math.sin(a), 0.0), chest)
    x, y, _ = project(v)
    return round(math.degrees(math.atan2(y, x)), 1)


def lift(poses: list[Pose], **extra) -> list[Pose3]:
    return [Pose3(base=p, **extra) for p in poses]


SMASH_ACROSS = 63.0   # the two-handed swing turned so it travels along the picture (with the chest's -18: profile)
SMASH_FRAMES = (4, 5, 6)   # the swing, smash and impact: arms reaching out, the elbow between shoulder and hand
SPEAR_TURN = 10.0     # the spear's chest turned a little toward the lead (near) hand
SPEAR_ACROSS = -35.0  # the lead arm swings across the body, so it reaches along the picture, not at the viewer
KICK_ACROSS = -40.0   # degrees the kicking leg swings across the body: its line then runs along the picture
BOW_TURN = 32.0       # the archer's chest turned toward profile: the arrow (and the draw) level across the picture

# Per template: anything set by hand on top of the automatic turns (frame index -> Pose3 fields; "base_<field>"
# changes the side pose itself for this view).
TWEAKS: dict[str, dict[int, dict]] = {
    # The kick swings across the picture, full length, and a little higher than the side one (owner, 2026-10-31).
    "a1b": {2: dict(r_leg_out=KICK_ACROSS / 2, hips=4.0),
            3: dict(r_leg_out=KICK_ACROSS, hips=8.0, base_r_hip=100.0),
            4: dict(r_leg_out=KICK_ACROSS, hips=8.0, base_r_hip=102.0),
            5: dict(r_leg_out=KICK_ACROSS / 2, hips=4.0)},
    # Two hands on one grip: the near shoulder back, the chest turned toward profile so both arms reach the hilt.
    # The far arm over the head and body; the swing runs across the picture, not at the viewer; the near elbow in
    # front of the shoulder and the hands while the hands are held in, the arms reaching out on the smash (owner,
    # 2026-10-31).
    "a3": {i: dict(chest=-18.0, arm_turn=SMASH_ACROSS, joined=True, near_elbow_forward=i not in SMASH_FRAMES,
                   far_arm_front=True) for i in range(8)},
    "a4": {i: dict(joined=True, far_behind=True, chest=SPEAR_TURN, r_out=SPEAR_ACROSS) for i in range(8)},
    # The bow: the body side-on to the shot, the draw hand pulling level across the picture; the hands stay fists
    # (owner, 2026-10-31).
    "a5": {i: dict(chest=BOW_TURN, hips=BOW_TURN / 3, head=BOW_TURN * 0.6, base_open=()) for i in range(8)},
}
SPEAR_REACH = 0.12    # the spear's near hand this much further forward than in the side view, the arm reaching out
SPEAR_LEAST = 0.13    # and never closer in than this
SPEAR_DRAW = 0.08     # in the wind-up frames the right forearm and hand come back this far, the elbow behind
SPEAR_DRAWN = (0, 1, 2, 6, 7)   # (owner, 2026-10-31: only the right arm back, the side pose otherwise as it was)


def poses34(tid: str) -> list[Pose3]:
    out = lift(side.TEMPLATES[tid]["fn"]())
    for i, kw in TWEAKS.get(tid, {}).items():
        for k, v in kw.items():
            if k.startswith("base_"):
                setattr(out[i].base, k[5:], v)
            else:
                setattr(out[i], k, v)
    if tid == "a4":
        for p3 in out:
            sp = _side_points(p3.base)
            x = max(sp[RWRI][0] - sp[RSHO][0] + SPEAR_REACH, SPEAR_LEAST)
            y = sp[RWRI][1] - sp[RSHO][1]
            p3.base.r_sh, p3.base.r_el = side.reach(x, y)
        for i in SPEAR_DRAWN:
            out[i].near_draw = SPEAR_DRAW
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
        if p3.base.open:
            fr["open"] = list(p3.base.open)
        else:
            fr.pop("open", None)
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
