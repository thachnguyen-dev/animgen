"""Authoring of the motion templates from joint angles (forward kinematics).

The 8 poses of every template are written here as joint angles, which are easier to read and tune than raw
points. ``python -m animgen.authoring`` rewrites ``template_data/*.json`` and draws a contact sheet per template
into ``docs/templates/`` for the owner to approve. Every pose here is Proposed until the owner approves it (§5.2).

Angle conventions (template faces right, image y points down):
- limbs: degrees from hanging straight down; positive swings the limb forward (toward the facing direction).
  So 90 = pointing forward, 180 = straight up, 200 = up and slightly back.
- knee: bend in degrees (the shin swings backward). Elbow: bend in degrees (the forearm swings forward/up).
- spine: lean from vertical, positive = forward. head: tilt, positive = chin down.
- weapon angle: image-space degrees, 0 = pointing right, -90 = up, 90 = down.
The near side of a right-facing figure is its RIGHT side (r_ limbs).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .skeleton import (LANK, LEAR, LELB, LEYE, LHIP, LKNE, LSHO, LWRI, N, NECK, NOSE, RANK, REAR, RELB, REYE,
                       RHIP, RKNE, RSHO, RWRI, Skeleton)

# Rest proportions in template units (figure height from ground to top of head = 1.0).
ANKLE_H = 0.04
SHIN = 0.23
THIGH = 0.24
SPINE = 0.29          # pelvis to neck
UPPER_ARM = 0.16
FOREARM = 0.15
HIP_DEPTH = 0.008     # tiny x offset between near and far hip / shoulder (side view)
SHO_DROP = 0.02
HEAD_NOSE = (0.075, -0.085)  # from the neck, head upright
HEAD_EYE = (0.055, -0.125)
HEAD_EAR = (-0.005, -0.115)
# Top of the head is about 0.20 above the neck, which gives a 1.0 tall figure at rest.


@dataclass
class Pose:
    spine: float = 2.0
    head: float = 0.0
    r_sh: float = 5.0
    r_el: float = 10.0
    l_sh: float = -5.0
    l_el: float = 10.0
    r_hip: float = 2.0
    r_knee: float = 3.0
    l_hip: float = -2.0
    l_knee: float = 3.0
    dx: float = 0.0       # pelvis shift forward (template units)
    lift: float = 0.0     # height of the lowest ankle above the ground (frames in the air)
    weapon: dict | None = field(default=None)
    weapon2: dict | None = field(default=None)    # the other hand's weapon (dual wielding)
    open: tuple = ()      # hands drawn open (palm) in this frame: "r" near, "l" far; others are fists
    floor: bool = False   # lying or kneeling: the lowest joint, not an ankle, rests on the ground


def _limb(origin, angle_deg, length):
    a = math.radians(angle_deg)
    return (origin[0] + math.sin(a) * length, origin[1] + math.cos(a) * length)


def _rot(v, deg):
    a = math.radians(deg)
    return (v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a))


def solve(p: Pose) -> tuple[list, bool]:
    pelvis = (p.dx, 0.0)
    s = math.radians(p.spine)
    neck = (pelvis[0] + math.sin(s) * SPINE, pelvis[1] - math.cos(s) * SPINE)
    pts: list = [None] * N
    pts[NECK] = neck
    head_rot = p.spine * 0.5 + p.head
    pts[NOSE] = tuple(a + b for a, b in zip(neck, _rot(HEAD_NOSE, head_rot)))
    pts[REYE] = tuple(a + b for a, b in zip(neck, _rot(HEAD_EYE, head_rot)))
    pts[REAR] = tuple(a + b for a, b in zip(neck, _rot(HEAD_EAR, head_rot)))
    pts[LEYE] = None   # far eye and ear are hidden in a side view
    pts[LEAR] = None
    pts[RSHO] = (neck[0] + HIP_DEPTH, neck[1] + SHO_DROP)
    pts[LSHO] = (neck[0] - HIP_DEPTH, neck[1] + SHO_DROP)
    pts[RELB] = _limb(pts[RSHO], p.r_sh, UPPER_ARM)
    pts[RWRI] = _limb(pts[RELB], p.r_sh + p.r_el, FOREARM)
    pts[LELB] = _limb(pts[LSHO], p.l_sh, UPPER_ARM)
    pts[LWRI] = _limb(pts[LELB], p.l_sh + p.l_el, FOREARM)
    pts[RHIP] = (pelvis[0] + HIP_DEPTH, pelvis[1])
    pts[LHIP] = (pelvis[0] - HIP_DEPTH, pelvis[1])
    pts[RKNE] = _limb(pts[RHIP], p.r_hip, THIGH)
    pts[RANK] = _limb(pts[RKNE], p.r_hip - p.r_knee, SHIN)
    pts[LKNE] = _limb(pts[LHIP], p.l_hip, THIGH)
    pts[LANK] = _limb(pts[LKNE], p.l_hip - p.l_knee, SHIN)
    # Stand the lowest ankle on the ground (y = -ANKLE_H), then lift frames in the air. A body on the floor rests on
    # its lowest joint, which is as far above the ground as an ankle is (the flesh around it).
    low = max(q[1] for q in pts if q is not None) if p.floor else max(pts[RANK][1], pts[LANK][1])
    dy = -ANKLE_H - low - p.lift
    pts = [None if q is None else (round(q[0], 4), round(q[1] + dy, 4)) for q in pts]
    return pts, p.lift <= 0.0


def forearm_angle(p: Pose, side: str) -> float:
    """Image-space angle of a forearm (0 = right, 90 = down)."""
    a = (p.r_sh + p.r_el) if side == "r" else (p.l_sh + p.l_el)
    return 90.0 - a


def held(p: Pose, offset: float = -70.0, layer: str = "front") -> Pose:
    """A one-hand carry for optional weapons: the weapon follows the near forearm, turned by ``offset``."""
    p.weapon = {"hand": "r", "angle": round(forearm_angle(p, "r") + offset, 1), "layer": layer, "stage": 1}
    return p


def w(hand: str, angle: float, layer: str = "front", stage: int = 1) -> dict:
    return {"hand": hand, "angle": angle, "layer": layer, "stage": stage}


def reach(x: float, y: float, elbow: str = "down") -> tuple[float, float]:
    """Shoulder and elbow angles that put the wrist at (x, y) from the shoulder (x forward, y down, template
    units). ``elbow``: "down" puts the elbow below the shoulder-wrist line, "back" behind it (a drawn bow)."""
    d = min(math.hypot(x, y), UPPER_ARM + FOREARM - 1e-6)
    to = math.atan2(x, y)                                       # direction to the wrist, as limb angles are measured
    bend = math.acos((UPPER_ARM ** 2 + d ** 2 - FOREARM ** 2) / (2 * UPPER_ARM * d))
    options = []
    for sign in (1, -1):
        up = to + sign * bend
        ex, ey = math.sin(up) * UPPER_ARM, math.cos(up) * UPPER_ARM
        fore = math.atan2(x - ex, y - ey)
        options.append((ex, ey, math.degrees(up), math.degrees(fore)))
    ex, ey, up, fore = min(options, key=lambda o: o[0]) if elbow == "back" else max(options, key=lambda o: o[1])
    el = (fore - up + 180.0) % 360.0 - 180.0
    return round(up, 1), round(el, 1)


def _near(x: float, y: float, elbow: str = "down") -> dict:
    sh, el = reach(x, y, elbow)
    return {"r_sh": sh, "r_el": el}


def _far(x: float, y: float, elbow: str = "down") -> dict:
    """The far wrist at (x, y) from the NEAR shoulder (the far shoulder sits a little behind it)."""
    sh, el = reach(x + 2 * HIP_DEPTH, y, elbow)
    return {"l_sh": sh, "l_el": el}


def _both(x: float, y: float, angle: float, gap: float, elbow: str = "down") -> dict:
    """Both hands on one handle: the near hand at (x, y) from the near shoulder, the far hand ``gap`` further along
    the handle toward the tip (``angle``, image degrees; a negative gap is toward the pommel)."""
    a = math.radians(angle)
    return {**_near(x, y, elbow), **_far(x + math.cos(a) * gap, y + math.sin(a) * gap)}


def _both_back(x: float, y: float, angle: float, gap: float) -> dict:
    """``_both`` with the near elbow behind the body, as the rear hand on a spear shaft."""
    return _both(x, y, angle, gap, "back")


# ------------------------------------------------------------------------------------------------------------
# Templates. Cyclic ones (idle, walk, run) are written so that frame 8 leads back into frame 1.
# ------------------------------------------------------------------------------------------------------------

def _cyc(i: int) -> float:
    return 2.0 * math.pi * i / 8.0


def idle() -> list[Pose]:
    out = []
    for i in range(8):
        t = _cyc(i)
        b = (1 - math.cos(t)) / 2  # 0..1..0 breathing
        out.append(held(Pose(spine=2 + 1.5 * b, head=-1 * b, r_sh=6 + 3 * b, r_el=12 + 4 * b, l_sh=-6 - 2 * b,
                             l_el=12 + 3 * b, r_hip=4, r_knee=4 + 3 * b, l_hip=-4, l_knee=4 + 3 * b), offset=-35))
    return out


def walk() -> list[Pose]:
    out = []
    for i in range(8):
        t = _cyc(i)
        sw = math.sin(t)
        r_knee = 5 + 35 * max(0.0, math.cos(t)) ** 1.5   # near leg bends while it swings forward
        l_knee = 5 + 35 * max(0.0, -math.cos(t)) ** 1.5
        out.append(held(Pose(spine=4, head=2, r_hip=24 * sw, r_knee=r_knee, l_hip=-24 * sw, l_knee=l_knee,
                             r_sh=-20 * sw, r_el=15 + 10 * max(0.0, -sw), l_sh=20 * sw,
                             l_el=15 + 10 * max(0.0, sw)), offset=-35))
    return out


def run() -> list[Pose]:
    out = []
    for i in range(8):
        t = _cyc(i)
        sw = math.sin(t)
        r_knee = 25 + 75 * max(0.0, math.cos(t))
        l_knee = 25 + 75 * max(0.0, -math.cos(t))
        lift = 0.035 if i in (2, 6) else 0.0   # short flight between steps
        out.append(held(Pose(spine=12, head=4, r_hip=42 * sw, r_knee=r_knee, l_hip=-42 * sw, l_knee=l_knee,
                             r_sh=-45 * sw, r_el=85, l_sh=45 * sw, l_el=85, lift=lift)))
    return out


_STANCE = dict(r_hip=18, r_knee=12, l_hip=-16, l_knee=8)


def a1a() -> list[Pose]:
    """Overhand punch with the near (right) arm (owner, 2026-10-30): the arm swings from behind to the front while
    the fist comes down from above the head. A negative elbow folds the forearm up, which is how the raised,
    cocked arm looks in a side view."""
    g = dict(r_sh=45, r_el=105, l_sh=35, l_el=115)   # guard, fists near the chin
    return [
        Pose(spine=6, head=6, **g, **_STANCE),
        Pose(spine=0, head=4, r_sh=-40, r_el=-120, l_sh=40, l_el=110, dx=-0.005, **_STANCE),   # draw back, fist up
        Pose(spine=-4, head=2, r_sh=-70, r_el=-130, l_sh=45, l_el=105, dx=-0.012, **_STANCE),  # load behind the head
        Pose(spine=4, head=4, r_sh=140, r_el=-20, l_sh=40, l_el=110, **_STANCE),               # over the top
        Pose(spine=12, head=8, r_sh=100, r_el=-10, l_sh=30, l_el=115, dx=0.015, **_STANCE),    # forward, coming down
        Pose(spine=16, head=10, r_sh=70, r_el=4, l_sh=20, l_el=120, dx=0.025, **_STANCE),      # down and through
        Pose(spine=10, head=7, r_sh=55, r_el=70, l_sh=30, l_el=115, dx=0.012, **_STANCE),      # recovery
        Pose(spine=6, head=6, **g, **_STANCE),
    ]


def a1b() -> list[Pose]:
    g = dict(r_sh=45, r_el=105, l_sh=35, l_el=115)
    return [
        Pose(spine=6, head=6, **g, **_STANCE),
        Pose(spine=0, head=6, r_sh=35, r_el=100, l_sh=30, l_el=110, r_hip=10, r_knee=20, l_hip=-8, l_knee=10),
        Pose(spine=-6, head=8, r_sh=25, r_el=95, l_sh=40, l_el=110, r_hip=70, r_knee=105, l_hip=-4, l_knee=8),
        Pose(spine=-14, head=10, r_sh=10, r_el=80, l_sh=50, l_el=100, r_hip=88, r_knee=8, l_hip=-2, l_knee=6),
        Pose(spine=-15, head=10, r_sh=10, r_el=80, l_sh=50, l_el=100, r_hip=90, r_knee=4, l_hip=-2, l_knee=6),
        Pose(spine=-8, head=8, r_sh=25, r_el=95, l_sh=40, l_el=110, r_hip=65, r_knee=95, l_hip=-4, l_knee=8),
        Pose(spine=0, head=6, r_sh=40, r_el=100, l_sh=35, l_el=110, r_hip=25, r_knee=30, l_hip=-10, l_knee=10),
        Pose(spine=6, head=6, **g, **_STANCE),
    ]


def a2() -> list[Pose]:
    """One-handed weapon in the near (right) hand: wind-up, diagonal slash, recovery."""
    lg = dict(l_sh=25, l_el=70)
    return [
        Pose(spine=5, head=4, r_sh=35, r_el=55, **lg, **_STANCE, weapon=w("r", -50)),          # ready
        Pose(spine=0, head=4, r_sh=150, r_el=60, **lg, **_STANCE, weapon=w("r", -130)),        # raise
        Pose(spine=-4, head=2, r_sh=205, r_el=70, **lg, dx=-0.01, **_STANCE, weapon=w("r", -170)),  # wind-up
        Pose(spine=6, head=6, r_sh=160, r_el=30, **lg, **_STANCE, weapon=w("r", -70)),         # slash starts
        Pose(spine=14, head=8, r_sh=100, r_el=5, l_sh=-10, l_el=40, dx=0.02, **_STANCE, weapon=w("r", 5)),
        Pose(spine=16, head=10, r_sh=45, r_el=10, l_sh=-20, l_el=40, dx=0.025, **_STANCE, weapon=w("r", 60)),
        Pose(spine=10, head=6, r_sh=30, r_el=35, **lg, dx=0.01, **_STANCE, weapon=w("r", 25)),  # recovery
        Pose(spine=5, head=4, r_sh=35, r_el=55, **lg, **_STANCE, weapon=w("r", -45)),
    ]


def a5() -> list[Pose]:
    """Bow in the far (left) hand, string drawn by the near hand. Stages: 1 resting, 2 half, 3 full.
    The shot cycle of archery guides and photos (owner, 2026-10-31: the old draw pushed both arms forward): bow
    low; set-up; bow arm straight forward at shoulder height; the draw hand pulls back along the arrow to the jaw
    with the elbow going back, level with the arrow; hold; release, the hand sliding back past the ear, fingers
    relaxed; follow-through with the bow arm still up."""
    st = dict(r_hip=16, r_knee=8, l_hip=-14, l_knee=6)
    aim = dict(l_sh=95, l_el=0)                       # bow hand about 0.29 ahead of the near shoulder, 0.03 up
    return [
        Pose(spine=2, head=2, l_sh=25, l_el=15, **_near(0.10, 0.20), **st, weapon=w("l", -70, "behind", 1)),  # rest
        Pose(spine=1, head=1, l_sh=60, l_el=10, **_near(0.18, 0.08), **st, weapon=w("l", -82, "behind", 1)),  # set-up
        Pose(spine=0, head=1, **aim, **_near(0.24, -0.035), **st, weapon=w("l", -90, "behind", 1)),  # pre-draw
        Pose(spine=-1, head=2, **aim, **_near(0.10, -0.05), **st, weapon=w("l", -90, "behind", 2)),  # draw: elbow still low in front
        # (a flat side view cannot show the elbow swinging out toward the viewer on its way back)
        Pose(spine=-2, head=3, **aim, **_near(0.02, -0.07, "back"), **st, weapon=w("l", -90, "behind", 3)),  # anchor
        Pose(spine=-2, head=3, **aim, **_near(0.015, -0.07, "back"), **st, weapon=w("l", -90, "behind", 3)),  # aim
        Pose(spine=-2, head=2, **aim, **_near(-0.06, -0.06, "back"), **st, weapon=w("l", -90, "behind", 1),
             open=("r",)),                                                                               # release
        Pose(spine=-1, head=1, l_sh=88, l_el=2, **_near(-0.05, 0.03, "back"), **st,
             weapon=w("l", -88, "behind", 1), open=("r",)),                                              # follow
    ]


def a6() -> list[Pose]:
    """Magic staff in the near hand (owner, 2026-10-31): the staff is raised forward to about 40 degrees above
    the horizontal, held there, then brought back to where it started."""
    st = dict(r_hip=14, r_knee=8, l_hip=-12, l_knee=6)
    far = dict(l_sh=-5, l_el=20)
    rest = dict(r_sh=20, r_el=60)
    up = dict(r_sh=75, r_el=15)                       # arm forward, forearm level
    return [
        Pose(spine=3, head=2, **rest, **far, **st, weapon=w("r", -88)),                          # start
        Pose(spine=3, head=1, r_sh=45, r_el=45, **far, **st, weapon=w("r", -70)),                # lifting
        Pose(spine=5, head=0, r_sh=70, r_el=20, l_sh=-8, l_el=22, dx=0.005, **st, weapon=w("r", -44)),
        Pose(spine=6, head=0, **up, l_sh=-10, l_el=24, dx=0.008, **st, weapon=w("r", -40)),      # held
        Pose(spine=6, head=0, r_sh=76, r_el=14, l_sh=-10, l_el=24, dx=0.008, **st, weapon=w("r", -39)),
        Pose(spine=6, head=0, **up, l_sh=-10, l_el=24, dx=0.008, **st, weapon=w("r", -40)),
        Pose(spine=4, head=1, r_sh=45, r_el=45, **far, dx=0.003, **st, weapon=w("r", -68)),      # lowering
        Pose(spine=3, head=2, **rest, **far, **st, weapon=w("r", -88)),                          # back to start
    ]


def a7() -> list[Pose]:
    """Skill cast with the far (left) hand (owner, 2026-10-30): the near arm keeps holding the weapon as in Idle while
    the far arm pulls in, then thrusts forward, open hand, and comes back."""
    st = dict(r_hip=8, r_knee=6, l_hip=-8, l_knee=6)
    near = dict(r_sh=6, r_el=12)                       # Idle's weapon arm
    return [held(p, offset=-35) for p in (
        Pose(spine=2, head=0, **near, l_sh=-6, l_el=12, **st),
        Pose(spine=0, head=-2, **near, l_sh=0, l_el=50, dx=-0.005, **st),      # pull the hand in
        Pose(spine=-3, head=-3, **near, l_sh=20, l_el=110, dx=-0.01, **st),     # gather at the chest
        Pose(spine=6, head=2, **near, l_sh=70, l_el=30, dx=0.008, **st, open=("l",)),        # thrust
        Pose(spine=9, head=3, r_sh=4, r_el=12, l_sh=88, l_el=2, dx=0.015, **st, open=("l",)),   # the cast
        Pose(spine=9, head=3, r_sh=4, r_el=12, l_sh=86, l_el=4, dx=0.015, **st, open=("l",)),   # hold
        Pose(spine=5, head=2, **near, l_sh=40, l_el=45, dx=0.006, **st),        # draw back
        Pose(spine=2, head=0, **near, l_sh=-4, l_el=16, **st),
    )]


# --- The rest of the Confirmed list (2026-10-31: owner asked to finish them before the workflow work) -----------
def jump() -> list[Pose]:
    """A whole jump in place: crouch, push off, rise with the knees tucking, peak, come down, land, absorb, stand."""
    return [held(p, offset=-35) for p in (
        Pose(spine=18, head=4, r_sh=-40, r_el=15, l_sh=-45, l_el=15, r_hip=42, r_knee=72, l_hip=36, l_knee=66),
        Pose(spine=8, head=0, r_sh=110, r_el=30, l_sh=100, l_el=30, r_hip=12, r_knee=12, l_hip=6, l_knee=8),
        Pose(spine=2, head=-4, r_sh=160, r_el=20, l_sh=150, l_el=20, r_hip=-4, r_knee=14, l_hip=-10, l_knee=20,
             lift=0.07),
        Pose(spine=4, head=-2, r_sh=140, r_el=40, l_sh=130, l_el=40, r_hip=50, r_knee=80, l_hip=40, l_knee=90,
             lift=0.17),
        Pose(spine=6, head=0, r_sh=110, r_el=40, l_sh=100, l_el=40, r_hip=60, r_knee=95, l_hip=50, l_knee=100,
             lift=0.21),
        Pose(spine=6, head=2, r_sh=80, r_el=20, l_sh=70, l_el=20, r_hip=20, r_knee=30, l_hip=10, l_knee=25,
             lift=0.10),
        Pose(spine=20, head=6, r_sh=50, r_el=30, l_sh=40, l_el=30, r_hip=45, r_knee=75, l_hip=40, l_knee=70),
        Pose(spine=6, head=2, r_sh=10, r_el=15, l_sh=-8, l_el=15, r_hip=8, r_knee=10, l_hip=-4, l_knee=8),
    )]


def fall() -> list[Pose]:
    """Falling, looped while in the air: arms up and out, legs hanging a little bent, slowly paddling."""
    out = []
    for i in range(8):
        sw = math.sin(_cyc(i))
        out.append(held(Pose(spine=-4, head=-6, r_sh=135 + 12 * sw, r_el=35, l_sh=125 - 12 * sw, l_el=35,
                             r_hip=14 + 8 * sw, r_knee=30 - 8 * sw, l_hip=4 - 8 * sw, l_knee=38 + 8 * sw,
                             lift=0.12), offset=-35))
    return out


def hit() -> list[Pose]:
    """Hit from the front: the head snaps back, the body recoils with the arms flung forward, then recovers."""
    st = dict(r_hip=8, r_knee=6, l_hip=-8, l_knee=6)
    return [held(p, offset=-35) for p in (
        Pose(spine=3, head=2, r_sh=10, r_el=20, l_sh=-6, l_el=20, **st),
        Pose(spine=-22, head=-24, r_sh=45, r_el=30, l_sh=35, l_el=30, dx=-0.03, **st),               # impact
        Pose(spine=-30, head=-26, r_sh=55, r_el=40, l_sh=45, l_el=40, dx=-0.045, r_hip=12, r_knee=20,
             l_hip=-12, l_knee=16),                                                                   # recoil
        Pose(spine=-24, head=-16, r_sh=40, r_el=40, l_sh=30, l_el=40, dx=-0.045, r_hip=12, r_knee=22, l_hip=-12,
             l_knee=18),
        Pose(spine=-10, head=-4, r_sh=25, r_el=35, l_sh=15, l_el=35, dx=-0.04, **st),                # catch it
        Pose(spine=2, head=4, r_sh=15, r_el=30, l_sh=0, l_el=30, dx=-0.018, **st),
        Pose(spine=5, head=4, r_sh=12, r_el=22, l_sh=-4, l_el=22, dx=-0.008, **st),
        Pose(spine=3, head=2, r_sh=10, r_el=20, l_sh=-6, l_el=20, **st),
    )]


def death() -> list[Pose]:
    """Struck, the knees give way and the body falls backward, lands, bounces a little and lies still on its back
    (face up)."""
    return [held(p, offset=-35) for p in (
        Pose(spine=-12, head=-15, r_sh=35, r_el=30, l_sh=25, l_el=30, dx=-0.01, r_hip=8, r_knee=8, l_hip=-8,
             l_knee=8),
        Pose(spine=-6, head=12, r_sh=10, r_el=20, l_sh=0, l_el=20, dx=-0.015, r_hip=20, r_knee=40, l_hip=12,
             l_knee=45),                                                                             # knees give
        Pose(spine=-24, head=-6, r_sh=40, r_el=25, l_sh=30, l_el=25, dx=-0.02, r_hip=34, r_knee=72, l_hip=26,
             l_knee=78),
        Pose(spine=-52, head=-20, r_sh=110, r_el=30, l_sh=100, l_el=30, dx=-0.03, r_hip=45, r_knee=55, l_hip=38,
             l_knee=60, floor=True),                                                                 # falling
        Pose(spine=-86, head=-40, r_sh=150, r_el=20, l_sh=140, l_el=20, dx=-0.04, r_hip=70, r_knee=40, l_hip=62,
             l_knee=50, floor=True),                                                                 # lands
        Pose(spine=-80, head=-38, r_sh=120, r_el=20, l_sh=115, l_el=25, dx=-0.04, r_hip=62, r_knee=34, l_hip=55,
             l_knee=42, floor=True),                                                                 # bounce
        Pose(spine=-90, head=-45, r_sh=95, r_el=6, l_sh=92, l_el=6, dx=-0.04, r_hip=86, r_knee=12, l_hip=80,
             l_knee=22, floor=True),
        Pose(spine=-90, head=-45, r_sh=94, r_el=4, l_sh=91, l_el=4, dx=-0.04, r_hip=88, r_knee=8, l_hip=82,
             l_knee=18, floor=True),                                                                 # still
    )]


_WIDE = dict(r_hip=22, r_knee=14, l_hip=-20, l_knee=8)
_LUNGE = dict(r_hip=40, r_knee=34, l_hip=-30, l_knee=4)


def a3() -> list[Pose]:
    """Two-handed heavy weapon: both hands on the hilt (the far one toward the pommel), a slow lift over the head,
    a held wind-up, then a downward smash with the body following, and recovery."""
    g = -0.06
    return [
        Pose(spine=5, head=3, **_both(0.10, 0.20, -60, g), **_WIDE, weapon=w("r", -60)),             # ready
        Pose(spine=2, head=1, **_both(0.08, 0.04, -100, g), **_WIDE, weapon=w("r", -100)),           # lift
        Pose(spine=-6, head=-2, **_both(-0.03, -0.22, -200, g), dx=-0.01, **_WIDE, weapon=w("r", -200)),  # over
        Pose(spine=-8, head=-3, **_both(-0.04, -0.23, -205, g), dx=-0.012, **_WIDE, weapon=w("r", -205)),  # hold
        Pose(spine=6, head=2, **_both(0.10, -0.20, -80, g), dx=0.005, **_WIDE, weapon=w("r", -80)),   # swing
        Pose(spine=22, head=8, **_both(0.22, 0.10, 40, g), dx=0.03, r_hip=34, r_knee=40, l_hip=-24, l_knee=12,
             weapon=w("r", 40)),                                                                     # smash
        Pose(spine=25, head=10, **_both(0.21, 0.15, 62, g), dx=0.032, r_hip=36, r_knee=44, l_hip=-24, l_knee=14,
             weapon=w("r", 62)),                                                                     # impact
        Pose(spine=10, head=5, **_both(0.12, 0.16, -30, g), dx=0.012, **_WIDE, weapon=w("r", -30)),   # recover
    ]


def a4() -> list[Pose]:
    """Polearm (spear): the near hand at the back of the shaft, the far hand in front; draw back, step, thrust
    forward with the body behind it, hold, pull back. The near elbow stays behind the body (owner, 2026-10-31)."""
    return [
        Pose(spine=4, head=2, **_both_back(0.0, 0.17, -8, 0.14), **_STANCE, weapon=w("r", -8)),                # ready
        Pose(spine=-4, head=0, **_both_back(-0.08, 0.15, -6, 0.13), dx=-0.012, **_STANCE, weapon=w("r", -6)),   # draw
        Pose(spine=-2, head=0, **_both_back(-0.06, 0.15, -5, 0.13), r_hip=34, r_knee=22, l_hip=-18, l_knee=6,
             weapon=w("r", -5)),                                                                         # step
        Pose(spine=12, head=6, **_both_back(0.10, 0.06, -3, 0.16), dx=0.04, **_LUNGE, weapon=w("r", -3)),      # thrust
        Pose(spine=16, head=8, **_both_back(0.14, 0.04, -2, 0.15), dx=0.045, **_LUNGE, weapon=w("r", -2)),      # reach
        Pose(spine=16, head=8, **_both_back(0.14, 0.045, -2, 0.15), dx=0.045, **_LUNGE, weapon=w("r", -2)),     # hold
        Pose(spine=6, head=3, **_both_back(0.0, 0.14, -6, 0.14), dx=0.03, **_STANCE, weapon=w("r", -6)),       # pull
        Pose(spine=4, head=2, **_both_back(0.0, 0.17, -8, 0.14), dx=0.01, **_STANCE, weapon=w("r", -8)),
    ]


# --- More attacks from common 2D attack sets (2026-10-31, owner asked; Proposed, not in the Confirmed list) -------
_GUARD = dict(l_sh=25, l_el=70)          # the free far hand up in front of the body


def a2b() -> list[Pose]:
    """One-handed thrust (stab): pull the blade back level at the hip, step into a lunge and stab straight forward,
    the far arm swinging back for balance, then withdraw."""
    return [
        Pose(spine=5, head=4, **_near(0.12, 0.14), **_GUARD, **_STANCE, weapon=w("r", -30)),             # ready
        Pose(spine=-4, head=2, **_near(-0.06, 0.10, "back"), l_sh=50, l_el=30, dx=-0.01, **_STANCE,
             weapon=w("r", -5)),                                                                          # cock
        Pose(spine=-6, head=2, **_near(-0.08, 0.08, "back"), l_sh=55, l_el=25, dx=-0.01, r_hip=30, r_knee=20,
             l_hip=-16, l_knee=6, weapon=w("r", -3)),                                                     # step
        Pose(spine=12, head=6, **_near(0.22, 0.04), l_sh=-30, l_el=20, dx=0.04, **_LUNGE, weapon=w("r", -2)),
        Pose(spine=16, head=8, **_near(0.29, 0.03), l_sh=-40, l_el=15, dx=0.045, **_LUNGE, weapon=w("r", 0)),
        Pose(spine=16, head=8, **_near(0.28, 0.03), l_sh=-38, l_el=16, dx=0.045, **_LUNGE, weapon=w("r", 0)),
        Pose(spine=8, head=5, **_near(0.10, 0.10), l_sh=0, l_el=40, dx=0.025, **_STANCE, weapon=w("r", -15)),
        Pose(spine=5, head=4, **_near(0.12, 0.14), **_GUARD, dx=0.008, **_STANCE, weapon=w("r", -30)),
    ]


def a2c() -> list[Pose]:
    """One-handed overhead chop: the blade straight up over the head and a little behind, then cut vertically down
    in front, the body bending into it."""
    return [
        Pose(spine=5, head=4, **_near(0.12, 0.14), **_GUARD, **_STANCE, weapon=w("r", -50)),             # ready
        Pose(spine=-2, head=0, **_near(0.02, -0.26), **_GUARD, **_STANCE, weapon=w("r", -100)),          # raise
        Pose(spine=-6, head=-2, **_near(-0.04, -0.27), **_GUARD, dx=-0.01, **_STANCE, weapon=w("r", -150)),
        Pose(spine=6, head=3, **_near(0.14, -0.22), **_GUARD, **_STANCE, weapon=w("r", -70)),            # chop
        Pose(spine=16, head=8, **_near(0.26, 0.02), l_sh=-10, l_el=40, dx=0.025, **_STANCE, weapon=w("r", 10)),
        Pose(spine=20, head=10, **_near(0.22, 0.14), l_sh=-15, l_el=40, dx=0.03, **_STANCE, weapon=w("r", 55)),
        Pose(spine=10, head=6, **_near(0.14, 0.14), **_GUARD, dx=0.01, **_STANCE, weapon=w("r", -10)),
        Pose(spine=5, head=4, **_near(0.12, 0.14), **_GUARD, **_STANCE, weapon=w("r", -50)),
    ]


def a2d() -> list[Pose]:
    """One-handed rising slash: the blade drops low behind the body, then sweeps up in front to above the head. While
    the blade is low the near elbow is behind the body (owner, 2026-10-31)."""
    low = dict(r_hip=24, r_knee=24, l_hip=-18, l_knee=14)
    return [
        Pose(spine=5, head=4, **_near(0.12, 0.14), **_GUARD, **_STANCE, weapon=w("r", -50)),             # ready
        Pose(spine=8, head=5, **_near(-0.10, 0.18, "back"), **_GUARD, **low, weapon=w("r", 140)),                # drop
        Pose(spine=12, head=6, **_near(-0.04, 0.24, "back"), **_GUARD, dx=0.005, **low, weapon=w("r", 120)),
        Pose(spine=6, head=4, **_near(0.18, 0.16, "back"), **_GUARD, dx=0.015, **_STANCE, weapon=w("r", 30)),    # rising
        Pose(spine=-4, head=0, **_near(0.26, -0.06), l_sh=-10, l_el=40, dx=0.025, **_STANCE, weapon=w("r", -50)),
        Pose(spine=-8, head=-3, **_near(0.16, -0.24), l_sh=-15, l_el=40, dx=0.025, **_STANCE,
             weapon=w("r", -100)),                                                                         # top
        Pose(spine=0, head=2, **_near(0.14, 0.06), **_GUARD, dx=0.012, **_STANCE, weapon=w("r", -60)),
        Pose(spine=5, head=4, **_near(0.12, 0.14), **_GUARD, **_STANCE, weapon=w("r", -50)),
    ]


def _dual(r: tuple, ra: float, lft: tuple, la: float, **kw) -> Pose:
    """A dual-wield pose: the near hand's blade (r: wrist from the near shoulder, angle) in front of the body, the
    far hand's behind it."""
    return Pose(**_near(*r), **_far(*lft), weapon=w("r", ra), weapon2=w("l", la, "behind"), **kw)


def a8a() -> list[Pose]:
    """Dual wielding, alternating slashes: the near blade cuts down while the far one guards, then the far blade
    cuts as the near one pulls back (the common two-hit combo; Spine's way: one arm's keys shifted in time)."""
    rd = dict(r=(0.12, 0.12), ra=-40, lft=(0.14, 0.10), la=-30)
    return [
        _dual(**rd, spine=5, head=4, **_STANCE),                                                          # ready
        _dual((-0.02, -0.24), -150, (0.14, 0.08), -30, spine=-4, head=1, **_STANCE),                     # near up
        _dual((0.26, 0.04), 30, (0.12, 0.10), -40, spine=12, head=6, dx=0.02, **_STANCE),                # near cut
        _dual((0.12, 0.20), 80, (0.0, -0.24), -140, spine=4, head=3, dx=0.02, **_STANCE),                # far up
        _dual((0.04, 0.16), -20, (0.27, 0.0), 20, spine=14, head=7, dx=0.035, **_STANCE),                # far cut
        _dual((0.06, 0.16), -25, (0.14, 0.20), 75, spine=10, head=6, dx=0.035, **_STANCE),
        _dual((0.10, 0.14), -40, (0.12, 0.12), -30, spine=7, head=5, dx=0.015, **_STANCE),               # recover
        _dual(**rd, spine=5, head=4, dx=0.005, **_STANCE),
    ]


def a8b() -> list[Pose]:
    """Dual wielding, cross slash: both blades raised over the head, then cut down together on crossing paths."""
    rd = dict(r=(0.12, 0.12), ra=-40, lft=(0.14, 0.10), la=-30)
    return [
        _dual(**rd, spine=5, head=4, **_STANCE),                                                          # ready
        _dual((0.0, -0.25), -130, (0.04, -0.25), -60, spine=-2, head=0, **_STANCE),                      # raise
        _dual((-0.04, -0.22), -160, (0.0, -0.24), -110, spine=-6, head=-2, dx=-0.01, **_STANCE),         # wind
        _dual((0.22, -0.06), -10, (0.20, -0.02), -60, spine=10, head=5, dx=0.02, **_STANCE),             # cut
        _dual((0.24, 0.12), 50, (0.22, 0.14), 10, spine=18, head=9, dx=0.035, **_STANCE),                # cross
        _dual((0.16, 0.22), 80, (0.18, 0.22), 40, spine=16, head=8, dx=0.035, **_STANCE),
        _dual((0.12, 0.12), -40, (0.14, 0.10), -30, spine=8, head=5, dx=0.012, **_STANCE),              # recover
        _dual(**rd, spine=5, head=4, **_STANCE),
    ]


TEMPLATES = {
    "idle": dict(fn=idle, animation="idle", variant=None, name_vi="Đứng yên", grip="optional",
                 kinds=["one_handed", "two_handed", "polearm", "bow", "staff"], prompt="standing, idle"),
    "walk": dict(fn=walk, animation="walk", variant=None, name_vi="Đi bộ", grip="optional",
                 kinds=["one_handed", "two_handed", "polearm", "bow", "staff"], prompt="walking"),
    "run": dict(fn=run, animation="run", variant=None, name_vi="Chạy", grip="optional",
                kinds=["one_handed", "two_handed", "polearm", "bow", "staff"], prompt="running"),
    "a1a": dict(fn=a1a, animation="attack", variant="a1a", name_vi="Tay không — đấm", grip="never", kinds=[],
                prompt="punching, overhand punch, fighting stance, clenched hands"),
    "a1b": dict(fn=a1b, animation="attack", variant="a1b", name_vi="Tay không — đá", grip="never", kinds=[],
                prompt="kicking, fighting stance"),
    "a2": dict(fn=a2, animation="attack", variant="a2", name_vi="Vũ khí một tay", grip="always",
               kinds=["one_handed"], prompt="attacking, swinging arm, clenched hand"),
    "a5": dict(fn=a5, animation="attack", variant="a5", name_vi="Cung", grip="always", kinds=["bow"],
               prompt="archery, drawing a bow, bow arm extended forward, drawing hand at the cheek, clenched hands"),
    "a6": dict(fn=a6, animation="attack", variant="a6", name_vi="Gậy phép", grip="always", kinds=["staff"],
               prompt="holding a staff, arm raised forward, clenched hand"),
    "a7": dict(fn=a7, animation="attack", variant="a7", name_vi="Tay trái tung chiêu", grip="always",
               kinds=["one_handed", "staff"], prompt="casting spell, arm outstretched forward, open palm, other hand clenched"),
    "jump": dict(fn=jump, animation="jump", variant=None, name_vi="Nhảy", grip="optional",
                 kinds=["one_handed", "two_handed", "polearm", "bow", "staff"], prompt="jumping"),
    "fall": dict(fn=fall, animation="fall", variant=None, name_vi="Rơi", grip="optional",
                 kinds=["one_handed", "two_handed", "polearm", "bow", "staff"], prompt="falling, arms raised"),
    "hit": dict(fn=hit, animation="hit", variant=None, name_vi="Bị đánh", grip="optional",
                kinds=["one_handed", "two_handed", "polearm", "bow", "staff"], prompt="getting hit, recoiling"),
    "death": dict(fn=death, animation="death", variant=None, name_vi="Gục ngã", grip="optional",
                  kinds=["one_handed", "two_handed", "polearm", "bow", "staff"],
                  prompt="falling backward, lying on back, defeated"),
    "a3": dict(fn=a3, animation="attack", variant="a3", name_vi="Vũ khí nặng hai tay", grip="always",
               kinds=["two_handed"], prompt="two-handed overhead smash, both hands on the hilt, clenched hands"),
    "a4": dict(fn=a4, animation="attack", variant="a4", name_vi="Vũ khí cán dài", grip="always", kinds=["polearm"],
               prompt="spear thrust, lunging forward, both hands holding a shaft, clenched hands"),
    "a2b": dict(fn=a2b, animation="attack", variant="a2b", name_vi="Một tay — đâm", grip="always",
                kinds=["one_handed"], prompt="lunging stab, arm thrust straight forward, clenched hand"),
    "a2c": dict(fn=a2c, animation="attack", variant="a2c", name_vi="Một tay — bổ từ trên xuống", grip="always",
                kinds=["one_handed"], prompt="overhead chop, arm raised high then swung down, clenched hand"),
    "a2d": dict(fn=a2d, animation="attack", variant="a2d", name_vi="Một tay — chém hất lên", grip="always",
                kinds=["one_handed"], prompt="rising upward slash, arm swung up from low, clenched hand"),
    "a8a": dict(fn=a8a, animation="attack", variant="a8a", name_vi="Song kiếm — chém luân phiên", grip="always",
                kinds=["one_handed"], prompt="dual wielding, alternating slashes, both hands clenched"),
    "a8b": dict(fn=a8b, animation="attack", variant="a8b", name_vi="Song kiếm — chém chéo", grip="always",
                kinds=["one_handed"], prompt="dual wielding, cross slash with both arms, both hands clenched"),
}


# Squash/stretch, part sizes, reach and bends per frame (shape.py; owner, 2026-10-31: on every built-in pose, both
# views). Frame index (0-based) -> shape. Squash + is shorter and wider (a landing), - taller and thinner (a leap);
# bend + arches forward (the back arched), - hunches. Kept small: about 10 % squash, 15 % size, 8 degrees.
def _breathe() -> dict[int, dict]:
    return {i: {"squash": round(-0.02 * (1 - math.cos(_cyc(i))) / 2, 4)} for i in range(1, 8)}


SHAPES: dict[str, dict[int, dict]] = {
    "idle": _breathe(),
    "walk": {0: {"squash": -0.015}, 2: {"squash": 0.03}, 4: {"squash": -0.015}, 6: {"squash": 0.03}},
    "run": {0: {"squash": 0.06, "bend": {"torso": -4}}, 1: {"squash": 0.02, "bend": {"torso": -4}},
            2: {"squash": -0.05, "bend": {"torso": -4}}, 3: {"bend": {"torso": -4}},
            4: {"squash": 0.06, "bend": {"torso": -4}}, 5: {"squash": 0.02, "bend": {"torso": -4}},
            6: {"squash": -0.05, "bend": {"torso": -4}}, 7: {"bend": {"torso": -4}}},
    "jump": {0: {"squash": 0.10}, 1: {"squash": -0.08}, 2: {"squash": -0.10}, 3: {"squash": -0.04},
             4: {"bend": {"torso": 6}}, 5: {"squash": -0.05}, 6: {"squash": 0.10}, 7: {"squash": 0.03}},
    # Fall loops in the air (no landing frame): only a stretch.
    "fall": {i: {"squash": round(-0.05 + 0.01 * math.sin(_cyc(i)), 4)} for i in range(8)},
    "hit": {1: {"squash": 0.06, "size": {"head": 0.92}, "bend": {"torso": 8}},
            2: {"squash": -0.02, "size": {"head": 0.96}, "bend": {"torso": 10}},
            3: {"bend": {"torso": 6}}, 4: {"bend": {"torso": 2}}},
    "death": {1: {"squash": 0.04}, 2: {"bend": {"torso": -8}}, 3: {"bend": {"torso": -6}},
              4: {"squash": 0.06}, 5: {"squash": -0.02}},
    "a1a": {1: {"squash": 0.04}, 2: {"squash": 0.05}, 3: {"bend": {"arm_r": 8}},
            4: {"squash": -0.04, "size": {"hand_r": 1.08}, "bend": {"arm_r": 5}},
            5: {"squash": -0.05, "size": {"hand_r": 1.15}}},
    "a1b": {1: {"squash": 0.03}, 2: {"size": {"foot_r": 1.06}},
            3: {"size": {"foot_r": 1.15}, "bend": {"torso": 4}}, 4: {"size": {"foot_r": 1.15}, "bend": {"torso": 4}},
            5: {"size": {"foot_r": 1.06}}},
    "a2": {2: {"squash": 0.05}, 3: {"bend": {"arm_r": -8}}, 4: {"squash": -0.04, "bend": {"arm_r": -5}},
           5: {"squash": -0.03}},
    "a2b": {1: {"squash": 0.04}, 2: {"squash": 0.04}, 3: {"reach": {"arm_r": 1.05}},
            4: {"squash": -0.03, "reach": {"arm_r": 1.10}}, 5: {"reach": {"arm_r": 1.08}}},
    "a2c": {1: {"squash": -0.04}, 2: {"squash": -0.04}, 3: {"bend": {"arm_r": -8}}, 4: {"squash": 0.05},
            5: {"squash": 0.05}},
    "a2d": {1: {"squash": 0.05}, 2: {"squash": 0.05}, 3: {"bend": {"arm_r": -8}},
            4: {"squash": -0.05, "bend": {"arm_r": -8}}, 5: {"squash": -0.05}},
    "a3": {1: {"squash": 0.04}, 2: {"squash": -0.04}, 3: {"squash": -0.04},
           5: {"squash": 0.07, "bend": {"torso": -8}}, 6: {"squash": 0.09, "bend": {"torso": -10}}},
    "a4": {1: {"squash": 0.05}, 3: {"squash": -0.04, "bend": {"torso": -5}},
           4: {"squash": -0.05, "bend": {"torso": -5}}, 5: {"bend": {"torso": -5}}},
    "a5": {3: {"bend": {"torso": 2}}, 4: {"bend": {"torso": 5}}, 5: {"bend": {"torso": 5}}, 6: {"bend": {"torso": 3}}},
    "a6": {2: {"squash": -0.03}, 3: {"squash": -0.03}, 4: {"squash": -0.03}, 5: {"squash": -0.03}},
    "a7": {1: {"squash": 0.04}, 2: {"squash": 0.04}, 3: {"squash": -0.03, "size": {"hand_l": 1.08}},
           4: {"squash": -0.04, "size": {"hand_l": 1.15}}, 5: {"size": {"hand_l": 1.10}}},
    "a8a": {1: {"squash": 0.03}, 2: {"squash": -0.03, "bend": {"arm_r": -6}}, 3: {"squash": 0.03},
            4: {"squash": -0.03, "bend": {"arm_l": -6}}},
    "a8b": {2: {"squash": -0.03}, 4: {"squash": 0.05}, 5: {"squash": 0.03}},
}


def build(tid: str, shapes: bool = True) -> dict:
    spec = TEMPLATES[tid]
    poses: list[Pose] = spec["fn"]()
    assert len(poses) == 8, tid
    rest_pts, _ = solve(Pose())
    frames = []
    for i, p in enumerate(poses):
        pts, grounded = solve(p)
        fr = {"points": [None if q is None else [q[0], q[1]] for q in pts],
              "contact": "ground" if grounded else "air"}
        if shapes and SHAPES.get(tid, {}).get(i):
            fr["shape"] = SHAPES[tid][i]
        if p.open:
            fr["open"] = list(p.open)
        if spec["grip"] != "never":
            fr["weapon"] = p.weapon or {"hand": "r", "angle": 0.0, "layer": "front", "stage": 1}
            if p.weapon2:
                fr["weapon2"] = p.weapon2
        frames.append(fr)
    return {"id": tid, "animation": spec["animation"], "variant": spec["variant"], "name_vi": spec["name_vi"],
            "status": spec.get("status", "Proposed"), "grip": spec["grip"], "weapon_kinds": spec["kinds"],
            "prompt": spec["prompt"], "rest": [None if q is None else [q[0], q[1]] for q in rest_pts],
            "frames": frames}


def write_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for tid in TEMPLATES:
        path = out_dir / f"{tid}.json"
        with open(path, "w", encoding="utf-8", newline="\r\n") as f:
            json.dump(build(tid), f, ensure_ascii=False, indent=1)
            f.write("\n")
        paths.append(path)
    return paths


def contact_sheet(tid: str, cell: int = 256):
    """Draw the 8 poses of a template side by side (skeleton plus a line for the weapon direction)."""
    import cv2
    import numpy as np
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
            length = 0.35 * scale if wd.get("stage", 1) and data["weapon_kinds"] != ["bow"] else 0.3 * scale
            cv2.line(img, (int(hx), int(hy)), (int(hx + math.cos(a) * length), int(hy + math.sin(a) * length)),
                     col, 2)
            if data["weapon_kinds"] == ["bow"]:
                cv2.line(img, (int(hx), int(hy)), (int(hx - math.cos(a) * length), int(hy - math.sin(a) * length)),
                         col, 2)
                cv2.putText(img, f"stage {wd['stage']}", (6, cell - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (220, 220, 220), 1)
        sheet.paste(Image.fromarray(img), (i * cell, 24))
        d.text((i * cell + 6, 5), f"{tid} {i + 1}/8 {fr['contact']}", fill=(230, 230, 230))
    return sheet


if __name__ == "__main__":
    import sys
    here = Path(__file__).parent
    for p in write_all(here / "template_data"):
        print("wrote", p)
    docs = Path(sys.argv[1]) if len(sys.argv) > 1 else here.parent.parent / "docs" / "templates"
    docs.mkdir(parents=True, exist_ok=True)
    for tid in TEMPLATES:
        contact_sheet(tid).save(docs / f"{tid}.png")
        print("drew", docs / f"{tid}.png")
