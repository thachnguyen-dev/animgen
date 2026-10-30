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
    # Stand the lowest ankle on the ground (y = -ANKLE_H), then lift frames in the air.
    low = max(pts[RANK][1], pts[LANK][1])
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
    """Bow in the far (left) hand, string drawn by the near hand. Stages: 1 resting, 2 half, 3 full."""
    st = dict(r_hip=16, r_knee=8, l_hip=-14, l_knee=6)
    return [
        Pose(spine=2, head=2, l_sh=25, l_el=15, r_sh=10, r_el=20, **st, weapon=w("l", -70, "behind", 1)),
        Pose(spine=2, head=0, l_sh=60, l_el=10, r_sh=40, r_el=60, **st, weapon=w("l", -80, "behind", 1)),
        Pose(spine=0, head=0, l_sh=88, l_el=2, r_sh=80, r_el=40, **st, weapon=w("l", -90, "behind", 1)),
        Pose(spine=-2, head=0, l_sh=90, l_el=0, r_sh=82, r_el=110, **st, weapon=w("l", -90, "behind", 2)),
        Pose(spine=-4, head=0, l_sh=90, l_el=0, r_sh=78, r_el=150, **st, weapon=w("l", -90, "behind", 3)),
        Pose(spine=-4, head=0, l_sh=90, l_el=0, r_sh=78, r_el=152, **st, weapon=w("l", -90, "behind", 3)),
        Pose(spine=-3, head=0, l_sh=90, l_el=0, r_sh=45, r_el=120, **st, weapon=w("l", -88, "behind", 1)),
        Pose(spine=0, head=0, l_sh=85, l_el=4, r_sh=25, r_el=80, **st, weapon=w("l", -85, "behind", 1)),
    ]


def a6() -> list[Pose]:
    """Magic staff in the near hand: raise, gather, cast forward."""
    st = dict(r_hip=14, r_knee=8, l_hip=-12, l_knee=6)
    return [
        Pose(spine=3, head=2, r_sh=20, r_el=60, l_sh=-5, l_el=20, **st, weapon=w("r", -88)),
        Pose(spine=0, head=-2, r_sh=70, r_el=55, l_sh=20, l_el=60, **st, weapon=w("r", -95)),
        Pose(spine=-4, head=-6, r_sh=140, r_el=30, l_sh=60, l_el=70, **st, weapon=w("r", -100)),
        Pose(spine=-6, head=-8, r_sh=165, r_el=15, l_sh=120, l_el=40, dx=-0.01, **st, weapon=w("r", -95)),
        Pose(spine=-6, head=-8, r_sh=168, r_el=12, l_sh=125, l_el=35, dx=-0.01, **st, weapon=w("r", -95)),
        Pose(spine=8, head=4, r_sh=125, r_el=10, l_sh=60, l_el=50, dx=0.01, **st, weapon=w("r", -45)),
        Pose(spine=14, head=6, r_sh=95, r_el=2, l_sh=30, l_el=60, dx=0.025, **st, weapon=w("r", -10)),
        Pose(spine=6, head=3, r_sh=40, r_el=50, l_sh=0, l_el=25, dx=0.01, **st, weapon=w("r", -75)),
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
        Pose(spine=6, head=2, **near, l_sh=70, l_el=30, dx=0.008, **st),        # thrust
        Pose(spine=9, head=3, r_sh=4, r_el=12, l_sh=88, l_el=2, dx=0.015, **st),   # full reach, the cast
        Pose(spine=9, head=3, r_sh=4, r_el=12, l_sh=86, l_el=4, dx=0.015, **st),   # hold
        Pose(spine=5, head=2, **near, l_sh=40, l_el=45, dx=0.006, **st),        # draw back
        Pose(spine=2, head=0, **near, l_sh=-4, l_el=16, **st),
    )]


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
               prompt="archery stance, aiming, arms raised, clenched hands"),
    "a6": dict(fn=a6, animation="attack", variant="a6", name_vi="Gậy phép", grip="always", kinds=["staff"],
               prompt="casting, raising arm, clenched hand"),
    "a7": dict(fn=a7, animation="attack", variant="a7", name_vi="Tay trái tung chiêu", grip="always",
               kinds=["one_handed", "staff"], prompt="casting spell, arm outstretched forward, open palm, other hand clenched"),
}


def build(tid: str) -> dict:
    spec = TEMPLATES[tid]
    poses: list[Pose] = spec["fn"]()
    assert len(poses) == 8, tid
    rest_pts, _ = solve(Pose())
    frames = []
    for p in poses:
        pts, grounded = solve(p)
        fr = {"points": [None if q is None else [q[0], q[1]] for q in pts],
              "contact": "ground" if grounded else "air"}
        if spec["grip"] != "never":
            fr["weapon"] = p.weapon or {"hand": "r", "angle": 0.0, "layer": "front", "stage": 1}
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
        wd = fr.get("weapon")
        if wd:
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
