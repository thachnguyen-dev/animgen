"""Small deformations on top of a pose (owner, 2026-10-31): squash and stretch of the whole figure, the size of a
part, and a gentle curve through a whole part. They are the classic animation exaggerations: a landing squashes, a
leap stretches, a fist grows on the frame it lands.

A template frame may carry them (``"shape"``)::

    "shape": {"squash": 0.08,                          # + squashes (shorter, wider), - stretches; the area is kept
              "size": {"hand_r": 1.15, "head": 0.92},   # a part bigger or smaller about the joint it hangs on
              "reach": {"arm_r": 1.1},                  # an arm longer (or shorter) along its length
              "bend": {"torso": 6, "arm_r": -8}}        # degrees: the part curves, both its ends where they were

Bend sign: positive pushes the middle of the part forward (the way the figure faces, with the part hanging or
standing as at rest), so a positive torso is an arched back and a negative one a hunch.

Where each happens:
- ``reach`` changes the frame's skeleton (``apply_reach``) right after fitting, so everything that reads the skeleton
  (the cloth, the hand anchors) follows it;
- ``size`` and ``bend`` move the rig's mesh vertices after the bones have moved them (``Warp``), by the bones the
  vertices hang on, so a cape on the torso bends with the torso and long hair grows with the head;
- ``squash`` is one transform of the finished frame (``squash_matrix``) about the feet on the ground, or about the
  middle of the figure in the air.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .skeleton import LELB, LSHO, LWRI, NECK, RELB, RSHO, RWRI, Skeleton, hand_point

# What can be set, and how far (a frame outside these is refused when a template is read).
SQUASH_MAX = 0.2
SIZES = ("head", "hand_r", "hand_l", "foot_r", "foot_l")
SIZE_RANGE = (0.7, 1.4)
REACHES = ("arm_r", "arm_l")
REACH_RANGE = (0.8, 1.25)
BENDS = ("torso", "arm_r", "arm_l", "leg_r", "leg_l")
BEND_MAX = 25.0

HAND_FROM = 0.8           # of the forearm from the elbow: the hand starts here (rig.HAND_CUT)
FOOT_FROM = 0.85          # of the shin from the knee: the foot starts here
REGION_SOFT = 0.12        # of the bone: how gradually a hand or a foot grows out of the forearm or the shin
GROUND_NEAR = 0.25        # of the torso (neck to hips): a frame whose lowest joint is this close to the ground stands
                          # on it (an ankle stands about 0.14 torso above it; a jump's first lift is about 0.38).
                          # The torso, not the figure's height, as a crouch is short (landing test, 2026-10-31).

# The frame bone and where along it the part starts (it is scaled about that point); the head is all of its bone,
# about the neck.
_SIZE_BONES = {"head": ("head", 0.0), "hand_r": ("larm", HAND_FROM), "hand_l": ("larm_far", HAND_FROM),
               "foot_r": ("lleg", FOOT_FROM), "foot_l": ("lleg_far", FOOT_FROM)}
# The bones a bend runs through, top joint first (the torso bone runs from the neck down to the hips).
_BEND_CHAINS = {"torso": ("torso",), "arm_r": ("uarm", "larm"), "arm_l": ("uarm_far", "larm_far"),
                "leg_r": ("uleg", "lleg"), "leg_l": ("uleg_far", "lleg_far")}
_REACH_JOINTS = {"arm_r": (RSHO, RELB, RWRI), "arm_l": (LSHO, LELB, LWRI)}


@dataclass
class Shape:
    squash: float = 0.0
    size: dict[str, float] = field(default_factory=dict)
    reach: dict[str, float] = field(default_factory=dict)
    bend: dict[str, float] = field(default_factory=dict)

    def empty(self) -> bool:
        return (abs(self.squash) < 1e-6 and all(abs(v - 1) < 1e-6 for v in self.size.values())
                and all(abs(v - 1) < 1e-6 for v in self.reach.values())
                and all(abs(v) < 1e-6 for v in self.bend.values()))

    def to_json(self) -> dict:
        out: dict = {}
        if abs(self.squash) >= 1e-6:
            out["squash"] = round(self.squash, 4)
        for key, values, neutral in (("size", self.size, 1.0), ("reach", self.reach, 1.0), ("bend", self.bend, 0.0)):
            kept = {k: round(v, 4) for k, v in values.items() if abs(v - neutral) >= 1e-6}
            if kept:
                out[key] = kept
        return out


def parse(data: object) -> Shape | None:
    """A frame's ``"shape"``, checked; None when absent or neutral. Raises ValueError for a bad one."""
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("shape is an object")

    def number(v: object) -> float:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v):
            raise ValueError("shape values are numbers")
        return float(v)

    def group(key: str, names: tuple, low: float, high: float) -> dict[str, float]:
        raw = data.get(key) or {}
        if not isinstance(raw, dict) or not set(raw) <= set(names):
            raise ValueError(f"shape {key}: one of {', '.join(names)}")
        out = {k: number(v) for k, v in raw.items()}
        if any(not low <= v <= high for v in out.values()):
            raise ValueError(f"shape {key}: between {low} and {high}")
        return out

    squash = number(data.get("squash", 0.0))
    if abs(squash) > SQUASH_MAX:
        raise ValueError(f"shape squash: within ±{SQUASH_MAX}")
    s = Shape(squash, group("size", SIZES, *SIZE_RANGE), group("reach", REACHES, *REACH_RANGE),
              group("bend", BENDS, -BEND_MAX, BEND_MAX))
    return None if s.empty() else s


# --------------------------------------------------------------------------------------------------------------
# Reach: the skeleton
# --------------------------------------------------------------------------------------------------------------
def apply_reach(sk: Skeleton, shape: Shape | None) -> Skeleton:
    """The arms longer or shorter along their length, from the shoulder; the rest of the skeleton as it was."""
    if shape is None or not shape.reach:
        return sk
    pts = list(sk.points)
    for name, k in shape.reach.items():
        sho, elb, wri = _REACH_JOINTS[name]
        if pts[sho] is None or pts[elb] is None:
            continue
        e = (pts[sho][0] + (pts[elb][0] - pts[sho][0]) * k, pts[sho][1] + (pts[elb][1] - pts[sho][1]) * k)
        if pts[wri] is not None:
            pts[wri] = (e[0] + (pts[wri][0] - pts[elb][0]) * k, e[1] + (pts[wri][1] - pts[elb][1]) * k)
        pts[elb] = e
    return Skeleton(pts)


# --------------------------------------------------------------------------------------------------------------
# Size and bend: the mesh vertices
# --------------------------------------------------------------------------------------------------------------
def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _along(verts: np.ndarray, a, b) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    return ((verts - a) @ d) / (float(d @ d) + 1e-9)


class Warp:
    """Size and bend for one frame: ``apply`` takes a mesh's vertices after the bones moved them."""

    def __init__(self, shape: Shape, bind: dict, frame: dict) -> None:
        self.shape, self.bind, self.frame = shape, bind, frame

    def active(self) -> bool:
        return bool(self.shape.size or self.shape.bend)

    def apply(self, bones: tuple, weights: np.ndarray, verts: np.ndarray, moved: np.ndarray) -> np.ndarray:
        """``bones``: the mesh's (bind bone, frame bone) pairs; ``verts`` in the bind pose, ``moved`` in the frame."""
        out = moved.copy()
        frame_bones = [f for _, f in bones]
        for name, k in self.shape.size.items():
            bone, start = _SIZE_BONES[name]
            if bone not in frame_bones or abs(k - 1) < 1e-6:
                continue
            j = frame_bones.index(bone)
            a, b = self.bind.get(bones[j][0], (None, None))
            fa, fb = self.frame.get(bone, (None, None))
            if a is None or fa is None or fb is None:
                continue
            t = _along(verts, a, b)
            region = weights[:, j] * (1.0 if start <= 0 else _smooth((t - start + REGION_SOFT) / REGION_SOFT))
            pivot = np.asarray(fa, float) + (np.asarray(fb, float) - np.asarray(fa, float)) * start
            out += (out - pivot) * ((k - 1) * region)[:, None]
        for name, deg in self.shape.bend.items():
            if abs(deg) < 1e-6:
                continue
            out += self._bend(_BEND_CHAINS[name], deg, bones, frame_bones, weights, verts)
        return out

    def _bend(self, chain: tuple, deg: float, bones: tuple, frame_bones: list, weights: np.ndarray,
              verts: np.ndarray) -> np.ndarray:
        """An arc through the chain's bones that keeps its two ends: each vertex moves square to the line from the
        chain's top joint to its end, by how far along the chain it hangs."""
        shift = np.zeros_like(verts)
        ends = [self.frame.get(b, (None, None)) for b in chain]
        if any(a is None or b is None for a, b in ends):
            return shift
        lengths = [math.dist(a, b) for a, b in ends]
        total = sum(lengths)
        top, end = np.asarray(ends[0][0], float), np.asarray(ends[-1][1], float)
        chord = end - top
        span = float(np.linalg.norm(chord))
        if total < 1e-6 or span < 1e-6:
            return shift
        u = chord / span
        normal = np.array([u[1], -u[0]])           # forward for a part hanging down at rest; it turns with the part
        height = span / 2 * math.tan(math.radians(deg) / 2)
        before = 0.0
        for bone, length in zip(chain, lengths):
            if bone in frame_bones:
                j = frame_bones.index(bone)
                a, b = self.bind.get(bones[j][0], (None, None))
                if a is not None:
                    s = (before + np.clip(_along(verts, a, b), 0.0, 1.0) * length) / total
                    shift += (weights[:, j] * 4 * height * s * (1 - s))[:, None] * normal
            before += length
        return shift


# --------------------------------------------------------------------------------------------------------------
# Squash: the finished frame
# --------------------------------------------------------------------------------------------------------------
def squash_matrix(shape: Shape | None, sk: Skeleton, ground: float | None) -> np.ndarray | None:
    """The 2×3 transform of a frame's squash or stretch: the height times 1 - squash, the width divided by it, about
    the pelvis's x and the ground (a frame standing or lying on it) or the middle of the figure (in the air)."""
    if shape is None or abs(shape.squash) < 1e-6:
        return None
    box = sk.bounds()
    pelvis = sk.pelvis()
    if box is None or pelvis is None:
        return None
    x0, y0, x1, y1 = box
    neck = sk.points[NECK]
    torso = math.dist(neck, pelvis) if neck is not None else (y1 - y0) / 3
    on_ground = ground is not None and ground - y1 < GROUND_NEAR * max(torso, 1.0)
    py = ground if on_ground else (y0 + y1) / 2
    sy = 1.0 - shape.squash
    sx = 1.0 / sy
    return np.array([[sx, 0.0, pelvis[0] * (1 - sx)], [0.0, sy, py * (1 - sy)]])



def map_point(m: np.ndarray | None, p: tuple[float, float]) -> tuple[float, float]:
    if m is None:
        return p
    return (float(m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2]), float(m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2]))


def map_angle(m: np.ndarray | None, deg: float) -> float:
    """An image-space direction (degrees) after the transform."""
    if m is None:
        return deg
    a = math.radians(deg)
    x, y = m[0, 0] * math.cos(a) + m[0, 1] * math.sin(a), m[1, 0] * math.cos(a) + m[1, 1] * math.sin(a)
    return math.degrees(math.atan2(y, x))


def shaped_hand(sk: Skeleton, hand: str, shape: Shape | None, ground: float | None) -> tuple[float, float] | None:
    """The middle of a hand in the drawn frame: ``hand_point`` of the (reached) skeleton, moved as the hand grew and
    as the frame squashed."""
    p = hand_point(sk, hand)
    if p is None or shape is None:
        return p
    k = shape.size.get("hand_r" if hand == "r" else "hand_l", 1.0)
    elb, wri = (sk.points[RELB], sk.points[RWRI]) if hand == "r" else (sk.points[LELB], sk.points[LWRI])
    if abs(k - 1) >= 1e-6 and elb is not None and wri is not None:
        pivot = (elb[0] + (wri[0] - elb[0]) * HAND_FROM, elb[1] + (wri[1] - elb[1]) * HAND_FROM)
        p = (pivot[0] + (p[0] - pivot[0]) * k, pivot[1] + (p[1] - pivot[1]) * k)
    return map_point(squash_matrix(shape, sk, ground), p)


