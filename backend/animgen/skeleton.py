"""Body skeletons in the 18-point OpenPose layout used by ControlNet, plus fitting and drawing.

Point order (OpenPose COCO-18):
 0 nose, 1 neck, 2 R shoulder, 3 R elbow, 4 R wrist, 5 L shoulder, 6 L elbow, 7 L wrist,
 8 R hip, 9 R knee, 10 R ankle, 11 L hip, 12 L knee, 13 L ankle, 14 R eye, 15 L eye, 16 R ear, 17 L ear.

Coordinates are image pixels, y pointing down. A missing point is None.
All templates are authored facing right; for a character facing right the viewer sees its RIGHT side, so the
right limbs are the near limbs (drawn in front).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
         "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
N = 18
NOSE, NECK, RSHO, RELB, RWRI, LSHO, LELB, LWRI, RHIP, RKNE, RANK, LHIP, LKNE, LANK, REYE, LEYE, REAR, LEAR = range(18)

# Parent -> child edges, in an order where every parent is placed before its children (tree rooted at the neck).
EDGES: list[tuple[int, int]] = [
    (NECK, NOSE), (NOSE, REYE), (NOSE, LEYE), (REYE, REAR), (LEYE, LEAR),
    (NECK, RSHO), (RSHO, RELB), (RELB, RWRI),
    (NECK, LSHO), (LSHO, LELB), (LELB, LWRI),
    (NECK, RHIP), (RHIP, RKNE), (RKNE, RANK),
    (NECK, LHIP), (LHIP, LKNE), (LKNE, LANK),
]
# Left/right counterpart of each edge, used when one side of the character was not seen (side view).
_MIRROR = {RSHO: LSHO, RELB: LELB, RWRI: LWRI, RHIP: LHIP, RKNE: LKNE, RANK: LANK, REYE: LEYE, REAR: LEAR}
_MIRROR.update({v: k for k, v in list(_MIRROR.items())})

# Drawing (same colours and limb order as the OpenPose preprocessor that ControlNet was trained on).
_LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
          (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]
_COLOURS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
            (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
            (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]

Point = tuple[float, float] | None

# Templates are authored so that the rest figure (ground to top of the head) is exactly this tall.
TEMPLATE_HEIGHT = 1.0


@dataclass
class Skeleton:
    points: list[Point]
    confidence: list[float] = field(default_factory=lambda: [1.0] * N)

    def __post_init__(self) -> None:
        if len(self.points) != N:
            raise ValueError(f"a skeleton has {N} points, got {len(self.points)}")
        if len(self.confidence) != N:
            self.confidence = [1.0 if p is not None else 0.0 for p in self.points]

    # ---- conversion -------------------------------------------------------------------------------------------
    def to_json(self) -> list:
        return [None if p is None else [round(float(p[0]), 2), round(float(p[1]), 2)] for p in self.points]

    @staticmethod
    def from_json(data: list) -> "Skeleton":
        pts: list[Point] = [None if p is None else (float(p[0]), float(p[1])) for p in data]
        return Skeleton(pts)

    def copy(self) -> "Skeleton":
        return Skeleton(list(self.points), list(self.confidence))

    # ---- geometry ---------------------------------------------------------------------------------------------
    def get(self, i: int) -> Point:
        return self.points[i]

    def mid(self, a: int, b: int) -> Point:
        pa, pb = self.points[a], self.points[b]
        if pa is None and pb is None:
            return None
        if pa is None:
            return pb
        if pb is None:
            return pa
        return ((pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0)

    def pelvis(self) -> Point:
        return self.mid(RHIP, LHIP)

    def lowest_ankle_y(self) -> float | None:
        ys = [p[1] for p in (self.points[RANK], self.points[LANK]) if p is not None]
        return max(ys) if ys else None

    def translated(self, dx: float, dy: float) -> "Skeleton":
        return Skeleton([None if p is None else (p[0] + dx, p[1] + dy) for p in self.points], list(self.confidence))

    def mirrored(self, width: float) -> "Skeleton":
        """Mirror horizontally inside an image of the given width. Left/right labels stay as they are: a mirrored
        right-facing figure is a left-facing figure whose near side is still the same limbs."""
        return Skeleton([None if p is None else (width - 1 - p[0], p[1]) for p in self.points],
                        list(self.confidence))

    def bounds(self) -> tuple[float, float, float, float] | None:
        pts = [p for p in self.points if p is not None]
        if not pts:
            return None
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def facing(self) -> str | None:
        """'right' or 'left' from where the face points relative to the neck; None when unclear."""
        neck = self.points[NECK]
        face = [p for p in (self.points[NOSE], self.points[REYE], self.points[LEYE]) if p is not None]
        if neck is None or not face:
            return None
        fx = sum(p[0] for p in face) / len(face)
        ears = [p for p in (self.points[REAR], self.points[LEAR]) if p is not None]
        ref = sum(p[0] for p in ears) / len(ears) if ears else neck[0]
        d = fx - ref
        if abs(d) < 2.0:
            return None
        return "right" if d > 0 else "left"


def bone_length(s: Skeleton, a: int, b: int) -> float | None:
    pa, pb = s.points[a], s.points[b]
    if pa is None or pb is None:
        return None
    return math.hypot(pb[0] - pa[0], pb[1] - pa[1])


@dataclass
class Proportions:
    """Bone lengths of one character, in canvas pixels, plus where its feet stand."""

    lengths: dict[tuple[int, int], float]
    ankle_height: float     # distance from the ankle points to the foot line
    height: float           # figure height in pixels

    def to_json(self) -> dict:
        return {"lengths": {f"{a}-{b}": round(v, 2) for (a, b), v in self.lengths.items()},
                "ankle_height": round(self.ankle_height, 2), "height": round(self.height, 2)}


def measure(reference: Skeleton, template_rest: Skeleton, figure_height: float, foot_line: float) -> Proportions:
    """Read the character's limb lengths from its skeleton.

    A bone the pose reader did not see is taken from its left/right counterpart, and failing that from the rest
    pose of the template scaled to the character's height.
    """
    scale = figure_height / TEMPLATE_HEIGHT
    lengths: dict[tuple[int, int], float] = {}
    for a, b in EDGES:
        L = bone_length(reference, a, b)
        if L is None or L < 1.0:
            ma, mb = _MIRROR.get(a, a), _MIRROR.get(b, b)
            L = bone_length(reference, ma, mb)
        if L is None or L < 1.0:
            tl = bone_length(template_rest, a, b)
            L = (tl or 0.0) * scale
        lengths[(a, b)] = L
    # Paired limbs get the longer of the two readings: in a side view the far limb is foreshortened or hidden.
    for a, b in EDGES:
        ma, mb = _MIRROR.get(a), _MIRROR.get(b)
        if ma is not None and mb is not None and (ma, mb) in lengths and a in (RSHO, RELB, RHIP, RKNE):
            m = max(lengths[(a, b)], lengths[(ma, mb)])
            lengths[(a, b)] = lengths[(ma, mb)] = m
    low = reference.lowest_ankle_y()
    ankle_h = (foot_line - low) if low is not None else 0.04 * figure_height
    ankle_h = float(min(max(ankle_h, 0.0), 0.12 * figure_height))
    return Proportions(lengths=lengths, ankle_height=ankle_h, height=float(figure_height))


def fit_frame(template_frame: Skeleton, template_rest: Skeleton, props: Proportions, anchor: tuple[float, float],
              grounded: bool) -> Skeleton:
    """Fit one template pose to a character.

    Bone directions come from the template, bone lengths from the character. The result is anchored at the same
    foot point in every frame (§6.3): horizontally the pelvis keeps the template's offset from the anchor, and
    vertically the lowest ankle stands on the foot line (grounded frames) or keeps the template's height above it
    (frames in the air).

    Template units: the anchor is (0, 0) and y points down, so the ground is y = 0.
    """
    scale = props.height / TEMPLATE_HEIGHT

    pts: list[Point] = [None] * N
    neck_t = template_frame.points[NECK]
    if neck_t is None:
        raise ValueError("template frame has no neck point")
    pts[NECK] = (0.0, 0.0)
    for a, b in EDGES:
        pa_t, pb_t = template_frame.points[a], template_frame.points[b]
        if pb_t is None or pts[a] is None or pa_t is None:
            continue
        dx, dy = pb_t[0] - pa_t[0], pb_t[1] - pa_t[1]
        d = math.hypot(dx, dy)
        L = props.lengths[(a, b)]
        if d < 1e-9:
            pts[b] = pts[a]
            continue
        # Keep the template's foreshortening: a bone drawn shorter than at rest (pointing toward the viewer)
        # stays proportionally shorter.
        rest_d = bone_length(template_rest, a, b) or d
        ratio = min(d / rest_d, 1.25) if rest_d > 1e-9 else 1.0
        pa = pts[a]
        pts[b] = (pa[0] + dx / d * L * ratio, pa[1] + dy / d * L * ratio)
    fitted = Skeleton(pts)

    # Horizontal anchor: pelvis at the template's pelvis offset, scaled.
    p_t = template_frame.pelvis()
    p_f = fitted.pelvis()
    assert p_t is not None and p_f is not None
    target_x = anchor[0] + p_t[0] * scale
    dx = target_x - p_f[0]

    # Vertical anchor: how far the template's lowest ankle is above its resting ankle height.
    rest_ankle = template_rest.lowest_ankle_y()
    frame_ankle = template_frame.lowest_ankle_y()
    clearance_t = 0.0
    if not grounded and rest_ankle is not None and frame_ankle is not None:
        clearance_t = max(0.0, rest_ankle - frame_ankle)
    low = fitted.lowest_ankle_y()
    assert low is not None
    target_low = anchor[1] - props.ankle_height - clearance_t * scale
    dy = target_low - low
    return fitted.translated(dx, dy)


def draw_openpose(skel: Skeleton, size: tuple[int, int], canvas: np.ndarray | None = None,
                  stick_width: int | None = None) -> np.ndarray:
    """Draw a skeleton the way the OpenPose preprocessor does (black background, coloured limbs). RGB uint8."""
    w, h = size
    img = np.zeros((h, w, 3), np.uint8) if canvas is None else canvas
    sw = stick_width or max(2, int(round(4 * h / 512)))
    for i, (a, b) in enumerate(_LIMBS):
        pa, pb = skel.points[a], skel.points[b]
        if pa is None or pb is None:
            continue
        mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        length = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        angle = math.degrees(math.atan2(pa[1] - pb[1], pa[0] - pb[0]))
        poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), sw), int(angle), 0, 360, 1)
        overlay = img.copy()
        cv2.fillConvexPoly(overlay, poly, _COLOURS[i])
        img[:] = (img * 0.4 + overlay * 0.6).astype(np.uint8)
    for i, p in enumerate(skel.points):
        if p is None:
            continue
        cv2.circle(img, (int(p[0]), int(p[1])), sw, _COLOURS[i], thickness=-1)
    return img


def hand_point(skel: Skeleton, hand: str) -> tuple[float, float] | None:
    """The middle of the closed hand ('r' near, 'l' far): the wrist pushed a little further along the forearm."""
    from . import config
    elb, wri = (skel.points[RELB], skel.points[RWRI]) if hand == "r" else (skel.points[LELB], skel.points[LWRI])
    if wri is None:
        return None
    if elb is None:
        return wri
    k = config.HAND_FROM_WRIST
    return (wri[0] + (wri[0] - elb[0]) * k, wri[1] + (wri[1] - elb[1]) * k)
