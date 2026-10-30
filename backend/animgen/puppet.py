"""Rough reposing of the reference character by its skeleton: the start image for each AI frame.

Local finalisation, 2026-10-30: drawing every frame from noise (strength 1.0) kept the pose but not the character
(outfit, line style and colours drifted, IP-Adapter alone could not hold them). Starting from this rough repose and
letting the image model repaint only part of it (``diffusers_engine.STRENGTH``) keeps the owner's own artwork.

The figure is cut into head, torso, upper/lower arm and upper/lower leg by the nearest bone; each part is moved with a
similarity transform from its reference bone to the target bone. Characters are generated facing right, so the right
limbs are the near ones. In a side view the far limbs are mostly hidden behind the near ones, so they are drawn as
darker copies of the near limbs, behind the body; the image model then repaints the joints and the far side.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .skeleton import LANK, LELB, LHIP, LKNE, LSHO, LWRI, NECK, NOSE, RANK, RELB, RHIP, RKNE, RSHO, RWRI, Skeleton

Point = tuple[float, float]

FAR_DARKEN = 0.72          # far limbs: colour multiplied by this
PART_GROW = 5              # px: each part's mask is grown a little so the moved parts overlap at the joints
# Distance weights for cutting: below 1 a part claims pixels from farther away.
WEIGHT = {"head": 0.6, "torso": 1.0, "uarm": 0.85, "larm": 0.75, "uleg": 1.0, "lleg": 1.0}
# In a standing side view the arm hangs exactly over the torso, so nearest-bone cutting ties and the hand stays behind
# on the hip. Arms therefore claim every pixel within this share of the skeleton's height from their bone. Where the
# arm then moves away, the gap it leaves in the torso is filled from the surrounding torso colours (cv2.inpaint);
# keeping a copy of the arm on the torso instead risked a duplicate arm.
ARM_RADIUS = 0.035
FILL_RADIUS = 5
DRAW_ORDER_FAR = (("uleg_far", "uleg"), ("lleg_far", "lleg"), ("uarm_far", "uarm"), ("larm_far", "larm"))
DRAW_ORDER_NEAR = ("uleg", "lleg", "torso", "head", "uarm", "larm")


def _mid(p: Point | None, q: Point | None) -> Point | None:
    if p is None or q is None:
        return p or q
    return ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)


def _reference_bones(sk: Skeleton) -> dict[str, tuple[Point | None, Point | None]]:
    p = sk.points
    hip = _mid(p[RHIP], p[LHIP])
    return {
        "head": (p[NECK], p[NOSE]),
        "torso": (p[NECK], hip),
        "uarm": (p[RSHO] or p[LSHO], p[RELB] or p[LELB]),
        "larm": (p[RELB] or p[LELB], p[RWRI] or p[LWRI]),
        "uleg": (p[RHIP] or hip, p[RKNE] or p[LKNE]),
        "lleg": (p[RKNE] or p[LKNE], p[RANK] or p[LANK]),
    }


def _target_bones(sk: Skeleton) -> dict[str, tuple[Point | None, Point | None]]:
    p = sk.points
    return {
        "head": (p[NECK], p[NOSE]), "torso": (p[NECK], _mid(p[RHIP], p[LHIP])),
        "uarm": (p[RSHO], p[RELB]), "larm": (p[RELB], p[RWRI]),
        "uleg": (p[RHIP], p[RKNE]), "lleg": (p[RKNE], p[RANK]),
        "uarm_far": (p[LSHO], p[LELB]), "larm_far": (p[LELB], p[LWRI]),
        "uleg_far": (p[LHIP], p[LKNE]), "lleg_far": (p[LKNE], p[LANK]),
    }


def _segment_distance(xs: np.ndarray, ys: np.ndarray, a: Point, b: Point) -> np.ndarray:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = np.clip(((xs - a[0]) * dx + (ys - a[1]) * dy) / (dx * dx + dy * dy + 1e-6), 0, 1)
    return np.hypot(xs - (a[0] + t * dx), ys - (a[1] + t * dy))


def _similarity(a: Point, b: Point, a2: Point, b2: Point) -> np.ndarray:
    """2×3 matrix taking bone a→b onto a2→b2 (rotation, a length change kept within ±25 %, translation)."""
    v1 = np.subtract(b, a, dtype=float)
    v2 = np.subtract(b2, a2, dtype=float)
    s = float(np.clip(np.linalg.norm(v2) / (np.linalg.norm(v1) + 1e-6), 0.8, 1.25))
    ang = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    m = np.array([[np.cos(ang) * s, -np.sin(ang) * s, 0.0], [np.sin(ang) * s, np.cos(ang) * s, 0.0]])
    m[:, 2] = np.asarray(a2, float) - m[:, :2] @ np.asarray(a, float)
    return m


def repose(reference: Image.Image, ref_skeleton: Skeleton, target: Skeleton) -> Image.Image:
    """The reference figure (RGBA, facing right) moved onto ``target``; same canvas size, transparent around."""
    arr = np.array(reference.convert("RGBA"))
    h, w = arr.shape[:2]
    alpha = arr[..., 3] > 20
    ref_bones = {k: v for k, v in _reference_bones(ref_skeleton).items() if v[0] is not None and v[1] is not None}
    tgt_bones = _target_bones(target)
    if "torso" not in ref_bones or tgt_bones["torso"][0] is None or tgt_bones["torso"][1] is None:
        return reference.convert("RGBA").copy()
    names = list(ref_bones)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.stack([_segment_distance(xs, ys, *ref_bones[n]) * WEIGHT[n] for n in names])
    label = np.argmin(dist, axis=0)
    ys_all = [pt[1] for pt in ref_skeleton.points if pt is not None]
    radius = ARM_RADIUS * (max(ys_all) - min(ys_all))
    torso_extra = np.zeros((h, w), bool)
    for arm in ("larm", "uarm"):  # forearm last so it wins where both are close
        if arm in ref_bones:
            near = _segment_distance(xs, ys, *ref_bones[arm]) < radius
            torso_extra |= near & (label == names.index("torso"))
            label[near] = names.index(arm)
    out = np.zeros_like(arr)
    grow = np.ones((PART_GROW, PART_GROW), np.uint8)

    def paste(target_name: str, part: str, darken: float = 1.0) -> None:
        a2, b2 = tgt_bones.get(target_name, (None, None))
        if part not in ref_bones or a2 is None or b2 is None:
            return
        m = (cv2.dilate(((label == names.index(part)) & alpha).astype(np.uint8), grow) > 0) & alpha
        src = arr.copy()
        src[~m] = 0
        if darken != 1.0:
            src[..., :3] = (src[..., :3] * darken).astype(np.uint8)
        a, b = ref_bones[part]
        if part == "head":  # the head stays upright and follows the neck
            mat = np.array([[1.0, 0.0, a2[0] - a[0]], [0.0, 1.0, a2[1] - a[1]]])
        else:
            mat = _similarity(a, b, a2, b2)
        moved = cv2.warpAffine(src, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        k = moved[..., 3:4].astype(np.float32) / 255
        out[..., :3] = (moved[..., :3] * k + out[..., :3] * (1 - k)).astype(np.uint8)
        out[..., 3] = np.maximum(out[..., 3], moved[..., 3])

    for far, part in DRAW_ORDER_FAR:
        paste(far, part, FAR_DARKEN)
    for part in DRAW_ORDER_NEAR:
        if part == "uarm":  # torso, head and legs are in place: fill what the arm left uncovered in the torso
            _fill_torso_gap(out, torso_extra, ref_bones["torso"], tgt_bones["torso"])
        paste(part, part)
    return Image.fromarray(out, "RGBA")


def _fill_torso_gap(out: np.ndarray, gap_ref: np.ndarray, bone: tuple, bone_target: tuple) -> None:
    """Inpaint the torso pixels the arm took, at the torso's new place, where nothing covers them yet."""
    if not gap_ref.any():
        return
    h, w = gap_ref.shape
    mat = _similarity(bone[0], bone[1], bone_target[0], bone_target[1])
    gap = cv2.warpAffine(gap_ref.astype(np.uint8) * 255, mat, (w, h), flags=cv2.INTER_NEAREST) > 0
    gap &= out[..., 3] < 128
    if not gap.any():
        return
    filled = cv2.inpaint(out[..., :3].copy(), gap.astype(np.uint8), FILL_RADIUS, cv2.INPAINT_TELEA)
    out[gap, :3] = filled[gap]
    out[gap, 3] = 255
