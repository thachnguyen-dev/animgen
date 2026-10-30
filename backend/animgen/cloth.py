"""Hanging parts that swing (owner, 2026-10-30: capes and long garments need their own bones): a cape or anything
else hanging behind the back, a skirt or tunic tail below the hips, long hair below the neck.

Each is found in the bind pose (rig.py) and hangs from a pivot: the upper back, the hips, the neck. Over an
animation it swings as a damped pendulum, pushed by how the pivot moves and by the air when the character runs. The
swing bends the part gradually: nothing at the pivot, the full angle at the hem, so it stays attached. Templates
walk and run in place, so the ground speed is read from how fast the planted foot slides back.

Angles are in degrees, positive = the hem swings toward the back (the character faces right, canvas y down).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .skeleton import LANK, RANK, Skeleton

FPS = 12                  # nominal playback rate for the physics (the game sets the real one)
REAL_HEIGHT = 1.75        # metres: the figure's height, to scale gravity to pixels
CAPE_DEPTH = 0.07         # of the figure height: body pixels further behind the spine than this hang from the back
CAPE_TOP = 0.15           # of the torso length below the neck: the cape's pivot
SKIRT_START = 1.02        # of the torso length below the neck: body pixels lower than this hang from the hips
HAIR_START = 0.03         # of the figure height below the neck: head pixels lower and behind the neck are long hair
HAIR_LENGTH = 0.12        # of the figure height: shorter "hair" is a beard, a chin or a helmet's rim and stays still
MIN_SHARE = 0.01          # a hanging part smaller than this share of the figure is left still
EDGE_BLUR = 0.015         # of the figure height: the swing fades out over this across the part's border
SWING_LENGTH = 0.6        # a hanging cloth swings like a pendulum this fraction of its length
DAMPING = 0.35            # of critical damping
WIND = 0.12               # trailing angle (tangent) per (figure heights per second)², as air drag grows with speed²:
                          # about 12° walking, 30° running
WIND_SHARE = {"cape": 1.0, "skirt": 0.4, "hair": 0.8}   # a skirt wraps the legs and catches less air
SUBSTEPS = 20
LOOP_CYCLES = 4           # a looping animation is simulated this many times; the last one is used
LIMITS = {"cape": (-10.0, 70.0), "skirt": (-25.0, 25.0), "hair": (-10.0, 50.0)}


@dataclass
class Hanging:
    name: str                   # "cape", "skirt" or "hair"
    bone: str                   # bind bone that carries the pivot: "torso" or "head"
    pivot: np.ndarray           # bind pose
    length: float               # px from the pivot to the hem
    factor: np.ndarray          # per vertex of its mesh: 0 = still, 1 = turns the full swing angle


def find_hanging(verts: np.ndarray, torso_mask: np.ndarray, head_mask: np.ndarray, bind: dict,
                 figure: float) -> list[Hanging]:
    """The hanging parts of the body mesh (torso and head pixels in the bind pose)."""
    a, b = (np.asarray(p, float) for p in bind["torso"])
    length = float(np.linalg.norm(b - a)) + 1e-6
    d = (b - a) / length                                  # down the body
    n = np.array([-d[1], d[0]])                           # toward the back
    h, w = torso_mask.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    rel_x, rel_y = xs - a[0], ys - a[1]
    along = (rel_x * d[0] + rel_y * d[1]) / length
    back = rel_x * n[0] + rel_y * n[1]
    total = float(torso_mask.sum() + head_mask.sum())
    body = (torso_mask | head_mask).astype(np.float32)

    found = []
    cape = torso_mask & (back > CAPE_DEPTH * figure) & (along > CAPE_TOP)
    skirt = torso_mask & (along > SKIRT_START) & ~cape
    hair = head_mask & (along * length > HAIR_START * figure) & (back > 0)
    for name, mask, bone, pivot in (
            ("cape", cape, "torso", a + n * CAPE_DEPTH * figure + d * CAPE_TOP * length),
            ("skirt", skirt, "torso", b),
            ("hair", hair, "head", a)):
        if mask.sum() < MIN_SHARE * total:
            continue
        py, px = np.nonzero(mask)
        reach = float(((px - pivot[0]) * d[0] + (py - pivot[1]) * d[1]).max())
        if reach <= 1 or (name == "hair" and reach < HAIR_LENGTH * figure):
            continue
        # Normalised over the figure's own pixels: a vertex just outside the outline goes with the part beside it
        # (knight test, 2026-10-30: with a plain blur the hem's outer vertices stayed behind and combed the edge).
        sigma = EDGE_BLUR * figure
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) / (
            cv2.GaussianBlur(body, (0, 0), sigma) + 1e-6)
        vx = np.clip(verts[:, 0].astype(int), 0, w - 1)
        vy = np.clip(verts[:, 1].astype(int), 0, h - 1)
        s = np.clip(((verts - pivot) @ d) / reach, 0, 1)
        found.append(Hanging(name, bone, pivot, reach, soft[vy, vx] * s * s * (3 - 2 * s)))
    return found


def bend(verts: np.ndarray, hanging: Hanging, angle: float) -> np.ndarray:
    """The bind-pose vertices with the part turned ``angle`` degrees about its pivot, gradually toward the hem."""
    t = np.radians(angle) * hanging.factor
    c, s = np.cos(t), np.sin(t)
    rel = verts - hanging.pivot
    out = verts.copy()
    out[:, 0] = hanging.pivot[0] + c * rel[:, 0] - s * rel[:, 1]
    out[:, 1] = hanging.pivot[1] + s * rel[:, 0] + c * rel[:, 1]
    return out


def ground_speed(targets: list[Skeleton], figure: float) -> float:
    """How fast the character would move over the ground, in figure heights per second: the planted (lower) foot
    slides back in a cycle that walks in place. 0 when the feet stay put."""
    steps = []
    for s1, s2 in zip(targets, targets[1:] + targets[:1]):
        feet = [(s1.points[i], s2.points[i]) for i in (RANK, LANK) if s1.points[i] and s2.points[i]]
        if not feet:
            continue
        p1, p2 = max(feet, key=lambda f: f[0][1] + f[1][1])
        steps.append(p1[0] - p2[0])
    if not steps:
        return 0.0
    return max(0.0, float(np.median(steps))) * FPS / figure


def simulate(pivots: list[np.ndarray], carried: list[float], hanging: Hanging, figure: float, speed: float,
             loop: bool) -> list[float]:
    """The part's swing angle in each frame, relative to the bone that carries it (what ``bend`` takes).

    ``pivots``: the pivot's position in each frame; ``carried``: how far the carrying bone turned from the bind pose
    in each frame (degrees, same sign as the swing)."""
    n = len(pivots)
    if n == 0:
        return []
    dt = 1.0 / FPS
    g = 9.8 * figure / REAL_HEIGHT
    arm = SWING_LENGTH * hanging.length
    omega = math.sqrt(g / arm)
    share = WIND_SHARE[hanging.name]

    def wind(v: float) -> float:                          # px/s² of push toward the back, for speed v px/s forward
        return share * WIND * g * (v / figure) * abs(v / figure)

    p = np.array(pivots, float)
    idx = np.arange(n)
    nxt, prv = (idx + 1) % n, (idx - 1) % n
    if not loop:
        nxt, prv = np.minimum(idx + 1, n - 1), np.maximum(idx - 1, 0)
    vel = (p[nxt] - p[prv]) / (2 * dt)
    acc = (p[nxt] - 2 * p + p[prv]) / (dt * dt)
    lo, hi = LIMITS[hanging.name]
    run = speed * figure

    phi = math.atan(wind(run) / g)                        # start where the steady wind would hold it
    rate = 0.0
    out = []
    for cycle in range(LOOP_CYCLES if loop else 1):
        out = []
        for i in range(n):
            out.append(phi)
            j = (i + 1) % n if loop else min(i + 1, n - 1)
            for k in range(SUBSTEPS):
                f = k / SUBSTEPS
                ax = acc[i, 0] * (1 - f) + acc[j, 0] * f
                ay = acc[i, 1] * (1 - f) + acc[j, 1] * f
                vx = vel[i, 0] * (1 - f) + vel[j, 0] * f
                push = (ax + wind(run + vx)) * math.cos(phi) + (ay - g) * math.sin(phi)
                rate += (push / arm - 2 * DAMPING * omega * rate) * dt / SUBSTEPS
                phi += rate * dt / SUBSTEPS
                if not math.radians(lo) < phi < math.radians(hi):
                    phi = min(max(phi, math.radians(lo)), math.radians(hi))
                    rate = 0.0
    return [float(np.clip(math.degrees(x) - c, lo, hi)) for x, c in zip(out, carried)]
