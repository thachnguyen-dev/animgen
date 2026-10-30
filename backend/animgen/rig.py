"""Layered cut-out character (owner, 2026-10-30): the character is made once from layers and every frame is the layers
moved onto the template's skeleton, with no AI drawing per frame. Face and hair always come from the owner's image.

Layers:
- head and body from the owner's image (the original pixels); where the near arm hid the body, pixels from the rig
  image fill in;
- the limbs from the rig image: one AI drawing of the character in a spread pose (near arm forward, far arm back,
  wide stride, closed fists), where the legs do not cover each other, so the far leg has pixels of its own;
- the far arm is a darker copy of the near arm: in a side view it is behind the body even in the rig image, and its
  cut picked up the quiver or the cape instead (prototype, 2026-10-30).

All images are canvas-sized RGBA facing right (as the templates); skeletons are canvas coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .puppet import PART_GROW, _cut, _drop_specks, _reference_bones, _segment_distance, _similarity, _target_bones
from .skeleton import Skeleton

# Limb parts and the bones they hang on (names as puppet._target_bones).
LIMBS = ("uarm", "larm", "uleg", "lleg", "uarm_far", "larm_far", "uleg_far", "lleg_far")
# Back to front. The body goes over the legs (a skirt or tunic covers the hips), the near arm over everything.
DRAW_ORDER = ("uarm_far", "larm_far", "uleg_far", "lleg_far", "uleg", "lleg", "torso", "head", "uarm", "larm")
FAR_ARM_SOURCE = {"uarm_far": "uarm", "larm_far": "larm"}
FAR_DARKEN = 0.85         # far limbs a little darker, as they are further from the light and the viewer
# Pixel claim weights for cutting the rig image: lower claims more. The body is wide, limbs are narrow.
CUT_WEIGHT = {"torso": 0.55, "head": 0.6, "uarm": 1.0, "larm": 1.0, "uleg": 0.9, "lleg": 0.9}
LIMB_RADIUS = 0.09        # of the figure height: a limb pixel further than this from its bone stays with the body


@dataclass
class Layer:
    name: str
    rgba: np.ndarray                    # canvas-sized, only this part's pixels
    bone: tuple                         # (a, b) in the layer's own image


@dataclass
class Rig:
    layers: dict[str, Layer]
    size: tuple[int, int]

    def render(self, target: Skeleton) -> Image.Image:
        """The character in the target pose (RGBA, canvas-sized, transparent around)."""
        w, h = self.size
        out = np.zeros((h, w, 4), np.uint8)
        bones = _target_bones(target)
        for name in DRAW_ORDER:
            layer = self.layers.get(name)
            a2, b2 = bones.get(name, (None, None))
            if layer is None or a2 is None or b2 is None:
                continue
            a, b = layer.bone
            if name == "head":   # upright, follows the neck
                mat = np.array([[1.0, 0.0, a2[0] - a[0]], [0.0, 1.0, a2[1] - a[1]]])
            else:
                mat = _similarity(a, b, a2, b2)
            moved = cv2.warpAffine(layer.rgba, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            k = moved[..., 3:4].astype(np.float32) / 255
            out[..., :3] = (moved[..., :3] * k + out[..., :3] * (1 - k)).astype(np.uint8)
            out[..., 3] = np.maximum(out[..., 3], moved[..., 3])
        _drop_specks(out)
        return Image.fromarray(out, "RGBA")


def _figure_height(sk: Skeleton) -> float:
    ys = [p[1] for p in sk.points if p is not None]
    return max(ys) - min(ys)


def cut_limbs(rig_image: Image.Image, rig_sk: Skeleton) -> dict[str, np.ndarray]:
    """Masks of the eight limb parts and the body in the rig image, by nearest bone."""
    arr = np.array(rig_image.convert("RGBA"))
    h, w = arr.shape[:2]
    alpha = arr[..., 3] > 20
    bones = {k: v for k, v in _target_bones(rig_sk).items() if v[0] is not None and v[1] is not None}
    names = list(bones)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    raw = np.stack([_segment_distance(xs, ys, *bones[n]) for n in names])
    weighted = np.stack([raw[i] * CUT_WEIGHT[n.replace("_far", "")] for i, n in enumerate(names)])
    label = np.argmin(weighted, axis=0)
    far_from_bone = np.take_along_axis(raw, label[None], 0)[0] > LIMB_RADIUS * _figure_height(rig_sk)
    masks = {}
    for i, n in enumerate(names):
        m = alpha & (label == i)
        if n in LIMBS:
            m &= ~far_from_bone
        masks[n] = m
    masks["torso"] = alpha & ~np.any([masks[n] for n in LIMBS if n in masks], axis=0)
    return masks


def _grown(mask: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (cv2.dilate(mask.astype(np.uint8), np.ones((PART_GROW, PART_GROW), np.uint8)) > 0) & alpha


def build_rig(original: Image.Image, original_sk: Skeleton, rig_image: Image.Image, rig_sk: Skeleton) -> Rig:
    orig = np.array(original.convert("RGBA"))
    rig = np.array(rig_image.convert("RGBA"))
    h, w = orig.shape[:2]
    layers: dict[str, Layer] = {}

    # Head and body: the owner's pixels. The near arm is cut away (it comes from the rig image).
    ref_bones = {k: v for k, v in _reference_bones(original_sk).items() if v[0] is not None and v[1] is not None}
    names, label, alpha, _ = _cut(orig, original_sk, ref_bones)
    head = _grown(label == names.index("head"), alpha)
    body = _grown(label == names.index("torso"), alpha)
    layers["head"] = Layer("head", np.where(head[..., None], orig, 0).astype(np.uint8), ref_bones["head"])

    # Where the arm hid the body, the rig image's body shows through (moved onto the original's torso bone).
    rig_masks = cut_limbs(rig_image, rig_sk)
    rig_bones = _target_bones(rig_sk)
    torso_rgba = np.where(body[..., None], orig, 0).astype(np.uint8)
    rig_body = np.where(rig_masks["torso"][..., None], rig, 0).astype(np.uint8)
    mat = _similarity(*rig_bones["torso"], *ref_bones["torso"])
    rig_body = cv2.warpAffine(rig_body, mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    arms = np.isin(label, [names.index(a) for a in ("uarm", "larm") if a in names]) & alpha
    hidden = arms & (torso_rgba[..., 3] == 0) & (rig_body[..., 3] > 127)
    torso_rgba[hidden] = rig_body[hidden]
    layers["torso"] = Layer("torso", torso_rgba, ref_bones["torso"])

    # Limbs: the rig image's pixels; the far arm copies the near one.
    ralpha = rig[..., 3] > 20
    for n in LIMBS:
        src = FAR_ARM_SOURCE.get(n, n)
        a, b = rig_bones.get(src, (None, None))
        if a is None or b is None or not rig_masks.get(src, np.zeros(1, bool)).any():
            continue
        part = np.where(_grown(rig_masks[src], ralpha)[..., None], rig, 0).astype(np.uint8)
        if n.endswith("_far"):
            part[..., :3] = (part[..., :3] * FAR_DARKEN).astype(np.uint8)
        layers[n] = Layer(n, part, (a, b))
    return Rig(layers, (w, h))
