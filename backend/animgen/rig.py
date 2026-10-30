"""Layered cut-out character (owner, 2026-10-30): the character is made once and every frame is that character bent
onto the template's skeleton, with no AI drawing per frame. Face and hair always come from the owner's image.

Pieces:
- head and body from the owner's image (the original pixels); where the near arm hid the body, pixels from the rig
  image fill in;
- the limbs from the rig image: one AI drawing of the character in a spread pose (near arm forward, far arm back,
  wide stride, closed fists), where the legs do not cover each other, so the far leg has pixels of its own;
- the far arm is a darker copy of the near arm: in a side view it is behind the body even in the rig image, and its
  cut picked up the quiver or the cape instead (prototype, 2026-10-30).

Bending (owner, 2026-10-30: no seams at the joints, the head moves): the pieces are put together once in the rig
image's pose (the bind pose) as five meshes, back to front: far arm, far leg, near leg, body (torso and head), near
arm. A limb mesh bends across its elbow or knee: its vertices blend the two bones within JOINT_BLEND of the joint
and move rigidly elsewhere. The head is rigid and turns with the template's head bone (spine lean and head tilt),
within HEAD_TURN; only the body just around it follows it partly (the neck). Shoulders and hips are not blended: the
arm is drawn over the shoulder and the body over the hip, and blending them pulled the tunic and the cape along with
the legs (knight test, 2026-10-30: smears at the feet and the hips).

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
FAR_DARKEN = 0.85         # far limbs a little darker, as they are further from the light and the viewer
# Pixel claim weights for cutting the rig image: lower claims more. The body is wide, limbs are narrow.
CUT_WEIGHT = {"torso": 0.55, "head": 0.6, "uarm": 1.0, "larm": 1.0, "uleg": 0.9, "lleg": 0.9}
LIMB_RADIUS = 0.09        # of the figure height: a limb pixel further than this from its bone stays with the body
FILL_LOCAL = 0.04         # of the figure height: how far around the owner's body colours are averaged for a fill
FILL_MATCH = 70.0         # RGB distance: a rig pixel further than this from that average is not used for the fill
MESH_STEP = 6            # px between mesh vertices
JOINT_BLEND = 0.03        # of the figure height: how far across a border between parts the two bones blend
HEAD_TURN = 30.0          # degrees: the most the head turns away from the bind pose

# The meshes, back to front: name, its two pieces in drawing order, parent first (the piece, the bind bone it hangs
# on, the frame bone that moves it), and whether it is darkened as far.
GROUPS = (
    ("arm_far", (("uarm", "uarm", "uarm_far"), ("larm", "larm", "larm_far")), True),
    ("leg_far", (("uleg_far", "uleg_far", "uleg_far"), ("lleg_far", "lleg_far", "lleg_far")), True),
    ("leg", (("uleg", "uleg", "uleg"), ("lleg", "lleg", "lleg")), False),
    ("body", (("torso", "torso", "torso"), ("head", "head", "head")), False),
    ("arm", (("uarm", "uarm", "uarm"), ("larm", "larm", "larm")), False),
)


@dataclass
class Mesh:
    name: str
    rgba: np.ndarray                    # canvas-sized, in the bind pose
    verts: np.ndarray                   # (n, 2) bind-pose vertex positions
    weights: np.ndarray                 # (n, k) per vertex, per bone; rows sum to 1
    tris: np.ndarray                    # (m, 3) vertex indices, in drawing order (back first)
    bones: tuple                        # k × (bind bone, frame bone)


@dataclass
class Rig:
    meshes: list[Mesh]
    bind: dict                          # bind bone name -> (a, b)
    size: tuple[int, int]

    def render(self, target: Skeleton) -> Image.Image:
        """The character in the target pose (RGBA, canvas-sized, transparent around)."""
        w, h = self.size
        out = np.zeros((h, w, 4), np.uint8)
        frame = _target_bones(target)
        for mesh in self.meshes:
            mats = [self._matrix(b, f, frame) for b, f in mesh.bones]
            if any(m is None for m in mats):
                continue
            moved = _deform(mesh, mats, (h, w))
            k = moved[..., 3:4].astype(np.float32) / 255
            out[..., :3] = (moved[..., :3] * k + out[..., :3] * (1 - k)).astype(np.uint8)
            out[..., 3] = np.maximum(out[..., 3], moved[..., 3])
        _drop_specks(out)
        return Image.fromarray(out, "RGBA")

    def _matrix(self, bind_bone: str, frame_bone: str, frame: dict) -> np.ndarray | None:
        a, b = self.bind[bind_bone]
        a2, b2 = frame.get(frame_bone, (None, None))
        if a2 is None or b2 is None:
            return None
        if bind_bone == "head":
            return _head_matrix(a, b, a2, b2)
        return _similarity(a, b, a2, b2)


def _head_matrix(a, b, a2, b2) -> np.ndarray:
    """The head turns about the neck with its bone, within HEAD_TURN, and keeps its size."""
    v1, v2 = np.subtract(b, a, dtype=float), np.subtract(b2, a2, dtype=float)
    turn = np.degrees(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))
    ang = np.radians(np.clip((turn + 180) % 360 - 180, -HEAD_TURN, HEAD_TURN))
    m = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0]])
    m[:, 2] = np.asarray(a2, float) - m[:, :2] @ np.asarray(a, float)
    return m


def _deform(mesh: Mesh, mats: list[np.ndarray], size: tuple[int, int]) -> np.ndarray:
    """The mesh's image bent by its bones: each vertex moves by the weighted blend of its bones' moves, then every
    pixel of each moved triangle takes the colour of the same place in the bind-pose triangle."""
    h, w = size
    homo = np.hstack([mesh.verts, np.ones((len(mesh.verts), 1))])
    moved = np.einsum("nk,kij,nj->ni", mesh.weights, np.stack(mats), homo)
    ids = np.full((h, w), -1, np.float32)
    for t, tri in enumerate(mesh.tris):
        cv2.fillConvexPoly(ids, np.round(moved[tri] * 16).astype(np.int32), float(t), cv2.LINE_8, 4)
    ys, xs = np.nonzero(ids >= 0)
    if len(xs) == 0:
        return np.zeros((h, w, 4), np.uint8)
    tri = mesh.tris[ids[ys, xs].astype(np.int64)]
    p0, p1, p2 = moved[tri[:, 0]], moved[tri[:, 1]], moved[tri[:, 2]]
    det = (p1[:, 1] - p2[:, 1]) * (p0[:, 0] - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (p0[:, 1] - p2[:, 1])
    det = np.where(np.abs(det) < 1e-6, 1e-6, det)
    l0 = ((p1[:, 1] - p2[:, 1]) * (xs - p2[:, 0]) + (p2[:, 0] - p1[:, 0]) * (ys - p2[:, 1])) / det
    l1 = ((p2[:, 1] - p0[:, 1]) * (xs - p2[:, 0]) + (p0[:, 0] - p2[:, 0]) * (ys - p2[:, 1])) / det
    l2 = 1 - l0 - l1
    src = (l0[:, None] * mesh.verts[tri[:, 0]] + l1[:, None] * mesh.verts[tri[:, 1]]
           + l2[:, None] * mesh.verts[tri[:, 2]])
    mapx = np.full((h, w), -1, np.float32)
    mapy = np.full((h, w), -1, np.float32)
    mapx[ys, xs], mapy[ys, xs] = src[:, 0], src[:, 1]
    return cv2.remap(mesh.rgba, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


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


def _pieces(original: Image.Image, original_sk: Skeleton, rig_image: Image.Image,
            rig_sk: Skeleton) -> dict[str, np.ndarray]:
    """Every piece as a canvas-sized RGBA image in the bind pose (the rig image's)."""
    orig = np.array(original.convert("RGBA"))
    rig = np.array(rig_image.convert("RGBA"))
    h, w = orig.shape[:2]
    rig_bones = _target_bones(rig_sk)
    pieces: dict[str, np.ndarray] = {}

    # Head and body: the owner's pixels. The near arm is cut away (it comes from the rig image).
    ref_bones = {k: v for k, v in _reference_bones(original_sk).items() if v[0] is not None and v[1] is not None}
    names, label, alpha, _ = _cut(orig, original_sk, ref_bones)
    head = _grown(label == names.index("head"), alpha)
    body = _grown(label == names.index("torso"), alpha)

    # Where the arm hid the body, the rig image's body shows through (moved onto the original's torso bone).
    rig_masks = cut_limbs(rig_image, rig_sk)
    torso_rgba = np.where(body[..., None], orig, 0).astype(np.uint8)
    rig_body = np.where(rig_masks["torso"][..., None], rig, 0).astype(np.uint8)
    to_orig = _similarity(*rig_bones["torso"], *ref_bones["torso"])
    rig_body = cv2.warpAffine(rig_body, to_orig, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    arms = np.isin(label, [names.index(a) for a in ("uarm", "larm") if a in names]) & alpha
    hidden = arms & (torso_rgba[..., 3] == 0) & (rig_body[..., 3] > 127)
    # The rig image may draw the back differently (archer test, 2026-10-30: the quiver became a pale bag, which showed
    # as a pale patch on the back): a rig pixel far in colour from the owner's body around it takes that colour.
    have = (torso_rgba[..., 3] > 127).astype(np.float32)
    sigma = FILL_LOCAL * _figure_height(original_sk)
    weight = cv2.GaussianBlur(have, (0, 0), sigma) + 1e-6
    local = np.stack([cv2.GaussianBlur(torso_rgba[..., c].astype(np.float32) * have, (0, 0), sigma) / weight
                      for c in range(3)], axis=-1)
    off = np.linalg.norm(rig_body[..., :3].astype(np.float32) - local, axis=-1) > FILL_MATCH
    torso_rgba[hidden] = rig_body[hidden]
    torso_rgba[hidden & off, :3] = local[hidden & off].astype(np.uint8)

    # Into the bind pose: the body onto the rig image's torso, the head onto its neck (upright, as drawn).
    to_bind = _similarity(*ref_bones["torso"], *rig_bones["torso"])
    pieces["torso"] = cv2.warpAffine(torso_rgba, to_bind, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    (nx, ny), (bx, by) = ref_bones["head"][0], rig_bones["head"][0]
    shift = np.array([[1.0, 0.0, bx - nx], [0.0, 1.0, by - ny]])
    pieces["head"] = cv2.warpAffine(np.where(head[..., None], orig, 0).astype(np.uint8), shift, (w, h),
                                    flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    # Limbs: the rig image's pixels, already in the bind pose.
    ralpha = rig[..., 3] > 20
    for n in LIMBS:
        if n.endswith("_far") and n.replace("_far", "") in ("uarm", "larm"):
            continue                                    # the far arm is drawn from the near arm's pixels
        a, b = rig_bones.get(n, (None, None))
        if a is None or b is None or not rig_masks.get(n, np.zeros(1, bool)).any():
            continue
        pieces[n] = np.where(_grown(rig_masks[n], ralpha)[..., None], rig, 0).astype(np.uint8)
    return pieces


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _child_weight(verts: np.ndarray, child: str, bind: dict, child_mask: np.ndarray, blend: float) -> np.ndarray:
    """How much the child bone moves each vertex. A limb: 0 before the joint, 1 after it, blended over ``blend`` px
    across the joint. The head: 1 on the head, fading out over the neck below it."""
    h, w = child_mask.shape
    if child == "head":
        vx = np.clip(verts[:, 0].astype(int), 0, w - 1)
        vy = np.clip(verts[:, 1].astype(int), 0, h - 1)
        near = cv2.GaussianBlur(child_mask.astype(np.float32), (0, 0), blend / 2)[vy, vx]
        return np.where(child_mask[vy, vx], 1.0, np.clip(2 * near, 0, 1))
    a, b = (np.asarray(p, float) for p in bind[child])
    d = (b - a) / (np.linalg.norm(b - a) + 1e-6)
    t = (verts - a) @ d
    return _smoothstep((t + blend / 2) / blend)


def _mesh(name: str, rgba: np.ndarray, owner: np.ndarray, child_mask: np.ndarray, bones: tuple,
          bind: dict, blend: float) -> Mesh | None:
    """A grid mesh over the image's pixels, moved by a parent and a child bone (see _child_weight)."""
    h, w = rgba.shape[:2]
    alpha = rgba[..., 3] > 0
    if not alpha.any():
        return None
    s = MESH_STEP
    ys, xs = np.nonzero(alpha)
    x0, y0 = max(0, xs.min() - s), max(0, ys.min() - s)
    nx, ny = (xs.max() + 2 * s - x0) // s + 1, (ys.max() + 2 * s - y0) // s + 1
    pad = np.zeros(((ny - 1) * s, (nx - 1) * s), bool)
    part = alpha[y0:y0 + pad.shape[0], x0:x0 + pad.shape[1]]
    pad[:part.shape[0], :part.shape[1]] = part
    cells = pad.reshape(ny - 1, s, nx - 1, s).any(axis=(1, 3))
    gy, gx = np.mgrid[0:ny, 0:nx]
    verts = np.stack([x0 + gx.ravel() * s, y0 + gy.ravel() * s], axis=1).astype(np.float64)

    if len(bones) == 1:
        weights = np.ones((len(verts), 1))
    else:
        c = _child_weight(verts, bones[1][0], bind, child_mask, blend)
        weights = np.stack([1 - c, c], axis=1)

    j, i = np.nonzero(cells)
    v00, v01, v10, v11 = j * nx + i, j * nx + i + 1, (j + 1) * nx + i, (j + 1) * nx + i + 1
    tris = np.concatenate([np.stack([v00, v01, v11], 1), np.stack([v00, v11, v10], 1)])
    # Back first: a triangle is as far front as the piece that owns the pixels around it (a fold puts the forearm
    # over the upper arm, the head over the body).
    cx = np.clip(verts[tris, 0].mean(1).astype(int), 0, w - 1)
    cy = np.clip(verts[tris, 1].mean(1).astype(int), 0, h - 1)
    tris = tris[np.argsort(owner[cy, cx], kind="stable")]
    return Mesh(name, rgba, verts, weights, tris, bones)


def build_rig(original: Image.Image, original_sk: Skeleton, rig_image: Image.Image, rig_sk: Skeleton) -> Rig:
    pieces = _pieces(original, original_sk, rig_image, rig_sk)
    bind = {k: v for k, v in _target_bones(rig_sk).items() if v[0] is not None and v[1] is not None}
    h, w = original.size[1], original.size[0]
    blend = JOINT_BLEND * _figure_height(rig_sk)
    meshes = []
    for name, drawn, far in GROUPS:
        rgba = np.zeros((h, w, 4), np.uint8)
        owner = np.zeros((h, w), np.int32)                 # which piece shows at each pixel, as its drawing order
        bones = []
        for order, (piece, bone, frame_bone) in enumerate(drawn, start=1):
            if piece not in pieces or bone not in bind:
                continue
            p = pieces[piece]
            k = p[..., 3:4].astype(np.float32) / 255
            rgba[..., :3] = (p[..., :3] * k + rgba[..., :3] * (1 - k)).astype(np.uint8)
            rgba[..., 3] = np.maximum(rgba[..., 3], p[..., 3])
            owner[p[..., 3] > 127] = order
            bones.append((bone, frame_bone))
        if not bones:
            continue
        if far:
            rgba[..., :3] = (rgba[..., :3] * FAR_DARKEN).astype(np.uint8)
        mesh = _mesh(name, rgba, owner, owner == 2, tuple(bones), bind, blend)
        if mesh is not None:
            meshes.append(mesh)
    return Rig(meshes, bind, (w, h))
