"""Body-part layers of the character (owner, 2026-10-31): one reference image (the owner's own character image, as
placed on the canvas) and a group of separate body parts made from it, which the owner corrects in Photoshop before
any frame is made.

- The parts are cut from the reference by what is drawn, not by distance to the bones: Segment Anything 2 follows
  the outlines (a pauldron goes with the upper arm, a gauntlet with the forearm), and a clothes-parsing model says
  which pixels are legs (trousers, shoes) and which are garments, so a cape, a skirt, a tabard or a quiver never
  goes with an arm or a leg (owner: "many applications put them with the limbs, and they move with the limbs").
  Garments left over become their own parts by where they hang from: the back (cape), the hips (skirt), behind the
  head (long hair).
- The joints are the owner's confirmed skeleton (``bind_skeleton`` reads them on an AI drawing instead, for rigs
  drawn in a spread pose before 2026-10-31).
- What a part hides is rebuilt by the AI (owner: "keep the visible portion, have the AI reconstruct the missing
  section"): the part's expected outline (a capsule along its bones as wide as the part is drawn) minus what shows of
  it, within the figure, is repainted with the whole character as context, starting from a guess: the near limb moved
  onto a far limb's bones (darker), or the part's own colours spread in.
- The head is cut level at the neck, so the face never comes apart from itself.

The layers are canvas-sized RGBA in the bind pose (the reference's), facing right. ``write_psd`` and ``read_psd`` turn
them into one Photoshop file facing as the owner's character: the reference at the bottom as a guide, then the parts
in groups by kind (limbs, accessories, body, head), back to front.

Two Photoshop files per character (owner, 2026-10-31: correct the split before the AI spends time on it):
- the part map: each part as cut from the reference, nothing repainted; the owner moves pixels between layers and
  reorders them; a hidden colour layer shows the split at a glance;
- the parts: the same layers with the hidden sections rebuilt (``rebuild``), for the last touches.
The order the owner leaves the layers in is the order the parts are drawn in (a cape wrapping the front, an arm
behind it).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .engines.base import Engine, EngineError, FrameRequest
from .puppet import _segment_distance, _similarity, _target_bones
from .skeleton import (LANK, LELB, LHIP, LKNE, LSHO, LWRI, NECK, NOSE, RANK, REAR, RELB, RHIP, RKNE, RSHO, RWRI,
                       Skeleton)

# Back to front: the part, its Photoshop layer name.
LAYERS = (
    ("cape", "Áo choàng"),
    ("far_arm_upper", "Cánh tay trên (xa)"),
    ("far_arm_lower", "Cẳng tay và bàn tay (xa)"),
    ("far_leg_upper", "Đùi (xa)"),
    ("far_leg_lower", "Cẳng chân và bàn chân (xa)"),
    ("leg_upper", "Đùi (gần)"),
    ("leg_lower", "Cẳng chân và bàn chân (gần)"),
    ("torso", "Thân"),
    ("skirt", "Váy, vạt áo"),
    ("head", "Đầu"),
    ("hair", "Tóc dài"),
    ("arm_upper", "Cánh tay trên (gần)"),
    ("arm_lower", "Cẳng tay và nắm tay (gần)"),
    ("open_hand", "Bàn tay mở (gần)"),
)
NAMES = [n for n, _ in LAYERS]
LABELS = dict(LAYERS)
GUIDE = "Hình mẫu (chỉ để xem)"
OVERLAY = "Màu từng bộ phận (chỉ để xem)"
# The groups of the Photoshop files, back to front, and the parts in each by default.
GROUPS = (
    ("Phụ kiện phía sau", ("cape",)),
    ("Tay chân phía xa", ("far_arm_upper", "far_arm_lower", "far_leg_upper", "far_leg_lower")),
    ("Chân phía gần", ("leg_upper", "leg_lower")),
    ("Thân", ("torso",)),
    ("Phụ kiện phía trước", ("skirt",)),
    ("Đầu", ("head", "hair")),
    ("Tay phía gần", ("arm_upper", "arm_lower", "open_hand")),
)
DEFAULT_ORDER = [(g, n) for g, names in GROUPS for n in names]
# A colour for each part in the map's overview (the app shows it; the Photoshop file has it as a hidden layer).
MAP_COLOURS = {"cape": (230, 120, 0), "far_arm_upper": (120, 70, 200), "far_arm_lower": (170, 120, 255),
               "far_leg_upper": (0, 110, 110), "far_leg_lower": (60, 190, 190), "leg_upper": (0, 150, 40),
               "leg_lower": (120, 220, 60), "torso": (220, 30, 40), "skirt": (0, 120, 255), "head": (255, 220, 0),
               "hair": (200, 0, 200), "arm_upper": (140, 60, 20), "arm_lower": (240, 150, 110),
               "open_hand": (255, 255, 255)}
MAP_TINT = 0.6            # how strongly the overview colours the parts
HIDDEN_IN_PSD = ("open_hand",)        # drawn only in the frames that open the hand
BONES = {"far_arm_upper": "uarm_far", "far_arm_lower": "larm_far", "far_leg_upper": "uleg_far",
         "far_leg_lower": "lleg_far", "leg_upper": "uleg", "leg_lower": "lleg", "arm_upper": "uarm",
         "arm_lower": "larm"}
LIMB_GROUPS = (("far_arm_upper", "far_arm_lower"), ("far_leg_upper", "far_leg_lower"), ("leg_upper", "leg_lower"),
               ("arm_upper", "arm_lower"))
COUNTERPART = {"far_arm_upper": "arm_upper", "far_arm_lower": "arm_lower", "far_leg_upper": "leg_upper",
               "far_leg_lower": "leg_lower"}

CONFIDENT = 0.3           # a joint read with less confidence is taken from the pose that was asked for
SAM_MAX_SHARE = 0.45      # a part mask bigger than this share of the figure is the whole figure, not a part
GARMENT_MIN = 0.01        # of the figure's pixels: smaller leftovers stay with the torso
HANG_NECK = 0.12          # of the figure height below the neck: a garment starting above this hangs from the back
SKIRT_BELOW = 0.03        # of the figure height below the hips: garments lower than this are a skirt
WIDTH_SLACK = 1.05        # the expected outline of a limb is this much wider than its drawn average
MIN_HIDDEN = 0.003        # of the figure's pixels: smaller hidden parts are not rebuilt
REBUILD_STRENGTH = 0.75   # how much of the guess the AI repaints
BACKGROUND_NEAR = 24      # RGB: a repainted pixel this close to the background colour is left out
TORSO_SLACK = 1.15        # the body hidden behind the near arm is at most this much wider than the torso drawn
FAR_GUESS = 0.85          # a far limb guessed from the near one is this much darker
REBUILD_PROMPT = "standing, clenched fists"
LEG_LABELS = ("Pants", "Left-leg", "Right-leg", "Left-shoe", "Right-shoe")
GARMENT_LABELS = ("Upper-clothes", "Skirt", "Dress", "Scarf", "Belt")


# ------------------------------------------------------------------------------------------------------------------
# Joints
# ------------------------------------------------------------------------------------------------------------------
def bind_skeleton(found: Skeleton | None, asked: Skeleton) -> tuple[Skeleton, set[int]]:
    """The joints as drawn in the reference (``found``, from the pose reader), the asked pose where they were not
    read. Also the joints that were not read: a far limb that was not found is hidden and is rebuilt whole."""
    pts, missing = [], set()
    for i, a in enumerate(asked.points):
        f = found.points[i] if found is not None else None
        ok = f is not None and found.confidence[i] > CONFIDENT
        pts.append(tuple(f) if ok else a)
        if found is not None and not ok:
            missing.add(i)
    # The spread pose has the near (right) arm and leg forward: the reader's left and right can be swapped.
    for near, far in (((RSHO, RELB, RWRI), (LSHO, LELB, LWRI)), ((RHIP, RKNE, RANK), (LHIP, LKNE, LANK))):
        if pts[far[2]] is not None and pts[near[2]] is not None and pts[far[2]][0] > pts[near[2]][0]:
            for a, b in zip(near, far):
                pts[a], pts[b] = pts[b], pts[a]
                if (a in missing) != (b in missing):
                    missing ^= {a, b}
    return Skeleton(pts), missing


# ------------------------------------------------------------------------------------------------------------------
# Finding the parts
# ------------------------------------------------------------------------------------------------------------------
class PartFinder:
    """Segment Anything 2.1 (small) and a clothes-parsing model. Folders: ``<models>/sam2``,
    ``<models>/clothes-parsing``."""

    def __init__(self, models: Path) -> None:
        self.models = Path(models)
        self._sam = None
        self._parse = None

    def missing_models(self) -> list[str]:
        need = [self.models / "sam2" / "model.safetensors", self.models / "clothes-parsing" / "model.safetensors"]
        return [str(p) for p in need if not p.is_file()]

    def _load(self) -> None:
        if self._sam is not None:
            return
        if self.missing_models():
            raise EngineError("models_missing", ", ".join(self.missing_models()))
        import torch
        from transformers import AutoModelForSemanticSegmentation, Sam2Model, Sam2Processor, SegformerImageProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self._dev = dev
        self._sam = (Sam2Processor.from_pretrained(str(self.models / "sam2")),
                     Sam2Model.from_pretrained(str(self.models / "sam2")).to(dev).eval())
        self._parse = (SegformerImageProcessor.from_pretrained(str(self.models / "clothes-parsing")),
                       AutoModelForSemanticSegmentation.from_pretrained(str(self.models / "clothes-parsing"))
                       .to(dev).eval())

    def unload(self) -> None:
        self._sam = self._parse = None

    def labels(self, rgb: Image.Image) -> tuple[np.ndarray, dict[str, int]]:
        """The clothes-parsing label of every pixel, and the label numbers by name."""
        import torch
        self._load()
        proc, model = self._parse
        with torch.no_grad():
            logits = model(**proc(images=rgb, return_tensors="pt").to(self._dev)).logits
        up = torch.nn.functional.interpolate(logits, size=rgb.size[::-1], mode="bilinear", align_corners=False)
        return up.argmax(1)[0].cpu().numpy(), {v: int(k) for k, v in model.config.id2label.items()}

    def masks(self, rgb: Image.Image, pos: list, neg: list) -> tuple[np.ndarray, np.ndarray]:
        """Segment Anything's three mask choices for the points (``pos`` in the part, ``neg`` outside) and scores."""
        import torch
        self._load()
        proc, model = self._sam
        pts = [[[float(x), float(y)] for x, y in pos + neg]]
        labels = [[1] * len(pos) + [0] * len(neg)]
        inputs = proc(images=rgb, input_points=[pts], input_labels=[labels], return_tensors="pt").to(self._dev)
        with torch.no_grad():
            out = model(**inputs, multimask_output=True)
        masks = proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"])[0][0].numpy()
        return masks, out.iou_scores[0, 0].cpu().numpy()


def find_parts(image: Image.Image, sk: Skeleton, missing: set[int], finder: PartFinder | None) -> dict[str, np.ndarray]:
    """Masks of the parts in the reference (bind pose). Without a finder (the demo engine, tests) the parts are cut
    by distance to the bones."""
    arr = np.array(image.convert("RGBA"))
    alpha = arr[..., 3] > 127
    if finder is None:
        return _cut_by_bones(alpha, sk)
    h, w = alpha.shape
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.alpha_composite(image.convert("RGBA"))
    rgb = bg.convert("RGB")
    P = sk.points
    label, ids = finder.labels(rgb)
    legish = np.isin(label, [ids[n] for n in LEG_LABELS if n in ids]) & alpha
    hairish = (label == ids.get("Hair", -1)) & alpha
    garment = np.isin(label, [ids[n] for n in GARMENT_LABELS if n in ids])
    total = float(alpha.sum())
    ys = np.mgrid[0:h, 0:w][0]

    def seg(pos, neg):
        masks, scores = finder.masks(rgb, pos, neg)

        def covered(m):
            return np.mean([m[min(h - 1, max(0, int(q[1]))), min(w - 1, max(0, int(q[0])))] for q in pos])
        ok = [k for k in range(masks.shape[0]) if 50 < (masks[k] & alpha).sum() <= SAM_MAX_SHARE * total]
        if not ok:
            return np.zeros_like(alpha)
        k = max(ok, key=lambda k: (round(covered(masks[k] & alpha), 2), scores[k]))
        return masks[k] & alpha

    def along(a, b, ts):
        return [(P[a][0] + (P[b][0] - P[a][0]) * t, P[a][1] + (P[b][1] - P[a][1]) * t) for t in ts]

    def beyond(a, b, t=0.25):
        return (P[b][0] + (P[b][0] - P[a][0]) * t, P[b][1] + (P[b][1] - P[a][1]) * t)

    def free(points, taken):
        return [p for p in points if 0 <= int(p[1]) < h and 0 <= int(p[0]) < w and alpha[int(p[1]), int(p[0])]
                and not taken[int(p[1]), int(p[0])]]

    xs_, ys_ = np.mgrid[0:h, 0:w][1].astype(np.float32), ys.astype(np.float32)

    def dist(a, b):
        return _segment_distance(xs_, ys_, P[a], P[b])

    hip = ((P[RHIP][0] + P[LHIP][0]) / 2, (P[RHIP][1] + P[LHIP][1]) / 2)
    chest = (P[NECK][0] + (hip[0] - P[NECK][0]) * 0.35, P[NECK][1] + (hip[1] - P[NECK][1]) * 0.35)
    belly = (P[NECK][0] + (hip[0] - P[NECK][0]) * 0.75, P[NECK][1] + (hip[1] - P[NECK][1]) * 0.75)
    ear = P[REAR] or P[NOSE]
    head = ((P[NOSE][0] + ear[0]) / 2, (P[NOSE][1] + ear[1]) / 2 - 0.02 * _figure(sk))
    figure = _figure(sk)

    parts: dict[str, np.ndarray] = {}
    parts["head"] = seg([head], [chest]) | (hairish & (ys < P[NECK][1]))
    upper = seg(along(RSHO, RELB, [.4, .7]), [chest, along(RELB, RWRI, [.6])[0], head])
    lower = seg(along(RELB, RWRI, [.3, .7]) + [beyond(RELB, RWRI)], [along(RSHO, RELB, [.4])[0], chest])
    arm = (upper | lower) & ~parts["head"]
    closer = dist(RSHO, RELB) < dist(RELB, RWRI)
    parts["arm_upper"], parts["arm_lower"] = arm & closer, arm & ~closer
    taken = parts["head"] | arm
    if not {LELB, LWRI} & missing:
        pos = free(along(LELB, LWRI, [.3, .5, .7, .95]) + [beyond(LELB, LWRI)], taken)
        if pos:
            far = seg(pos, [chest, belly, head]) & ~taken
            closer = dist(LSHO, LELB) < dist(LELB, LWRI)
            parts["far_arm_upper"], parts["far_arm_lower"] = far & closer, far & ~closer
            taken |= far
    # Legs: only what the parsing calls leg, shared between the four leg bones by distance.
    leg = legish & ~taken
    bones = {"leg_upper": (RHIP, RKNE), "leg_lower": (RKNE, RANK), "far_leg_upper": (LHIP, LKNE),
             "far_leg_lower": (LKNE, LANK)}
    nearest = np.stack([dist(*b) for b in bones.values()]).argmin(0)
    for k, name in enumerate(bones):
        parts[name] = leg & (nearest == k)
    taken |= leg
    parts["torso"] = seg([chest, belly], [along(RSHO, RELB, [.6])[0], along(RELB, RWRI, [.5])[0], head,
                                          along(RHIP, RKNE, [.7])[0]]) & ~taken
    taken |= parts["torso"]
    # Garments left over, by where they hang from.
    rest = cv2.morphologyEx((alpha & ~taken).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest)
    for i in range(1, n):
        m = lab == i
        if m.sum() < GARMENT_MIN * total:
            continue
        py, px = np.nonzero(m)
        top, behind = py.min(), np.median(px) < P[NECK][0]
        if top < P[NECK][1] + HANG_NECK * figure and behind:
            kind = "cape"
        elif top < hip[1] + 0.05 * figure and py.max() > hip[1]:
            kind = "skirt" if not behind or np.median(py) > hip[1] else "cape"
        elif top < P[NECK][1]:
            kind = "hair"
        else:
            continue
        parts[kind] = parts.get(kind, np.zeros_like(alpha)) | m
    _give_leftovers(parts, alpha)
    # Garments below the hips (a tunic's hem, a tabard) swing as a skirt, whoever claimed them.
    hem = parts["torso"] & garment & (ys > hip[1] + SKIRT_BELOW * figure)
    if hem.sum() >= GARMENT_MIN * total:
        parts["skirt"] = parts.get("skirt", np.zeros_like(alpha)) | hem
        parts["torso"] &= ~hem
    return {k: v for k, v in parts.items() if v.any()}


def _give_leftovers(parts: dict[str, np.ndarray], alpha: np.ndarray) -> None:
    """Pixels of the figure no part claimed (thin outlines, specks between parts) go to the nearest part (knight
    test, 2026-10-31: given to the torso, its layer carried the whole figure's outline)."""
    names = list(parts)
    label = np.zeros(alpha.shape, np.int32)
    for i, n in enumerate(names, start=1):
        label[parts[n] & (label == 0)] = i
    rest = alpha & (label == 0)
    if not rest.any() or not (label > 0).any():
        return
    # Each unclaimed pixel takes the label of the nearest claimed pixel.
    _, nearest = cv2.distanceTransformWithLabels((label == 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                 labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(label > 0)
    by_seed = np.zeros(int(nearest.max()) + 1, np.int32)
    by_seed[nearest[ys, xs]] = label[ys, xs]
    got = by_seed[nearest]
    for i, n in enumerate(names, start=1):
        parts[n] = parts[n] | (rest & (got == i))


def _cut_by_bones(alpha: np.ndarray, sk: Skeleton) -> dict[str, np.ndarray]:
    h, w = alpha.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    bones = {k: v for k, v in _target_bones(sk).items() if v[0] is not None and v[1] is not None}
    names = list(bones)
    # The near limbs are in front: where a near and a far bone are as close, the pixels are the near limb's (the
    # owner's standing image has the far arm and leg behind the near ones).
    weight = {"torso": 0.55, "head": 0.6, "uarm_far": 1.4, "larm_far": 1.4, "uleg_far": 1.4, "lleg_far": 1.4}
    d = np.stack([_segment_distance(xs, ys, *bones[n]) * weight.get(n, 1.0) for n in names])
    nearest = d.argmin(0)
    by_bone = {n: alpha & (nearest == i) for i, n in enumerate(names)}
    out = {part: by_bone[bone] for part, bone in BONES.items() if bone in by_bone}
    out["head"] = by_bone.get("head", np.zeros_like(alpha))
    out["torso"] = by_bone.get("torso", np.zeros_like(alpha))
    return {k: v for k, v in out.items() if v.any()}


def _figure(sk: Skeleton) -> float:
    ys = [p[1] for p in sk.points if p is not None]
    return max(ys) - min(ys)


# ------------------------------------------------------------------------------------------------------------------
# Rebuilding what is hidden
# ------------------------------------------------------------------------------------------------------------------
def rebuild(engine: Engine, image: Image.Image, parts: dict[str, np.ndarray], sk: Skeleton, reference: Image.Image,
            reference_sk: Skeleton, background: tuple[int, int, int], seed: int,
            progress=None, start: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Every part as a whole layer: its own pixels from the reference (or ``start``, the part map's layers as the
    owner left them), its hidden section repainted by the AI."""
    arr = np.array(image.convert("RGBA"))
    alpha = arr[..., 3] > 127
    total = float(alpha.sum())
    if start is not None:
        layers = {n: v.copy() for n, v in start.items()}
    else:
        layers = {n: np.where(m[..., None], arr, 0).astype(np.uint8) for n, m in parts.items()}
    jobs = _rebuild_jobs(arr, alpha, parts, sk)
    jobs = [j for j in jobs if j[1].sum() >= MIN_HIDDEN * total]
    for k, (names, hidden, guess) in enumerate(jobs):
        if progress:
            progress(k, len(jobs))
        rough = arr.copy()
        rough[hidden] = guess[hidden]
        rough[hidden, 3] = 255
        mask = cv2.dilate(hidden.astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
        req = FrameRequest(reference=reference, reference_skeleton=reference_sk, target_skeleton=sk,
                           background=background, prompt=REBUILD_PROMPT, seed=seed + k, hands_gripping=False,
                           extra={"rough": Image.fromarray(rough, "RGBA"), "strength": REBUILD_STRENGTH,
                                  "mask": Image.fromarray(mask, "L")})
        out = np.array(engine.drawer.draw(req).convert("RGB").resize(image.size))
        # Where the AI painted background, the part does not reach (knight test: white streaks on the far shin).
        drawn = np.abs(out.astype(np.int16) - np.array(background, np.int16)).max(-1) > BACKGROUND_NEAR
        drawn = cv2.morphologyEx(drawn.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
        # The repainted hidden section goes to the part whose bone is nearest (a limb group has two).
        owner = _nearest_part(names, sk, hidden)
        for name in names:
            layer = layers.get(name, np.zeros_like(arr))
            m = hidden & drawn & (owner == names.index(name))
            layer[m, :3] = out[m]
            layer[m, 3] = 255
            layers[name] = layer
    if progress:
        progress(len(jobs), len(jobs))
    return {k: v for k, v in layers.items() if (v[..., 3] > 0).any()}


def _rebuild_jobs(arr: np.ndarray, alpha: np.ndarray, parts: dict, sk: Skeleton) -> list:
    """(parts, hidden mask, guess image) for each limb and the torso."""
    h, w = alpha.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    bones = _target_bones(sk)
    empty = np.zeros_like(alpha)
    jobs = []
    for group in LIMB_GROUPS:
        if any(bones[BONES[n]][0] is None or bones[BONES[n]][1] is None for n in group):
            continue
        own = parts.get(group[0], empty) | parts.get(group[1], empty)
        expect = np.zeros_like(alpha)
        for n in group:
            src = n if parts.get(n, empty).any() else COUNTERPART.get(n, n)
            r = _half_width(parts.get(src, empty), bones[BONES[src]])
            expect |= _segment_distance(xs, ys, *bones[BONES[n]]) <= r
        hidden = expect & ~own & alpha
        if group[0] in COUNTERPART:
            guess = np.zeros_like(arr)
            for n in group:
                c = COUNTERPART[n]
                if parts.get(c, empty).any():
                    moved = _moved(arr, parts[c], bones[BONES[c]], bones[BONES[n]])
                    k = moved[..., 3:4] / 255.0
                    guess[..., :3] = (moved[..., :3] * k * FAR_GUESS + guess[..., :3] * (1 - k)).astype(np.uint8)
                    guess[..., 3] = np.maximum(guess[..., 3], moved[..., 3])
            fill = _spread(arr, own, hidden)
            guess = np.where((guess[..., 3:4] > 127), guess, fill)
        else:
            guess = _spread(arr, own, hidden)
        jobs.append((group, hidden, guess))
    torso = parts.get("torso", empty)
    if torso.any():
        hull = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.argwhere(torso)[:, ::-1].astype(np.int32)), 1)
        front = parts.get("arm_upper", empty) | parts.get("arm_lower", empty)
        # Only as wide as the body (knight test: the hull reached along the arm and the arm was painted again).
        a, b = bones["torso"]
        body = _segment_distance(xs, ys, a, b) <= _half_width(torso, (a, b)) * TORSO_SLACK
        hidden = (hull > 0) & ~torso & front & body
        jobs.append((("torso",), hidden, _spread(arr, torso, hidden)))
    # The cape is behind everything: what of its outline the body and the legs cover (knight test: the hem had holes
    # where the legs were, which showed as tears once the legs moved).
    cape = parts.get("cape", empty)
    if cape.any():
        hull = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(hull, cv2.convexHull(np.argwhere(cape)[:, ::-1].astype(np.int32)), 1)
        # Only behind the spine: in front of it the hull is the body, and the AI painted armour into the cape.
        a, b = (np.asarray(p, float) for p in bones["torso"])
        d = (b - a) / (np.linalg.norm(b - a) + 1e-6)
        behind = (xs - a[0]) * -d[1] + (ys - a[1]) * d[0] > 0
        hidden = (hull > 0) & ~cape & alpha & behind
        jobs.append((("cape",), hidden, _spread(arr, cape, hidden)))
    return jobs


def _half_width(mask: np.ndarray, bone: tuple) -> float:
    a, b = bone
    length = float(np.hypot(b[0] - a[0], b[1] - a[1])) + 1e-6
    return max(4.0, float(mask.sum()) / length / 2 * WIDTH_SLACK)


def _moved(arr: np.ndarray, mask: np.ndarray, src: tuple, dst: tuple) -> np.ndarray:
    h, w = mask.shape
    layer = np.where(mask[..., None], arr, 0).astype(np.uint8)
    return cv2.warpAffine(layer, _similarity(*src, *dst), (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _spread(arr: np.ndarray, own: np.ndarray, hidden: np.ndarray) -> np.ndarray:
    """The part's own colours spread into the hidden section (a rough start for the AI)."""
    out = np.zeros_like(arr)
    if not own.any() or not hidden.any():
        return out
    src = np.where(own[..., None], arr[..., :3], 0).astype(np.uint8)
    hole = (~own).astype(np.uint8) * 255
    filled = cv2.inpaint(np.ascontiguousarray(src), hole, 9, cv2.INPAINT_TELEA)
    out[hidden, :3] = filled[hidden]
    out[hidden, 3] = 255
    return out


def _nearest_part(names: tuple, sk: Skeleton, region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    if len(names) == 1:
        return np.zeros((h, w), np.int64)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    bones = _target_bones(sk)
    return np.stack([_segment_distance(xs, ys, *bones[BONES[n]]) for n in names]).argmin(0)


# ------------------------------------------------------------------------------------------------------------------
# Photoshop file
# ------------------------------------------------------------------------------------------------------------------
def full_order(order: list | None, layers) -> list[tuple[str, str]]:
    """(group, part) back to front for every part in ``layers``: the owner's order, then parts it does not have (a
    new open hand) where they are by default."""
    out = [(g, n) for g, n in (order or []) if n in layers]
    have = {n for _, n in out}
    for g, n in DEFAULT_ORDER:
        if n in layers and n not in have:
            later = [k for k, (_, m) in enumerate(out) if NAMES.index(m) > NAMES.index(n)]
            out.insert(later[0] if later else len(out), (g, n))
    return out


def map_view(layers: dict[str, np.ndarray], order: list | None = None, tint: float = MAP_TINT) -> np.ndarray:
    """The layers drawn back to front, each tinted in its colour (the part map at a glance; ``tint`` 0: as drawn)."""
    h, w = next(iter(layers.values())).shape[:2]
    out = np.zeros((h, w, 4), np.uint8)
    for _, name in full_order(order, layers):
        if name in HIDDEN_IN_PSD:
            continue
        p = layers[name].astype(np.float32)
        p[..., :3] = p[..., :3] * (1 - tint) + np.array(MAP_COLOURS[name], np.float32) * tint
        k = p[..., 3:4] / 255
        out[..., :3] = (p[..., :3] * k + out[..., :3] * (1 - k)).astype(np.uint8)
        out[..., 3] = np.maximum(out[..., 3], layers[name][..., 3])
    return out


def write_psd(path: Path, guide: Image.Image, layers: dict[str, np.ndarray], mirrored: bool,
              order: list | None = None, overlay: bool = False) -> None:
    """One Photoshop file: the reference at the bottom (a guide), then the parts in their groups, back to front
    (``order``: (group, part) as the owner left them), cropped to their pixels. ``overlay``: a hidden layer on top
    colours each part (the part map). ``mirrored``: the owner's character faces left, so the file does too."""
    from psd_tools import PSDImage
    from psd_tools.api.layers import Group, PixelLayer

    def own_facing(img: Image.Image) -> Image.Image:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mirrored else img

    psd = PSDImage.new("RGBA", guide.size)
    g = PixelLayer.frompil(own_facing(guide.convert("RGBA")), psd, "guide", 0, 0)
    g.name = GUIDE
    group, group_name = None, None
    for gname, name in full_order(order, layers):
        img = own_facing(Image.fromarray(layers[name], "RGBA"))
        box = img.getchannel("A").getbbox()
        if box is None:
            continue
        if gname != group_name:              # a part moved out of its group starts a group of that name again
            group, group_name = Group.new(psd, "group", True), gname
            group.name = gname or "Bộ phận"
        layer = PixelLayer.frompil(img.crop(box), group, name, box[1], box[0])
        layer.name = LABELS[name]
        layer.visible = name not in HIDDEN_IN_PSD
    if overlay:
        tint = own_facing(Image.fromarray(map_view(layers, order), "RGBA"))
        top = PixelLayer.frompil(tint, psd, "overlay", 0, 0)
        top.name = OVERLAY
        top.visible = False
    # Old programs read the short name in a Mac character set; Photoshop shows the Unicode name set above.
    for record in psd._record.layer_and_mask_information.layer_info.layer_records:
        record.name = record.name.encode("ascii", "replace").decode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    psd.save(str(tmp))
    tmp.replace(path)


def read_psd(path: Path, size: tuple[int, int], mirrored: bool) -> tuple[dict[str, np.ndarray], list]:
    """The part layers of a Photoshop file written by ``write_psd`` (and edited by the owner), back in the bind
    pose facing right, and their order back to front as (group, part). Layers are found by name anywhere in the
    file; a part whose layer is gone or empty is left out. Raises ValueError when the file cannot be read."""
    from psd_tools import PSDImage
    psd = PSDImage.open(str(path))
    if tuple(psd.size) != tuple(size):
        raise ValueError("size")
    by_label = {label: name for name, label in LAYERS}
    out: dict[str, np.ndarray] = {}
    order: list = []
    for layer in psd.descendants():         # bottom to top
        name = by_label.get(layer.name)
        if name is None or layer.is_group() or name in out:
            continue
        img = layer.topil()
        if img is None:
            continue
        canvas = Image.new("RGBA", size, (0, 0, 0, 0))
        canvas.paste(img.convert("RGBA"), (layer.left, layer.top))
        if mirrored:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        arr = np.array(canvas)
        if (arr[..., 3] > 0).any():
            out[name] = arr
            parent = layer.parent
            order.append((parent.name if parent is not None and parent is not psd else "", name))
    return out, order


def save_layers(folder: Path, layers: dict[str, np.ndarray]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.png"):
        if old.stem not in layers:
            old.unlink()
    for name, arr in layers.items():
        Image.fromarray(arr, "RGBA").save(folder / f"{name}.png")


def load_layers(folder: Path) -> dict[str, np.ndarray]:
    out = {}
    for name in NAMES:
        p = folder / f"{name}.png"
        if p.is_file():
            with Image.open(p) as im:
                out[name] = np.array(im.convert("RGBA"))
    return out
