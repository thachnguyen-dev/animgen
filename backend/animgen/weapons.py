"""Weapons combined by the app (§7). The AI never draws or redraws a weapon.

A weapon has a kind (one_handed, two_handed, polearm, bow, staff) and one or more skins. Each skin is its own image
(bows: three images, one per draw stage). The owner clicks the grip point and the tip once per image; a new skin
whose images have the same size as an existing skin reuses that skin's points (§4.2).

Store layout: ``<weapons_dir>/<weapon_id>/weapon.json`` and the copied images next to it.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config
from .errors import AppError
from .imaging import has_alpha_channel
from .skeleton import LELB, LWRI, RELB, RWRI, Skeleton
from .templates import WEAPON_KINDS


def slug(text: str) -> str:
    """ASCII file-name part: diacritics removed (Vietnamese đ handled), spaces to '-', other characters dropped."""
    t = text.replace("đ", "d").replace("Đ", "D")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^A-Za-z0-9_-]+", "-", t).strip("-").lower()
    return t or "x"


@dataclass
class Points:
    grip: tuple[float, float]
    tip: tuple[float, float]

    def to_json(self) -> dict:
        return {"grip": list(self.grip), "tip": list(self.tip)}


@dataclass
class Skin:
    name: str
    images: list[str]                         # file names inside the weapon folder, 1 or 3 (bow)
    points: list[Points | None] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"name": self.name, "images": self.images,
                "points": [None if p is None else p.to_json() for p in self.points]}


@dataclass
class Weapon:
    id: str
    name: str
    kind: str
    skins: list[Skin]
    folder: Path

    @property
    def stages(self) -> int:
        return 3 if self.kind == "bow" else 1

    def skin(self, name: str | None) -> Skin:
        if name is None:
            return self.skins[0]
        for s in self.skins:
            if s.name == name:
                return s
        raise AppError("weapon_not_found")

    def ready(self, skin: Skin) -> bool:
        return len(skin.points) == len(skin.images) and all(p is not None for p in skin.points)

    def to_json(self) -> dict:
        return {"id": self.id, "name": self.name, "kind": self.kind,
                "skins": [dict(s.to_json(), ready=self.ready(s),
                               paths=[str(self.folder / f) for f in s.images]) for s in self.skins]}


def _check_weapon_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as raw:
            raw.load()
            alpha_present = has_alpha_channel(raw)
            img = raw.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AppError("weapon_unreadable", name=path.name) from exc
    a = np.asarray(img.getchannel("A"))
    if not alpha_present or (a == 0).mean() < 0.02 or (a > config.ALPHA_OPAQUE).mean() < 0.0005:
        raise AppError("weapon_no_alpha", name=path.name)
    return img


class WeaponStore:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)

    # ---- persistence --------------------------------------------------------------------------------------------
    def _save(self, w: Weapon) -> None:
        data = {"id": w.id, "name": w.name, "kind": w.kind, "skins": [s.to_json() for s in w.skins]}
        tmp = w.folder / "weapon.json.part"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        tmp.replace(w.folder / "weapon.json")

    def _load(self, folder: Path) -> Weapon:
        with open(folder / "weapon.json", encoding="utf-8") as f:
            d = json.load(f)
        skins = [Skin(s["name"], list(s["images"]),
                      [None if p is None else Points(tuple(p["grip"]), tuple(p["tip"])) for p in s["points"]])
                 for s in d["skins"]]
        return Weapon(d["id"], d["name"], d["kind"], skins, folder)

    def list(self) -> list[Weapon]:
        out = []
        for sub in sorted(self.folder.iterdir()):
            if (sub / "weapon.json").is_file():
                try:
                    out.append(self._load(sub))
                except (OSError, ValueError, KeyError):
                    continue
        return out

    def get(self, weapon_id: str) -> Weapon:
        folder = self.folder / slug(weapon_id)
        if not (folder / "weapon.json").is_file():
            raise AppError("weapon_not_found")
        return self._load(folder)

    # ---- editing ------------------------------------------------------------------------------------------------
    def add(self, name: str, kind: str, skin_name: str, image_paths: list[str]) -> Weapon:
        """Create a weapon, or add a skin to an existing weapon with the same name."""
        if kind not in WEAPON_KINDS:
            raise AppError("bad_request", detail=f"kind {kind}")
        need = 3 if kind == "bow" else 1
        if len(image_paths) != need:
            if kind == "bow":
                raise AppError("weapon_bow_needs_three")
            raise AppError("bad_request", detail="one image per skin")
        imgs = [_check_weapon_image(Path(p)) for p in image_paths]

        wid = slug(name)
        folder = self.folder / wid
        if (folder / "weapon.json").is_file():
            w = self._load(folder)
            if w.kind != kind:
                raise AppError("weapon_wrong_kind")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            w = Weapon(wid, name, kind, [], folder)
        sname = skin_name.strip() or f"skin{len(w.skins) + 1}"
        replaced = [f for s in w.skins if s.name == sname for f in s.images]
        w.skins = [s for s in w.skins if s.name != sname]
        files = []
        for i, img in enumerate(imgs):
            fname = f"{slug(sname)}_{i + 1}_{uuid.uuid4().hex[:6]}.png"
            img.save(folder / fname, format="PNG")
            files.append(fname)
        skin = Skin(sname, files, [None] * len(files))
        # Reuse the points of an existing skin whose images have the same sizes.
        sizes = [im.size for im in imgs]
        for other in w.skins:
            if self.ready_sizes(w, other) == sizes and w.ready(other):
                skin.points = list(other.points)
                break
        w.skins.append(skin)
        self._save(w)
        for f in replaced:  # the skin's earlier images, now unused
            (folder / f).unlink(missing_ok=True)
        return w

    @staticmethod
    def ready_sizes(w: Weapon, skin: Skin) -> list[tuple[int, int]]:
        out = []
        for f in skin.images:
            with Image.open(w.folder / f) as im:
                out.append(im.size)
        return out

    def set_points(self, weapon_id: str, skin_name: str | None, stage: int, grip: tuple[float, float],
                   tip: tuple[float, float], apply_to_all_skins: bool = True) -> Weapon:
        w = self.get(weapon_id)
        skin = w.skin(skin_name)
        if not 1 <= stage <= len(skin.images):
            raise AppError("bad_request", detail="stage")
        with Image.open(w.folder / skin.images[stage - 1]) as im:
            iw, ih = im.size
        if not (0 <= grip[0] < iw and 0 <= grip[1] < ih and 0 <= tip[0] < iw and 0 <= tip[1] < ih) \
                or math.hypot(tip[0] - grip[0], tip[1] - grip[1]) < 2:
            raise AppError("weapon_points_invalid")
        pts = Points((float(grip[0]), float(grip[1])), (float(tip[0]), float(tip[1])))
        targets = [skin]
        if apply_to_all_skins:
            size = self.ready_sizes(w, skin)[stage - 1]
            targets += [s for s in w.skins if s is not skin and len(s.images) >= stage
                        and self.ready_sizes(w, s)[stage - 1] == size]
        for s in targets:
            if len(s.points) != len(s.images):
                s.points = [None] * len(s.images)
            s.points[stage - 1] = pts
        self._save(w)
        return w

    def remove(self, weapon_id: str) -> None:
        folder = self.folder / slug(weapon_id)
        if folder.is_dir() and folder.parent == self.folder:
            shutil.rmtree(folder)

    def stage_image(self, w: Weapon, skin: Skin, stage: int) -> tuple[Image.Image, Points]:
        idx = min(max(stage, 1), len(skin.images)) - 1
        p = skin.points[idx] if idx < len(skin.points) else None
        if p is None:
            raise AppError("weapon_points_missing")
        with Image.open(w.folder / skin.images[idx]) as im:
            im.load()
            return im.convert("RGBA"), p


# ---------------------------------------------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------------------------------------------

def hand_point(skel: Skeleton, hand: str) -> tuple[float, float] | None:
    """The middle of the closed hand: the wrist pushed a little further along the forearm."""
    elb, wri = (skel.points[RELB], skel.points[RWRI]) if hand == "r" else (skel.points[LELB], skel.points[LWRI])
    if wri is None:
        return None
    if elb is None:
        return wri
    k = config.HAND_FROM_WRIST
    return (wri[0] + (wri[0] - elb[0]) * k, wri[1] + (wri[1] - elb[1]) * k)


def transform_weapon(weapon: Image.Image, pts: Points, hand: tuple[float, float], angle_deg: float, scale: float,
                     size: tuple[int, int]) -> Image.Image:
    """Return a canvas-sized RGBA layer with the weapon's grip on ``hand``, pointing at ``angle_deg``.

    The rotation is done on premultiplied colour so the edges do not get dark fringes.
    """
    native = math.degrees(math.atan2(pts.tip[1] - pts.grip[1], pts.tip[0] - pts.grip[0]))
    theta = math.radians(angle_deg - native)
    c, s = math.cos(theta) * scale, math.sin(theta) * scale
    gx, gy = pts.grip
    hx, hy = hand
    # x' = R*S*(x - grip) + hand
    m = np.array([[c, -s, hx - (c * gx - s * gy)],
                  [s, c, hy - (s * gx + c * gy)]], dtype=np.float64)
    arr = np.asarray(weapon.convert("RGBA"), dtype=np.float32) / 255.0
    pre = arr.copy()
    pre[..., :3] *= pre[..., 3:4]
    out = cv2.warpAffine(pre, m, size, flags=cv2.INTER_LINEAR if scale >= 0.75 else cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = out[..., 3:4]
    rgb = np.where(a > 1e-6, out[..., :3] / np.maximum(a, 1e-6), 0.0)
    res = np.concatenate([np.clip(rgb, 0, 1), np.clip(a, 0, 1)], axis=-1)
    return Image.fromarray((res * 255 + 0.5).astype(np.uint8), mode="RGBA")


def combine(character: Image.Image, weapon_layer: Image.Image, layer: str) -> Image.Image:
    """Stack the weapon in front of or behind the whole character (§7: no finger wrapping)."""
    base = Image.new("RGBA", character.size, (0, 0, 0, 0))
    if layer == "behind":
        base.alpha_composite(weapon_layer)
        base.alpha_composite(character)
    else:
        base.alpha_composite(character)
        base.alpha_composite(weapon_layer)
    return base


def weapon_clipped(weapon: Image.Image, pts: Points, hand: tuple[float, float], angle_deg: float, scale: float,
                   size: tuple[int, int]) -> bool:
    """True when part of the placed weapon would fall outside the canvas (the owner is told; nothing is moved)."""
    a = np.asarray(weapon.getchannel("A"))
    ys, xs = np.nonzero(a > config.ALPHA_OPAQUE)
    if len(xs) == 0:
        return False
    native = math.degrees(math.atan2(pts.tip[1] - pts.grip[1], pts.tip[0] - pts.grip[0]))
    theta = math.radians(angle_deg - native)
    c, s = math.cos(theta) * scale, math.sin(theta) * scale
    corners = [(xs.min(), ys.min()), (xs.max(), ys.min()), (xs.min(), ys.max()), (xs.max(), ys.max())]
    for x, y in corners:
        dx, dy = x - pts.grip[0], y - pts.grip[1]
        X, Y = hand[0] + c * dx - s * dy, hand[1] + s * dx + c * dy
        if X < 0 or Y < 0 or X > size[0] - 1 or Y > size[1] - 1:
            return True
    return False
