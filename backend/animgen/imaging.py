"""Small image helpers shared by the pipeline (all CPU, no models)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import config


def load_rgba(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return im.convert("RGBA")


def alpha_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.getchannel("A"), dtype=np.uint8)


def has_alpha_channel(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        return True
    return img.mode == "P" and "transparency" in img.info


def opaque_mask(alpha: np.ndarray) -> np.ndarray:
    return alpha > config.ALPHA_OPAQUE


@dataclass
class Regions:
    count_significant: int
    areas: list[int]           # areas of all regions, largest first
    labels: np.ndarray         # label image
    order: list[int]           # label ids sorted by area, largest first


def find_regions(mask: np.ndarray, significant_fraction: float) -> Regions:
    """Connected opaque regions. A small closing joins pieces that touch only by a pixel or two."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return Regions(0, [], labels, [])
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = [int(i) + 1 for i in np.argsort(-areas)]
    sorted_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in order]
    total = float(sum(sorted_areas))
    significant = sum(1 for a in sorted_areas if a >= significant_fraction * total)
    return Regions(int(significant), sorted_areas, labels, order)


@dataclass
class Bounds:
    left: int
    top: int
    right: int   # exclusive
    bottom: int  # exclusive

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def mask_bounds(mask: np.ndarray) -> Bounds | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return Bounds(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def premultiplied_resize(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize RGBA without dark/bright fringes: scale premultiplied colour, then un-premultiply."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32) / 255.0
    rgb, a = arr[..., :3], arr[..., 3:4]
    pre = np.concatenate([rgb * a, a], axis=-1)
    chans = [Image.fromarray(pre[..., i], mode="F").resize(size, Image.Resampling.LANCZOS) for i in range(4)]
    out = np.stack([np.asarray(c) for c in chans], axis=-1)
    out = np.clip(out, 0.0, 1.0)
    a2 = out[..., 3:4]
    rgb2 = np.where(a2 > 1e-6, out[..., :3] / np.maximum(a2, 1e-6), 0.0)
    res = np.concatenate([np.clip(rgb2, 0, 1), a2], axis=-1)
    return Image.fromarray((res * 255.0 + 0.5).astype(np.uint8), mode="RGBA")


def flatten_on(img: Image.Image, colour: tuple[int, int, int]) -> Image.Image:
    bg = Image.new("RGBA", img.size, colour + (255,))
    bg.alpha_composite(img.convert("RGBA"))
    return bg.convert("RGB")


def pick_background_colour(img: Image.Image) -> tuple[int, int, int]:
    """A flat background far from the character's own colours (§6.5 step 3: background drawn flat and plain)."""
    arr = np.asarray(img.convert("RGBA"), dtype=np.float32)
    m = arr[..., 3] > config.ALPHA_OPAQUE
    if not m.any():
        return (255, 255, 255)
    px = arr[..., :3][m]
    candidates = [(255, 255, 255), (128, 128, 128), (40, 40, 40)]
    best, best_d = candidates[0], -1.0
    for c in candidates:
        d = np.sqrt(((px - np.array(c, np.float32)) ** 2).sum(axis=1))
        # share of character pixels that are close to this background; fewer is better
        close = float((d < 40).mean())
        score = -close
        if score > best_d:
            best, best_d = c, score
    return best


def save_png(img: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    img.save(tmp, format="PNG", optimize=False)
    tmp.replace(path)
