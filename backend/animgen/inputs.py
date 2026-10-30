"""Input checks for the character image (§4.1, §4.4).

Confirmed: the file opens; it has an alpha channel with real transparency (a checkerboard drawn into the pixels is
rejected). Proposed (cut C5): exactly one figure, i.e. one connected opaque region of meaningful size. Full body and
side view are judged by the owner by eye.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config
from .errors import AppError
from .imaging import Bounds, alpha_array, find_regions, has_alpha_channel, mask_bounds, opaque_mask


@dataclass
class CharacterInput:
    path: Path
    image: Image.Image   # RGBA, original size
    bounds: Bounds       # bounding box of the figure
    foot_line: int       # lowest opaque row (exclusive bottom - 1)
    height: int          # figure height in pixels

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "width": self.image.width,
            "height": self.image.height,
            "figure": {"left": self.bounds.left, "top": self.bounds.top,
                       "right": self.bounds.right, "bottom": self.bounds.bottom},
            "figure_height": self.height,
            "foot_line": self.foot_line,
        }


def looks_like_checkerboard(rgb: np.ndarray) -> bool:
    """True when the image border is a two-tone grid, the usual "fake transparency" pattern.

    Looks at a strip around the border (where a figure rarely is), splits it into two tones, and checks whether
    the tones follow the (x//b + y//b) % 2 parity for some block size b.
    """
    h, w = rgb.shape[:2]
    if h < 16 or w < 16:
        return False
    grey = rgb[..., :3].astype(np.float32).mean(axis=-1)
    strip = max(8, min(h, w) // 8)
    border = np.zeros((h, w), bool)
    border[:strip, :] = True
    border[-strip:, :] = True
    border[:, :strip] = True
    border[:, -strip:] = True
    vals = grey[border]
    lo, hi = np.percentile(vals, 10), np.percentile(vals, 90)
    if hi - lo < 6:          # one flat colour: an opaque background, not a checkerboard
        return False
    if hi < 150:             # checkerboards are light (white / light grey)
        return False
    # Pixels must sit near one of the two tones (allow antialiasing at a few percent)
    near = (np.abs(vals - lo) < 12) | (np.abs(vals - hi) < 12)
    if near.mean() < 0.85:
        return False
    tone = (grey > (lo + hi) / 2.0)[border]
    ys, xs = np.nonzero(border)
    best = 0.0
    for b in (4, 5, 6, 8, 10, 12, 16, 20, 24, 25, 30, 32, 40, 48, 50, 64):
        for ox in range(0, b, max(1, b // 4)):
            for oy in range(0, b, max(1, b // 4)):
                parity = (((xs + ox) // b + (ys + oy) // b) % 2).astype(bool)
                agree = float((parity == tone).mean())
                best = max(best, agree, 1.0 - agree)
        if best > 0.95:
            return True
    return best > 0.95


def check_character(path: str | Path) -> CharacterInput:
    p = Path(path)
    if not p.is_file():
        raise AppError("input_not_found")
    try:
        with Image.open(p) as raw:
            raw.load()
            alpha_present = has_alpha_channel(raw)
            img = raw.convert("RGBA")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AppError("input_unreadable") from exc

    if min(img.size) < config.MIN_INPUT_SIDE:
        raise AppError("input_too_small", min=config.MIN_INPUT_SIDE)
    if max(img.size) > config.MAX_INPUT_SIDE:
        raise AppError("input_too_large", max=config.MAX_INPUT_SIDE)

    rgb = np.asarray(img, dtype=np.uint8)
    alpha = alpha_array(img)
    transparent_fraction = float((alpha == 0).mean())

    if not alpha_present or transparent_fraction < config.MIN_TRANSPARENT_FRACTION:
        # Distinguish the common "checkerboard baked in" mistake from a plain opaque picture.
        if looks_like_checkerboard(rgb):
            raise AppError("input_checkerboard")
        if not alpha_present:
            raise AppError("input_no_alpha")
        raise AppError("input_no_transparency")

    mask = opaque_mask(alpha)
    if mask.mean() < config.MIN_FIGURE_FRACTION:
        raise AppError("input_empty")

    regions = find_regions(mask, config.SIGNIFICANT_REGION_FRACTION)
    if regions.count_significant != 1:
        raise AppError("input_many_figures", count=regions.count_significant)

    # The figure is the largest region plus the small detached bits that belong to it (hair strands etc.).
    bounds = mask_bounds(mask)
    assert bounds is not None
    return CharacterInput(path=p, image=img, bounds=bounds, foot_line=bounds.bottom - 1, height=bounds.height)
