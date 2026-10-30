"""Canvas placement, edge cleaning, canvas fit and technical validation of frames (§6.5 steps 4, 5 and 7; §8.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from . import config
from .imaging import alpha_array, find_regions, mask_bounds, opaque_mask, premultiplied_resize
from .inputs import CharacterInput


@dataclass
class Placement:
    """Where the reference character sits on the canvas. Every frame uses the same scale and foot anchor."""

    image: Image.Image        # canvas-sized RGBA with the reference character
    scale: float              # original pixels -> canvas pixels (the hand anchor scale too)
    anchor: tuple[float, float]  # (x, foot line y) on the canvas
    figure_height: float      # in canvas pixels
    mirrored: bool            # the reference was mirrored so the figure faces right

    def to_json(self) -> dict:
        return {"scale": round(self.scale, 5), "anchor": [round(self.anchor[0], 2), round(self.anchor[1], 2)],
                "figure_height": round(self.figure_height, 2), "mirrored": self.mirrored}


def place_reference(ch: CharacterInput, mirror: bool, canvas: int = config.CANVAS_SIZE) -> Placement:
    img = ch.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if mirror else ch.image
    b = ch.bounds
    if mirror:
        left, right = img.width - b.right, img.width - b.left
    else:
        left, right = b.left, b.right
    crop = img.crop((left, b.top, right, b.bottom))
    target_h = canvas * config.CHARACTER_HEIGHT_FRACTION
    scale = target_h / b.height
    new_w = max(1, int(round(crop.width * scale)))
    new_h = max(1, int(round(crop.height * scale)))
    max_w = int(canvas * 0.95)
    if new_w > max_w:   # a very wide figure: shrink to fit the width
        scale = max_w / crop.width
        new_w, new_h = max_w, max(1, int(round(crop.height * scale)))
    small = premultiplied_resize(crop, (new_w, new_h))
    foot_y = canvas * config.FOOT_LINE_FRACTION
    anchor_x = canvas * config.ANCHOR_X_FRACTION
    x0 = int(round(anchor_x - new_w / 2))
    y0 = int(round(foot_y - new_h))
    out = Image.new("RGBA", (canvas, canvas), (0, 0, 0, 0))
    out.alpha_composite(small, (x0, y0))
    return Placement(out, scale, (anchor_x, foot_y), float(new_h), mirror)


def clean_alpha(img: Image.Image, near: tuple[float, float, float, float] | None = None) -> Image.Image:
    """Edge cleaning after background removal: drop near-zero alpha and detached specks.

    ``near`` is the box where the figure should be (the target skeleton's bounds). Detached regions that do not reach
    it, grown by ``config.FIGURE_BOX_MARGIN`` of its height, are removed too: a second figure or a prop the model drew
    beside the character. Skipped when no region reaches the box, so a misplaced skeleton never empties the frame."""
    arr = np.array(img.convert("RGBA"))
    a = arr[..., 3]
    a[a <= config.ALPHA_FLOOR] = 0
    mask = a > config.ALPHA_OPAQUE
    if mask.any():
        n, labels, stats, _ = cv2.connectedComponentsWithStats((a > 0).astype(np.uint8), connectivity=8)
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep_min = config.STRAY_REGION_FRACTION * float(areas.max())
            drop = {i for i, area in enumerate(areas, start=1) if area < keep_min}
            if near is not None:
                m = config.FIGURE_BOX_MARGIN * (near[3] - near[1])
                x0, y0, x1, y1 = near[0] - m, near[1] - m, near[2] + m, near[3] + m
                outside = set()
                for i in range(1, n):
                    l, t = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                    r, b = l + stats[i, cv2.CC_STAT_WIDTH], t + stats[i, cv2.CC_STAT_HEIGHT]
                    if r < x0 or l > x1 or b < y0 or t > y1:
                        outside.add(i)
                if len(outside) < n - 1:
                    drop |= outside
            for i in drop:
                a[labels == i] = 0
    arr[..., 3] = a
    arr[a == 0, :3] = 0
    return Image.fromarray(arr, mode="RGBA")


def fit_to_canvas(img: Image.Image, foot_line: float, figure_height: float, grounded: bool,
                  canvas: int = config.CANVAS_SIZE) -> tuple[Image.Image, int]:
    """Make the frame exactly canvas-sized, and stand grounded frames on the foot line if they drifted a little.

    Returns the frame and the vertical shift applied. Larger drifts are left alone: moving them would hide a
    pose problem rather than fix it.
    """
    if img.size != (canvas, canvas):
        img = img.resize((canvas, canvas), Image.Resampling.LANCZOS)
    shift = 0
    if grounded:
        b = mask_bounds(opaque_mask(alpha_array(img)))
        if b is not None:
            drift = (foot_line - (b.bottom - 1))
            if abs(drift) <= config.MAX_FOOT_SNAP_FRACTION * figure_height:
                shift = int(round(drift))
    if shift:
        src = np.asarray(img.convert("RGBA"))
        moved = np.zeros_like(src)
        if shift > 0:
            moved[shift:] = src[:-shift]
        else:
            moved[:shift] = src[-shift:]
        img = Image.fromarray(moved, mode="RGBA")
    return img, shift


@dataclass
class Check:
    ok: bool
    reason: str = ""

    def to_json(self) -> dict:
        return {"ok": self.ok, "reason": self.reason}


def validate_image(img: Image.Image, size: tuple[int, int]) -> Check:
    """§8.1: correct size, a proper alpha channel and real transparency, and something actually drawn."""
    if img.mode != "RGBA":
        return Check(False, "no alpha channel")
    if img.size != size:
        return Check(False, f"size {img.size} != {size}")
    a = alpha_array(img)
    transparent = float((a == 0).mean())
    figure = float(opaque_mask(a).mean())
    if transparent < config.FRAME_MIN_TRANSPARENT_FRACTION:
        return Check(False, "not enough transparency")
    if figure < config.FRAME_MIN_FIGURE_FRACTION:
        return Check(False, "frame is empty")
    regions = find_regions(opaque_mask(a), config.SIGNIFICANT_REGION_FRACTION)
    if regions.count_significant < 1:
        return Check(False, "frame is empty")
    return Check(True)


def validate_file(path: str | Path, size: tuple[int, int]) -> Check:
    """§8.1: the file opens, then the image checks."""
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode != "RGBA":
                return Check(False, f"mode {im.mode}")
            img = im.copy()
    except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
        return Check(False, f"cannot open: {exc}")
    return validate_image(img, size)
