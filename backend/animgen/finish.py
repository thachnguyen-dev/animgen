"""AI finishing pass (owner, 2026-10-30: optional, off by default): a frame made from the layers (rig.py) is repainted
lightly by the drawer, so it looks drawn rather than cut out. The pose, the proportions and the limb order are
already right in the frame, so a low strength keeps them; the owner's head is put back on top afterwards, so the
face and the hair never drift.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .engines.base import Engine, FrameRequest
from .skeleton import Skeleton

FINISH_STRENGTH = 0.3     # TUNE: share of the frame that is repainted
HEAD_FEATHER = 4          # px: the pasted head's edge is softened over this, inside its outline


def finish(engine: Engine, frame: Image.Image, head: Image.Image, reference: Image.Image, reference_sk: Skeleton,
           target_sk: Skeleton, background: tuple[int, int, int], prompt: str, seed: int,
           gripping: bool) -> Image.Image:
    """The frame repainted lightly, with the owner's head from ``head`` (the rig's head layer in this pose)."""
    req = FrameRequest(reference=reference, reference_skeleton=reference_sk, target_skeleton=target_sk,
                       background=background, prompt=prompt, seed=seed, hands_gripping=gripping,
                       extra={"rough": frame, "strength": FINISH_STRENGTH})
    out = np.array(engine.remover.remove(engine.drawer.draw(req), background).convert("RGBA"))
    return Image.fromarray(paste_head(out, np.array(head.convert("RGBA"))), "RGBA")


def paste_head(out: np.ndarray, head: np.ndarray) -> np.ndarray:
    """``head`` over ``out`` (both RGBA arrays), its edge softened inward so the seam does not show."""
    inside = (head[..., 3] > 127).astype(np.uint8)
    if HEAD_FEATHER > 0:
        inside = cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=HEAD_FEATHER // 2)
        k = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), HEAD_FEATHER / 2)
    else:
        k = inside.astype(np.float32)
    k = (k * head[..., 3] / 255.0)[..., None]
    res = out.copy()
    res[..., :3] = (head[..., :3] * k + out[..., :3] * (1 - k)).astype(np.uint8)
    res[..., 3] = np.maximum(out[..., 3], (k[..., 0] * 255).astype(np.uint8))
    return res
