"""AI finishing pass (owner, 2026-10-30: optional, off by default): a frame made from the layers (rig.py) is repainted
lightly by the drawer, so it looks drawn rather than cut out. The pose, the proportions and the limb order are
already right in the frame, so a low strength keeps them; the owner's head and the hands are put back from the frame
afterwards, so the face, the hair and the hand shapes never drift (archer test, 2026-10-30: even at 0.3 the drawer
opened fists).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .engines.base import Engine, FrameRequest
from .skeleton import Skeleton

FINISH_STRENGTH = 0.3     # TUNE: share of the frame that is repainted
OUTLINE_SLACK = 3         # px: the repainted figure may reach this far past the frame's outline
KEEP_NEAR = 15            # px: around the head and the hands it may not reach past it at all (archer test,
                          # 2026-10-30: the drawer painted fingers around the fists put back)
HEAD_FEATHER = 4          # px: the pasted head's edge is softened over this, inside its outline


def finish(engine: Engine, frame: Image.Image, keep: np.ndarray, reference: Image.Image, reference_sk: Skeleton,
           target_sk: Skeleton, background: tuple[int, int, int], prompt: str, seed: int,
           gripping: bool, open_hands: tuple = ()) -> Image.Image:
    """The frame repainted lightly, with the frame's own pixels where ``keep`` (Rig.render_keep) is set."""
    req = FrameRequest(reference=reference, reference_skeleton=reference_sk, target_skeleton=target_sk,
                       background=background, prompt=frame_prompt(prompt, open_hands), seed=seed,
                       hands_gripping=gripping,
                       extra={"rough": frame, "strength": FINISH_STRENGTH})
    out = np.array(engine.remover.remove(engine.drawer.draw(req), background).convert("RGBA"))
    layer = np.array(frame.convert("RGBA"))
    out[..., 3] = np.minimum(out[..., 3], outline_limit(layer[..., 3], keep))
    layer[~keep] = 0
    return Image.fromarray(paste_head(out, layer), "RGBA")


def outline_limit(alpha: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """The most alpha the repainted frame may have at each pixel: the frame's own outline, a little wider except
    around the head and the hands."""
    inside = (alpha > 127).astype(np.uint8)
    wide = cv2.dilate(inside, np.ones((2 * OUTLINE_SLACK + 1,) * 2, np.uint8)) > 0
    near = cv2.dilate(keep.astype(np.uint8), np.ones((2 * KEEP_NEAR + 1,) * 2, np.uint8)) > 0
    return np.where(near, alpha, np.where(wide, 255, 0)).astype(np.uint8)


def frame_prompt(prompt: str, open_hands: tuple) -> str:
    """The template's prompt for one frame: a frame without an open hand does not ask for one (archer test,
    2026-10-30: A7's "open palm" opened the fists of the other frames too)."""
    if open_hands:
        return prompt
    parts = [p.strip() for p in prompt.split(",") if "open palm" not in p and "fingers" not in p]
    if not any("clenched" in p for p in parts):
        parts.append("clenched fists")
    return ", ".join(p for p in parts if p)


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
