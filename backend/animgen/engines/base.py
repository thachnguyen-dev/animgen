"""Interfaces between the pipeline and anything that needs models or a GPU (§0.2: kept behind clear interfaces).

Two engines implement them:
- ``demo``: CPU only, no models. Draws a flat figure from the skeleton so the whole app (UI flow, retries,
  weapons, export) can be exercised without a GPU. It is NOT the AI and its frames are not real output.
- ``diffusers``: the real pipeline (Animagine XL 4.0 + ControlNet Union + IP-Adapter Plus, DWPose, ToonOut).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from PIL import Image

from ..skeleton import Skeleton

ProgressFn = Callable[[str], None]


class EngineError(Exception):
    """A failure inside an engine. ``kind`` is 'oom', 'models_missing' or 'inference'."""

    def __init__(self, kind: str, detail: str = "") -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass
class FrameRequest:
    reference: Image.Image            # canvas-sized RGBA, reference character facing right
    reference_skeleton: Skeleton      # skeleton of the reference, canvas coordinates
    target_skeleton: Skeleton         # the pose to draw, canvas coordinates
    background: tuple[int, int, int]  # flat background colour to draw on
    prompt: str                       # short motion description from the template
    seed: int
    hands_gripping: bool              # hands closed around a weapon attached later; no weapon is drawn
    extra: dict = field(default_factory=dict)


class PoseReader(Protocol):
    def read(self, image: Image.Image) -> Skeleton | None:
        """Skeleton of the single figure in an RGBA image (canvas coordinates), or None."""


class FrameDrawer(Protocol):
    def draw(self, req: FrameRequest) -> Image.Image:
        """Draw the new frame. Returns an RGB canvas-sized image on the flat background colour."""


class BackgroundRemover(Protocol):
    def remove(self, image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
        """Return RGBA with real alpha."""


class Engine:
    name = "base"
    is_demo = False

    def __init__(self) -> None:
        self.loaded = False

    def load(self, progress: ProgressFn | None = None) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    # Subclasses provide these three parts.
    pose: PoseReader
    drawer: FrameDrawer
    remover: BackgroundRemover
    # Finds the body parts in the rig image (parts.PartFinder); None cuts them by distance to the bones.
    parts = None

    def free_memory(self) -> None:
        """Called after an out-of-memory error, before a retry."""
