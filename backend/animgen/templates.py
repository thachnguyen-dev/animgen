"""Motion templates (§6.5 step 2): each animation or attack variant is 8 side-view skeletons, facing right.

File format (``template_data/<id>.json``)::

    {
      "id": "a2", "animation": "attack", "variant": "a2",
      "name_vi": "...", "status": "Proposed",
      "grip": "never" | "optional" | "always",     # hands closed around a weapon attached later in the game
      "weapon_kinds": ["one_handed", ...],          # which weapon kinds the motion is made for
      "prompt": "walking",                          # extra description for the image model
      "rest": [18 points],                          # the standing pose the template was authored on
      "frames": [                                   # exactly 8
        {"points": [18 points], "contact": "ground" | "air",
         "weapon": {"hand": "r" | "l", "angle": 0.0, "layer": "front" | "behind", "stage": 1},   # grip != never
         "open": ["l"]}                            # optional: hands drawn as an open palm, others are fists
      ]
    }

Template units: height of the rest pose is about 1.0, the ground anchor is (0, 0), y points down.
Weapon angle: degrees in image space (x right, y down), 0 = pointing right, 90 = pointing down.
The app draws characters only (owner, 2026-10-30): the frame "weapon" data is not drawn, it is exported as the hand
anchors for attaching a weapon in the game. Hand "r" is the near arm, "l" the far arm (the template faces right).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config
from .skeleton import N, Skeleton

TEMPLATE_DIR = Path(__file__).parent / "template_data"

WEAPON_KINDS = ("one_handed", "two_handed", "polearm", "bow", "staff")

# Confirmed lists (§5.1, §5.2) with the Proposed cuts C1 and C2: which ones are in this version.
ANIMATIONS = [
    # id, animation, variant, Vietnamese name, in version 1
    ("idle", "idle", None, "Đứng yên", True),
    ("walk", "walk", None, "Đi bộ", True),
    ("run", "run", None, "Chạy", True),
    ("jump", "jump", None, "Nhảy", False),
    ("fall", "fall", None, "Rơi", False),
    ("hit", "hit", None, "Bị đánh", False),
    ("death", "death", None, "Gục ngã", False),
    ("a1a", "attack", "a1a", "Tay không — đấm", True),
    ("a1b", "attack", "a1b", "Tay không — đá", True),
    ("a2", "attack", "a2", "Vũ khí một tay", True),
    ("a3", "attack", "a3", "Vũ khí nặng hai tay", False),
    ("a4", "attack", "a4", "Vũ khí cán dài", False),
    ("a5", "attack", "a5", "Cung", True),
    ("a6", "attack", "a6", "Gậy phép", True),
    ("a7", "attack", "a7", "Tay trái tung chiêu", True),
]


@dataclass
class WeaponPose:
    hand: str      # "r" or "l"
    angle: float   # degrees, image space
    layer: str     # "front" or "behind"
    stage: int     # bow draw stage 1..3; 1 for other weapons

    def to_json(self) -> dict:
        return {"hand": self.hand, "angle": self.angle, "layer": self.layer, "stage": self.stage}


@dataclass
class TemplateFrame:
    skeleton: Skeleton
    grounded: bool
    weapon: WeaponPose | None
    open_hands: tuple[str, ...] = ()    # "r" near, "l" far: drawn as an open palm in this frame


@dataclass
class MotionTemplate:
    id: str
    animation: str
    variant: str | None
    name_vi: str
    status: str
    grip: str
    weapon_kinds: list[str]
    prompt: str
    rest: Skeleton
    frames: list[TemplateFrame]

    def gripping(self, asked: bool | None) -> bool:
        """Whether the hands are drawn closed around a weapon: fixed by the template, or the owner's choice."""
        if self.grip == "never":
            return False
        return self.grip == "always" or bool(asked)


class TemplateError(ValueError):
    pass


def _parse_points(data: object, where: str) -> Skeleton:
    if not isinstance(data, list) or len(data) != N:
        raise TemplateError(f"{where}: expected {N} points")
    for p in data:
        if p is not None and (not isinstance(p, list) or len(p) != 2):
            raise TemplateError(f"{where}: each point is [x, y] or null")
    s = Skeleton.from_json(data)
    for idx in (1, 8, 11, 10, 13):  # neck, hips, ankles are required for fitting
        if s.points[idx] is None:
            raise TemplateError(f"{where}: point {idx} is required")
    return s


def parse_template(data: dict) -> MotionTemplate:
    tid = str(data.get("id", "?"))
    frames_raw = data.get("frames")
    if not isinstance(frames_raw, list) or len(frames_raw) != config.FRAME_COUNT:
        raise TemplateError(f"{tid}: a template has exactly {config.FRAME_COUNT} frames")
    grip = data.get("grip", "never")
    if grip not in ("never", "optional", "always"):
        raise TemplateError(f"{tid}: grip must be never/optional/always")
    kinds = list(data.get("weapon_kinds", []))
    for k in kinds:
        if k not in WEAPON_KINDS:
            raise TemplateError(f"{tid}: unknown weapon kind {k}")
    frames: list[TemplateFrame] = []
    for i, fr in enumerate(frames_raw):
        sk = _parse_points(fr.get("points"), f"{tid} frame {i + 1}")
        contact = fr.get("contact", "ground")
        if contact not in ("ground", "air"):
            raise TemplateError(f"{tid} frame {i + 1}: contact is ground or air")
        wp = None
        w = fr.get("weapon")
        if grip != "never":
            if not isinstance(w, dict):
                raise TemplateError(f"{tid} frame {i + 1}: weapon data is required when the hands can grip")
            hand, layer, stage = w.get("hand"), w.get("layer"), int(w.get("stage", 1))
            if hand not in ("r", "l") or layer not in ("front", "behind") or stage not in (1, 2, 3):
                raise TemplateError(f"{tid} frame {i + 1}: bad weapon data")
            wp = WeaponPose(hand=hand, angle=float(w.get("angle", 0.0)), layer=layer, stage=stage)
        open_hands = fr.get("open", [])
        if not isinstance(open_hands, list) or not set(open_hands) <= {"r", "l"}:
            raise TemplateError(f"{tid} frame {i + 1}: open is a list of \"r\" and \"l\"")
        frames.append(TemplateFrame(skeleton=sk, grounded=(contact == "ground"), weapon=wp,
                                    open_hands=tuple(open_hands)))
    return MotionTemplate(
        id=tid, animation=str(data.get("animation", tid)), variant=data.get("variant"),
        name_vi=str(data.get("name_vi", tid)), status=str(data.get("status", "Proposed")),
        grip=grip, weapon_kinds=kinds, prompt=str(data.get("prompt", "")),
        rest=_parse_points(data.get("rest"), f"{tid} rest"), frames=frames)


def load_template(tid: str, directory: Path = TEMPLATE_DIR) -> MotionTemplate:
    path = directory / f"{tid}.json"
    with open(path, encoding="utf-8") as f:
        return parse_template(json.load(f))


def available(directory: Path = TEMPLATE_DIR) -> list[dict]:
    """The animation list for the UI, with whether each one can be generated in this version."""
    out = []
    for tid, anim, variant, name, in_v1 in ANIMATIONS:
        path = directory / f"{tid}.json"
        entry = {"id": tid, "animation": anim, "variant": variant, "name_vi": name,
                 "ready": bool(in_v1 and path.is_file()), "grip": "never", "weapon_kinds": [], "status": None}
        if path.is_file():
            t = load_template(tid, directory)
            entry.update(grip=t.grip, weapon_kinds=t.weapon_kinds, status=t.status)
        out.append(entry)
    return out
