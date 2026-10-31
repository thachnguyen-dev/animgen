"""One working session: the chosen character, the current animation, its 8 frames, regeneration and export.

All frames are generated facing right (templates face right). A character facing left is mirrored on the way in
and every finished frame is mirrored back on the way out, so the owner always gets their character's own facing.

The app draws characters without weapons (owner, 2026-10-30): the owner attaches weapons in the game. Animations
made for a weapon draw the hands closed, and the export says where the weapon hand is in every frame.

Frames come from the character's rig (owner, 2026-10-30, DECISIONS.md "Layered cut-out character"): every frame
is the character's body-part layers bent onto the template (rig.py), in seconds, optionally repainted lightly
(finish.py). The rig is saved by image content like the skeleton, and used while its skeleton is the confirmed one.

The layers come from the owner's own image (owner, 2026-10-31, parts.py; no redrawing of the whole character): the
parts are found on it and given to the owner as the part map, a Photoshop file to correct; then the AI paints what
each part hides, and the parts go to a second Photoshop file for the last touches. Either file rebuilds the rig when
it is saved. The AI draws only an open hand, for the frames that open it.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .authoring import Pose, solve
from .engines.base import Engine, EngineError, FrameRequest
from .finish import finish as finish_frame
from . import parts as parts_mod
from .rig import Rig, below_neck, build_rig, build_rig_from_layers, open_hand_layer, owner_head
from .errors import AppError
from .frames import Placement, clean_alpha, fit_to_canvas, place_reference, validate_image
from .imaging import pick_background_colour, save_png
from .inputs import CharacterInput, check_character
from .shape import Shape, apply_reach, map_angle, shaped_hand, squash_matrix
from .skeleton import (LANK, LELB, LEYE, LHIP, LKNE, LSHO, LWRI, NECK, RANK, RELB, REYE, RHIP, RKNE, RSHO, RWRI,
                       Proportions, Skeleton, fit_frame, measure)
from .tasks import Task
from .templates import ANGLES, ANIMATIONS, CUSTOM_PREFIX, MotionTemplate, load_template
from . import export as export_mod


@dataclass
class Character:
    input: CharacterInput
    placement: Placement
    skeleton: Skeleton          # on the canvas, facing right
    facing: str                 # 'right' or 'left' (the character's own facing)
    facing_source: str          # 'detected', 'assumed', 'owner' or 'saved'
    name: str
    key: str = ""               # content hash of the image file: where its confirmed skeleton is saved
    skeleton_source: str = "detected"   # 'detected' (DWPose), 'proportions' (fallback) or 'saved' (owner)
    skeleton_confirmed: bool = False
    view_image: Path | None = None      # the canvas image in the character's own facing, for the skeleton editor
    angle: str = "side"                 # 'side' or '34' (turned three-quarters toward the viewer): the pose set
    angle_source: str = "detected"      # 'detected', 'owner' or 'saved'
    angle_owner: bool = False           # the owner chose the angle: never guessed again

    def view_points(self) -> list:
        """The skeleton on ``view_image`` (the character's own facing), as the editor shows it."""
        sk = self.skeleton.mirrored(self.placement.image.width) if self.placement.mirrored else self.skeleton
        return sk.to_json()

    def to_json(self) -> dict:
        return {**self.input.summary(), "facing": self.facing, "facing_source": self.facing_source,
                "angle": self.angle, "angle_source": self.angle_source,
                "name": self.name, "placement": self.placement.to_json(),
                "skeleton": {"image": str(self.view_image) if self.view_image else None,
                             "size": self.placement.image.width, "points": self.view_points(),
                             "source": self.skeleton_source, "confirmed": self.skeleton_confirmed}}


# The rig drawings (rig_proto.py prototype, 2026-10-30): the spread pose keeps every limb clear of the others; the
# hand image has the near arm straight forward with an open palm.
# The spread pose rigs were drawn in before 2026-10-31 (they still load).
RIG_POSE = Pose(spine=2, head=0, r_sh=40, r_el=20, l_sh=-40, l_el=20, r_hip=28, r_knee=6, l_hip=-26, l_knee=10)
HAND_POSE = Pose(spine=2, head=0, r_sh=90, r_el=0, l_sh=-40, l_el=20, r_hip=28, r_knee=6, l_hip=-26, l_knee=10)
HAND_PROMPT = "standing, arm outstretched forward, open palm, fingers spread"
HAND_STRENGTH = 0.75
LOOPING = ("idle", "walk", "run", "fall")      # animations that repeat (the others play once)
HANGING_TINT = {"cape": (255, 140, 0), "skirt": (0, 160, 255), "hair": (200, 0, 255)}
# The body-part layers (owner, 2026-10-31): one PNG per part, and the Photoshop file the owner corrects them in.
LAYERS_DIR = "layers"
LAYERS_INFO = "layers.json"
PSD_FILE = "parts.psd"
MAP_DIR = "map"
MAP_FILE = "map.psd"
FAR_BEHIND = 0.02         # of the figure height: a far limb the owner did not place is this far behind the near one
EDIT_MIN_DRAWN = 0.25     # the edited layers keep at least this share of the drawn pixels


def guess_angle(sk: Skeleton) -> str:
    """'34' when the skeleton shows a body turned toward the viewer: both eyes found and the shoulders or hips well
    apart across the picture (in a side view they nearly overlap); 'side' otherwise. The owner can change it."""
    pts = [p for p in sk.points if p is not None]
    if len(pts) < 4:
        return "side"
    height = max(p[1] for p in pts) - min(p[1] for p in pts)
    if height <= 0 or sk.points[REYE] is None or sk.points[LEYE] is None:
        return "side"

    def gap(a: int, b: int) -> float:
        pa, pb = sk.points[a], sk.points[b]
        return abs(pa[0] - pb[0]) / height if pa is not None and pb is not None else 0.0
    return "34" if gap(RSHO, LSHO) > 0.07 or gap(RHIP, LHIP) > 0.05 else "side"


def _skeleton(points: list) -> Skeleton:
    return Skeleton([None if p is None else (float(p[0]), float(p[1])) for p in points])


@dataclass
class RigState:
    rig: Rig
    seed: int
    still: set = field(default_factory=set)      # hanging parts the owner turned off: they stay as drawn
    view: Path | None = None                     # the assembled character, hanging parts tinted, own facing
    hand_view: Path | None = None                # the hand image, own facing
    editing: list = field(default_factory=list)  # "parts" while the Photoshop file is out for the owner
    pose: Skeleton | None = None                 # the bind pose the view is drawn in
    psd: Path | None = None                      # the Photoshop file of the stage (None: an older rig without)
    stage: str | None = None                     # "map" (parts as found), "parts" (hidden sections painted), None: old


@dataclass
class Slot:
    skeleton: Skeleton | None = None
    base: Path | None = None         # accepted frame, character only, facing right
    candidate: Path | None = None    # regenerated frame waiting for accept / keep
    status: str = "empty"            # empty, ok, failed
    candidate_status: str = "none"   # none, ok, failed
    error: dict | None = None
    attempts: int = 0
    preview: Path | None = None
    candidate_preview: Path | None = None
    swing: dict = field(default_factory=dict)    # rig frames: the hanging parts' angles in this frame
    open_hands: tuple = ()
    front: tuple = ()                # meshes in front of the body in this frame (3/4 view)
    back: tuple = ()
    shape: Shape | None = None       # squash, part sizes and bends of this frame (its reach is in the skeleton)


class Session:
    def __init__(self, engine: Engine, data_dir: Path, auto_retries: int = config.DEFAULT_AUTO_RETRIES,
                 require_confirmed_skeleton: bool = False, require_rig: bool = False) -> None:
        self.engine = engine
        self.data_dir = Path(data_dir)
        # The app asks the owner to confirm the character's skeleton before generating (DECISIONS.md, "Character
        # skeleton set by the owner"); scripts and tests use the detected one as it is.
        self.require_confirmed_skeleton = require_confirmed_skeleton
        # The app makes every frame from the rig; scripts may still draw each frame with the AI (phase0).
        self.require_rig = require_rig
        self.skeleton_dir = self.data_dir / "skeletons"
        self.rig_dir = self.data_dir / "rigs"
        self.rig: RigState | None = None
        self.finished = False           # the frames had the AI finishing pass
        temp_root = self.data_dir / "temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        # Proposed retention (§10.6): this session's frames live until exit; older sessions are removed on start.
        for old in temp_root.glob("session-*"):
            shutil.rmtree(old, ignore_errors=True)
        self.temp = temp_root / f"session-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.temp.mkdir(parents=True)
        self.auto_retries = max(0, min(int(auto_retries), config.MAX_AUTO_RETRIES))
        self.character: Character | None = None
        self.template: MotionTemplate | None = None
        self.props: Proportions | None = None
        self.grip = False               # hands drawn closed around a weapon attached later
        self.slots: list[Slot] = [Slot() for _ in range(config.FRAME_COUNT)]
        self.run_seed = 0
        self.background = (255, 255, 255)
        self._lock = threading.RLock()

    def close(self) -> None:
        shutil.rmtree(self.temp, ignore_errors=True)

    # ------------------------------------------------------------------------------------------------------------
    # Character
    # ------------------------------------------------------------------------------------------------------------
    def set_character(self, path: str, facing: str | None = None, name: str | None = None) -> Character:
        """Check the image (§4.4), place it on the canvas and read its skeleton. Fast enough to run inline for the
        demo engine; with real models the pose reader needs DWPose (small, CPU is fine)."""
        ch_in = check_character(path)
        key = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:24]
        saved = self._load_skeleton(key)
        source = "owner" if facing in ("left", "right") else "assumed"
        if facing not in ("left", "right") and saved is not None:
            facing, source = saved["facing"], "saved"
        placement = place_reference(ch_in, mirror=False)
        skel = self._read_pose(placement.image)
        if facing not in ("left", "right"):
            can_detect = getattr(self.engine.pose, "detects_facing", True)
            detected = skel.facing() if (skel is not None and can_detect) else None
            if detected is not None:
                facing, source = detected, "detected"
            else:
                facing = "right"
        if facing == "left":
            placement = place_reference(ch_in, mirror=True)
            skel = self._read_pose(placement.image)
        skel_source = "detected"
        if skel is None or skel.points[1] is None:
            # No skeleton found: fall back to the standard proportions inside the figure's box.
            from .engines.demo import ProportionalPoseReader
            skel = ProportionalPoseReader().read(placement.image)
            skel_source = "proportions"
        confirmed = False
        if saved is not None:
            view = Skeleton([None if p is None else (float(p[0]), float(p[1])) for p in saved["points"]])
            skel = view.mirrored(placement.image.width) if placement.mirrored else view
            skel_source, confirmed = "saved", True
        assert skel is not None
        if saved is not None and saved.get("angle") in ANGLES:
            # Saved before the owner's choice was recorded: taken as the owner's, so it is never guessed over.
            angle, angle_source, angle_owner = saved["angle"], "saved", bool(saved.get("angle_owner", True))
        elif self.character is not None and self.character.key == key and self.character.angle_owner:
            angle, angle_source, angle_owner = self.character.angle, "owner", True    # same image, facing flipped
        else:
            angle, angle_source, angle_owner = guess_angle(skel), "detected", False
        view_image = self.temp / f"skeleton_view_{uuid.uuid4().hex[:8]}.png"
        save_png(placement.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if placement.mirrored
                 else placement.image, view_image)
        with self._lock:
            self.character = Character(ch_in, placement, skel, facing, source,
                                       name or Path(path).stem, key, skel_source, confirmed, view_image,
                                       angle, angle_source, angle_owner)
            self.background = pick_background_colour(placement.image)
            self._reset_frames()
            self.template = None
            self.rig = None
        self._load_rig()
        return self.character

    def confirm_skeleton(self, points: list) -> Character:
        """The owner's skeleton, on the view image (the character's own facing). Saved by image content, so the same
        image is not asked again; current frames are dropped because they were drawn from the old skeleton."""
        ch = self.character
        if ch is None:
            raise AppError("no_character")
        size = ch.placement.image.width
        if not isinstance(points, list) or len(points) != len(ch.skeleton.points):
            raise AppError("skeleton_invalid")
        pts: list = []
        for p in points:
            if p is None:
                pts.append(None)
                continue
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise AppError("skeleton_invalid") from exc
            if not (0 <= x < size and 0 <= y < size):
                raise AppError("skeleton_invalid")
            pts.append((x, y))
        if pts[NECK] is None or (pts[RHIP] is None and pts[LHIP] is None) or (pts[RANK] is None and pts[LANK] is None):
            raise AppError("skeleton_incomplete")
        view = Skeleton(pts)
        with self._lock:
            ch.skeleton = view.mirrored(size) if ch.placement.mirrored else view
            ch.skeleton_source, ch.skeleton_confirmed = "saved", True
            if not ch.angle_owner:
                # Guessed again from the owner's corrected joints (owner, 2026-10-31), unless they chose the view.
                ch.angle, ch.angle_source = guess_angle(ch.skeleton), "detected"
            self._reset_frames()
            self.template = None
            self.rig = None
        self._save_skeleton(ch)
        self._load_rig()                # a rig drawn for this very skeleton (the owner confirmed it unchanged)
        return ch

    def _save_skeleton(self, ch: Character) -> None:
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        with open(self.skeleton_dir / f"{ch.key}.json", "w", encoding="utf-8") as f:
            json.dump({"facing": ch.facing, "angle": ch.angle, "angle_owner": ch.angle_owner,
                       "points": ch.view_points()}, f, ensure_ascii=False)

    def set_angle(self, angle: str) -> Character:
        """The owner's choice of the character's angle: the pose set every animation is made from (owner,
        2026-10-31). Kept with the confirmed skeleton; the frames made in the other angle are dropped."""
        ch = self.character
        if ch is None:
            raise AppError("no_character")
        if angle not in ANGLES:
            raise AppError("bad_request")
        with self._lock:
            ch.angle, ch.angle_source, ch.angle_owner = angle, "owner", True
            self._reset_frames()
            self.template = None
        if ch.skeleton_confirmed:
            self._save_skeleton(ch)
        return ch

    def _load_skeleton(self, key: str) -> dict | None:
        try:
            with open(self.skeleton_dir / f"{key}.json", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        if d.get("facing") in ("left", "right") and isinstance(d.get("points"), list):
            return d
        return None

    def _read_pose(self, image: Image.Image) -> Skeleton | None:
        """The pose reader's skeleton, or None (missing model files, nothing found): the caller then falls back to
        standard proportions, and the facing is reported as assumed rather than detected."""
        try:
            return self.engine.pose.read(image)
        except Exception as exc:  # noqa: BLE001
            import sys
            print(f"pose reader failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return None

    def _reset_frames(self) -> None:
        self.slots = [Slot() for _ in range(config.FRAME_COUNT)]

    # ------------------------------------------------------------------------------------------------------------
    # Rig
    # ------------------------------------------------------------------------------------------------------------
    def _rig_skeletons(self) -> tuple[Skeleton, Skeleton]:
        """The spread pose (rigs made before 2026-10-31 were drawn in it) and the hand pose, fitted to the character
        as the templates are."""
        ch = self.character
        assert ch is not None
        rest = load_template("idle").rest
        props = measure(ch.skeleton, rest, ch.placement.figure_height, ch.placement.anchor[1])
        out = []
        for pose in (RIG_POSE, HAND_POSE):
            pts, _ = solve(pose)
            tpl = Skeleton([None if p is None else (p[0], p[1]) for p in pts])
            out.append(fit_frame(tpl, rest, props, ch.placement.anchor, True))
        return out[0], out[1]

    def make_rig(self, task: Task, seed: int | None = None, part: str | None = None) -> dict:
        """The character's rig from the owner's own image (owner, 2026-10-31: no redrawing of the whole character):
        the AI draws only an open hand (about a minute), the parts are found on the image itself and given to the
        owner as the part map to correct; ``rebuild_hidden`` then has the AI paint what each part hides.
        ``part`` 'parts': the parts are found again, the hand kept; 'hand': only the hand is drawn again."""
        ch = self.character
        if ch is None:
            raise AppError("no_character")
        if self.require_confirmed_skeleton and not ch.skeleton_confirmed:
            raise AppError("skeleton_unconfirmed")
        if part not in (None, "parts", "hand"):
            raise AppError("unknown_part")
        if part is not None and self.rig is None:
            raise AppError("rig_missing")
        self._ensure_loaded(task)
        seed = seed if seed is not None else random.randint(1, 2**31 - 1)
        folder = self.rig_dir / ch.key
        folder.mkdir(parents=True, exist_ok=True)
        task.update(stage="generating", total=1 if part == "parts" else 2, done=0)
        if part != "parts":
            _, hand_sk = self._rig_skeletons()
            task.update(message="Đang vẽ bàn tay mở…")
            req = FrameRequest(reference=ch.placement.image, reference_skeleton=ch.skeleton, target_skeleton=hand_sk,
                               background=self.background, prompt=HAND_PROMPT, seed=seed % (2**31 - 1),
                               hands_gripping=False, extra={"strength": HAND_STRENGTH})
            try:
                hand = self.engine.remover.remove(self.engine.drawer.draw(req), self.background)
            except EngineError as exc:
                raise self._engine_error(exc) from exc
            save_png(hand, folder / "hand.png")
            task.update(done=1)
        info = {"seed": seed, "skeleton": ch.skeleton.to_json(), "still": []}
        if part is not None:
            info = self._rig_info() or info
            info["hand_seed" if part == "hand" else "seed"] = seed
            if part == "parts":
                info["still"] = []                   # the hanging parts are found again
        # The reference is the owner's own image, as placed on the canvas.
        save_png(ch.placement.image, folder / "rig.png")
        self._write_rig_info(info)
        self._make_layers(task, folder, whole=part != "hand")
        task.update(done=task.total)
        with self._lock:
            self._reset_frames()
            self.template = None
        self._load_rig()
        if self.rig is None:
            raise AppError("rig_failed")
        return self.state()

    def _bind_skeleton(self) -> Skeleton:
        """The confirmed skeleton with every limb: a far arm or leg the owner did not place (hidden behind the body)
        is taken as the near one, a little further back."""
        ch = self.character
        pts = list(ch.skeleton.points)
        back = FAR_BEHIND * ch.placement.figure_height
        for near, far in zip((RSHO, RELB, RWRI, RHIP, RKNE, RANK), (LSHO, LELB, LWRI, LHIP, LKNE, LANK)):
            if pts[far] is None and pts[near] is not None:
                pts[far] = (pts[near][0] - back, pts[near][1])
        return Skeleton(pts)

    def _make_layers(self, task: Task, folder: Path, whole: bool) -> None:
        """The part map (parts.py): each part cut from the owner's image, nothing repainted yet, for the owner to
        correct before ``rebuild_hidden`` (owner, 2026-10-31). ``whole`` False: only the open hand is made again,
        the other layers (perhaps corrected by the owner) are kept."""
        ch = self.character
        _, hand_sk = self._rig_skeletons()
        hand_img = None
        if (folder / "hand.png").exists():
            with Image.open(folder / "hand.png") as im:
                hand_img = im.convert("RGBA")
        meta = None if whole else self._layers_info()
        if meta is None:
            task.update(message="Đang tách các bộ phận…")
            image = ch.placement.image
            bind = self._bind_skeleton()
            missing = {i for i, p in enumerate(ch.skeleton.points) if p is None}
            finder = self.engine.parts
            try:
                masks = parts_mod.find_parts(image, bind, missing, finder)
            except EngineError as exc:
                raise self._engine_error(exc) from exc
            finally:
                if finder is not None:
                    finder.unload()
                    self.engine.free_memory()
            arr = np.array(image.convert("RGBA"))
            layers = {n: np.where(m[..., None], arr, 0).astype(np.uint8) for n, m in masks.items()}
            # The head is cut level at the neck (the face never moves apart); below it, the neck joins the body.
            head = owner_head(image, ch.skeleton, bind)
            found_head = layers.get("head")
            if found_head is not None and "torso" in layers:
                neck = below_neck(head.shape[:2], bind) & (found_head[..., 3] > 0) & (head[..., 3] == 0)
                layers["torso"] = layers["torso"].copy()
                layers["torso"][neck] = found_head[neck]
            layers["head"] = head
            for name in [n for n in layers if n != "head"]:
                layers[name] = layers[name].copy()
                layers[name][head[..., 3] > 0] = 0     # nothing else keeps the head's pixels
                if not (layers[name][..., 3] > 0).any():
                    del layers[name]
            meta = {"bind": bind.to_json(), "stage": "map", "order": None}
        else:
            layers = parts_mod.load_layers(folder / LAYERS_DIR)
            bind = _skeleton(meta["bind"])
        layers.pop("open_hand", None)
        hand = open_hand_layer(hand_img, hand_sk, bind) if hand_img is not None else None
        if hand is not None:
            layers["open_hand"] = hand
        if meta.get("stage", "parts") == "map":
            parts_mod.save_layers(folder / MAP_DIR, layers)
        elif hand is not None and (folder / MAP_DIR).is_dir():
            Image.fromarray(hand, "RGBA").save(folder / MAP_DIR / "open_hand.png")   # back to the map keeps it
        with Image.open(folder / "rig.png") as im:
            guide = im.convert("RGBA")
        self._save_layers(folder, guide, layers, meta)

    def _engine_error(self, exc: EngineError) -> AppError:
        if exc.kind == "oom":
            self.engine.free_memory()
            return AppError("gpu_out_of_memory")
        if exc.kind == "models_missing":
            return AppError("models_missing", detail=exc.detail)
        return AppError("inference_failed", detail=exc.detail or exc.kind)

    def _save_layers(self, folder: Path, guide: Image.Image, layers: dict, meta: dict) -> None:
        """The layers the rig is built from, the Photoshop file of the stage (the part map, or the parts with the
        hidden sections rebuilt) and ``meta`` (bind pose, stage, the owner's layer order, the file's time)."""
        parts_mod.save_layers(folder / LAYERS_DIR, layers)
        psd = self._stage_file(folder, meta)
        try:
            parts_mod.write_psd(psd, guide, layers, self.character.placement.mirrored, meta.get("order"),
                                overlay=meta.get("stage") == "map")
        except OSError as exc:
            raise AppError("rig_psd_busy") from exc
        meta["psd_time"] = psd.stat().st_mtime_ns
        with open(folder / LAYERS_INFO, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @staticmethod
    def _stage_file(folder: Path, meta: dict | None) -> Path:
        """The Photoshop file of the stage: the part map, or the parts (rigs made before the map have only those)."""
        return folder / (MAP_FILE if meta is not None and meta.get("stage") == "map" else PSD_FILE)

    def rebuild_hidden(self, task: Task) -> dict:
        """The AI paints what each part of the (corrected) part map hides; the parts then go to their own Photoshop
        file for the last touches (about 5 minutes on the RTX 4060)."""
        ch = self.character
        if ch is None or self.rig is None:
            raise AppError("rig_missing")
        meta = self._layers_info()
        if meta is None or meta.get("stage") != "map":
            raise AppError("rig_map_missing")
        folder = self.rig_dir / ch.key
        psd = folder / MAP_FILE
        if psd.exists() and psd.stat().st_mtime_ns != meta.get("psd_time"):
            self._import_psd(folder)                 # saved in Photoshop and not taken back yet
            meta = self._layers_info()
        self._ensure_loaded(task)
        layers = parts_mod.load_layers(folder / MAP_DIR)
        hand = layers.pop("open_hand", None)
        bind = _skeleton(meta["bind"])
        order = parts_mod.full_order(meta.get("order"), layers)
        context = Image.fromarray(parts_mod.map_view(layers, order, tint=0.0), "RGBA")
        masks = {n: v[..., 3] > 127 for n, v in layers.items()}
        task.update(stage="generating", total=1, done=0, message="Đang vẽ phần bị che…")

        def progress(k: int, n: int) -> None:
            task.check_cancel()
            if k < n:
                task.update(message=f"Đang vẽ phần bị che ({k + 1}/{n})…")
        try:
            out = parts_mod.rebuild(self.engine, context, masks, bind, ch.placement.image, ch.skeleton,
                                    self.background, self.rig.seed, progress, start=layers)
        except EngineError as exc:
            raise self._engine_error(exc) from exc
        if hand is not None:
            out["open_hand"] = hand
        with Image.open(folder / "rig.png") as im:
            guide = im.convert("RGBA")
        meta["stage"] = "parts"
        self._save_layers(folder, guide, out, meta)
        task.update(done=1)
        with self._lock:
            self._reset_frames()
            self.template = None
        self._load_rig()
        if self.rig is None:
            raise AppError("rig_failed")
        return self.state()

    def back_to_map(self) -> dict:
        """Back to the part map as the owner last left it (the painted parts are dropped)."""
        ch = self.character
        if ch is None or self.rig is None:
            raise AppError("rig_missing")
        meta = self._layers_info()
        folder = self.rig_dir / ch.key
        if meta is None or meta.get("stage") != "parts" or not (folder / MAP_FILE).exists():
            raise AppError("rig_map_missing")
        layers = parts_mod.load_layers(folder / MAP_DIR)
        if not layers:
            raise AppError("rig_map_missing")
        meta["stage"] = "map"
        meta["psd_time"] = None                      # the map file is read again: it may have been saved since
        parts_mod.save_layers(folder / LAYERS_DIR, layers)
        with open(folder / LAYERS_INFO, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        with self._lock:
            self._reset_frames()
            self.template = None
        self._load_rig()
        if self.rig is None:
            raise AppError("rig_failed")
        return self.state()

    def _layers_info(self) -> dict | None:
        try:
            with open(self.rig_dir / self.character.key / LAYERS_INFO, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            return None
        return d if isinstance(d.get("bind"), list) else None

    def _rig_info(self) -> dict | None:
        try:
            with open(self.rig_dir / self.character.key / "rig.json", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write_rig_info(self, info: dict) -> None:
        with open(self.rig_dir / self.character.key / "rig.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)

    def rig_edit_file(self) -> dict:
        """The Photoshop file of the stage (the part map, or the parts), for the owner to correct (owner,
        2026-10-31). ``reload_rig_edits`` takes it back; the app calls it when the file is saved."""
        ch = self.character
        if ch is None or self.rig is None:
            raise AppError("rig_missing")
        if self.rig.psd is None:
            raise AppError("rig_layers_missing")
        with self._lock:
            if "parts" not in self.rig.editing:
                self.rig.editing.append("parts")
        return {"file": str(self.rig.psd), "state": self.state()}

    def reload_rig_edits(self) -> dict:
        """The rig rebuilt from the Photoshop file the owner saved. The old layers stay if it cannot be built."""
        ch = self.character
        if ch is None or self.rig is None:
            raise AppError("rig_missing")
        if self.rig.psd is None or not self.rig.psd.exists():
            raise AppError("rig_edit_missing")
        meta = self._layers_info()
        if meta is not None and self.rig.psd.stat().st_mtime_ns == meta.get("psd_time"):
            return self.state()                      # not saved since it was written: the frames stay
        self._import_psd(self.rig_dir / ch.key)
        with self._lock:
            self._reset_frames()
            self.template = None
        editing = list(self.rig.editing)
        self._load_rig()
        if self.rig is None:
            raise AppError("rig_failed")
        self.rig.editing = editing
        return self.state()

    def _import_psd(self, folder: Path) -> None:
        """The owner's Photoshop file of the stage into the layers, with the layer order the owner left. Raises
        AppError and keeps the old layers when it is not usable."""
        ch = self.character
        meta = self._layers_info()
        if meta is None:
            raise AppError("rig_layers_missing")
        psd = self._stage_file(folder, meta)
        old = parts_mod.load_layers(folder / LAYERS_DIR)
        size = ch.placement.image.size
        try:
            new, order = parts_mod.read_psd(psd, size, ch.placement.mirrored)
        except ValueError as exc:
            raise AppError("rig_edit_size" if str(exc) == "size" else "rig_edit_unreadable") from exc
        except Exception as exc:  # noqa: BLE001
            raise AppError("rig_edit_unreadable") from exc

        def drawn(layers: dict) -> int:
            return sum(int((v[..., 3] > 127).sum()) for k, v in layers.items() if k != "open_hand")
        if "torso" not in new or drawn(new) < EDIT_MIN_DRAWN * drawn(old):
            raise AppError("rig_edit_failed")        # most of the character erased, or the body layer gone
        bind = _skeleton(meta["bind"])
        try:
            build_rig_from_layers(new, bind, [n for _, n in order])
        except Exception as exc:  # noqa: BLE001
            raise AppError("rig_edit_failed") from exc
        parts_mod.save_layers(folder / LAYERS_DIR, new)
        if meta.get("stage") == "map":
            parts_mod.save_layers(folder / MAP_DIR, new)
        meta["order"] = [list(o) for o in order]
        meta["psd_time"] = psd.stat().st_mtime_ns
        with open(folder / LAYERS_INFO, "w", encoding="utf-8") as f:
            json.dump(meta, f)

    def _load_rig(self) -> None:
        """The saved rig of this image, if it was made for the current skeleton."""
        ch = self.character
        with self._lock:
            self.rig = None
        if ch is None:
            return
        folder = self.rig_dir / ch.key
        try:
            with open(folder / "rig.json", encoding="utf-8") as f:
                info = json.load(f)
            saved = [None if p is None else (float(p[0]), float(p[1])) for p in info["skeleton"]]
        except (OSError, ValueError, KeyError, TypeError):
            return
        # Saved to 2 decimals: rounding both to whole pixels could still differ (archer test, 541.4996 vs 541.5).
        same = len(saved) == len(ch.skeleton.points) and all(
            (a is None) == (b is None) and (a is None or max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1.0)
            for a, b in zip(saved, ch.skeleton.points))
        if not same or not (folder / "rig.png").exists():
            return
        rig_sk, hand_sk = self._rig_skeletons()
        meta = self._layers_info()
        psd = self._stage_file(folder, meta)
        if meta is not None and psd.exists() and psd.stat().st_mtime_ns != meta.get("psd_time"):
            try:                                     # saved in Photoshop while the app was closed
                self._import_psd(folder)
                meta = self._layers_info()
            except AppError as exc:
                import sys
                print(f"parts file not taken: {exc.code}", file=sys.stderr)
        stage = None
        try:
            hand_img = None
            if (folder / "hand.png").exists():
                with Image.open(folder / "hand.png") as im:
                    hand_img = im.convert("RGBA")
            if meta is not None:
                pose = _skeleton(meta["bind"])
                layers = parts_mod.load_layers(folder / LAYERS_DIR)
                order = [n for _, n in parts_mod.full_order(meta.get("order"), layers)]
                rig = build_rig_from_layers(layers, pose, order)
                stage = meta.get("stage", "parts")
                view = parts_mod.map_view(layers, meta.get("order")) if stage == "map" else None
            else:                                    # a rig made before the layers (2026-10-31)
                pose = rig_sk
                with Image.open(folder / "rig.png") as im:
                    rig_img = im.convert("RGBA")
                rig = build_rig(ch.placement.image, ch.skeleton, rig_img, rig_sk, hand_img, hand_sk)
                view = None
        except Exception as exc:  # noqa: BLE001
            import sys
            print(f"rig not usable: {type(exc).__name__}: {exc}", file=sys.stderr)
            return
        state = RigState(rig, int(info.get("seed", 0)), set(info.get("still", [])), pose=pose,
                         psd=psd if meta is not None and psd.exists() else None, stage=stage)
        state.view = self.temp / f"rig_view_{uuid.uuid4().hex[:8]}.png"
        shown = Image.fromarray(view, "RGBA") if view is not None else self._rig_view(rig, pose, state.still)
        save_png(self._own_facing(shown), state.view)
        if hand_img is not None:
            state.hand_view = self.temp / f"hand_view_{uuid.uuid4().hex[:8]}.png"
            save_png(self._own_facing(hand_img), state.hand_view)
        with self._lock:
            self.rig = state

    def _rig_view(self, rig: Rig, rig_sk: Skeleton, still: set) -> Image.Image:
        """The assembled character in the spread pose, each hanging part that swings tinted in its colour."""
        import numpy as np
        img = np.array(rig.render(rig_sk))
        for mesh in rig.meshes:
            for part in mesh.hanging:
                if part.name in still or part.mask is None:
                    continue
                m = part.mask & (img[..., 3] > 0)
                tint = np.array(HANGING_TINT.get(part.name, (255, 255, 0)), np.float32)
                img[m, :3] = (img[m, :3] * 0.55 + tint * 0.45).astype(np.uint8)
        return Image.fromarray(img, "RGBA")

    def _own_facing(self, image: Image.Image) -> Image.Image:
        ch = self.character
        if ch is not None and ch.placement.mirrored:
            return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return image

    def set_hanging(self, name: str, on: bool) -> dict:
        """Turn a hanging part's swing on or off (the owner's correction of the automatic detection)."""
        ch, state = self.character, self.rig
        if ch is None or state is None:
            raise AppError("rig_missing")
        if name not in {p.name for m in state.rig.meshes for p in m.hanging}:
            raise AppError("unknown_part")
        with self._lock:
            if on:
                state.still.discard(name)
            else:
                state.still.add(name)
        info = self._rig_info()
        if info is not None:
            info["still"] = sorted(state.still)
            try:
                self._write_rig_info(info)
            except OSError:
                pass
        state.view = self.temp / f"rig_view_{uuid.uuid4().hex[:8]}.png"
        save_png(self._own_facing(self._rig_view(state.rig, state.pose, state.still)), state.view)
        return self.state()

    # ------------------------------------------------------------------------------------------------------------
    # Animation choice
    # ------------------------------------------------------------------------------------------------------------
    def prepare(self, animation_id: str) -> MotionTemplate:
        if self.character is None:
            raise AppError("no_character")
        if self.require_confirmed_skeleton and not self.character.skeleton_confirmed:
            raise AppError("skeleton_unconfirmed")
        known = {a[0]: a for a in ANIMATIONS}
        custom = animation_id.startswith(CUSTOM_PREFIX)        # the owner's own pose (pose editor)
        if animation_id not in known and not custom:
            raise AppError("unknown_animation")
        if not custom and not known[animation_id][4]:
            raise AppError("animation_not_ready")
        try:
            template = load_template(animation_id, angle=self.character.angle)
        except FileNotFoundError as exc:
            # The 3/4 set leaves some out (A3, owner 2026-10-31).
            raise AppError("animation_not_for_angle" if self.character.angle == "34" else "animation_not_ready")                 from exc
        return template

    # ------------------------------------------------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------------------------------------------------
    def generate(self, task: Task, animation_id: str, grip: bool | None = None, seed: int | None = None,
                 only: list[int] | None = None, finish: bool = False) -> dict:
        """Make the animation. ``grip``: the owner's choice where the template leaves it open. ``only``: frame
        indices (0-based) to make, for cheaper test runs; the others stay empty and the animation cannot export.
        ``finish``: with a rig, also the AI finishing pass (off by default, owner 2026-10-30)."""
        template = self.prepare(animation_id)
        ch = self.character
        assert ch is not None
        if self.require_rig and self.rig is None:
            raise AppError("rig_missing")
        with self._lock:
            self.template = template
            self.grip = template.gripping(grip)
            self.finished = bool(finish) and self.rig is not None
            self._reset_frames()
            self.run_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
            self.props = measure(ch.skeleton, template.rest, ch.placement.figure_height, ch.placement.anchor[1])
            for i, tf in enumerate(template.frames):
                self.slots[i].skeleton = apply_reach(fit_frame(tf.skeleton, template.rest, self.props,
                                                               ch.placement.anchor, tf.grounded), tf.shape)
                self.slots[i].shape = tf.shape
                self.slots[i].open_hands = tf.open_hands
                self.slots[i].front, self.slots[i].back = tf.front, tf.back
            if self.rig is not None:
                # The hanging parts swing over the whole animation, so their angles are worked out for all frames.
                swings = self.rig.rig.swings([s.skeleton for s in self.slots], loop=template.animation in LOOPING)
                for slot, sw in zip(self.slots, swings):
                    slot.swing = {k: v for k, v in sw.items() if k not in self.rig.still}
        if self.rig is None or self.finished:
            self._ensure_loaded(task)
        indices = list(range(config.FRAME_COUNT)) if only is None else sorted({self._index(i + 1) for i in only})
        self._run_frames(task, indices, regenerate=False)
        return self.state()

    def regenerate(self, task: Task, frame: int | None) -> dict:
        """Regenerate one frame (1..8) or the whole animation (None). Good frames are never overwritten: the new
        attempt is kept as a candidate next to the current frame until the owner accepts it (§8.3). Rig frames
        without the finishing pass come out the same every time, so there is nothing to redraw."""
        if self.template is None:
            raise AppError("no_frames")
        if self.rig is not None and not self.finished:
            raise AppError("nothing_to_redraw")
        indices = list(range(config.FRAME_COUNT)) if frame is None else [self._index(frame)]
        self._ensure_loaded(task)
        with self._lock:
            self.run_seed = random.randint(1, 2**31 - 1)
        self._run_frames(task, indices, regenerate=True)
        return self.state()

    def _index(self, frame: int) -> int:
        if not isinstance(frame, int) or not 1 <= frame <= config.FRAME_COUNT:
            raise AppError("frame_index")
        return frame - 1

    def _ensure_loaded(self, task: Task) -> None:
        if not self.engine.loaded:
            task.update(stage="loading_models", message="Đang tải mô hình…")
            try:
                self.engine.load(lambda msg: task.update(message=msg))
            except EngineError as exc:
                if exc.kind == "models_missing":
                    raise AppError("models_missing", detail=exc.detail) from exc
                raise AppError("inference_failed", detail=exc.detail) from exc

    def _run_frames(self, task: Task, indices: list[int], regenerate: bool) -> None:
        task.update(frames_total=config.FRAME_COUNT, total=len(indices), done=0)
        for n, i in enumerate(indices):
            task.check_cancel()
            slot = self.slots[i]
            into_candidate = regenerate and slot.status == "ok"
            task.update(stage="generating", frame=i + 1, attempt=0, message=f"Khung {i + 1} / 8")
            ok, path, err, attempts = self._generate_one(task, i)
            with self._lock:
                if into_candidate:
                    slot.candidate = path if ok else None
                    slot.candidate_status = "ok" if ok else "failed"
                    slot.candidate_preview = None
                    if not ok:
                        slot.error = err
                else:
                    slot.base = path if ok else None
                    slot.status = "ok" if ok else "failed"
                    slot.error = None if ok else err
                    slot.candidate, slot.candidate_status, slot.candidate_preview = None, "none", None
                slot.attempts = attempts
                self._refresh_preview(i)
            if not ok:
                task.warnings.append(err or {})
            task.update(done=n + 1)

    def _generate_one(self, task: Task, i: int) -> tuple[bool, Path | None, dict | None, int]:
        ch, tpl, slot = self.character, self.template, self.slots[i]
        assert ch is not None and tpl is not None and slot.skeleton is not None
        tf = tpl.frames[i]
        last_reason = ""
        tries = 1 + self.auto_retries
        for attempt in range(tries):
            task.check_cancel()
            if attempt:
                task.update(stage="retrying", attempt=attempt,
                            message=f"Khung {i + 1} lỗi ({last_reason}), thử lại lần {attempt}/{self.auto_retries}")
            seed = (self.run_seed + i * 7919 + attempt * 104729) % (2**31 - 1)
            if self.rig is not None:
                try:
                    rgba = self._rig_frame(slot, tpl, seed)
                except EngineError as exc:
                    if exc.kind == "models_missing":
                        raise AppError("models_missing", detail=exc.detail) from exc
                    last_reason = "hết bộ nhớ GPU" if exc.kind == "oom" else exc.detail or exc.kind
                    if exc.kind == "oom":
                        self.engine.free_memory()
                    continue
                check = validate_image(rgba, (config.CANVAS_SIZE, config.CANVAS_SIZE))
                if not check.ok:
                    last_reason = check.reason
                    continue
                path = self.temp / "frames" / f"f{i + 1}_{uuid.uuid4().hex[:8]}.png"
                save_png(rgba, path)
                return True, path, None, attempt + 1
            req = FrameRequest(reference=ch.placement.image, reference_skeleton=ch.skeleton,
                               target_skeleton=slot.skeleton, background=self.background, prompt=tpl.prompt,
                               seed=(self.run_seed + i * 7919 + attempt * 104729) % (2**31 - 1),
                               hands_gripping=self.grip,
                               extra={"frame_index": i, "template": tpl.id})
            try:
                rgb = self.engine.drawer.draw(req)
                rgba = self.engine.remover.remove(rgb, self.background)
            except EngineError as exc:
                if exc.kind == "models_missing":
                    raise AppError("models_missing", detail=exc.detail) from exc
                last_reason = "hết bộ nhớ GPU" if exc.kind == "oom" else exc.detail or exc.kind
                if exc.kind == "oom":
                    self.engine.free_memory()
                continue
            rgba = clean_alpha(rgba, slot.skeleton.bounds() if slot.skeleton else None)
            rgba, _shift = fit_to_canvas(rgba, ch.placement.anchor[1], ch.placement.figure_height, tf.grounded)
            check = validate_image(rgba, (config.CANVAS_SIZE, config.CANVAS_SIZE))
            if not check.ok:
                last_reason = check.reason
                continue
            path = self.temp / "frames" / f"f{i + 1}_{uuid.uuid4().hex[:8]}.png"
            save_png(rgba, path)
            return True, path, None, attempt + 1
        err = AppError("frame_failed_after_retries", frame=i + 1, tries=tries).to_json()
        err["params"]["reason"] = last_reason
        return False, None, err, tries

    def _rig_frame(self, slot: Slot, tpl: MotionTemplate, seed: int) -> Image.Image:
        """One frame from the rig, with the finishing pass when it was asked for. Placed by the template already."""
        ch, state = self.character, self.rig
        assert ch is not None and state is not None and slot.skeleton is not None
        image, keep = state.rig.render_keep(slot.skeleton, slot.swing, slot.open_hands, ch.placement.anchor[1],
                                            slot.front, slot.back, slot.shape)
        if not self.finished:
            return image
        return finish_frame(self.engine, image, keep, ch.placement.image, ch.skeleton, slot.skeleton,
                            self.background, tpl.prompt, seed, self.grip, slot.open_hands)

    # ------------------------------------------------------------------------------------------------------------
    # Accept / keep
    # ------------------------------------------------------------------------------------------------------------
    def accept(self, frame: int) -> dict:
        i = self._index(frame)
        with self._lock:
            s = self.slots[i]
            if s.candidate is None or s.candidate_status != "ok":
                raise AppError("no_candidate")
            s.base, s.status, s.error = s.candidate, "ok", None
            s.candidate, s.candidate_status = None, "none"
            s.preview, s.candidate_preview = s.candidate_preview, None
            if s.preview is None:
                self._refresh_preview(i)
        return self.state()

    def keep(self, frame: int) -> dict:
        i = self._index(frame)
        with self._lock:
            s = self.slots[i]
            if s.candidate is None and s.candidate_status == "none":
                raise AppError("no_candidate")
            s.candidate, s.candidate_status, s.candidate_preview = None, "none", None
            if s.status == "ok":
                s.error = None
        return self.state()

    def accept_all(self) -> dict:
        with self._lock:
            for n, s in enumerate(self.slots, start=1):
                if s.candidate is not None and s.candidate_status == "ok":
                    self.accept(n)
        return self.state()

    # ------------------------------------------------------------------------------------------------------------
    # Previews
    # ------------------------------------------------------------------------------------------------------------
    def compose(self, base_path: Path, i: int) -> Image.Image:
        """The finished frame in the character's own facing."""
        ch = self.character
        assert ch is not None
        with Image.open(base_path) as im:
            frame = im.convert("RGBA")
        if ch.placement.mirrored:
            frame = frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return frame

    def _refresh_preview(self, i: int) -> None:
        s = self.slots[i]
        s.preview = None
        s.candidate_preview = None
        if s.base is not None:
            p = self.temp / "preview" / f"p{i + 1}_{uuid.uuid4().hex[:8]}.png"
            save_png(self.compose(s.base, i), p)
            s.preview = p
        if s.candidate is not None:
            p = self.temp / "preview" / f"c{i + 1}_{uuid.uuid4().hex[:8]}.png"
            save_png(self.compose(s.candidate, i), p)
            s.candidate_preview = p

    # ------------------------------------------------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------------------------------------------------
    def export(self, task: Task, out_dir: str, height: int) -> dict:
        if self.template is None or self.character is None:
            raise AppError("no_frames")
        if any(s.status != "ok" or s.base is None for s in self.slots):
            raise AppError("export_incomplete")
        task.update(stage="exporting", total=config.FRAME_COUNT + 1, done=0, message="Đang xuất…")
        frames = [self.compose(s.base, i) for i, s in enumerate(self.slots)]
        tpl = self.template
        anim = tpl.animation if tpl.variant is None else f"{tpl.animation}-{tpl.variant}"
        name = export_mod.base_name(self.character.name, anim)
        paths = export_mod.export_animation(frames, out_dir, height, name,
                                            progress=lambda d, t: task.update(done=d, total=t),
                                            anchors=self.weapon_anchors(height))
        return {"files": [str(p) for p in paths], "folder": str(Path(out_dir)), "height": height}

    def weapon_anchors(self, height: int) -> dict | None:
        """Where the weapon hand is in each exported frame, for attaching a weapon in the game; None when the hands
        were not drawn gripping."""
        tpl, ch = self.template, self.character
        if tpl is None or ch is None or not self.grip or not any(f.weapon for f in tpl.frames):
            return None
        canvas = ch.placement.image.width
        k = height / canvas
        frames = []
        def anchor(wp, slot) -> dict | None:
            # Where the hand is drawn: grown with the hand and squashed with the frame (shape.py), rig frames only.
            sk, shape = slot.skeleton, slot.shape if self.rig is not None else None
            hand = shaped_hand(sk, wp.hand, shape, ch.placement.anchor[1]) if wp is not None and sk is not None \
                else None
            if hand is None:
                return None
            x, y = hand
            angle = map_angle(squash_matrix(shape, sk, ch.placement.anchor[1]), wp.angle)
            if ch.placement.mirrored:      # back to the character's own facing, as the frames are exported
                x, angle = canvas - 1 - x, 180.0 - angle
            return {"x": round(x * k, 2), "y": round(y * k, 2), "angle": round((angle + 180.0) % 360.0 - 180.0, 2),
                    "layer": wp.layer, "bow_stage": wp.stage, "hand": "near" if wp.hand == "r" else "far"}

        for i, (tf, slot) in enumerate(zip(tpl.frames, self.slots)):
            main = anchor(tf.weapon, slot)
            if main is None:
                frames.append({"index": i + 1, "weapon": False})
                continue
            entry = {"index": i + 1, "weapon": True, **main}
            second = anchor(tf.weapon2, slot)
            if second is not None:          # dual wielding: the other hand's weapon
                entry["second"] = second
            frames.append(entry)
        return {
            "animation": tpl.id, "weapon_kinds": tpl.weapon_kinds,
            "frame_size": [height, height],
            "weapon_scale": round(ch.placement.scale * k, 5),
            "about": "x, y: the middle of the weapon hand, pixels from the frame's top left. angle: degrees from the "
                     "grip towards the tip, 0 = pointing right, 90 = pointing down (in Unity, z rotation = -angle). "
                     "layer: weapon in front of or behind the character. weapon_scale: multiply a weapon drawn at "
                     "the character image's pixel scale by this. hand: near = the arm on the viewer's side, far = the "
                     "arm behind the body. second: dual wielding, the other hand's weapon (same fields).",
            "frames": frames,
        }

    # ------------------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        with self._lock:
            tpl = self.template
            return {
                "engine": self.engine.name, "demo": self.engine.is_demo, "models_loaded": self.engine.loaded,
                "character": self.character.to_json() if self.character else None,
                "animation": None if tpl is None else {"id": tpl.id, "name_vi": tpl.name_vi, "status": tpl.status,
                                                       "grip": tpl.grip},
                "grip": self.grip,
                "finished": self.finished,
                "rig": None if self.rig is None else {
                    "seed": self.rig.seed,
                    "view": str(self.rig.view) if self.rig.view else None,
                    "hand_view": str(self.rig.hand_view) if self.rig.hand_view else None,
                    "editing": list(self.rig.editing),
                    "psd": str(self.rig.psd) if self.rig.psd else None,
                    "stage": self.rig.stage,
                    "hanging": [{"name": p.name, "on": p.name not in self.rig.still}
                                for m in self.rig.rig.meshes for p in m.hanging],
                },
                "auto_retries": self.auto_retries,
                "frames": [{
                    "index": n + 1, "status": s.status, "attempts": s.attempts, "error": s.error,
                    "preview": str(s.preview) if s.preview else None,
                    "candidate_status": s.candidate_status,
                    "candidate_preview": str(s.candidate_preview) if s.candidate_preview else None,
                } for n, s in enumerate(self.slots)],
                "complete": all(s.status == "ok" for s in self.slots) and tpl is not None,
                "export_heights": list(config.EXPORT_HEIGHTS),
            }
