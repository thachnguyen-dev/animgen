"""One working session: the chosen character, the current animation, its 8 frames, regeneration and export.

All frames are generated facing right (templates face right). A character facing left is mirrored on the way in
and every finished frame is mirrored back on the way out, so the owner always gets their character's own facing.

The app draws characters without weapons (owner, 2026-10-30): the owner attaches weapons in the game. Animations
made for a weapon draw the hands closed, and the export says where the weapon hand is in every frame.

Frames come from the character's rig (owner, 2026-10-30, DECISIONS.md "Layered cut-out character"): after the
skeleton, the AI draws the character once in a spread pose and once with an open hand (``make_rig``); every frame
is then those pieces bent onto the template (rig.py), in seconds, optionally repainted lightly (finish.py). The rig
is saved by image content like the skeleton, and used while its skeleton is the confirmed one.
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

from PIL import Image

from . import config
from .authoring import Pose, solve
from .engines.base import Engine, EngineError, FrameRequest
from .finish import finish as finish_frame
from .rig import Rig, build_rig
from .errors import AppError
from .frames import Placement, clean_alpha, fit_to_canvas, place_reference, validate_image
from .imaging import pick_background_colour, save_png
from .inputs import CharacterInput, check_character
from .skeleton import LANK, LHIP, NECK, RANK, RHIP, Proportions, Skeleton, fit_frame, hand_point, measure
from .tasks import Task
from .templates import MotionTemplate, load_template, ANIMATIONS
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

    def view_points(self) -> list:
        """The skeleton on ``view_image`` (the character's own facing), as the editor shows it."""
        sk = self.skeleton.mirrored(self.placement.image.width) if self.placement.mirrored else self.skeleton
        return sk.to_json()

    def to_json(self) -> dict:
        return {**self.input.summary(), "facing": self.facing, "facing_source": self.facing_source,
                "name": self.name, "placement": self.placement.to_json(),
                "skeleton": {"image": str(self.view_image) if self.view_image else None,
                             "size": self.placement.image.width, "points": self.view_points(),
                             "source": self.skeleton_source, "confirmed": self.skeleton_confirmed}}


# The rig drawings (rig_proto.py prototype, 2026-10-30): the spread pose keeps every limb clear of the others; the
# hand image has the near arm straight forward with an open palm.
RIG_POSE = Pose(spine=2, head=0, r_sh=40, r_el=20, l_sh=-40, l_el=20, r_hip=28, r_knee=6, l_hip=-26, l_knee=10)
RIG_PROMPT = "standing, wide stride, legs apart, one arm forward, one arm back, clenched fists"
RIG_STRENGTH = 0.6
HAND_POSE = Pose(spine=2, head=0, r_sh=90, r_el=0, l_sh=-40, l_el=20, r_hip=28, r_knee=6, l_hip=-26, l_knee=10)
HAND_PROMPT = "standing, arm outstretched forward, open palm, fingers spread"
HAND_STRENGTH = 0.75
HANGING_TINT = {"cape": (255, 140, 0), "skirt": (0, 160, 255), "hair": (200, 0, 255)}


@dataclass
class RigState:
    rig: Rig
    seed: int
    still: set = field(default_factory=set)      # hanging parts the owner turned off: they stay as drawn
    view: Path | None = None                     # the assembled character, hanging parts tinted, own facing
    hand_view: Path | None = None                # the hand image, own facing


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
        view_image = self.temp / f"skeleton_view_{uuid.uuid4().hex[:8]}.png"
        save_png(placement.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if placement.mirrored
                 else placement.image, view_image)
        with self._lock:
            self.character = Character(ch_in, placement, skel, facing, source,
                                       name or Path(path).stem, key, skel_source, confirmed, view_image)
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
            self._reset_frames()
            self.template = None
            self.rig = None
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)
        with open(self.skeleton_dir / f"{ch.key}.json", "w", encoding="utf-8") as f:
            json.dump({"facing": ch.facing, "points": view.to_json()}, f, ensure_ascii=False)
        self._load_rig()                # a rig drawn for this very skeleton (the owner confirmed it unchanged)
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
        """The spread pose and the hand pose, fitted to the character as the templates are."""
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

    def make_rig(self, task: Task, seed: int | None = None) -> dict:
        """The AI draws the rig image and the hand image (two drawings, about a minute each on the RTX 4060)."""
        ch = self.character
        if ch is None:
            raise AppError("no_character")
        if self.require_confirmed_skeleton and not ch.skeleton_confirmed:
            raise AppError("skeleton_unconfirmed")
        self._ensure_loaded(task)
        seed = seed if seed is not None else random.randint(1, 2**31 - 1)
        rig_sk, hand_sk = self._rig_skeletons()
        task.update(stage="generating", total=2, done=0, message="Đang vẽ khung nhân vật (1/2)…")
        images = []
        for n, (sk, prompt, strength) in enumerate(((rig_sk, RIG_PROMPT, RIG_STRENGTH),
                                                    (hand_sk, HAND_PROMPT, HAND_STRENGTH))):
            task.check_cancel()
            if n:
                task.update(message="Đang vẽ bàn tay mở (2/2)…")
            req = FrameRequest(reference=ch.placement.image, reference_skeleton=ch.skeleton, target_skeleton=sk,
                               background=self.background, prompt=prompt, seed=(seed + n) % (2**31 - 1),
                               hands_gripping=False, extra={"strength": strength})
            try:
                images.append(self.engine.remover.remove(self.engine.drawer.draw(req), self.background))
            except EngineError as exc:
                if exc.kind == "oom":
                    self.engine.free_memory()
                    raise AppError("gpu_out_of_memory") from exc
                raise AppError("inference_failed", detail=exc.detail or exc.kind) from exc
            task.update(done=n + 1)
        folder = self.rig_dir / ch.key
        folder.mkdir(parents=True, exist_ok=True)
        save_png(images[0], folder / "rig.png")
        save_png(images[1], folder / "hand.png")
        with open(folder / "rig.json", "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "skeleton": ch.skeleton.to_json(), "still": []}, f, ensure_ascii=False)
        with self._lock:
            self._reset_frames()
            self.template = None
        self._load_rig()
        if self.rig is None:
            raise AppError("rig_failed")
        return self.state()

    def _load_rig(self) -> None:
        """The saved rig of this image, if it was drawn for the current skeleton."""
        ch = self.character
        with self._lock:
            self.rig = None
        if ch is None:
            return
        folder = self.rig_dir / ch.key
        try:
            with open(folder / "rig.json", encoding="utf-8") as f:
                info = json.load(f)
            saved = [None if p is None else (round(float(p[0])), round(float(p[1]))) for p in info["skeleton"]]
        except (OSError, ValueError, KeyError, TypeError):
            return
        now = [None if p is None else (round(p[0]), round(p[1])) for p in ch.skeleton.points]
        if saved != now or not (folder / "rig.png").exists():
            return
        rig_sk, hand_sk = self._rig_skeletons()
        try:
            with Image.open(folder / "rig.png") as im:
                rig_img = im.convert("RGBA")
            hand_img = None
            if (folder / "hand.png").exists():
                with Image.open(folder / "hand.png") as im:
                    hand_img = im.convert("RGBA")
            rig = build_rig(ch.placement.image, ch.skeleton, rig_img, rig_sk, hand_img, hand_sk)
        except Exception as exc:  # noqa: BLE001
            import sys
            print(f"rig not usable: {type(exc).__name__}: {exc}", file=sys.stderr)
            return
        state = RigState(rig, int(info.get("seed", 0)), set(info.get("still", [])))
        state.view = self.temp / f"rig_view_{uuid.uuid4().hex[:8]}.png"
        save_png(self._own_facing(self._rig_view(rig, rig_sk, state.still)), state.view)
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
        path = self.rig_dir / ch.key / "rig.json"
        try:
            with open(path, encoding="utf-8") as f:
                info = json.load(f)
            info["still"] = sorted(state.still)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False)
        except (OSError, ValueError):
            pass
        rig_sk, _ = self._rig_skeletons()
        state.view = self.temp / f"rig_view_{uuid.uuid4().hex[:8]}.png"
        save_png(self._own_facing(self._rig_view(state.rig, rig_sk, state.still)), state.view)
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
        if animation_id not in known:
            raise AppError("unknown_animation")
        if not known[animation_id][4]:
            raise AppError("animation_not_ready")
        try:
            template = load_template(animation_id)
        except FileNotFoundError as exc:
            raise AppError("animation_not_ready") from exc
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
                self.slots[i].skeleton = fit_frame(tf.skeleton, template.rest, self.props, ch.placement.anchor,
                                                   tf.grounded)
                self.slots[i].open_hands = tf.open_hands
            if self.rig is not None:
                # The hanging parts swing over the whole animation, so their angles are worked out for all frames.
                swings = self.rig.rig.swings([s.skeleton for s in self.slots], loop=template.animation != "attack")
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
        image, keep = state.rig.render_keep(slot.skeleton, slot.swing, slot.open_hands)
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
        for i, (tf, slot) in enumerate(zip(tpl.frames, self.slots)):
            wp = tf.weapon
            hand = hand_point(slot.skeleton, wp.hand) if wp is not None and slot.skeleton is not None else None
            if wp is None or hand is None:
                frames.append({"index": i + 1, "weapon": False})
                continue
            x, y, angle = hand[0], hand[1], wp.angle
            if ch.placement.mirrored:      # back to the character's own facing, as the frames are exported
                x, angle = canvas - 1 - x, 180.0 - angle
            frames.append({"index": i + 1, "weapon": True, "x": round(x * k, 2), "y": round(y * k, 2),
                           "angle": round((angle + 180.0) % 360.0 - 180.0, 2), "layer": wp.layer,
                           "bow_stage": wp.stage, "hand": "near" if wp.hand == "r" else "far"})
        return {
            "animation": tpl.id, "weapon_kinds": tpl.weapon_kinds,
            "frame_size": [height, height],
            "weapon_scale": round(ch.placement.scale * k, 5),
            "about": "x, y: the middle of the weapon hand, pixels from the frame's top left. angle: degrees from the "
                     "grip towards the tip, 0 = pointing right, 90 = pointing down (in Unity, z rotation = -angle). "
                     "layer: weapon in front of or behind the character. weapon_scale: multiply a weapon drawn at "
                     "the character image's pixel scale by this. hand: near = the arm on the viewer's side, far = the "
                     "arm behind the body.",
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
