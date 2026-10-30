"""One working session: the chosen character, the current animation, its 8 frames, regeneration and export.

All frames are generated facing right (templates face right). A character facing left is mirrored on the way in
and every finished frame is mirrored back on the way out, so the owner always gets their character's own facing.

The character is drawn once without a weapon (§7). The weapon is placed when a frame is shown or exported, so a
new skin is only a recombine and all skins have exactly the same motion.
"""

from __future__ import annotations

import random
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from . import config
from .engines.base import Engine, EngineError, FrameRequest
from .errors import AppError
from .frames import Placement, clean_alpha, fit_to_canvas, place_reference, validate_image
from .imaging import pick_background_colour, save_png
from .inputs import CharacterInput, check_character
from .skeleton import Proportions, Skeleton, fit_frame, measure
from .tasks import Task
from .templates import MotionTemplate, load_template, ANIMATIONS
from .weapons import WeaponStore, combine, hand_point, transform_weapon, weapon_clipped
from . import export as export_mod


@dataclass
class Character:
    input: CharacterInput
    placement: Placement
    skeleton: Skeleton          # on the canvas, facing right
    facing: str                 # 'right' or 'left' (the character's own facing)
    facing_source: str          # 'detected', 'assumed' or 'owner'
    name: str

    def to_json(self) -> dict:
        return {**self.input.summary(), "facing": self.facing, "facing_source": self.facing_source,
                "name": self.name, "placement": self.placement.to_json()}


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
    weapon_clipped: bool = False


@dataclass
class WeaponChoice:
    weapon_id: str
    skin: str | None


class Session:
    def __init__(self, engine: Engine, data_dir: Path, auto_retries: int = config.DEFAULT_AUTO_RETRIES) -> None:
        self.engine = engine
        self.data_dir = Path(data_dir)
        self.weapons = WeaponStore(self.data_dir / "weapons")
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
        self.weapon: WeaponChoice | None = None
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
        source = "owner" if facing in ("left", "right") else "assumed"
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
        if skel is None or skel.points[1] is None:
            # No skeleton found: fall back to the standard proportions inside the figure's box.
            from .engines.demo import ProportionalPoseReader
            skel = ProportionalPoseReader().read(placement.image)
        assert skel is not None
        with self._lock:
            self.character = Character(ch_in, placement, skel, facing, source,
                                       name or Path(path).stem)
            self.background = pick_background_colour(placement.image)
            self._reset_frames()
            self.template = None
        return self.character

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
    # Animation and weapon choice
    # ------------------------------------------------------------------------------------------------------------
    def _check_choice(self, template: MotionTemplate, weapon: WeaponChoice | None) -> None:
        if template.weapon == "never" and weapon is not None:
            raise AppError("weapon_not_allowed")
        if template.weapon == "required" and weapon is None:
            raise AppError("weapon_required")
        if weapon is not None:
            w = self.weapons.get(weapon.weapon_id)
            if w.kind not in template.weapon_kinds:
                raise AppError("weapon_wrong_kind")
            skin = w.skin(weapon.skin)
            if not w.ready(skin):
                raise AppError("weapon_points_missing")

    def prepare(self, animation_id: str, weapon: WeaponChoice | None) -> MotionTemplate:
        if self.character is None:
            raise AppError("no_character")
        known = {a[0]: a for a in ANIMATIONS}
        if animation_id not in known:
            raise AppError("unknown_animation")
        if not known[animation_id][4]:
            raise AppError("animation_not_ready")
        try:
            template = load_template(animation_id)
        except FileNotFoundError as exc:
            raise AppError("animation_not_ready") from exc
        self._check_choice(template, weapon)
        return template

    def set_weapon(self, weapon: WeaponChoice | None) -> None:
        """Change weapon or skin for the current frames: recombine only, no AI work (§7)."""
        with self._lock:
            if self.template is None:
                raise AppError("no_frames")
            self._check_choice(self.template, weapon)
            self.weapon = weapon
            self._refresh_previews()

    # ------------------------------------------------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------------------------------------------------
    def generate(self, task: Task, animation_id: str, weapon: WeaponChoice | None, seed: int | None = None) -> dict:
        template = self.prepare(animation_id, weapon)
        ch = self.character
        assert ch is not None
        with self._lock:
            self.template = template
            self.weapon = weapon
            self._reset_frames()
            self.run_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
            self.props = measure(ch.skeleton, template.rest, ch.placement.figure_height, ch.placement.anchor[1])
            for i, tf in enumerate(template.frames):
                self.slots[i].skeleton = fit_frame(tf.skeleton, template.rest, self.props, ch.placement.anchor,
                                                   tf.grounded)
        self._ensure_loaded(task)
        self._run_frames(task, list(range(config.FRAME_COUNT)), regenerate=False)
        return self.state()

    def regenerate(self, task: Task, frame: int | None) -> dict:
        """Regenerate one frame (1..8) or the whole animation (None). Good frames are never overwritten: the new
        attempt is kept as a candidate next to the current frame until the owner accepts it (§8.3)."""
        if self.template is None:
            raise AppError("no_frames")
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
            req = FrameRequest(reference=ch.placement.image, reference_skeleton=ch.skeleton,
                               target_skeleton=slot.skeleton, background=self.background, prompt=tpl.prompt,
                               seed=(self.run_seed + i * 7919 + attempt * 104729) % (2**31 - 1),
                               hands_gripping=self.weapon is not None or tpl.weapon == "required",
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
            rgba = clean_alpha(rgba)
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
    # Weapon combination and previews
    # ------------------------------------------------------------------------------------------------------------
    def compose(self, base_path: Path, i: int) -> Image.Image:
        """The finished frame: the character, the weapon placed in the hand, and the original facing."""
        ch, tpl = self.character, self.template
        assert ch is not None and tpl is not None
        with Image.open(base_path) as im:
            frame = im.convert("RGBA")
        wp = tpl.frames[i].weapon
        if self.weapon is not None and wp is not None:
            w = self.weapons.get(self.weapon.weapon_id)
            skin = w.skin(self.weapon.skin)
            img, pts = self.weapons.stage_image(w, skin, wp.stage if w.kind == "bow" else 1)
            hand = hand_point(self.slots[i].skeleton, wp.hand) if self.slots[i].skeleton else None
            if hand is not None:
                self.slots[i].weapon_clipped = weapon_clipped(img, pts, hand, wp.angle, ch.placement.scale,
                                                              frame.size)
                layer = transform_weapon(img, pts, hand, wp.angle, ch.placement.scale, frame.size)
                frame = combine(frame, layer, wp.layer)
        if ch.placement.mirrored:
            frame = frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return frame

    def _refresh_preview(self, i: int) -> None:
        s = self.slots[i]
        s.preview = None
        s.candidate_preview = None
        s.weapon_clipped = False
        if s.base is not None:
            p = self.temp / "preview" / f"p{i + 1}_{uuid.uuid4().hex[:8]}.png"
            save_png(self.compose(s.base, i), p)
            s.preview = p
        if s.candidate is not None:
            p = self.temp / "preview" / f"c{i + 1}_{uuid.uuid4().hex[:8]}.png"
            save_png(self.compose(s.candidate, i), p)
            s.candidate_preview = p

    def _refresh_previews(self) -> None:
        for i in range(config.FRAME_COUNT):
            self._refresh_preview(i)

    def weapon_preview(self, weapon_id: str, skin: str | None, stage: int = 1) -> Path:
        """The weapon in the character's hand on the reference pose, for the owner to confirm (§4.2)."""
        ch = self.character
        if ch is None:
            raise AppError("no_character")
        w = self.weapons.get(weapon_id)
        sk = w.skin(skin)
        img, pts = self.weapons.stage_image(w, sk, stage)
        hand = hand_point(ch.skeleton, "l" if w.kind == "bow" else "r") or ch.placement.anchor
        angle = -90.0 if w.kind in ("bow", "staff") else -45.0
        layer = transform_weapon(img, pts, hand, angle, ch.placement.scale, ch.placement.image.size)
        frame = combine(ch.placement.image, layer, "behind" if w.kind == "bow" else "front")
        if ch.placement.mirrored:
            frame = frame.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        p = self.temp / "preview" / f"weapon_{uuid.uuid4().hex[:8]}.png"
        save_png(frame, p)
        return p

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
        wname = sname = None
        if self.weapon is not None:
            w = self.weapons.get(self.weapon.weapon_id)
            wname, sname = w.name, w.skin(self.weapon.skin).name
        name = export_mod.base_name(self.character.name, anim, wname, sname)
        paths = export_mod.export_animation(frames, out_dir, height, name,
                                            progress=lambda d, t: task.update(done=d, total=t))
        return {"files": [str(p) for p in paths], "folder": str(Path(out_dir)), "height": height}

    # ------------------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        with self._lock:
            tpl = self.template
            return {
                "engine": self.engine.name, "demo": self.engine.is_demo, "models_loaded": self.engine.loaded,
                "character": self.character.to_json() if self.character else None,
                "animation": None if tpl is None else {"id": tpl.id, "name_vi": tpl.name_vi, "status": tpl.status,
                                                       "weapon": tpl.weapon},
                "weapon": None if self.weapon is None else {"weapon_id": self.weapon.weapon_id,
                                                            "skin": self.weapon.skin},
                "auto_retries": self.auto_retries,
                "frames": [{
                    "index": n + 1, "status": s.status, "attempts": s.attempts, "error": s.error,
                    "weapon_clipped": s.weapon_clipped,
                    "preview": str(s.preview) if s.preview else None,
                    "candidate_status": s.candidate_status,
                    "candidate_preview": str(s.candidate_preview) if s.candidate_preview else None,
                } for n, s in enumerate(self.slots)],
                "complete": all(s.status == "ok" for s in self.slots) and tpl is not None,
                "export_heights": list(config.EXPORT_HEIGHTS),
            }
