"""The pose editor (owner, 2026-10-31): the owner makes a new animation by posing a sample character in 8 frames.

The sample character (sample.py) is opened in a session of its own, one per view ("side", "34"), with its rig, so a
frame is drawn exactly as generation would draw it. The editor works in the sample's canvas pixels; a pose is saved
as a motion template like the built-in ones (template units: the anchor at (0, 0), figure height 1), with the sample's
own skeleton as its rest pose. Fitting a frame to the sample then gives back the same pose, so what the owner sees
while editing is what generation makes; on another character the bone directions are kept and the lengths are its
own.

Saved poses live in the data folder (<data>/poses/u_*.json), not in the app, so an update keeps them. Each belongs to
the view it was made in, and shows in the animation list only for characters seen from that view.
"""
from __future__ import annotations

import json
import math
import re
import threading
import uuid
from pathlib import Path

from . import sample
from .errors import AppError
from .skeleton import EDGES, RELB, RWRI, Skeleton, fit_frame, measure
from .templates import (ANGLES, BACK_MESHES, FRONT_MESHES, WEAPON_KINDS, TemplateError, custom_dir, load_template,
                        parse_template)

NAME_MAX = 40
GRIPS = ("never", "optional", "always")
START = "idle"                # a new pose starts as the first frame of Idle in all 8 frames


def _tid(text: object) -> str:
    if not isinstance(text, str) or not re.fullmatch(r"u_[0-9a-f]{8}", text):
        raise AppError("pose_unknown")
    return text


class PoseEditor:
    def __init__(self, data_dir: Path) -> None:
        self.dir = custom_dir(data_dir)
        self.work = Path(data_dir) / "pose_editor"
        self._sessions: dict = {}
        self._lock = threading.RLock()
        self._last_preview: dict[str, Path] = {}

    # ------------------------------------------------------------------------------------------------------------
    # The sample character
    # ------------------------------------------------------------------------------------------------------------
    def _sample(self, angle: str):
        if angle not in ANGLES:
            raise AppError("bad_request", detail="angle must be side or 34")
        with self._lock:
            s = self._sessions.get(angle)
            if s is None:
                from .engines.demo import DemoEngine
                from .session import Session
                folder = self.work / angle            # one folder each: a session clears its folder's old temp files
                s = Session(DemoEngine(), folder, require_rig=True)
                sample.make(s, angle, folder / "sample.png")
                self._sessions[angle] = s
            return s

    def _unit(self, s) -> tuple[float, float, float]:
        """The anchor (foot point) and the pixels per template unit of the sample on its canvas."""
        pl = s.character.placement
        return pl.anchor[0], pl.anchor[1], float(pl.figure_height)

    def _to_template(self, s, points: list) -> list:
        ax, ay, h = self._unit(s)
        return [None if p is None else [round((p[0] - ax) / h, 4), round((p[1] - ay) / h, 4)] for p in points]

    def _props(self, s, rest_t: Skeleton):
        ch = s.character
        return measure(ch.skeleton, rest_t, ch.placement.figure_height, ch.placement.anchor[1])

    def _rest(self, s) -> Skeleton:
        """The rest pose of a saved pose: the sample's bind pose with every bone at its full length (in the 3/4 view
        some bones of the bind pose are drawn shorter, and a frame is measured against the full length)."""
        ch = s.character
        bind = Skeleton.from_json(self._to_template(s, ch.skeleton.to_json()))
        props = self._props(s, bind)
        h = float(ch.placement.figure_height)
        pts = list(bind.points)
        for a, b in EDGES:
            pa, pb = pts[a], bind.points[b]
            if pa is None or pb is None or bind.points[a] is None:
                continue
            dx, dy = pb[0] - bind.points[a][0], pb[1] - bind.points[a][1]
            d = math.hypot(dx, dy)
            L = props.lengths[(a, b)] / h
            pts[b] = pb if d < 1e-9 else (pa[0] + dx / d * L, pa[1] + dy / d * L)
        return Skeleton(pts)

    def _fit(self, s, points_t: Skeleton, rest_t: Skeleton, grounded: bool) -> Skeleton:
        ch = s.character
        return fit_frame(points_t, rest_t, self._props(s, rest_t), ch.placement.anchor, grounded)

    def info(self, angle: str) -> dict:
        s = self._sample(angle)
        ch = s.character
        w, h = ch.placement.image.size
        return {"angle": angle, "width": w, "height": h, "bones": [list(e) for e in EDGES],
                "rest": ch.skeleton.to_json(), "ground": ch.placement.anchor[1],
                "layers": {"front": list(FRONT_MESHES), "back": list(BACK_MESHES)}}

    # ------------------------------------------------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------------------------------------------------
    def _frames_from(self, s, tid: str, angle: str) -> list[dict]:
        try:
            tpl = load_template(tid, angle=angle)
        except (OSError, TemplateError) as exc:
            raise AppError("pose_unknown") from exc
        out = []
        for tf in tpl.frames:
            sk = self._fit(s, tf.skeleton, tpl.rest, tf.grounded)
            out.append({"points": sk.to_json(), "contact": "ground" if tf.grounded else "air",
                        "open": list(tf.open_hands), "front": list(tf.front), "back": list(tf.back)})
        return out

    def start(self, angle: str, base: str | None) -> dict:
        """A new pose: 8 frames copied from an existing animation, or the standing pose."""
        s = self._sample(angle)
        frames = self._frames_from(s, base or START, angle)
        if not base:
            frames = [dict(frames[0]) for _ in frames]
        return {"sample": self.info(angle), "frames": frames}

    def open(self, tid: str) -> dict:
        """A saved pose, to change it."""
        data = self._read(_tid(tid))
        angle = data.get("angle", "side")
        s = self._sample(angle)
        return {"sample": self.info(angle), "id": tid, "name": data.get("name_vi", ""), "grip": data.get("grip"),
                "frames": self._frames_from(s, tid, angle)}

    def _frame(self, s, fr: dict) -> tuple[Skeleton, bool, tuple, tuple, tuple]:
        """One frame from the editor, checked, and fitted to the sample: the bones keep their lengths (or are drawn
        shorter, pointing toward the viewer) and grounded feet stand on the ground."""
        if not isinstance(fr, dict):
            raise AppError("bad_request", detail="frame")
        pts = fr.get("points")
        if not isinstance(pts, list) or len(pts) != len(s.character.skeleton.points):
            raise AppError("bad_request", detail="points")
        clean = []
        for p in pts:
            if p is None:
                clean.append(None)
            elif isinstance(p, list) and len(p) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v)
                                                           for v in p):
                clean.append([float(p[0]), float(p[1])])
            else:
                raise AppError("bad_request", detail="points")
        grounded = fr.get("contact", "ground") != "air"
        opened = tuple(h for h in ("r", "l") if h in (fr.get("open") or []))
        front = tuple(m for m in FRONT_MESHES if m in (fr.get("front") or []))
        back = tuple(m for m in BACK_MESHES if m in (fr.get("back") or []))
        rest = self._rest(s)
        try:
            fitted = self._fit(s, Skeleton.from_json(self._to_template(s, clean)), rest, grounded)
        except (ValueError, AssertionError) as exc:
            raise AppError("pose_frame_bad") from exc
        return fitted, grounded, opened, front, back

    def preview(self, angle: str, fr: dict) -> dict:
        """The sample drawn in one frame's pose, and the joints where they ended up."""
        s = self._sample(angle)
        sk, _grounded, opened, front, back = self._frame(s, fr)
        image, _keep = s.rig.rig.render_keep(sk, {}, opened, s.character.placement.anchor[1], front, back)
        path = self.work / angle / f"preview_{uuid.uuid4().hex[:8]}.png"
        image.save(path)
        with self._lock:                  # only the latest preview of each view is kept
            old = self._last_preview.get(angle)
            self._last_preview[angle] = path
        if old is not None:
            old.unlink(missing_ok=True)
        return {"image": str(path), "points": sk.to_json()}

    # ------------------------------------------------------------------------------------------------------------
    # Saved poses
    # ------------------------------------------------------------------------------------------------------------
    def save(self, body: dict) -> dict:
        angle = body.get("angle")
        s = self._sample(angle)
        name = body.get("name")
        if not isinstance(name, str) or not name.strip():
            raise AppError("pose_name_missing")
        name = name.strip()[:NAME_MAX]
        grip = body.get("grip", "never")
        if grip not in GRIPS:
            raise AppError("bad_request", detail="grip")
        frames = body.get("frames")
        if not isinstance(frames, list) or len(frames) != 8:
            raise AppError("bad_request", detail="frames")
        tid = _tid(body["id"]) if body.get("id") else f"u_{uuid.uuid4().hex[:8]}"
        out_frames = []
        for fr in frames:
            sk, grounded, opened, front, back = self._frame(s, fr)
            entry = {"points": self._to_template(s, sk.to_json()), "contact": "ground" if grounded else "air"}
            if grip != "never":
                # The weapon points along the near forearm (0 = right, 90 = down), held in front.
                e, w = sk.points[RELB], sk.points[RWRI]
                ang = math.degrees(math.atan2(w[1] - e[1], w[0] - e[0])) if e and w else 0.0
                entry["weapon"] = {"hand": "r", "angle": round(ang, 1), "layer": "front", "stage": 1}
            if opened:
                entry["open"] = list(opened)
            if front:
                entry["front"] = list(front)
            if back:
                entry["back"] = list(back)
            out_frames.append(entry)
        data = {"id": tid, "animation": "custom", "variant": tid, "name_vi": name, "status": "Owner",
                "angle": angle, "grip": grip, "weapon_kinds": list(WEAPON_KINDS) if grip != "never" else [],
                "prompt": "full body, action pose" + (", clenched hand" if grip != "never" else ""),
                "rest": [None if p is None else [round(p[0], 4), round(p[1], 4)] for p in self._rest(s).points],
                "frames": out_frames}
        parse_template(data)                      # the same checks as the built-in poses
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / f"{tid}.json.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        tmp.replace(self.dir / f"{tid}.json")
        return {"id": tid, "name": name, "angle": angle, "grip": grip}

    def _read(self, tid: str) -> dict:
        try:
            with open(self.dir / f"{tid}.json", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError) as exc:
            raise AppError("pose_unknown") from exc

    def list(self) -> list[dict]:
        out = []
        for path in sorted(self.dir.glob("u_*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
            except (OSError, ValueError):
                continue
            out.append({"id": d.get("id", path.stem), "name": d.get("name_vi", ""), "angle": d.get("angle", "side"),
                        "grip": d.get("grip", "never")})
        return out

    def delete(self, tid: str) -> dict:
        path = self.dir / f"{_tid(tid)}.json"
        if not path.exists():
            raise AppError("pose_unknown")
        path.unlink()
        return {"id": tid, "deleted": True}

    def close(self) -> None:
        for s in self._sessions.values():
            s.close()
