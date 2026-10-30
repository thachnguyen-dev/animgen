"""Export (§10): 8 transparent PNG frames and one sprite sheet per animation.

Sheet layout (Proposed, cut C8): one row, 8 frames in play order, no padding, each cell exactly the frame size.
Unity: Sprite Mode Multiple, Sprite Editor > Slice > Grid By Cell Count, 1 row x 8 columns.
File names (Proposed, §10.3): ``<character>_<animation>_<01-08>.png`` and ``..._sheet.png``.
Animations drawn with gripping hands also get ``..._weapon.json``: where the weapon hand is in every frame, so the
owner attaches the weapon in the game (the app draws characters without weapons, owner 2026-10-30).
Files are written to a hidden folder first, checked, then moved into place, so a failed export never leaves a
mix of old and new files that looks complete.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
import uuid
from pathlib import Path

from PIL import Image

from . import config
from .errors import AppError
from .frames import validate_file
from .imaging import premultiplied_resize


def slug(text: str) -> str:
    """ASCII file-name part: diacritics removed (Vietnamese đ handled), spaces to '-', other characters dropped."""
    t = text.replace("đ", "d").replace("Đ", "D")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^A-Za-z0-9_-]+", "-", t).strip("-").lower()
    return t or "x"


def base_name(character: str, animation: str) -> str:
    return f"{slug(character)}_{slug(animation)}"


def sprite_sheet(frames: list[Image.Image]) -> Image.Image:
    w, h = frames[0].size
    sheet = Image.new("RGBA", (w * len(frames), h), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        if f.size != (w, h):
            raise AppError("export_verify_failed", detail="frames differ in size")
        sheet.paste(f, (i * w, 0))
    return sheet


def export_animation(frames: list[Image.Image], out_dir: str | Path, height: int, name: str,
                     progress=None, anchors: dict | None = None) -> list[Path]:
    if len(frames) != config.FRAME_COUNT or any(f is None for f in frames):
        raise AppError("export_incomplete")
    if height not in config.EXPORT_HEIGHTS:
        raise AppError("export_bad_size")
    out = Path(out_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AppError("export_bad_folder") from exc
    if not out.is_dir():
        raise AppError("export_bad_folder")

    size = (height, height)
    scaled = [f if f.size == size else premultiplied_resize(f, size) for f in frames]
    staging = out / f".animgen-export-{uuid.uuid4().hex[:8]}"
    written: list[tuple[Path, Path, tuple[int, int]]] = []
    try:
        staging.mkdir()
        for i, f in enumerate(scaled):
            fname = f"{name}_{i + 1:02d}.png"
            f.save(staging / fname, format="PNG")
            written.append((staging / fname, out / fname, size))
            if progress:
                progress(i + 1, config.FRAME_COUNT + 1)
        sheet = sprite_sheet(scaled)
        sname = f"{name}_sheet.png"
        sheet.save(staging / sname, format="PNG")
        written.append((staging / sname, out / sname, sheet.size))
        data: list[tuple[Path, Path]] = []
        if anchors is not None:
            jname = f"{name}_weapon.json"
            with open(staging / jname, "w", encoding="utf-8") as f:
                json.dump(anchors, f, ensure_ascii=False, indent=1)
            data.append((staging / jname, out / jname))
    except OSError as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise AppError("export_write_failed", detail=str(exc)) from exc

    for tmp, _final, sz in written:
        chk = validate_file(tmp, sz)
        if not chk.ok:
            shutil.rmtree(staging, ignore_errors=True)
            raise AppError("export_verify_failed", detail=f"{tmp.name}: {chk.reason}")
    try:
        for tmp, final in [(t, f) for t, f, _ in written] + data:
            tmp.replace(final)
    except OSError as exc:
        raise AppError("export_write_failed", detail=str(exc)) from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if progress:
        progress(config.FRAME_COUNT + 1, config.FRAME_COUNT + 1)
    return [final for _, final, _ in written] + [final for _, final in data]
