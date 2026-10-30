"""Reads the character's look as Danbooru-style tags (WD ViT tagger v3, ONNX Runtime, CPU), once per character.

Animagine is trained on such tags; without them the drawing prompt says nothing about who is drawn, and Phase 0
(2026-10-30) drifted a male knight towards a female figure. Only appearance tags are kept: pose, view, framing,
background and anything held are decided by the template and the app, not by the reference picture.

Model files: ``<models>/wd-tagger/model.onnx`` and ``selected_tags.csv``. Input: 448×448, padded white, BGR, 0–255.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

from .base import EngineError

SIZE = 448
THRESHOLD = 0.35               # general tags; the model card's suggested range is 0.3–0.4
MAX_TAGS = 16
# Tags that describe the picture rather than the character (see module doc). Matched as whole words.
DROP = re.compile(
    r"\b(background|standing|sitting|walking|running|from (side|behind|above|below)|profile|full body|cowboy shot|"
    r"upper body|solo|looking|facing|arms?|hands?|holding|weapon|sword|bow|staff|shield|spear|axe|polearm|dagger|"
    r"knife|arrow|pose|legs?|feet|leaning|turning|motion|shadow|border|signature|watermark|text|simple|white|transparent|"
    r"grey background|male focus|realistic|closed mouth|open mouth|teeth|expressionless|serious|smile)\b")
# Kept even when matched above: what makes the character recognisable.
KEEP = {"male focus"}


class WDTagger:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self._session = None
        self._names: list[str] = []
        self._general: np.ndarray | None = None

    def available(self) -> bool:
        return (self.folder / "model.onnx").is_file() and (self.folder / "selected_tags.csv").is_file()

    def _load(self) -> None:
        if self._session is not None:
            return
        model, tags = self.folder / "model.onnx", self.folder / "selected_tags.csv"
        missing = [str(p) for p in (model, tags) if not p.is_file()]
        if missing:
            raise EngineError("models_missing", ", ".join(missing))
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        with open(tags, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self._names = [r["name"] for r in rows]
        self._general = np.array([r["category"] == "0" for r in rows])

    def scores(self, image: Image.Image) -> dict[str, float]:
        """Every general tag's probability for the figure (RGBA, transparent around it)."""
        self._load()
        assert self._session is not None and self._general is not None
        x = prepare(image)
        inp = self._session.get_inputs()[0].name
        probs = self._session.run(None, {inp: x})[0][0]
        return {self._names[i]: float(p) for i, p in enumerate(probs) if self._general[i]}

    def tags(self, image: Image.Image) -> list[str]:
        return appearance_tags(self.scores(image))


def prepare(image: Image.Image) -> np.ndarray:
    """The figure cropped, padded to a white square, 448², BGR float32 0–255, NHWC."""
    rgba = image.convert("RGBA")
    box = rgba.getchannel("A").point(lambda v: 255 if v > 20 else 0).getbbox() or (0, 0, *rgba.size)
    fig = rgba.crop(box)
    side = max(fig.size)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    canvas.alpha_composite(fig, ((side - fig.width) // 2, (side - fig.height) // 2))
    rgb = canvas.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    return np.ascontiguousarray(np.asarray(rgb, np.float32)[:, :, ::-1])[None]


def appearance_tags(scores: dict[str, float]) -> list[str]:
    """The confident appearance tags, most confident first, as prompt words (underscores to spaces)."""
    out = []
    for name, p in sorted(scores.items(), key=lambda kv: -kv[1]):
        if p < THRESHOLD:
            break
        words = name.replace("_", " ")
        if words not in KEEP and DROP.search(words):
            continue
        out.append(words)
        if len(out) == MAX_TAGS:
            break
    return out
