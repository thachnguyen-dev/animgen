"""Phase 0 feasibility run (§13.4) on the owner's PC: animations on one character, one model load.

Measures what the cloud build could not: time per frame, peak GPU memory, and produces contact sheets to judge
frame consistency and the gripping hands by eye. Nothing here is scored automatically (§8.1).

  python -m animgen.phase0 --data C:\\AnimGen --character hero.png --anims idle walk a2 --frames 4 --out D:\\phase0

--frames 4 draws frames 1, 3, 5 and 7 only, to test for half the GPU time (owner, 2026-10-30); the app always
draws 8. --grip closes the hands where the animation leaves it open (Idle, Walk, Run).
Use --engine demo to rehearse the run without the GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .session import Session
from .tasks import TaskRunner

DEFAULT_ANIMS = ["idle", "walk", "a1a", "a2", "a5", "a6", "a7"]


def sheet(paths: list[str | None], cell: int = 384) -> Image.Image:
    out = Image.new("RGBA", (cell * 8, cell), (230, 230, 230, 255))
    for i, p in enumerate(paths):
        if p:
            with Image.open(p) as im:
                out.alpha_composite(im.convert("RGBA").resize((cell, cell), Image.Resampling.LANCZOS), (i * cell, 0))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", default="diffusers", choices=["diffusers", "demo"])
    ap.add_argument("--character", required=True)
    ap.add_argument("--facing", choices=["left", "right"])
    ap.add_argument("--anims", nargs="+", default=DEFAULT_ANIMS, help="animation ids to run")
    ap.add_argument("--grip", action="store_true", help="hands closed where the animation leaves it open")
    ap.add_argument("--frames", type=int, choices=[4, 8], default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(a.data)
    from .server import make_engine
    engine = make_engine(a.engine, data / "models")
    s = Session(engine, out / "work")
    report: dict = {"engine": a.engine, "runs": []}
    ch = s.set_character(a.character, a.facing)
    report["character"] = ch.to_json()

    only = [0, 2, 4, 6] if a.frames == 4 else None

    torch = None
    if a.engine == "diffusers":
        import torch as _t
        torch = _t
    runner = TaskRunner()
    for anim in a.anims:
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        task = runner.wait(runner.start(anim, lambda t, an=anim: s.generate(t, an, a.grip, a.seed, only)).id,
                           timeout=6 * 3600)
        dt = time.time() - t0
        st = s.state()
        entry = {"animation": anim, "grip": st["grip"], "state": task.state, "error": task.error, "seconds": round(dt, 1),
                 "frames": [{k: f[k] for k in ("index", "status", "attempts")} for f in st["frames"]]}
        if torch is not None:
            entry["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        report["runs"].append(entry)
        sheet([f["preview"] for f in st["frames"]]).save(out / f"{anim}.png")
        print(json.dumps(entry, ensure_ascii=False), flush=True)

    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    s.close()
    print(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
