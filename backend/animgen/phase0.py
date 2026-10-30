"""Phase 0 feasibility run (§13.4) on the owner's PC: Idle, Walk, A2 (one-handed) and A5 (bow) on the owner's art.

Measures what the cloud build could not: time per frame, peak GPU memory, and produces contact sheets to judge
frame consistency, empty-grip hands and the placed weapon by eye. Nothing here is scored automatically (§8.1).

  python -m animgen.phase0 --data D:\\AnimGen --character hero.png ^
      --sword sword.png --sword-grip 32,30 --sword-tip 350,30 ^
      --bow b1.png b2.png b3.png --bow-grip 66,210 --bow-tip 78,22 --out D:\\phase0

Grip and tip are pixel positions in the weapon image (the same points the app's weapon dialog records).
Use --engine demo to rehearse the run without the GPU.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .session import Session, WeaponChoice
from .tasks import TaskRunner


def _xy(s: str) -> tuple[float, float]:
    x, y = s.split(",")
    return float(x), float(y)


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
    ap.add_argument("--sword")
    ap.add_argument("--sword-grip", type=_xy)
    ap.add_argument("--sword-tip", type=_xy)
    ap.add_argument("--bow", nargs=3)
    ap.add_argument("--bow-grip", type=_xy, help="same points for the 3 stage images")
    ap.add_argument("--bow-tip", type=_xy)
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

    runs: list[tuple[str, WeaponChoice | None]] = [("idle", None), ("walk", None)]
    if a.sword:
        w = s.weapons.add("phase0-sword", "one_handed", "a", [a.sword])
        s.weapons.set_points(w.id, "a", 1, a.sword_grip, a.sword_tip)
        runs.append(("a2", WeaponChoice(w.id, "a")))
    if a.bow:
        w = s.weapons.add("phase0-bow", "bow", "a", list(a.bow))
        for st in (1, 2, 3):
            s.weapons.set_points(w.id, "a", st, a.bow_grip, a.bow_tip, apply_to_all_skins=False)
        runs.append(("a5", WeaponChoice(w.id, "a")))

    torch = None
    if a.engine == "diffusers":
        import torch as _t
        torch = _t
    runner = TaskRunner()
    for anim, weapon in runs:
        if torch is not None:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        task = runner.wait(runner.start(anim, lambda t, an=anim, wp=weapon: s.generate(t, an, wp, a.seed)).id,
                           timeout=6 * 3600)
        dt = time.time() - t0
        st = s.state()
        entry = {"animation": anim, "state": task.state, "error": task.error, "seconds": round(dt, 1),
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
