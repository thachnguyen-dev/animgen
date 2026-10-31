"""A test character made of simple shapes, for checking and tuning the motion templates (owner, 2026-10-31).

Every body part is its own flat-coloured shape with a dark outline, so the parts are known exactly: the script draws
the character image, confirms its skeleton and writes its rig (the body-part layers and their Photoshop file) straight
into the app's data folder. No AI drawing is needed: opening the image in the app finds the rig at once, and every
animation is made in seconds. Then it renders all the animations into sheets and GIFs.

usage: python shape_character.py [--view side|34] [--data C:/AnimGen] [--out D:/Game/ainm/pose_test]
(run with the app closed: it writes into the app's data folder)

--view 34 draws the character in a three-quarter view (owner, 2026-10-31): turned a little toward the viewer, so the
far shoulder and hip sit further back, the torso is wider, both eyes show and the far limbs are in plain view. It goes
to test_shapes_34.png and D:/Game/ainm/pose_test_34 by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / "backend"))
from animgen.engines.demo import DemoEngine  # noqa: E402
from animgen.sample import make  # noqa: E402  (the shapes live in the backend: the pose editor uses them too)
from animgen.session import Session  # noqa: E402
from animgen.tasks import TaskRunner  # noqa: E402
from animgen.templates import ANIMATIONS  # noqa: E402

IMAGE = HERE.parents[2] / "dev" / "test_art" / "test_shapes.png"
IMAGE_34 = HERE.parents[2] / "dev" / "test_art" / "test_shapes_34.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=("side", "34"), default="side")
    ap.add_argument("--data", default="C:/AnimGen")
    ap.add_argument("--out")
    args = ap.parse_args()
    view = args.view
    data = Path(args.data)
    out = Path(args.out or ("D:/Game/ainm/pose_test" if view == "side" else "D:/Game/ainm/pose_test_34"))
    image_path = IMAGE if view == "side" else IMAGE_34

    # 1-3. The character image, its skeleton confirmed and its rig written into the app's data folder.
    s = Session(DemoEngine(), data, require_rig=True)
    ch = make(s, view, image_path)
    print("character", image_path, "rig", s.rig_dir / ch.key)

    # 4. Every animation, as a sheet (one row each) and a GIF.
    out.mkdir(parents=True, exist_ok=True)
    runner = TaskRunner()
    rows = []
    for tid, _anim, _variant, name_vi, ready in ANIMATIONS:
        if not ready:
            continue
        t = runner.wait(runner.start("t", lambda task: s.generate(task, tid, None, seed=1)).id, 600)
        if t.state != "completed":
            print(tid, t.state, t.error)
            continue
        frames = [Image.open(f["preview"]).convert("RGBA") for f in s.state()["frames"] if f["preview"]]
        rows.append((tid, name_vi, frames))
    box = None
    for _, _, frames in rows:
        for im in frames:
            b = im.getchannel("A").getbbox()
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    box = (max(0, box[0] - 12), max(0, box[1] - 12), box[2] + 12, box[3] + 12)
    cw, chh = box[2] - box[0], box[3] - box[1]
    label = 34
    for tid, name_vi, frames in rows:
        tiles = []
        for im in frames:
            bg = Image.new("RGBA", im.size, (215, 215, 215, 255))
            bg.alpha_composite(im)
            tiles.append(bg.convert("RGB").crop(box))
        strip = Image.new("RGB", (cw * 8, chh + label), (255, 255, 255))
        d = ImageDraw.Draw(strip)
        d.text((8, 8), f"{tid} - {name_vi}", fill=(0, 0, 0))
        for i, t in enumerate(tiles):
            strip.paste(t, (i * cw, label))
            d.text((i * cw + 6, label + 4), str(i + 1), fill=(60, 60, 60))
        strip.save(out / f"{tid}.png")
        tiles[0].save(out / f"{tid}.gif", save_all=True, append_images=tiles[1:], duration=110, loop=0)
    sheet = Image.new("RGB", (cw * 8, (chh + label) * len(rows)), (255, 255, 255))
    for r, (tid, _, _) in enumerate(rows):
        sheet.paste(Image.open(out / f"{tid}.png"), (0, r * (chh + label)))
    sheet.save(out / "all.png")
    print("saved", out, len(rows), "animations")


if __name__ == "__main__":
    main()
