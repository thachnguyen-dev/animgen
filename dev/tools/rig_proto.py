"""Prototype of the layered cut-out approach (DECISIONS.md, "Layered cut-out character"): one AI rig image per
character, then every animation from layers with no AI per frame.
usage: python rig_proto.py <out_dir> <character.png>... [--str=0.6,0.85] [--seeds=1234] [--reuse=1] [--finish=a7,...]
--reuse=1 builds from the rig and hand images already in <out_dir> (no GPU, unless --finish).
--finish: animations that also get the AI finishing pass (finish.py), saved as <tag>_<anim>_fin.gif.
"""
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, r"D:\Game\ainm\animgen\backend")
import animgen.engines.diffusers_engine as de  # noqa: E402
from animgen.authoring import Pose, solve  # noqa: E402
from animgen.engines.base import FrameRequest  # noqa: E402
from animgen.finish import finish  # noqa: E402
from animgen.rig import build_rig  # noqa: E402
from animgen.server import make_engine  # noqa: E402
from animgen.session import Session  # noqa: E402
from animgen.skeleton import Skeleton, fit_frame, measure  # noqa: E402
from animgen.templates import load_template  # noqa: E402

ANIMS = ["idle", "walk", "run", "a1a", "a7"]
# Near arm forward, far arm back, wide stride: no limb covers another in a side view.
SPREAD = Pose(spine=2, head=0, r_sh=40, r_el=20, l_sh=-40, l_el=20, r_hip=28, r_knee=6, l_hip=-26, l_knee=10)
PROMPT = "standing, wide stride, legs apart, one arm forward, one arm back, clenched fists"
# The hand image: the near arm stretched forward with an open palm.
HAND = Pose(spine=2, head=0, r_sh=90, r_el=0, l_sh=-40, l_el=20, r_hip=28, r_knee=6, l_hip=-26, l_knee=10)
HAND_PROMPT = "standing, arm outstretched forward, open palm, fingers spread"
HAND_STRENGTH = 0.75

out = Path(sys.argv[1])
chars = [a for a in sys.argv[2:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--"))
strengths = [float(x) for x in opts.get("str", "0.6,0.85").split(",")]
seeds = [int(x) for x in opts.get("seeds", "1234").split(",")]
out.mkdir(parents=True, exist_ok=True)

REUSE = opts.get("reuse") == "1"
FINISH = [a for a in opts.get("finish", "").split(",") if a]
engine = make_engine("diffusers", Path(r"C:\AnimGen\models"))
if not REUSE or FINISH:
    engine.load(print)
rest = load_template("idle").rest
pts, _ = solve(SPREAD)
spread_tpl = Skeleton([None if p is None else (p[0], p[1]) for p in pts])
pts, _ = solve(HAND)
hand_tpl = Skeleton([None if p is None else (p[0], p[1]) for p in pts])


def strip(frames: list[Image.Image], height: int = 300) -> Image.Image:
    box = None
    for im in frames:
        b = im.getchannel("A").getbbox()
        box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    box = (box[0] - 10, box[1] - 10, box[2] + 10, box[3] + 10)
    tiles = []
    for im in frames:
        c = im.crop(box)
        bg = Image.new("RGBA", c.size, (235, 235, 235, 255))
        bg.alpha_composite(c)
        tiles.append(bg.convert("RGB").resize((int(c.width * height / c.height), height)))
    row = Image.new("RGB", (sum(t.width for t in tiles), height), "white")
    x = 0
    for t in tiles:
        row.paste(t, (x, 0))
        x += t.width
    return row, tiles


for char in chars:
    name = Path(char).stem
    s = Session(engine, out / f"work_{name}")
    ch = s.set_character(char, None)
    props = measure(ch.skeleton, rest, ch.placement.figure_height, ch.placement.anchor[1])
    rig_sk = fit_frame(spread_tpl, rest, props, ch.placement.anchor, True)
    hand_sk = fit_frame(hand_tpl, rest, props, ch.placement.anchor, True)
    hand_path = out / f"{name}_hand.png"
    if not hand_path.exists():
        de.STRENGTH = HAND_STRENGTH
        req = FrameRequest(reference=ch.placement.image, reference_skeleton=ch.skeleton, target_skeleton=hand_sk,
                           background=s.background, prompt=HAND_PROMPT, seed=seeds[0], hands_gripping=False)
        engine.remover.remove(engine.drawer.draw(req), s.background).save(hand_path)
    hand_img = Image.open(hand_path).convert("RGBA")
    for st in strengths:
        for seed in seeds:
            de.STRENGTH = st
            t0 = time.time()
            tag = f"{name}_s{int(st * 100)}_{seed}"
            if REUSE:
                rgba = Image.open(out / f"{tag}_rig.png").convert("RGBA")
            else:
                req = FrameRequest(reference=ch.placement.image, reference_skeleton=ch.skeleton,
                                   target_skeleton=rig_sk, background=s.background, prompt=PROMPT, seed=seed,
                                   hands_gripping=True)
                rgba = engine.remover.remove(engine.drawer.draw(req), s.background)
                rgba.save(out / f"{tag}_rig.png")
            rig = build_rig(ch.placement.image, ch.skeleton, rgba, rig_sk, hand_img, hand_sk)
            rows, rows_fin = [], []
            for anim in ANIMS:
                tpl = load_template(anim)
                p2 = measure(ch.skeleton, tpl.rest, ch.placement.figure_height, ch.placement.anchor[1])
                targets = [fit_frame(f.skeleton, tpl.rest, p2, ch.placement.anchor, f.grounded) for f in tpl.frames]
                hands = [f.open_hands for f in tpl.frames]
                frames = rig.render_sequence(targets, loop=tpl.animation != "attack", open_hands=hands)
                if anim in FINISH:
                    swings = rig.swings(targets, loop=tpl.animation != "attack")
                    fin = [finish(engine, im, rig.render(t, sw, o, head_only=True), ch.placement.image, ch.skeleton,
                                  t, s.background, tpl.prompt, seed, tpl.gripping(None))
                           for im, t, sw, o in zip(frames, targets, swings, hands)]
                    if ch.placement.mirrored:
                        fin = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in fin]
                    _, ftiles = strip(fin)
                    ftiles[0].save(out / f"{tag}_{anim}_fin.gif", save_all=True, append_images=ftiles[1:], duration=125,
                                   loop=0)
                    rows_fin.append(strip(fin)[0])
                if ch.placement.mirrored:
                    frames = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in frames]
                row, tiles = strip(frames)
                tiles[0].save(out / f"{tag}_{anim}.gif", save_all=True, append_images=tiles[1:], duration=125, loop=0)
                rows.append(row)
            sheet = Image.new("RGB", (max(r.width for r in rows), 300 * len(rows)), "white")
            for i, r in enumerate(rows):
                sheet.paste(r, (0, 300 * i))
            sheet.save(out / f"{tag}_sheet.png")
            if rows_fin:
                fs = Image.new("RGB", (max(r.width for r in rows_fin), 300 * len(rows_fin)), "white")
                for i, r in enumerate(rows_fin):
                    fs.paste(r, (0, 300 * i))
                fs.save(out / f"{tag}_fin_sheet.png")
            print(tag, round(time.time() - t0, 1), "s", flush=True)
    s.close()
print("done")
