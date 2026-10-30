"""Single-frame probes of drawing setups. usage: python probe.py <character.png> <animation> <frame 1-8> <out_dir> <variant>...

variants (each one image, saved as <variant>.png = raw output):
  base      two panels, pose + inpaint controls (the engine as is)
  pose2     two panels, pose control only (no inpaint control)
  posecn6   two panels, pose + inpaint, ControlNet 0.6
  single    one panel, pose control only, IP-Adapter carries the look
  single8   one panel, pose only, IP-Adapter 0.8
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r"D:\Game\ainm\animgen\backend")
import animgen.engines.diffusers_engine as de  # noqa: E402
from animgen import config  # noqa: E402
from animgen.server import make_engine  # noqa: E402
from animgen.session import Session  # noqa: E402
from animgen.skeleton import draw_openpose, fit_frame, measure  # noqa: E402
from animgen.templates import load_template  # noqa: E402

char, anim, fidx, out = sys.argv[1], sys.argv[2], int(sys.argv[3]) - 1, Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
engine = make_engine("diffusers", Path(r"C:\AnimGen\models"))
s = Session(engine, out / "work")
ch = s.set_character(char, None)
print("facing:", ch.facing, ch.facing_source if hasattr(ch, "facing_source") else "")
tpl = load_template(anim)
props = measure(ch.skeleton, tpl.rest, ch.placement.figure_height, ch.placement.anchor[1])
tf = tpl.frames[fidx]
target = fit_frame(tf.skeleton, tpl.rest, props, ch.placement.anchor, tf.grounded)
engine.load()
pipe = engine.drawer.pipe
k = de.DRAW_SIZE / config.CANVAS_SIZE
size = de.DRAW_SIZE
ref = ch.placement.image.resize((size, size), Image.Resampling.LANCZOS)
rsk, tsk = de._scaled(ch.skeleton, k), de._scaled(target, k)
bg = s.background
ref_flat = Image.new("RGB", ref.size, bg)
ref_flat.paste(ref, mask=ref.getchannel("A"))
hands = "clenched hands, empty hands, " if tpl.weapon in ("required", "optional") and anim.startswith("a") else ""
prompt = de.POSITIVE.format(motion=tpl.prompt, hands=hands, bg=de.BACKGROUND_WORDS.get(tuple(bg), "plain background"))
negative = de.NEGATIVE.format(weapon="weapon, holding weapon, sword, bow, staff, effects")
print("prompt:", prompt)


sys.path.insert(0, str(Path(__file__).parent))
from animgen.puppet import repose  # noqa: E402

rough = repose(ref, rsk, tsk)
rough_flat = Image.new("RGB", rough.size, bg)
rough_flat.paste(rough, mask=rough.getchannel("A"))
rough_flat.save(out / "rough.png")


def run(name, panels, inpaint_ctrl, cn, ip, tags="", strength=1.0):
    pipe.set_ip_adapter_scale(ip)
    pr = prompt.replace("simple background", (tags + ", " if tags else "") + "simple background")
    if panels == 2:
        width = size * 2
        canvas = Image.new("RGB", (width, size), bg)
        canvas.paste(ref_flat, (0, 0))
        if strength < 1.0:
            canvas.paste(rough_flat, (size, 0))
        mask = Image.new("L", (width, size), 0)
        mask.paste(255, (size, 0, width, size))
        pose = np.zeros((size, width, 3), np.uint8)
        draw_openpose(rsk, (size, size), canvas=pose[:, :size])
        draw_openpose(tsk, (size, size), canvas=pose[:, size:])
        masked = np.asarray(canvas).copy()
        masked[:, size:] = 0
    else:
        width = size
        canvas = Image.new("RGB", (size, size), bg)
        mask = Image.new("L", (size, size), 255)
        pose = draw_openpose(tsk, (size, size))
        masked = np.zeros((size, size, 3), np.uint8)
    if inpaint_ctrl:
        ctrl, modes = [Image.fromarray(pose), Image.fromarray(masked)], [0, 7]
    else:
        ctrl, modes = [Image.fromarray(pose)], [0]
    with torch.inference_mode():
        img = pipe(prompt=pr, negative_prompt=negative, image=canvas, mask_image=mask, control_image=ctrl,
                   control_mode=modes, ip_adapter_image=ref_flat, width=width, height=size, num_inference_steps=24,
                   guidance_scale=de.GUIDANCE, strength=strength, controlnet_conditioning_scale=cn,
                   generator=torch.Generator("cpu").manual_seed(1234)).images[0]
    img.save(out / f"{name}.png")
    print(name, "done", flush=True)


V = {"base": (2, True, 0.9, 0.6), "pose2": (2, False, 0.9, 0.6), "posecn6": (2, True, 0.6, 0.6),
     "single": (1, False, 0.9, 0.6), "single8": (1, False, 0.9, 0.8)}
for v in sys.argv[5:]:
    if v in V:
        run(v, *V[v])
    else:  # name:ip=..:cn=..:tags=a|b|c   (two panels, pose only)
        name, *opts = v.split(":")
        o = dict(x.split("=", 1) for x in opts)
        run(name, 2, False, float(o.get("cn", 0.9)), float(o.get("ip", 0.6)), o.get("tags", "").replace("|", ", "),
            float(o.get("st", 1.0)))
