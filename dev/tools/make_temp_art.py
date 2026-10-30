"""Temporary test art for AnimGen (not the owner's art): characters and weapons from Animagine XL 4.0,
background removed with the app's own ToonOut remover.

usage: python make_temp_art.py <out_dir> <job> [seeds...]
jobs: hero (no weapon), hero_sword (combined), sword, bow
"""
import sys
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

sys.path.insert(0, r"D:\Game\ainm\animgen\backend")
from animgen.engines.diffusers_engine import ToonOutRemover  # noqa: E402

M = Path(r"C:\AnimGen\models")
Q = "masterpiece, high score, great score, absurdres"
NEG = ("lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, out of frame, "
       "worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry, "
       "text, shadow, gradient background, scenery, aura, effects, splashes, cape")
CHAR = ("1boy, solo, full body, standing, from side, profile, facing right, young knight, light leather armor, "
        "brown hair, short hair, boots, arms at sides, thick outlines, bold lineart, flat color, cel shading, "
        "western cartoon, game sprite, ")
JOBS = {
    "hero": (CHAR + "empty hands, simple background, white background, " + Q,
             NEG + ", weapon, sword, holding, holding weapon, bow, staff, shield", (832, 1216)),
    "hero_sword": (CHAR + "holding sword, one-handed sword, simple background, white background, " + Q,
                   NEG + ", bow, staff, shield, two swords", (832, 1216)),
    "sword": ("no humans, sword, longsword, one-handed sword, horizontal, side view, object focus, simple background, "
              "white background, game asset, " + Q, NEG + ", 1boy, 1girl, hand, holding", (1216, 832)),
    "bow": ("no humans, bow (weapon), wooden recurve bow, longbow, vertical, side view, object focus, bowstring, "
            "simple background, white background, game asset, " + Q,
            NEG + ", arrow, quiver, 1boy, 1girl, hand, holding", (832, 1216)),
}


def main() -> None:
    out, job = Path(sys.argv[1]), sys.argv[2]
    seeds = [int(s) for s in sys.argv[3:]] or [1, 2, 3, 4]
    out.mkdir(parents=True, exist_ok=True)
    prompt, neg, (w, h) = JOBS[job]
    vae = AutoencoderKL.from_pretrained(str(M / "sdxl-vae-fp16-fix"), torch_dtype=torch.float16, local_files_only=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(str(M / "animagine-xl-4.0"), vae=vae, torch_dtype=torch.float16,
                                                     local_files_only=True)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    remover = ToonOutRemover(M)
    for s in seeds:
        t = time.time()
        img = pipe(prompt=prompt, negative_prompt=neg, width=w, height=h, num_inference_steps=28, guidance_scale=5.0,
                   generator=torch.Generator("cpu").manual_seed(s)).images[0]
        img.save(out / f"{job}_{s}_raw.png")
        rgba = remover.remove(img, (255, 255, 255))
        rgba.save(out / f"{job}_{s}.png")
        print(job, s, f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
