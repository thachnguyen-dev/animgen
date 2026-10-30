"""Time the AnimGen drawer under different GPU placements. usage: python bench_place.py <config> [steps]

configs:
  current   UNet streamed (leaf), everything else resident  (the cloud build)
  unet_res  UNet resident; ControlNet, both text encoders and the image encoder streamed (leaf); VAE resident
"""
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r"D:\Game\ainm\animgen\backend")
from diffusers.hooks import apply_group_offloading  # noqa: E402

import animgen.engines.diffusers_engine as de  # noqa: E402
from animgen.engines.base import FrameRequest  # noqa: E402
from animgen.skeleton import Skeleton  # noqa: E402

cfg = sys.argv[1]
de.STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
M = de.Path(r"C:\AnimGen\models")
cuda, cpu = torch.device("cuda"), torch.device("cpu")


def stream(m):
    apply_group_offloading(m, onload_device=cuda, offload_device=cpu, offload_type="leaf_level", use_stream=True,
                           low_cpu_mem_usage=True)


def stream_parts(model):
    """Stream a model leaf by leaf, except children holding nn.MultiheadAttention (its out_proj weights are read
    outside forward(), so the offload hooks never move them): those stay on the GPU, as do the root's own tensors."""
    for _, child in model.named_children():
        if any(isinstance(m, torch.nn.MultiheadAttention) for m in child.modules()):
            child.to(cuda)
        else:
            stream(child)
    for _, prm in model.named_parameters(recurse=False):
        prm.data = prm.data.to(cuda)
    for n, buf in model.named_buffers(recurse=False):
        model._buffers[n] = buf.to(cuda)


orig_apply = apply_group_offloading
if cfg in ("unet_res", "fp8"):
    # Load as the engine does but intercept the placement step.
    import diffusers.hooks as dh

    def fake_apply(module, **kw):
        pass
    dh.apply_group_offloading = fake_apply

d = de.SDXLDrawer(M)
t = time.time()
d.load(None)
p = d.pipe
if cfg == "unet_res":
    for part in (p.controlnet, p.text_encoder, p.text_encoder_2, p.image_encoder):
        part.to(cpu)
    torch.cuda.empty_cache()
    p.unet.to(cuda)
    stream_parts(p.controlnet)
    for part in (p.text_encoder, p.text_encoder_2, p.image_encoder):
        stream(part)
    p.vae.to(cuda)
if cfg == "fp8":
    from diffusers.hooks import apply_layerwise_casting
    for part in (p.unet, p.controlnet, p.text_encoder, p.text_encoder_2, p.image_encoder):
        part.to(cpu)
    torch.cuda.empty_cache()
    for part in (p.unet, p.controlnet, p.text_encoder_2, p.image_encoder):
        apply_layerwise_casting(part, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.float16,
                                skip_modules_pattern="auto", skip_modules_classes=(torch.nn.MultiheadAttention,))
    for part in (p.unet, p.controlnet, p.text_encoder, p.text_encoder_2, p.image_encoder, p.vae):
        part.to(cuda)
print(cfg, "load", round(time.time() - t), "s; resident GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)

SIZE = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
k = SIZE / 1024
ref = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
ref.paste((180, 90, 60, 255), tuple(int(v * k) for v in (400, 300, 600, 950)))
pts = [(500, 320), (500, 400), (460, 400), (450, 520), (450, 640), (540, 400), (550, 520), (550, 640),
       (480, 640), (470, 790), (470, 940), (520, 640), (530, 790), (530, 940), (490, 310), (510, 310), (480, 315),
       (520, 315)]
sk = Skeleton([(x * k, y * k) for x, y in pts])
req = FrameRequest(reference=ref, reference_skeleton=sk, target_skeleton=sk, background=(255, 255, 255),
                   prompt="standing", seed=1, hands_gripping=False)
for run in range(2):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    img = d.draw(req)
    torch.cuda.synchronize()
    dt = time.time() - t
    print(cfg, SIZE, f"run{run}", f"{dt:.1f}s", f"{dt / de.STEPS:.2f}s/step", "peak GB",
          round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
img.save(f"bench_{cfg}_{SIZE}.png")
