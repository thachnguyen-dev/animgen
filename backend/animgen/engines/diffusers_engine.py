"""The real engine (§6.5, §9.2). Needs an NVIDIA GPU and the model folders; NOT run in the cloud build.

Per frame (local finalisation, 2026-10-30; see DECISIONS.md "Drawing method"):
  1. A wide canvas: the reference character in the left half; in the right half the reference cut into body parts
     and moved onto the target skeleton (puppet.repose), on the flat background.
  2. Repaint the right half partly (STRENGTH) with Animagine XL 4.0 guided by ControlNet Union ProMax in pose mode,
     the pose control showing the reference skeleton on the left and the target skeleton on the right.
  3. IP-Adapter Plus also feeds the reference as an image prompt.
  4. ToonOut (BiRefNet fine-tuned for anime) gives real alpha.

Memory on 8 GB (§14, Inherited—Verify): the UNet is streamed with leaf-level group offloading; ControlNet, the VAE
(with tiling), both text encoders and the image encoder stay on the GPU. Moving the whole UNet at once spilled
into Windows shared memory in chargen (528 s per image).

Model folder layout (made by scripts/download_models.py):
  <models>/animagine-xl-4.0/               diffusers folder (model_index.json, unet/, vae/, text_encoder*/ ...)
  <models>/controlnet-union-sdxl-promax/   config.json + diffusion_pytorch_model.safetensors
  <models>/ip-adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors
  <models>/ip-adapter/models/image_encoder/ (CLIP ViT-H)
  <models>/sdxl-vae-fp16-fix/
  <models>/dwpose/yolox_l.onnx, dw-ll_ucoco_384.onnx
  <models>/birefnet/ (BiRefNet code + config) and <models>/toonout/birefnet_finetuned_toonout.pth

All values marked TUNE are Proposed starting points for Phase 0.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
from PIL import Image

from ..puppet import repose
from ..skeleton import Skeleton, draw_openpose
from .base import Engine, EngineError, FrameRequest, ProgressFn
from .dwpose import DWPoseReader

STEPS = 28                      # TUNE (with STRENGTH 0.6 about 17 of them run)
STRENGTH = 0.6                  # TUNE: share of the rough repose that is repainted. Probes: 0.45 leaves cut seams,
                                # 0.75 starts to redraw the outfit; 1.0 (from noise) lost the character entirely.
GUIDANCE = 5.0                  # TUNE (Animagine 4 recommends 4–7)
CONTROL_SCALE = 0.9             # TUNE
IP_ADAPTER_SCALE = 0.6          # TUNE
REFERENCE_PANEL = True          # cut C10 would set this False (IP-Adapter alone, faster, weaker consistency)
# Each frame is drawn at this size and scaled back to the canvas. Measured on the owner's 8 GB RTX 4060 laptop
# (two panels, 10 steps): 1024 -> 9.0 s/step (spills into shared memory), 896 -> 3.2 s/step, 768 -> 2.6 s/step.
# 768 since the owner's note that the sprites are for mobile (exports at most 512); SDXL degrades well below it.
DRAW_SIZE = 768

BACKGROUND_WORDS = {(255, 255, 255): "white background", (128, 128, 128): "grey background",
                    (40, 40, 40): "black background"}
# Templates face right, and the reference panel shows the character facing right (§ Facing in DECISIONS.md).
POSITIVE = ("solo, full body, from side, profile, facing right, {motion}, {hands}simple background, {bg}, "
            "masterpiece, high score, great score, absurdres")
# Phase 0 on test art (2026-10-30): frames turned away or faced left, and one held a second, faint figure.
NEGATIVE = ("lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, extra arms, extra legs, "
            "cropped, out of frame, worst quality, low quality, low score, bad score, average score, signature, "
            "watermark, username, blurry, text, shadow, gradient background, scenery, multiple views, "
            "character sheet, reference sheet, 2boys, 2girls, multiple boys, multiple girls, from behind, "
            "looking back, facing left, {weapon}")

REQUIRED = [
    "animagine-xl-4.0/model_index.json",
    "controlnet-union-sdxl-promax/config.json",
    "ip-adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
    "ip-adapter/models/image_encoder/config.json",
    "sdxl-vae-fp16-fix/config.json",
    "dwpose/yolox_l.onnx",
    "dwpose/dw-ll_ucoco_384.onnx",
    "birefnet/config.json",
    "toonout/birefnet_finetuned_toonout.pth",
]


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    return name == "OutOfMemoryError" or "out of memory" in str(exc).lower()


class ToonOutRemover:
    def __init__(self, models: Path) -> None:
        self.models = models
        self.model = None

    def _load(self):
        import torch
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(str(self.models / "birefnet"), trust_remote_code=True,
                                                              local_files_only=True)
        state = torch.load(self.models / "toonout" / "birefnet_finetuned_toonout.pth", map_location="cpu",
                           weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 10:
            raise EngineError("models_missing", f"ToonOut weights do not match BiRefNet ({len(missing)} missing)")
        model.eval().half()
        self.model = model

    def remove(self, image: Image.Image, background: tuple[int, int, int]) -> Image.Image:
        try:
            return self._remove(image)
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise EngineError("oom", str(exc)) from exc
            raise EngineError("inference", f"ToonOut: {type(exc).__name__}: {exc}") from exc

    def _remove(self, image: Image.Image) -> Image.Image:
        import torch
        if self.model is None:
            self._load()
        rgb = image.convert("RGB")
        w, h = rgb.size
        x = np.asarray(rgb.resize((1024, 1024), Image.Resampling.BILINEAR), np.float32) / 255.0
        x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        t = torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float16))
        self.model.to("cuda")
        try:
            with torch.no_grad():
                pred = self.model(t.to("cuda"))[-1].sigmoid()[0, 0].float().cpu().numpy()
        finally:
            self.model.to("cpu")
            torch.cuda.empty_cache()
        alpha = Image.fromarray((pred * 255).astype(np.uint8)).resize((w, h), Image.Resampling.BILINEAR)
        out = rgb.convert("RGBA")
        out.putalpha(alpha)
        return out


class SDXLDrawer:
    def __init__(self, models: Path) -> None:
        self.models = models
        self.pipe = None

    def load(self, progress: ProgressFn | None) -> None:
        import torch
        from diffusers import (AutoencoderKL, ControlNetUnionModel,
                               StableDiffusionXLControlNetUnionInpaintPipeline)
        from transformers import CLIPVisionModelWithProjection
        say = progress or (lambda m: None)
        dt = torch.float16
        say("Đang tải mô hình (1/4)…")
        controlnet = ControlNetUnionModel.from_pretrained(str(self.models / "controlnet-union-sdxl-promax"),
                                                          torch_dtype=dt, local_files_only=True)
        vae = AutoencoderKL.from_pretrained(str(self.models / "sdxl-vae-fp16-fix"), torch_dtype=dt,
                                            local_files_only=True)
        say("Đang tải mô hình (2/4)…")
        encoder = CLIPVisionModelWithProjection.from_pretrained(
            str(self.models / "ip-adapter" / "models" / "image_encoder"), torch_dtype=dt, local_files_only=True)
        say("Đang tải mô hình (3/4)…")
        pipe = StableDiffusionXLControlNetUnionInpaintPipeline.from_pretrained(
            str(self.models / "animagine-xl-4.0"), controlnet=controlnet, vae=vae, image_encoder=encoder,
            torch_dtype=dt, local_files_only=True, low_cpu_mem_usage=True)
        pipe.load_ip_adapter(str(self.models / "ip-adapter"), subfolder="sdxl_models",
                             weight_name="ip-adapter-plus_sdxl_vit-h.safetensors", image_encoder_folder=None,
                             local_files_only=True)
        pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
        say("Đang tải mô hình (4/4)…")
        # §14: leaf-level group offload of the UNet; everything else resident on the GPU.
        from diffusers.hooks import apply_group_offloading
        apply_group_offloading(pipe.unet, onload_device=torch.device("cuda"), offload_device=torch.device("cpu"),
                               offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True)
        for part in (pipe.controlnet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2, pipe.image_encoder):
            part.to("cuda")
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def unload(self) -> None:
        self.pipe = None

    def draw(self, req: FrameRequest) -> Image.Image:
        canvas_size = req.reference.size[0]
        if canvas_size == DRAW_SIZE:
            return self._draw(req)
        k = DRAW_SIZE / canvas_size
        small = FrameRequest(req.reference.resize((DRAW_SIZE, DRAW_SIZE), Image.Resampling.LANCZOS),
                             _scaled(req.reference_skeleton, k), _scaled(req.target_skeleton, k), req.background,
                             req.prompt, req.seed, req.hands_gripping, req.extra)
        return self._draw(small).resize((canvas_size, canvas_size), Image.Resampling.LANCZOS)

    def _draw(self, req: FrameRequest) -> Image.Image:
        import torch
        if self.pipe is None:
            raise EngineError("inference", "models not loaded")
        size = req.reference.size[0]
        bg = req.background
        ref_flat = Image.new("RGB", req.reference.size, bg)
        ref_flat.paste(req.reference, mask=req.reference.getchannel("A"))

        rough = repose(req.reference, req.reference_skeleton, req.target_skeleton)
        rough_flat = Image.new("RGB", rough.size, bg)
        rough_flat.paste(rough, mask=rough.getchannel("A"))

        if REFERENCE_PANEL:
            width = size * 2
            canvas = Image.new("RGB", (width, size), bg)
            canvas.paste(ref_flat, (0, 0))
            canvas.paste(rough_flat, (size, 0))
            mask = Image.new("L", (width, size), 0)
            mask.paste(255, (size, 0, width, size))
            pose = np.zeros((size, width, 3), np.uint8)
            draw_openpose(req.reference_skeleton, (size, size), canvas=pose[:, :size])
            draw_openpose(req.target_skeleton, (size, size), canvas=pose[:, size:])
        else:
            width = size
            canvas = rough_flat
            mask = Image.new("L", (size, size), 255)
            pose = draw_openpose(req.target_skeleton, (size, size))

        hands = "clenched hands, empty hands, " if req.hands_gripping else ""
        prompt = POSITIVE.format(motion=req.prompt or "standing", hands=hands,
                                 bg=BACKGROUND_WORDS.get(tuple(bg), "plain background"))
        negative = NEGATIVE.format(weapon="weapon, holding weapon, sword, bow, staff, effects")
        gen = torch.Generator(device="cpu").manual_seed(int(req.seed))
        try:
            with torch.inference_mode():
                out = self.pipe(prompt=prompt, negative_prompt=negative, image=canvas, mask_image=mask,
                                # Pose mode only: adding the inpaint mode (7) left the right half an empty flat
                                # field whenever the pose differed from the reference (Phase 0 on test art).
                                control_image=[Image.fromarray(pose)], control_mode=[0],
                                ip_adapter_image=ref_flat, width=width, height=size,
                                num_inference_steps=STEPS, guidance_scale=GUIDANCE, strength=STRENGTH,
                                controlnet_conditioning_scale=CONTROL_SCALE, generator=gen).images[0]
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise EngineError("oom", str(exc)) from exc
            raise EngineError("inference", f"{type(exc).__name__}: {exc}") from exc
        if REFERENCE_PANEL:
            out = out.crop((size, 0, width, size))
        return out.convert("RGB")


def _scaled(skel: Skeleton, k: float) -> Skeleton:
    return Skeleton([None if pt is None else (pt[0] * k, pt[1] * k) for pt in skel.points], list(skel.confidence))


class DiffusersEngine(Engine):
    name = "diffusers"
    is_demo = False

    def __init__(self, models_dir: Path) -> None:
        super().__init__()
        self.models = Path(models_dir)
        self.pose = DWPoseReader(self.models / "dwpose")
        self.drawer = SDXLDrawer(self.models)
        self.remover = ToonOutRemover(self.models)

    def missing_models(self) -> list[str]:
        return [r for r in REQUIRED if not (self.models / r).exists()]

    def load(self, progress: ProgressFn | None = None) -> None:
        missing = self.missing_models()
        if missing:
            raise EngineError("models_missing", ", ".join(missing))
        try:
            import torch
        except ImportError as exc:
            raise EngineError("models_missing", "PyTorch") from exc
        if not torch.cuda.is_available():
            raise EngineError("inference", "không thấy GPU NVIDIA (CUDA)")
        try:
            self.drawer.load(progress)
        except EngineError:
            raise
        except Exception as exc:  # noqa: BLE001
            if _is_oom(exc):
                raise EngineError("oom", str(exc)) from exc
            raise EngineError("inference", f"{type(exc).__name__}: {exc}") from exc
        self.loaded = True

    def unload(self) -> None:
        self.drawer.unload()
        self.loaded = False
        self.free_memory()

    def free_memory(self) -> None:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
