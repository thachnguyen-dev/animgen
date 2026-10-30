# What still needs testing on the owner's PC

The cloud build had **no GPU, no models, no owner art and no .NET SDK**. Nothing below has been verified:
image quality, frame consistency, speed, memory use, and even whether the WPF app compiles.

What *was* verified in the cloud (86 automated tests, `backend/tests`): input checks, checkerboard detection,
template parsing and fitting, weapon placement and layering, skins reuse, bow stages, canvas fit, technical
validation, retries (bounded), regenerate/compare/accept/keep, cancel, mirroring for left-facing characters,
export names/sizes/sprite sheet, temp clean-up, the HTTP server end to end (token, errors, full flow), the
DWPose pre/post-processing maths, and a clear error when model folders are missing. All of it with the demo
engine and synthetic images.

## 1. Build the app (first, ~10 min)

1. Install the .NET 9 SDK. `dotnet build app\AnimGen.sln`. **The C# has never been compiled**: expect a few
   compile errors to fix.
2. Run the demo without models: `scripts\run_dev_demo.ps1 -Python <python 3.12 with backend\requirements-dev.txt>`.
3. Walk the whole UI in demo mode (yellow banner on top):
   - choose a character (try an opaque PNG and a PNG with a baked checkerboard: both must be refused);
   - Walk → Tạo; watch "Khung n / 8"; play at 6/8/12/24 fps;
   - click a frame → "Tạo lại khung này" → compare → "Dùng bản mới" / "Giữ bản cũ";
   - "Thêm vũ khí hoặc màu…": a sword image, click grip then tip, "Xem vũ khí trên tay nhân vật";
   - Vũ khí một tay with the sword; add a second skin of the same size (points reused); switch skin after
     generating (should take seconds, no regeneration);
   - Cung with 3 images (one set of points per image);
   - Export at 256/512/1024 into a folder; import the sheet into Unity (Sprite Mode Multiple, Grid By Cell
     Count 1 × 8);
   - close the app while generating: the Python process must disappear (Task Manager).
4. UI automation, if used: paste text through the clipboard (Telex IME), capture only the app's window (§14).

## 2. Runtime and models (with internet, once)

1. `scripts\setup_runtime.ps1 -Data D:\AnimGen -Uv <uv.exe>` — Python 3.12, venv, PyTorch 2.7.1 cu126,
   Diffusers 0.35.1, …, then ~15 GB of models.
2. Check every repository and file name in `scripts\download_models.py` (they are Proposed), especially
   **ToonOut** (`joelseytre/toonout`, `birefnet_finetuned_toonout.pth`) and that **Animagine XL 4.0** ships a
   diffusers folder (`model_index.json`). If it only ships a single `.safetensors`, convert it once with
   diffusers' `from_single_file(...).save_pretrained(...)`.
3. Read the licences (§9.3): IP-Adapter Plus + CLIP ViT-H encoder, DWPose weights, ToonOut card. Record any
   restriction in `docs/DECISIONS.md`.
4. `scripts\publish.ps1 -Data D:\AnimGen`. Disconnect the network and start `D:\AnimGen\AnimGen.cmd`.
5. Move `D:\AnimGen` to another drive and start again: the app must repair the venv offline.

## 3. Phase 0 (go / no-go, §13.4)

Rehearse with `--engine demo`, then run for real:

```
D:\AnimGen\venv\Scripts\python.exe -m animgen.phase0 --data D:\AnimGen --character hero.png ^
   --sword sword.png --sword-grip X,Y --sword-tip X,Y ^
   --bow b1.png b2.png b3.png --bow-grip X,Y --bow-tip X,Y --out D:\phase0
```
(with `PYTHONPATH=D:\AnimGen\backend`). It writes one 8-frame sheet per animation and `report.json` with time
and peak VRAM per animation.

Judge by eye, per §6.6:
- **Frame consistency** — same clothes, colours, proportions across the 8 frames.
- **Empty grip** — closed hands that hold the placed weapon convincingly.
- **Placed weapon** — grip on the hand, angle believable; adjust templates if not.
- **Style range** — try one hand-drawn character close to cartoon.
- **Speed and memory** — expected ~2–3 min per frame, peak VRAM 4–5 GB, RAM 9–11 GB (all estimates).

Things in the real engine most likely to need fixing (unverified code paths):
- `StableDiffusionXLControlNetUnionInpaintPipeline` arguments: `control_image` list, `control_mode=[0, 7]`
  (openpose, repaint), and what the repaint control image should contain for ProMax.
- IP-Adapter Plus loading together with leaf-level group offloading of the UNet.
- DWPose detector output: the code decodes raw YOLOX output if needed (a heuristic); check one image.
- ToonOut weights loading into the BiRefNet code (`strict=False`, prefixes stripped).
- If the two-panel method is too slow: cut C10 (`REFERENCE_PANEL = False`).

## 4. Templates (owner approval, §5.2)

Contact sheets of every Proposed template are in `docs/templates/*.png`. They are written as joint angles in
`backend/animgen/authoring.py`; after a change run `python -m animgen.authoring` (rewrites the JSON and the
sheets). Approve each template, then mark its `status` as Confirmed there.

Tuning numbers live in `backend/animgen/config.py` (canvas, character size, retries, thresholds) and at the top of
`backend/animgen/engines/diffusers_engine.py` (steps, CFG, ControlNet and IP-Adapter strength).
