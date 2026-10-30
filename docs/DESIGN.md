# Local AI 2D Animation Generator — Consolidated Design

**Document status:** Consolidated design, version 2.0 (2026-10-30)
**Replaces:** App Design, Core Design Rules and Assets Art Direction, versions 1.0–1.5 (kept in the owner's
`doc` folder for reference). Where they differ, this document wins.
**Owner:** the sole user and final decision-maker (§2.11).

---

## 0. How to Read This Document

### 0.1. Status labels

Every decision carries one of these labels. **Do not treat a Proposed or Pending item as Confirmed.**

| Label | Meaning |
|---|---|
| **Confirmed** | Explicitly approved by the owner. |
| **Proposed** | Recommended, not yet approved. May be built as the default, but stays changeable and must be reported as Proposed. |
| **Pending** | Needs information or a decision from the owner. Do not invent an answer. |
| **Inherited—Verify** | Taken from the owner's earlier app (chargen, same laptop, some of the same models). Likely true here, but must be checked. |
| **Out of Scope** | Explicitly excluded. |

When a technical limitation conflicts with a Confirmed requirement, document the conflict and ask the owner. Never
change a Confirmed requirement silently.

### 0.2. Where the work happens (Confirmed)

1. **Cloud session (first).** It builds the project: the WPF app, the Python backend, motion templates, export, the
   offline package scripts. It has no GPU, no owner art and no models. It must:
   - keep every GPU and model step behind clear interfaces, so they can be tested later on the owner's PC;
   - test what it can without a GPU (input checks, template fitting, weapon placement, sprite sheets, validation,
     retry logic, UI flow), using synthetic test images it makes itself;
   - never claim that image quality, frame consistency, speed or memory use were verified;
   - leave a clear list of what still needs testing on the owner's PC.
2. **Local finalisation (after).** On the owner's PC (§3), with the owner's character and weapon art: Phase 0 (§11),
   model downloads, tuning, and the owner's approval of the motion templates.

---

## 1. Product Overview

### 1.1. Purpose (Confirmed)

A Windows desktop tool that uses locally running AI models to generate 2D character animation frames from a
reference image, for the owner's own 2D games made in Unity. Output quality comes first. The app works entirely
offline after installation. The workflow is simple: choose input, generate, review, regenerate, export.

### 1.2. User (Confirmed)

Personal use by the owner only. Never commercial, including in the future. Models with non-commercial licences are
acceptable if their licences allow this personal use (§9.3).

### 1.3. Platform (Confirmed)

| Item | Decision | Status |
|---|---|---|
| Operating system | Windows | Confirmed |
| Desktop UI | WPF, C#/.NET | Confirmed |
| .NET version | .NET 9 | Proposed |
| AI inference | Local, on the owner's computer | Confirmed |
| AI backend | Python with Diffusers, run as a separate local process (§9.1) | Proposed |
| Target engine | Unity (2D) | Confirmed |
| Internet | Not required after installation | Confirmed |
| UI language | Vietnamese | Confirmed |

### 1.4. Main objective (used to judge every cut)

From one character image (plus separate weapon images), entirely offline, produce Unity-ready transparent frames
and one sprite sheet per animation.

---

## 2. Core Rules (Confirmed unless marked)

These take precedence over implementation convenience.

1. **Personal use only.** No commercial infrastructure, subscriptions, licence servers, ads, analytics or
   monetisation.
2. **Quality first.** Long generation times are acceptable. Do not lower quality or swap the model only for speed
   without a demonstrated need.
3. **Local AI only.** All inference runs on the owner's computer; no remote AI service.
4. **Offline completeness.** The installation includes the models and every dependency. No online activation, no
   model downloads in normal use, no remote APIs. Any dependency that needs the network must be raised before it is
   adopted.
5. **Hardware fit.** The target is §3. Check peak VRAM, RAM, temporary disk and the whole pipeline. Downloadable
   weights do not mean a model fits.
6. **Minimal scope.** Build the smallest app that meets the Confirmed requirements.
7. **WPF.** Do not replace WPF without the owner's approval.
8. **Backend separation.** AI inference stays out of the UI layer. The UI starts generation, receives progress,
   receives frames and reports errors.
9. **One primary model.** Extra models only when they bring a real quality benefit and fit the hardware and offline
   rules. Never present a model as suitable until the real workflow has been tested.
10. **Licences.** Check the licences of models, weights, code and dependencies. Open-source code does not imply
    free weights. Record material restrictions before finalising the stack.
11. **Responsive UI.** Long inference must not freeze the window; progress and errors are always shown.
12. **Explicit errors.** Handle input-check, inference, file-writing and export failures explicitly. Never report
    success for invalid output.
13. **Owner's authority.** The owner decides requirements and scope. Routine technical choices that do not contradict
    Confirmed requirements may be made by the implementer and recorded.

**Prohibited without the owner's approval:** online inference; mandatory internet after installation; commercial
services; free-form text-prompt animation; batch generation of several animations at once; automatic aesthetic or
semantic quality scoring; Unity runtime animation management; Unity-specific configuration export; a project
library; mandatory cloud storage; unnecessary extra AI models; visual effects in the frames.

---

## 3. Target Hardware (Confirmed)

| Component | Specification |
|---|---|
| OS | Windows |
| RAM | 16 GB |
| GPU | NVIDIA GeForce RTX 4060 |
| VRAM | 8 GB |
| Free disk | ~100 GB |

Quality over speed. The app must handle GPU memory limits and inference failures without crashing.

---

## 4. Input

### 4.1. Character image (Confirmed)

- One human-shaped character per image.
- Full body, standing pose, side view.
- Real transparent background (alpha channel). A checkerboard drawn into the pixels is not transparency.
- **No weapon** in the character image.
- One character at a time.

Do not silently broaden this to other poses, several characters or opaque backgrounds.

### 4.2. Weapon images (Confirmed)

- Each weapon, and each colour variant ("skin") of a weapon, is its own image with real transparency.
- Weapons are placed into the frames by the app; they are never generated or redrawn by the AI (§7).
- **Bows:** 3 images for the draw stages: resting, half drawn, fully drawn. (Confirmed)

Proposed preparation:

- Drawn side-on, at the same pixel scale as the character art.
- The first time a weapon is used, the owner clicks its **grip point**, then its **tip**. The app shows the weapon in
  the character's hand for the owner to confirm. Skins of the same weapon can reuse the two points.

### 4.3. Supported art styles

- **Proposed:** anime and cartoon first (what the proposed model handles); hand-drawn styles close to cartoon tested in
  Phase 0; pixel art later, with its own downscaling and palette step.
- **Pending:** the final supported range is decided by test results. Do not promise a style before it is tested.

### 4.4. Input checks

- **Confirmed:** the file opens and can be processed; it has an alpha channel with real transparency (not a
  checkerboard). An invalid input is never treated as valid.
- **Proposed:** also check that there is exactly one figure (one connected opaque region of meaningful size). The owner
  judges full body and side view by eye (cut C5, §12).
- **Pending:** the exact behaviour and wording when an input fails.

---

## 5. Animations

### 5.1. Animation list (Confirmed)

| Animation | Intent |
|---|---|
| Idle | standing or resting |
| Walk | walking |
| Run | running |
| Jump | jumping |
| Fall | falling |
| Attack | attacking (variants, §5.2) |
| Hit | reacting to being hit |
| Death | dying |

Chosen from the list; no free-form text prompts. One animation is generated at a time.

### 5.2. Attack variants (Confirmed list)

| # | Variant | Movement idea (8 frames) | Weapon |
|---|---|---|---|
| A1a | Unarmed — hands | punch: wind-up, strike, recovery | never |
| A1b | Unarmed — feet | kick: lift, strike, recovery | never |
| A2 | One-handed weapon (sword, axe, mace, dagger) | wind-up, horizontal or diagonal slash, recovery | yes |
| A3 | Two-handed heavy weapon (greatsword, hammer) | slow overhead wind-up, downward smash | yes |
| A4 | Polearm (spear) | step and forward thrust | yes |
| A5 | Bow | draw, aim, release, follow-through | yes, 3 draw stages |
| A6 | Magic staff | raise, gather, cast forward | yes |

The list is Confirmed. The movement ideas are a starting point. The exact 8 poses of each variant are drawn as motion
templates and **approved by the owner during the build**. Until then they are Proposed.

### 5.3. Frame count (Confirmed)

**Every animation has exactly 8 frames.** The user never enters a frame count.

### 5.4. Weapons in non-attack animations (Confirmed)

Idle, Walk, Run, Jump, Fall, Hit and Death are made **without a weapon by default**. The user may choose a weapon for
them as an option; it is placed the same way as in attacks, so their templates also store hand and weapon data. The
two Unarmed attacks never use a weapon.

### 5.5. Effects (Confirmed: Out of Scope)

Frames show only the character's movement. Flying arrows, spell glows, projectiles and other effects are made in
Unity.

### 5.6. Scope boundary (Confirmed)

The app makes frames and exports them, with a preview for review. Unity owns runtime playback, looping and game
logic.

---

## 6. Generation

### 6.1. Method (Confirmed)

Individual image frames, not a video with frames extracted. Do not switch the main workflow to video without
approval.

### 6.2. Appearance (Confirmed)

Preserve the character's visual style, while allowing some details to change for natural movement. Do not
strengthen this into a promise of exact identity, and do not weaken it into unrestricted changes.

### 6.3. Position (Confirmed)

The character stays at a relatively fixed position in the frame; limbs move as the animation needs. No unnecessary
drift.

### 6.4. Priority (Confirmed)

Quality over time. Several minutes or longer per animation is acceptable.

### 6.5. Pipeline per frame (Proposed)

1. **Read the input.** Check the alpha. From the alpha, find the outline, height and foot line. Read the body skeleton
   with DWPose to get the character's proportions (limb lengths, head size).
2. **Motion template.** Each animation and attack variant is a stored sequence of **8 side-view body skeletons**. It is
   fitted to the character's proportions and anchored at the same foot point in every frame (keeps §6.3). Each frame
   of a template also stores the weapon data of §7.
3. **Draw the frame.** The reference image is placed in the left half of a wide canvas, and the new frame is drawn in
   the right half, guided by that frame's skeleton (ControlNet Union: inpaint + pose). The model copies the character
   from the picture it can see. IP-Adapter Plus also feeds the reference in as an image prompt. The background is drawn
   flat and plain. For armed animations the character is drawn **without a weapon**, hands closed in a grip.
4. **Remove the background.** ToonOut (BiRefNet fine-tuned for anime) produces real alpha; edges are cleaned.
5. **Fit to the canvas.** The same scale and foot anchor in every frame, on the fixed canvas.
6. **Add the weapon** (armed animations, §7).
7. **Check** (§8.1).

Each frame is still a separate image, so this follows §6.1.

### 6.6. Main risks (to be tested in Phase 0)

- **Frame consistency:** the same clothes, colours and proportions in all 8 frames.
- **Empty grip:** the model draws a hand closed around nothing; the placed weapon must sit convincingly in it.
- **Style range:** how far beyond anime/cartoon the model holds up.

---

## 7. Weapons — Combined by the App (Confirmed decision; details Proposed)

The character is animated once, without a weapon. The app places the owner's weapon image into each frame, so the
weapon is identical in every frame, thin parts (bow strings) stay intact, and a new skin needs only a recombine.

**Placing the weapon (Proposed).**

- For each frame, the motion template stores: hand position, weapon angle, layer order (weapon in front of or behind
  the hand/body), and for bows the draw stage (1–3).
- The app puts the weapon's grip point on the hand at the stored angle and layer order.
- Layering is only "in front of" or "behind"; fingers do not wrap around the handle. This looks right at game sprite
  sizes but may look pasted on in large close-ups.

**Weapon skins (Confirmed behaviour).** Each skin is its own weapon image. Changing the skin reuses the same character
frames and recombines them in seconds; all skins of a weapon have exactly the same motion. Each skin gives its own
exported files.

---

## 8. Validation, Retries, Regeneration

### 8.1. Technical validation (Confirmed)

Checks per frame: the file opens; the size is correct; a proper alpha channel and real transparency are present.
No automatic semantic or aesthetic scoring.

### 8.2. Automatic retries (Confirmed principle)

A frame that fails a technical check is regenerated automatically a limited number of times, then the user is told
and can regenerate by hand. Never unbounded.

- **Pending:** the retry count. **Proposed:** 2 automatic retries per frame.

### 8.3. Regeneration (Confirmed)

The user can regenerate the whole animation or a single frame. A regenerated frame is shown next to the current one,
and replaces it only when the user accepts. A good frame is never lost to a worse attempt.

---

## 9. Technical Proposal

### 9.1. Architecture (Proposed)

- **WPF app (.NET 9)** for the UI. **Python backend** (Diffusers, PyTorch CUDA, ONNX Runtime) as a separate local
  process, started by the app, talking over a loopback HTTP server with tasks, progress polling and cancel.
- The backend is reachable only from the app: a per-run secret token, bound to 127.0.0.1. The backend process ends
  with the app (Windows job object).
- Models are stored as plain folders in the app's data folder and loaded offline (`HF_HUB_OFFLINE=1`).
- The app's data folder (runtime, models, temp) **follows the installed app**: the app runs from `DATA\app`, so the
  whole folder can be moved (lesson from chargen, §14).

### 9.2. Models (Proposed)

| Role | Model | Size | Licence | Status |
|---|---|---|---|---|
| Main image model (the one primary model) | Animagine XL 4.0 (SDXL-based, anime/cartoon) | ~6.9 GB | CreativeML Open RAIL++-M, unmodified; allows personal use | Proposed |
| Pose and inpaint guidance | ControlNet Union SDXL 1.0 ProMax (xinsir) | ~2.5 GB | Apache-2.0 | Proposed; Inherited—Verify |
| Reference appearance | IP-Adapter Plus SDXL (ViT-H) + CLIP ViT-H image encoder | ~0.9 + 2.5 GB | Apache-2.0 (verify) | Proposed |
| Background removal | ToonOut (BiRefNet fine-tune for anime) | ~0.9 GB | MIT; training data CC-BY 4.0 | Proposed |
| Skeleton reading | DWPose (yolox_l + dw-ll_ucoco_384, ONNX) | ~0.35 GB | Apache-2.0 (verify) | Proposed; Inherited—Verify |
| Image decoder | sdxl-vae-fp16-fix | ~0.3 GB | MIT | Proposed; Inherited—Verify |

Considered and not proposed:

- **NoobAI-XL / Illustrious-based anime models:** strong anime quality, but their licences add conditions (no
  commercial use of the model or outputs; derivatives must be open-sourced). Personal use is allowed. Kept as the
  fallback if Animagine fails Phase 0. Pending.
- **Video models** (Animate Anyone-style, Wan, etc.): would break §6.1, and 8 GB VRAM is tight. Out of Scope unless the
  owner changes §6.1.
- **Per-character LoRA training:** one reference image is too little; adds 15–30 min per character. Not in version 1.
- **SAM 2 / segmentation:** not needed, since weapons are separate images.
- **Online services:** excluded by §2.3.

### 9.3. Licences to verify before finalising (Proposed)

IP-Adapter Plus and the CLIP image encoder; DWPose weights; ToonOut model card. Read the original model cards; record
any restriction.

---

## 10. Output and Export

### 10.1. Individual frames (Confirmed)

Each frame is a PNG with real transparency, at the fixed canvas size.

### 10.2. Sprite sheets (Confirmed)

One sprite sheet per animation (never all animations in one sheet by default). All frames use the same canvas size.

**Proposed layout:** one row, 8 frames in play order, no padding, each cell exactly the frame size. Unity slices it with
"Grid By Cell Count" (1 × 8). Cut C8.

### 10.3. File names (Proposed)

- Frames: `<character>_<animation>_<frame 01–08>.png`
- Sheet: `<character>_<animation>_sheet.png`
- With a weapon: `<character>_<animation>_<weapon>_<skin>_…`

### 10.4. Frame size

- **Proposed:** frames are generated at about 1024 px tall and scaled down to the export size.
- **Pending:** the export frame width and height (for example 256, 512 or 1024 px tall, depending on the game). Do
  not assume a size is approved.

### 10.5. Unity (Confirmed)

The output must import into Unity as 2D assets. The app does **not** create Animator Controllers, animation clips or
any Unity configuration files. PNGs and sprite sheets are the only outputs.

### 10.6. Projects and temporary files

- **Confirmed:** no project library for saving and reopening named projects.
- **Pending:** how long temporary files are kept and whether work can be recovered after a crash. **Proposed:** keep
  the current session's frames in a temp folder until export or exit; clean up on the next start.

---

## 11. User Interface

### 11.1. Principles (Confirmed)

Minimal, functional, easy to understand, Vietnamese, no unnecessary menus, advanced controls or decoration. No AI
jargon needed for the main workflow. Generate, regenerate and export buttons are visually distinct. Errors are clearly
distinguishable from success, never by colour alone.

### 11.2. Workflow (Confirmed)

1. Choose the character image (checked at once).
2. Choose the animation; for Attack, the variant; for armed animations, the weapon (and skin).
3. Generate.
4. Watch progress, including which frame is being made.
5. Preview the animation, with adjustable playback speed.
6. Regenerate the whole animation or one frame (compare, then accept or keep).
7. Export PNGs and the sprite sheet.

### 11.3. States to show (Confirmed)

Ready, generating, completed, failed, retrying a failed frame, exporting, export completed, export failed. Clear
Vietnamese status text.

### 11.4. Proposed details

- **Palette, font, icons:** a restrained light neutral palette with one accent colour, Segoe UI (full Vietnamese
  support), status shown with an icon and text. Pending the owner's approval.
- **Preview:** the animation plays on a checkerboard backing (shown behind transparency, never baked into the image),
  with the 8-frame strip below it for choosing a frame to regenerate. Speed as fixed choices: 6 / 8 / 12 / 24 fps (cut C7).
- **Weapon setup:** a small step when a new weapon image is added: click grip, click tip, confirm the preview (§4.2).
- **Progress:** "Frame n / 8" plus a progress bar; the first run of a session also shows "Loading models…".

### 11.5. Not required (Confirmed)

Decorative illustrations, marketing art, banners, animated backgrounds, bundled example characters, commercial
branding.

---

## 12. Possible Feature Cuts (Proposed — the owner decides)

Each cut keeps the main objective (§1.4).

| # | Cut | Saves | What is lost | Recommendation |
|---|---|---|---|---|
| C1 | Start with Idle, Walk, Run, Attack; add Jump, Fall, Hit, Death in a second step | ~40% of motion-template work and testing | the other 4 arrive later | Recommended |
| C2 | Attack variants in two steps: first A1a, A1b, A2, A5, A6; then A3, A4 | about a third of attack-template work at first | Two-handed and Polearm later | Recommended |
| C3 | Anime/cartoon first; pixel art later | a separate pixel-art pipeline | pixel art in version 1 | Recommended |
| C5 | Input check = opens + real alpha + one figure; owner judges full body / side view | automatic body-visibility checks | a wrong input is noticed later | Recommended |
| C6 | Manual retry button instead of automatic retries | retry logic | conflicts with Confirmed §8.2, so only with the owner's approval | Not recommended |
| C7 | Preview speed as fixed choices | a free slider | fine control | Optional |
| C8 | One fixed sprite-sheet layout (single row, no padding) | layout options | nothing Unity needs | Recommended |
| C9 | Copy a runtime folder with a publish script instead of a real installer | installer work | a proper installer | Recommended |
| C10 | Drop the two-panel method if too slow; IP-Adapter alone | ~40% of time per frame | consistency | Decide after Phase 0 |

(C4 was removed: the frame count is fixed at 8 by the owner.)

With C1–C3, C5, C8 and C9, version 1 is roughly half the work: Idle, Walk, Run and five attack variants, 8 animations
× 8 frames = 64 frames.

---

## 13. Cost Estimate (Proposed; figures marked Inherited—Verify come from chargen on the same laptop)

### 13.1. Disk

| Item | Size |
|---|---|
| Models (§9.2) | ~14 GB |
| Python runtime (PyTorch CUDA, Diffusers, ONNX Runtime) | ~6 GB |
| Application | < 0.3 GB |
| Temporary frames | up to ~2 GB |
| **Total** | **~22 GB of ~100 GB** |

### 13.2. Memory

- Peak VRAM about 4–5 GB of 8 GB. In chargen, SDXL + ControlNet Union peaked at 3.3 GB with the UNet streamed from
  RAM (Inherited—Verify); IP-Adapter adds about 1 GB.
- RAM about 9–11 GB of 16 GB while generating.

### 13.3. Time

| Work | Estimate |
|---|---|
| One frame, two-panel canvas (~2 MP) | ~2–3 min (chargen: ~73 s for a 1 MP SDXL + ControlNet image, Inherited—Verify) |
| One frame without the two-panel method | ~1.5 min, weaker consistency |
| One 8-frame animation | ~15–25 min |
| Everything: 7 non-attack animations + 7 attack variants (14 × 8 = 112 frames) | ~3.75–5.5 hours |
| Version 1 with the recommended cuts (64 frames) | ~2.5–3.5 hours |
| Another weapon skin (recombine, no AI work) | a few seconds |
| First generation after starting the app | + 1–2 min to load models |

### 13.4. Development phases

| Phase | Where | Content |
|---|---|---|
| Build | Cloud | Project skeleton, WPF app, backend server and task system, input checks, template format and fitting, weapon placement, export, preview, regenerate-and-compare, retries, package scripts; tests without GPU (§0.2) |
| 0. Feasibility test | Owner's PC | Idle, Walk, A2 (One-handed) and A5 (Bow) on the owner's character and weapon art: frame consistency, empty-grip hands, placed weapon, speed, memory. Go / no-go. |
| Templates | Owner's PC | Author and tune the 8-pose templates; owner approves each |
| Finalise | Owner's PC | Model packaging, offline install, end-to-end runs, fixes |

---

## 14. Inherited Lessons from chargen (Inherited—Verify)

The owner's earlier app (WPF + Python Diffusers, same laptop) learned these the hard way. Reuse them; verify here.

**Runtime and versions.** Python 3.12 managed by `uv` (bundled `uv.exe`), PyTorch 2.7.1 (CUDA 12.6 wheels), Diffusers
0.35.1, Transformers 4.56.1, ONNX Runtime 1.22.1, .NET 9 WPF with CommunityToolkit.Mvvm.

**Memory on 8 GB.**

- Moving the whole SDXL UNet to the GPU at once (`enable_model_cpu_offload`) spilled into Windows shared memory: 528 s
  per image.
- What worked: `unet.apply_group_offloading(onload_device="cuda", offload_type="leaf_level", use_stream=True,
  low_cpu_mem_usage=True)`, with ControlNet, VAE (with tiling) and both text encoders kept on the GPU: about 73 s per
  1 MP image, 3.3 GB peak.
- Block-level offload of the UNet still spilled (8.8 GB peak). Diffusers handles LoRA loading with group offload itself.

**DWPose (ONNX).** The person detector needs **BGR** input. Use NMS overlap 0.6, or two people hugging merge into one.
The pose model takes RGB normalised with ImageNet mean/std, 288×384 crops, SimCC decoding. Convert to the 18 OpenPose
body points for ControlNet.

**Offline models.** Download with `local_dir` into plain folders (Hugging Face cache symlinks fail on Windows), load
with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.

**Moving the install.** The data folder must follow the app (`DATA\app` → `DATA`). uv's `venv\Scripts\python.exe` has
the base interpreter's absolute path built in, so editing `pyvenv.cfg` is not enough after a move. Run the bundled
`uv venv <DATA>\venv --python <DATA>\python\cpython-…\python.exe --allow-existing` offline (`UV_OFFLINE=1`,
`UV_PYTHON_DOWNLOADS=never`, `UV_NO_CONFIG=1`); it keeps installed packages. Never show a first-run screen in a folder
that is not a real install.

**Backend process.** Loopback HTTP with a per-run token; the app attaches the Python process to a Windows job object so
it dies with the app; the backend prints `READY <port>` when started; tasks are polled for progress; cancel is honoured
between steps.

**Testing.** Test every new backend result end to end through the HTTP server, not by calling Python functions
directly: once, NumPy booleans in a reply broke the JSON and a release failed with "cause unknown".

**Windows UI automation.** The owner types with a Vietnamese Telex IME. Automated UI tests must paste text through the
clipboard, never send keystrokes for text. Never capture the whole screen; capture only the app's own windows. A
PowerShell 5.1 script containing Vietnamese text must be saved as UTF-8 **with BOM**.

**Files.** Write text files with the line endings already in the file (the owner's files use CRLF).

---

## 15. Open Items (Pending — ask the owner, do not invent)

1. Export frame width and height (§10.4).
2. Supported art styles beyond anime/cartoon (§4.3), after Phase 0.
3. Behaviour and wording when an input fails the checks (§4.4).
4. Automatic retry count (§8.2; Proposed: 2).
5. Temporary-file retention and crash recovery (§10.6).
6. Which feature cuts in §12 are accepted.
7. Approval of the Proposed technical choices: .NET 9, Python backend, models (§9), file names, sprite-sheet layout, UI
   palette and preview details.
8. The exact 8 poses of every animation and attack variant (approved template by template).
9. Installation packaging: Proposed a runtime folder plus publish script (C9); a real installer is Pending.

---

## 16. Decision History

| Version | Date | Decision |
|---|---|---|
| 1.0 | — | Owner's three documents. |
| 1.1 | 2026-10-30 | Technical proposals added (models, cost, cuts). |
| 1.2 | 2026-10-30 | Confirmed: 8 frames per animation. Confirmed: attack needs variants per weapon type and unarmed. |
| 1.3 | 2026-10-30 | Confirmed: attack list (A1 split into hands and feet, 7 variants); effects made in Unity. |
| 1.4 | 2026-10-30 | Confirmed: weapons are separate images that the app combines into the frames (replaces "weapon in the input image"). |
| 1.5 | 2026-10-30 | Confirmed: non-attack animations without a weapon by default (optional weapon); bows use 3 draw stages. |
| 2.0 | 2026-10-30 | Consolidated into this single document. Build in the cloud first; Phase 0 and finalisation on the owner's PC with the owner's art. |
