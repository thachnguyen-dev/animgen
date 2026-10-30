# Decisions log — cloud build, 2026-10-30

The owner chose, before going offline: **start the cloud build**, **accept the recommended cuts** (C1, C2, C3, C5,
C8, C9) and **make the export size selectable at export**. The owner then wrote (02:55): *"you can decide the
design rules yourself, just keep the application's purpose."*

So every item below was decided by the implementer under that delegation. None of them changes a Confirmed
requirement of `DESIGN.md`. Each stays changeable; the owner approves or reverses them.

Labels: **Owner** = chosen by the owner in this session. **Decided** = decided by the implementer, changeable.

## Scope (§12)

| # | Item | Decision | By |
|---|---|---|---|
| 1 | Cuts | C1, C2, C3, C5, C8, C9 accepted. C7 (fixed preview speeds 6/8/12/24) also used, as §11.4 already proposed it. C6 rejected (keeps Confirmed §8.2). C10 not applied (decide after Phase 0; a single switch in `engines/diffusers_engine.py`). | Owner (C7: Decided) |
| 2 | Version 1 animations | Idle, Walk, Run; attacks A1a, A1b, A2, A5, A6. Jump, Fall, Hit, Death, A3, A4 appear in the list marked "chưa có" and cannot be generated. | Owner (via cuts) |

## Pending items of §15

| §15 | Item | Decision | By |
|---|---|---|---|
| 1 | Export frame size | Chosen at export: 256, 512 or 1024 px tall. Frames are **square** (width = height). | Owner (size choice) / Decided (square) |
| 2 | Art styles | Not decided; waits for Phase 0. Nothing in the app promises a style. | — |
| 3 | Input-failure behaviour | The image is rejected, the previous character (if any) stays, and a Vietnamese message says why: not openable, no alpha, checkerboard drawn into the pixels, alpha but nothing transparent, empty, several figures, too small/large. Wording in `backend/animgen/errors.py`. | Decided |
| 4 | Automatic retries | 2 per frame (the Proposed value), hard ceiling 5, never unbounded. An out-of-memory error frees GPU memory before the retry. | Decided |
| 5 | Temp files | Current session's frames live in `DATA\temp\session-…` until exit; every older session folder is removed on the next start. No crash recovery. | Decided |
| 7 | Technical choices | .NET 9 WPF; Python backend; models of §9.2; file names of §10.3; one-row sprite sheet; palette below. All taken as proposed. | Decided |
| 8 | Template poses | Authored (see "Motion templates" below). Every template is still **Proposed**: the owner approves each one. | Proposed |
| 9 | Packaging | Runtime folder + `scripts\publish.ps1` (C9). | Owner (via C9) |

## Pipeline numbers

| Item | Decision | Why |
|---|---|---|
| Canvas | 1024 × 1024 per frame; the two-panel generation canvas is 2048 × 1024. | §10.4 and §13.3 (~2 MP). |
| Character size on the canvas | 52 % of the height, feet at 92 %, centred. | First tried 60 %/88 %: a sword raised overhead left the canvas. 52 % keeps a long sword inside. Trade-off: fewer pixels for the character. Tune in Phase 0. |
| Weapon leaving the canvas | Not moved or shrunk; the frame is marked "Vũ khí bị cắt" in the strip and the summary. | The owner sees it rather than a silent change. |
| Foot anchor | Grounded frames are moved vertically onto the foot line if they drifted ≤ 6 % of the figure height; larger drifts are left as drawn. | §6.3 without hiding a pose problem. |
| Facing | Templates face right. A left-facing character is mirrored in, generated facing right, and mirrored back. The facing is read from the skeleton (DWPose) or, when it cannot be read, assumed "right" and shown as assumed with a "Đổi hướng" button. | One set of templates; the owner keeps the character's own facing. |
| Weapon hold point | The wrist pushed 30 % of the forearm length further along the forearm. | Middle of a closed hand. Tune in Phase 0. |
| Weapon layers | Only in front of / behind the whole character (§7). | As designed. |
| Seeds | One random seed per run, offset per frame and per retry. | Reproducible within a run; a regeneration gets a new seed. |
| Background colour for drawing | White, mid grey or near black, whichever is least present in the character. | §6.5 step 3 ("flat and plain") with a background that the remover can separate. |
| Image-model settings | 28 steps, CFG 5, ControlNet 0.9, IP-Adapter 0.6, prompt tags per template. | Starting points; all marked TUNE in the code. |
| Drawing size (local, 2026-10-30) | The AI draws each frame at 768 × 768 (two-panel canvas 1536 × 768) and the result is scaled to the internal 1024 canvas; placement and skeletons stay at 1024. First set to 896, lowered to 768 after the owner's note that the sprites are for mobile 2D games (below). | Measured on the owner's RTX 4060 Laptop 8 GB, 16 GB RAM: 1024 → 9.0 s/step (spills into shared GPU memory, ~8.7 min per frame), 896 → 3.2 s/step (~1.5 min per frame, ~13 min per animation), 768 → 2.6 s/step. Keeping the UNet resident or storing weights in fp8 was slower (14–19 s/step). |

## Weapons

- A weapon store in `DATA\weapons` keeps each weapon's images and points, so a weapon is set up once (§4.2). It is
  not a project library (§10.6): it stores only weapon images and their two points.
- A new skin whose images have the same sizes as an existing skin of that weapon reuses its points automatically.
- Bows: three images (resting, half drawn, fully drawn); each gets its own two points.
- Non-attack animations carry the optional weapon in the near hand, pointing down-forward (walk, idle) or up
  (run). Bows in non-attack animations use the resting image.

## UI (§11.4)

- One window: steps 1–4 on the left (character, animation, generate, export); status line, preview, compare panel
  and the 8-frame strip on the right. A weapon dialog for adding weapons/skins and clicking grip and tip.
- Palette: warm light neutrals, one blue accent; success green, warning amber, error red, each always with an
  icon and words (never colour alone). Segoe UI; Segoe MDL2 Assets icons. All values in `Themes/Theme.xaml`.
- Three visibly different action buttons: Generate (filled blue), Regenerate (light blue with outline),
  Export (filled green).
- Preview on a checkerboard brush (never in the images); speeds 6/8/12/24 fps.
- A yellow banner says when the demo engine is running (no AI).
- Export is refused while a frame has an unanswered "bản mới".

## Engineering

| Item | Decision | Why |
|---|---|---|
| No NuGet packages in the app | Plain `INotifyPropertyChanged` + small command classes instead of CommunityToolkit.Mvvm. | Builds offline with the SDK alone; nothing to restore. |
| Backend HTTP | Python standard library (`http.server`), no web framework. | Fewer dependencies to bundle offline. |
| Demo engine | `--engine demo` / `AnimGen.exe --demo`: CPU only, draws a flat figure on the fitted skeleton. | §0.2: the whole app can be tested without GPU; the owner can rehearse on the PC before models are installed. |
| Self-contained publish | `dotnet publish --self-contained`. | The owner's PC needs no .NET runtime install (offline, §2.4). |
| Moving the install | The app detects a moved DATA folder (venv `home` outside `DATA\python`) and re-creates the venv launcher offline with the bundled uv (§14). | Lesson from chargen. |
| Line endings | CRLF for all text files; PowerShell scripts saved as UTF-8 with BOM. | §14 (owner's files use CRLF; PowerShell 5.1 and Vietnamese text). |

## Drawing method (local finalisation, 2026-10-30) — Decided, verified on single frames

- Phase 0 on temporary test art with the delivered method (each frame from noise; ControlNet pose + inpaint modes)
  failed: whenever the pose differed from the reference, the right panel stayed an empty flat field; frames that did
  come out drifted in outfit, line style and facing. Raising IP-Adapter (0.8, 1.0) lost the pose; descriptive tags
  helped the outfit but not the pose.
- Two changes, each tested on single frames: (1) ControlNet in **pose mode only** (dropping the inpaint mode is what
  brought the character back); (2) each frame **starts from a rough repose** of the owner's own art
  (`animgen/puppet.py`: parts cut by bone and moved onto the target skeleton; far limbs as darker copies) and the image
  model repaints part of it (`STRENGTH` 0.6). Result: correct pose, the outfit and colours kept, a clean white
  background. 0.45 left cut seams; 0.75 began to redraw the outfit.
- Still open: a full 8-frame run, line weight (thinner than the reference), tuning. See HANDOFF.md.

## Owner notes (2026-10-30) — Confirmed

- The app is for the owner's personal use only, not commercial. Licences are therefore not a selection criterion;
  a model only has to be suitable. (The table below is kept for the record.)
- The sprites are used in 2D games on mobile devices, so the output resolution can be much lower. Export heights
  are now 128, 256 and 512 px (default 256); 1024 was removed because it would only be an upscale.

## Model licences (verified 2026-10-30 on Hugging Face)

| Model | Licence |
|---|---|
| Animagine XL 4.0 | CreativeML Open RAIL++-M (use-based restrictions; commercial use allowed) |
| SDXL VAE fp16 fix | MIT |
| ControlNet Union SDXL ProMax (xinsir) | Apache-2.0 |
| IP-Adapter Plus SDXL (h94) | Apache-2.0; its image encoder LAION CLIP ViT-H is MIT |
| DWPose (yzd-v) | Apache-2.0 |
| BiRefNet code | MIT |
| ToonOut weights | MIT (training data CC-BY 4.0) |

## Character skeleton set by the owner (local finalisation, 2026-10-30)

Owner's suggestion: let the user define the skeletons of the character and the weapons with stick figures, for when
pose recognition is ineffective or too expensive. **Decided (changeable):**

- Pose recognition runs only once per character, on the reference image; every frame's skeleton comes from the
  motion template fitted to that skeleton's proportions. A wrong reference skeleton therefore spoils all frames,
  while the cost is small (DWPose, about 350 MB, CPU, a second or two). The risk is accuracy on stylised art
  (chibi, capes, heavy armour), not cost.
- So the stick figure is set **in the app, on the character image**, not drawn as a separate image file: after
  loading, the skeleton is shown over the character, pre-filled by DWPose (or by standard proportions when DWPose
  is missing or finds nothing), and the owner drags any wrong joint and confirms. Separate stick-figure files
  would need a strict drawing format and extra art work, and could not be checked against the character.
- The confirmed skeleton is saved next to the character (by image content), so it is asked once per image.
- Generation is refused until the skeleton is confirmed (one click when it is already right).
- Weapons keep their two clicked points (grip, tip): that already is the weapon's stick figure; bows have them per
  draw stage.
- DWPose stays as the pre-fill. If Phase 0 shows it is wrong on most of the owner's art, it can be dropped and the
  pre-fill comes from standard proportions alone.

## Not done on purpose

- No free-form prompts, no batch generation, no automatic quality scoring, no Unity configuration files, no
  project library, no effects (§2 prohibited list).
- No Jump/Fall/Hit/Death/A3/A4 templates in version 1 (cuts C1, C2).
- No pixel-art step (C3).
