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
- Full Phase 0 run (test art, 2026-10-30): Idle, Walk, A2 (sword) and A5 (bow, 3 draw stages) all 8/8 frames on the first attempt, about
  40–55 s per frame, peak 7.3 GB GPU memory. Same outfit and colours across frames, correct facing (checked with
  DWPose on every exported frame), poses follow the templates, the sword sits in the hand. Still weak: lines thinner
  and paler than the reference; small drift (hair, a missing sleeve in one frame); the model sometimes drew a prop at
  the waist, so the negative prompt now also lists holding, dagger, knife and shield.
- Line weight, A/B on Idle (same seed): IP-Adapter 0.8 kept the reference's colours and darker outlines better than
  0.6, with the same poses — **now the default**. Strength 0.5 changed little. Style tags ("thick outlines, bold
  lineart") gave the boldest lines but would force one line style on every character, so they are not used.

## Western test set (2026-10-30, owner: "prepare temporary images yourself")

The owner's old art is semi-realistic western (armoured knight with cape, bearded archer with quiver, often a
three-quarter view), harder than the first anime test character. Temporary set made with Animagine
(`dev/tools/make_temp_art.py`): knight with cape, archer with quiver, the archer in three-quarter view, a mage in a
long robe, a staff. Two faults found and fixed:

- **Puppet cut.** Each pixel went to the nearest bone, and arms got extra reach. On the knight the arm (hanging along
  the back) took most of the cape and a strip of the breastplate: in Run and A2 the cape tore into strips flying with
  the arms, and the strip was painted as a "tassel" in the hand. Now limbs keep only pixels within a limb's width of
  their bone (`LIMB_REACH`; the lower leg also keeps the foot, `FOOT_REACH`); capes, robes and quivers stay with the
  torso. Loose specks are dropped from the rough image. Tests in `tests/test_puppet.py`; viewer `dev/tools/puppet_labels.py`.
- **Who is drawn.** The prompt named nothing about the character, so the male knight drifted towards a female
  figure. Now a tagger (WD ViT tagger v3, Apache-2.0, 378 MB ONNX on the CPU) reads the reference once per character
  and the appearance tags ("1boy, beard, quiver, blue robe" …) go into the prompt, trimmed to CLIP's 77 tokens;
  pose, view, background and anything held are filtered out. Optional: an install without it draws as before.
  First result with both fixes (archer Idle and bow): the same man in every frame.

## Input images: separate character and weapons (tested 2026-10-30) — Recommended to the owner

- Test: the same pipeline with a temporary character already holding a sword (`dev/test_art/test_hero_sword.png`)
  against the separate character + sword. Combined: the sword is cut out with the body part nearest to it and stays
  roughly where it was while the arm moves — in Walk it floats beside an empty hand, in one frame it crosses the body
  (`result_combined_walk.png`, `result_combined_idle.png`); the drawn ground shadow became stray lines under the
  feet. Separate: the weapon follows the hand in every frame of A2 and the bow (`result_separate_a2.png`,
  `result_separate_bow.png`), and skins, bow draw stages and regenerating without a weapon all depend on it.
- Recommendation: character and weapons as separate images (the owner's own preference). Character: one figure,
  full body, standing, seen from the side, empty hands, real transparent background, no ground shadow, hands a little
  away from the body. Weapon: seen from the side, real transparent background, the same pixel scale as the character;
  a bow as 3 images (rest, half drawn, fully drawn). The owner's existing pictures can be reused by erasing the weapon
  from the hand and drawing the weapon on its own.

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
- Implemented: `Session.confirm_skeleton` and `POST /character/skeleton`; saved as
  `DATA/skeletons/<content hash>.json` in the image's own facing (facing saved with it); the app's backend refuses
  Generate with `skeleton_unconfirmed` until confirmed (scripts and tests are not gated). The app's skeleton dialog
  opens by itself after a character with an unconfirmed skeleton loads, shows only the figure (cropped from the
  canvas) with its 18 joints, lets the owner drag a joint or pick it in the list and click its place, remove a
  hidden joint, reset, and confirm. Confirming drops frames drawn from the old skeleton.
- DWPose stays as the pre-fill. If Phase 0 shows it is wrong on most of the owner's art, it can be dropped and the
  pre-fill comes from standard proportions alone.

## Weaponless character animation only (owner, 2026-10-30) — Confirmed

- Owner: "having users attach weapons themselves afterward is more effective. Therefore, the application will now
  focus on generating weaponless character animations." This replaces Option C (v1.4: the app composites separate
  weapon images per frame).
- Consequences decided by me (design delegated, purpose kept):
  - No weapon images as input, no weapon compositing, no weapon grip/tip clicks; the weapon controls leave the app.
  - The attack variants stay as motions (A1a, A1b unarmed; A2 one-handed, A5 bow, A6 staff), drawn without a weapon.
    In the weapon-type attacks the hands keep their grip pose, so a weapon attached later sits in a closed hand.
  - The per-frame hand anchor file (`<name>_weapon.json`: hand x/y in exported pixels, angle, layer, bow stage) is
    kept and exported for every animation whose template has weapon poses, as the aid for attaching weapons later.
  - Character images must be weapon-free (unchanged from "Input images").
- Implemented 2026-10-30 (commit d2b4210): weapon images, the weapon store, compositing, the weapon dialog and the
  weapon routes are gone. Templates carry `grip`: `never` (A1a, A1b), `optional` (Idle, Walk, Run: the owner ticks
  "Tay nắm" in the app) or `always` (A2, A5, A6, A7). The hand anchor file is written whenever the hands were drawn
  gripping; its `hand` is `near` (the arm on the viewer's side) or `far`, because for a character facing left the
  near arm is anatomically its left one.

## Owner requests on the attacks (2026-10-30) — Confirmed by the owner, poses Proposed

- New attack A7 "Tay trái tung chiêu": the left hand reaches forward to cast a skill while the main arm keeps
  holding the weapon in its Idle pose. In the template (facing right) the casting arm is the far arm and the weapon
  arm the near one; the hand anchors follow the near (weapon) hand. Prompt asks for an open palm on the cast.
- A1a punch reworked: the arm swings from behind to the front while the fist comes down from above (overhand):
  guard, draw back, load behind the head, over the top, forward, down and through, recovery, guard.
- Test runs may draw 4 frames (1, 3, 5, 7) to save GPU time: `phase0 --frames 4`, `ab_run.py --frames=4`.
  The app and exports always use 8 (Rule 16).

## Left/right accuracy: depth guide (2026-10-30) — Proposed, on test art only

- Owner: frames where the model does not tell the character's left side from its right. Walk on the test archer
  (4 frames): the model drew the same leg in front in frames 3 and 7, where the template swaps them, so the cycle
  looked like half a step repeated. Darker far limbs (0.5), pose guide 1.0 and repaint 0.5 changed nothing: the pose
  guide says where the limbs are, not which one is in front.
- Fix: a depth guide built from the puppet layers (far limbs dark, body mid, near legs and arms bright) as a second
  ControlNet Union control (mode 1) next to the pose (`puppet.repose_with_depth`, `DEPTH_SCALE`). A/B 0 / 0.5 / 0.8 /
  0.8 with repaint 0.7: at 0.8 the near arm and leg follow the template better (frame 3), looks unchanged; repaint
  0.7 drifts the face. Default 0.8. Not yet confirmed on the punch load frame (fist behind the head): that run was
  stopped by the system for low memory.
- Also fixed: a wide arm (armour) moving away left a background pocket between the cape and the back, painted as a
  white patch (knight). The body the arm hid is now filled before the model repaints (`BODY_CLOSE`).

## Layered cut-out character (2026-10-30) — owner inputs Confirmed, method Proposed (prototype)

- Owner, after testing: the 8 frames are consistent and the action is right, but the face and hairstyle differ from
  the image and most frames have distortions, wrongly oriented legs and broken proportions. Owner proposed learning
  per pose; I advised against (no training data for side-view poses; learning lowers the failure rate but cannot make
  a passed test hold for every new character; a per-character LoRA costs 20–40 min and fixes the face only).
- Owner, confirmed: only ONE complete character image is given; closed fists in all poses; the face keeps its overall
  look (hairstyle, hair colour), fine detail not needed; imperfections in secondary parts are acceptable (the user
  post-processes, not a bug).
- Method (mine, Proposed): AI draws the character once in a spread pose (rig image: near arm forward, far arm back,
  wide stride, closed fists), owner can redraw it; head and body come from the owner's image, limbs from the rig
  image; every frame is the layers moved onto the template skeleton (`animgen/rig.py`), no AI per frame. Face and
  hair are always the owner's pixels; proportions and limb order follow the template exactly; 8 frames in seconds.
  Prototype: `dev/tools/rig_proto.py`.
- Owner, confirmed 2026-10-30: the layered cut-out approach is ACCEPTED. Owner-named limits to reduce: head locked in
  place; cut-out look with seams at joints (needs mesh deformation); capes and long garments need their own bones;
  hand shapes need their own images. Remedies proposed to the owner (Proposed, order pending the owner's answer):
  head turns with the neck and spine; one skinned mesh with blended joint weights instead of rigid parts;
  automatic spring-driven chains for capes and long hair; an optional low-strength AI finishing pass with the
  owner's head pasted back; a per-character hand image only if open palms are wanted.
- Owner, confirmed 2026-10-30: all remedies agreed, in this order: (1) head movement, (2) skinned mesh, (3) cape and
  hair chains, detected automatically and correctable in the skeleton editor, (4) AI finishing pass as an option, off
  by default, (5) open palm hand image per character (one extra AI drawing, used where a template asks for it).
- Done (1)+(2), mine: five grid meshes in the rig image's pose (far arm, far leg, near leg, body+head, near arm);
  elbows and knees blend their two bones over JOINT_BLEND across the joint; the head is rigid, turns with the
  template's head bone within HEAD_TURN, and only the neck blends. Shoulders and hips are not blended: blending
  them pulled the tunic and cape along with the legs (knight: smears at feet and hips). A body fill from the rig
  image that does not match the owner's colours around it takes their average (archer: AI quiver as a pale bag).
- Done (3), mine (`animgen/cloth.py`): hanging parts found in the bind pose: body pixels well behind the spine
  (cape, back flap), body pixels below the hips (skirt, tunic tail), head pixels behind and well below the neck (long
  hair; short ones are beards or helmet rims and stay still). Each swings as a damped pendulum from its pivot over
  the whole animation (12 fps nominal; idle/walk/run looped to a steady cycle, attacks once), pushed by the pivot's
  motion and by air drag ∝ speed²; templates walk in place, so the speed is how fast the planted foot slides back.
  The swing bends the part gradually from the pivot to the hem. Knight: cape still in idle, ~12° walking, ~30°
  running. Correcting the parts in the skeleton editor comes with the app integration.
- Done (5), mine: a hand image per character (one AI drawing, near arm forward, open palm, strength 0.75); its hand
  past HAND_CUT of the forearm replaces the fist in a second version of both arm meshes. Template frames name
  their open hands (`"open": ["l"]`); A7 frames 4–6 open the far hand. Verified on knight and archer.
- Done (4), mine (`animgen/finish.py`): the drawer repaints the rig frame at FINISH_STRENGTH 0.3 (the frame is
  the starting picture, the pose guide as before). Afterwards the rig's head and visible hands are pasted back,
  and the outline is clipped to the rig frame's (3 px slack, none near head and hands): even at 0.3 the drawer
  opened fists and painted fingers around them (archer test). A frame without an open hand drops "open palm" from
  its prompt. About 25 s per frame on the RTX 4060.
- App integration (mine, owner said "go ahead", 2026-10-31): a new step 2 "Khung nhân vật" after the skeleton.
  The AI draws the rig and open-hand images once per character image (about 2 minutes, saved per image, reused
  while the confirmed skeleton is unchanged); the owner reviews both and can redraw. Generate requires a rig and
  takes seconds; the checkbox "Hoàn thiện bằng AI" (off) adds the finishing pass (about 25 s per frame). Rig frames
  come out the same every time, so regenerate/compare is offered only for finished frames.
- Hanging-part toggles sit in step 2, not in the skeleton editor as first proposed: the parts are detected from the
  rig, which does not exist yet when the skeleton is confirmed. Turned-off parts stay as drawn.
- The app no longer uses the per-frame diffusion path; it stays in the backend for dev scripts (phase0).
- Owner's questions on the rig step (2026-10-31), owner agreed my recommendation A+B:
  - Each drawing is captioned with what it gives (the spread pose: arms, legs, cape; the hand image: only the open
    hand; head and torso always from the owner's image).
  - A: each drawing is redrawn on its own (POST /rig {part}); a good body is never lost to fix the hand.
  - B: "Sửa…" copies a drawing (facing as the owner's character) to data/rigs/<key>/rig_edit.png or hand_edit.png
    and opens it in the owner's editor (Windows "edit" verb, else the default app); "Nạp hình đã sửa" rebuilds the
    rig. Same size required; a picture saved without transparency gets the edge-connected background keyed out;
    a drawing that lost over 75% of its pixels or does not build is refused and the old one kept. A redraw deletes
    the copy of the old drawing.
  - Not built: picking parts from earlier draws (C) and a no-AI fallback (D), until the owner finds them needed.
- A5 bow and A6 staff redone (owner, 2026-10-31):
  - A5 follows the archery shot cycle (archery guides and Wikimedia Commons photos): bow low, set-up, bow arm
    straight forward at shoulder height, draw hand pulled back to the jaw with the elbow behind and level with the
    arrow, hold, release (hand slides back past the ear, open), follow-through. Draw-hand positions are solved
    from target points (authoring.reach). Frame 4 keeps the elbow low in front: a flat side view cannot show the
    elbow swinging out toward the viewer on its way back.
  - A6, the owner's description: the staff is raised forward to about 40 degrees above horizontal, held for
    three frames, and brought back to where it started.

## Not done on purpose

- No free-form prompts, no batch generation, no automatic quality scoring, no Unity configuration files, no
  project library, no effects (§2 prohibited list).
- No Jump/Fall/Hit/Death/A3/A4 templates in version 1 (cuts C1, C2).
- No pixel-art step (C3).
