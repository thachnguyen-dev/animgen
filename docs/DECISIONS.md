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
- Body-part layers (owner, 2026-10-31; owner said "go ahead" to my plan):
  - The owner's request: one reference image plus a group of separate body parts, corrected in Adobe software;
    hidden sections rebuilt by the AI while the visible pixels stay; a cape, a skirt, clothing or accessories never
    belong to an arm or a leg.
  - Mine (backend/animgen/parts.py): after the two drawings, the parts are found in the spread-pose drawing.
    Segment Anything 2.1 small (Apache-2.0) follows each limb's outline from points along its bones; a
    clothes-parsing model (segformer_b2_clothes, MIT) says which pixels are legs and which are garments. Leftover
    garments become cape (hangs from the back), skirt (spans the hips; garments below the hips too) or long hair.
    The joints are read on the drawing itself (DWPose), with the forward arm and leg taken as the near ones.
  - Hidden sections: each limb's expected outline (a capsule along its bones as wide as it is drawn) minus what
    shows, within the figure, is repainted by the AI with the whole character as context (masked inpaint, strength
    0.75), from a guess (the near limb moved onto the far bones, darker). The torso's hidden part is only what the
    near arm covers, no wider than the body. The cape's hidden part is what the body and legs cover of its
    outline, behind the spine only (in front of it the AI painted armour into the cape). Pixels the AI paints as
    background are left out; pixels no part claimed (thin outlines) go to the nearest part.
  - The head layer is the owner's own head, cut at the level line through the neck (the face never drifts); the
    torso layer is the reference's, with the reference's neck below that line (in the owner's standing image the
    near arm hangs over the torso).
  - One Photoshop file per character image (data/rigs/<key>/parts.psd): the reference at the bottom as a guide,
    then the group "Bộ phận" with one named layer per part, back to front, facing as the owner's character; the
    open hand is a hidden layer. "Sửa trong Photoshop" opens it in Photoshop (App Paths registry, else the default
    program); each save rebuilds the rig (the app watches the file; a save while the app was closed is taken on the
    next start). Refused and kept as before: a resized file, an unreadable one, the "Thân" layer gone, or most of
    the character erased. Layer names are how parts are found, so renamed layers are left out.
  - The rig is built from the layers (rig.build_rig_from_layers): each limb on its own bones, the far ones as found
    in the reference; cape, skirt and long hair are meshes of their own that swing as a whole. Rigs made before
    keep the old builder.
  - The PNG "Sửa…" copies (answer B above) are replaced by the Photoshop file. A redraw of the spread pose makes
    every part again (the app asks first while the file is out); a redraw of the hand replaces only the open hand.
  - Time on the RTX 4060: about 5-6 minutes more per character (parts 20 s, up to six hidden sections about 50 s
    each; knight 380 s, archer 328 s with the drawings). Models: 280 MB more.
  - App: the step-2 caption says the character is put together from separate parts; "Sửa trong Photoshop" and
    "Nạp lại" (shown while the file is out) sit in one row under both drawings, as the file covers both (in the
    left column the button was clipped, seen in a window capture).
  - Known: on the knight the cape's frayed hem went to the far shin layer (the clothes parser called it leg); the
    owner can move it in Photoshop.
  - Fixed on the way: the rig was dropped when a saved joint rounded differently from the live one (archer: 541.5
    against 541.4996); the skeletons now match within 1 px.
- Every pose first (owner, 2026-10-31: "complete the remaining missing animation poses … prioritize completing
  these poses before optimizing the current workflow"):
  - The cuts C1 and C2 are lifted: Jump, Fall, Hit, Death, A3 (two-handed) and A4 (polearm) now have their 8 poses
    (Proposed until the owner approves them, as every pose). Jump is a whole jump in place (crouch to landing);
    Fall is a loop in the air; Hit recoils from a blow in front; Death falls backward and lies face up (a pose may
    now rest on its lowest joint instead of an ankle). A3 and A4 hold the weapon with both hands; A4's far hand is
    in front on the shaft.
  - More attacks, from common 2D attack sets (owner asked for research; sources: sprite packs listing thrusts,
    overhead and rising slashes, combo and dual-wield attacks): A2b one-handed thrust, A2c overhead chop, A2d rising
    slash, A8a dual wielding with alternating slashes, A8b dual wielding with a cross slash. Proposed: they extend the
    Confirmed attack list only if the owner keeps them.
  - Dual wielding: a template frame may carry "weapon2", the other hand's weapon; the hand anchor file then has
    "second" for it (same fields).
  - The character keeps within 5% of its height of where it stands (§6.3, no unnecessary drift), lunges and the hit
    recoil included.
  - The owner's custom poses (answer to "option 1", a pose editor) stay planned as the fallback for characters the
    built-in poses do not suit (owner, 2026-10-31).

## Part map from the owner's own image (owner, 2026-10-31) — owner direction Confirmed, method Proposed

- Why: the owner found the AI's spread-pose redraw of the whole character to be where most errors started; joint
  editing alone would not fix a wrong drawing. The owner chose to correct the parts in Photoshop (option A), with
  the layers grouped by kind, and to correct the reference before the expensive processing.
- The owner's placed image is the reference (rig.png), never redrawn. The AI draws only an open hand.
- Stage 1, the part map: the parts are found on the image itself against the confirmed skeleton (a far limb the
  owner did not place sits 2% of the figure behind the near one), every pixel in exactly one part, the head always
  whole. Saved as map.psd: groups back to front "Phụ kiện phía sau", "Tay chân phía xa", "Chân phía gần", "Thân",
  "Phụ kiện phía trước", "Đầu", "Tay phía gần", plus a hidden colour overlay layer for viewing. A provisional rig is
  built at once, so the owner can already preview poses.
- Stage 2: "Vẽ phần bị che" paints what each part hides (about 5 minutes) and writes parts.psd for the last
  touches. "Về bản đồ" goes back to the map as the owner left it; "Tách lại" splits again.
- The layer order in either Photoshop file is the drawing order of the rig (dragging a layer moves that part in
  front or behind). A map saved in Photoshop but not yet reloaded by the app is read before painting.
- Real-engine check on the knight test art: map and hand 101 s, then the hidden sections; the near arm was merged
  into the torso on the map, the kind of mistake the map stage exists for.
- The pose editor (owner's option 1) stays the planned fallback for special characters.

## Three-quarter view pose set (owner, 2026-10-31) — Confirmed (owner, 2026-10-31), without A3

- Why: the side-view poses on a 3/4 character lose the body's width and hide the far arm (tested on a 3/4 shape
  character); the owner chose a separate pose set over adapting the side one, and asked for it to follow common
  3/4 animation practice.
- Research used: 3/4 front is about 45 degrees (turnaround guides); in a walk the hips turn about 5-10 degrees with
  the legs and the chest 3-8 degrees against them, nearly parallel at the passing pose (walk-cycle course notes);
  a slash winds the torso one way and swings it through the other, the back foot shifting (pixel-art slash
  tutorials); sprite packs keep idle/walk/attack/death rows with the feet on one baseline.
- Method (authoring34.py): each side pose is lifted into 3D (its joint angles are swings in the body's forward-down
  plane), given shoulder and hip width, turned, and projected at 45 degrees with the camera 12 degrees above. The
  chest turns with the arm that leads, the hips with the legs (counter-turned in cycles); by hand: the kick turns
  the hips, two-handed and bow stances turn the near shoulder back. Each frame says which limbs are nearer the
  viewer than the body: the far arm reaching forward and the far leg stepping forward are drawn in front, the near
  arm swung far back behind ("front"/"back" in the template; Rig.drawing_order).
- The character's angle ("side" or "34") is guessed from the skeleton (both eyes found and the shoulders or hips
  well apart), shown in step 1 with "Đổi góc nhìn", and saved with the confirmed skeleton.
- Known limit: the torso is one flat layer, so a turn shows in the shoulders and limbs, not in a narrowing chest.
- Owner corrections (2026-10-31): the far (left) arm always stays behind the body; the kick swings across the
  picture at full length and a little higher (it shrank toward the viewer); the bow is drawn level across the
  picture (the chest turned toward profile) instead of back into it.
- Second round (owner, 2026-10-31): two-handed weapons (A3, A4) keep both hands together on one handle in every
  frame (the far hand placed on the handle in 3D, its arm solved to reach it), so a weapon can be attached; the
  spear's right hand leads with the arm extended forward and the left hand behind it on the shaft; the bow keeps
  fists (no open hand); a limb pointing toward the viewer is shortened along its length only and keeps its width
  (a leg looked enlarged next to the shrunk other one).
- Third round (owner, 2026-10-31): on the two-handed weapon (A3) the right elbow is always in front of both the
  right shoulder and the hand (the hands, with the weapon, shift the least distance that allows it) and the left
  arm is drawn over the head and body in every frame (it used to vanish behind the body in 5 of 8); on the spear
  (A4) the right arm sits further back throughout (the near hand only 0.03 ahead of the side pose, at least 0.05
  from the shoulder) and is never put behind the body.
- Fourth round (owner, 2026-10-31): the two-handed smash (A3) travels across the picture, not toward the viewer:
  the arms' swing is turned to profile (SMASH_ACROSS) while the chest keeps its turn. The elbow-forward rule now
  holds while the hands are held in (ready, lift, overhead, recover); on the swing, smash and impact the arms reach
  out and the elbow sits between the shoulder and the hand (still in front of the shoulder) — an arm reaching out
  cannot have its elbow past the hand, and forcing it pulled the smash back into the body. The spear (A4) is back
  to the second-round arm; only in the wind-up frames (1-3, 7-8) the right forearm and hand come back 0.08 with
  the elbow behind the shoulder (the third round had pulled the whole side pose back, which bent the left arm).
- Closed (owner, 2026-10-31): the current poses are confirmed, no further changes. A3 (two-handed weapon) still
  did not satisfy and is removed from the 3/4 set rather than fixed: authoring34.OMITTED, no view34/a3.json. The
  animation list follows the character's angle (GET /animations), so a 3/4 character sees A3 as "không có ở góc
  ba phần tư" and cannot generate it (error animation_not_for_angle); side characters keep A3. Special poses are
  made with the pose editor instead (next section).
- The view is guessed again when the owner confirms the skeleton (owner, 2026-10-31): the corrected joints decide
  side or 3/4, unless the owner chose the view ("Đổi góc nhìn"); that choice is saved with the skeleton
  (angle_owner) and never guessed over. Skeletons saved before this count as the owner's choice.
- Test character: dev/tools/shape_character.py --view 34 (drawn from the same 3D body) renders all animations to
  C:/AnimGen/project/pose_test_34.

## Pose editor (owner, 2026-10-31) — owner request Confirmed, plan approved by the owner ("yes approve all")

The owner makes new animations themselves: the app shows a sample character in 8 frames, the owner moves its
joints and body parts in every frame and saves the pose. Built as planned and approved:
- Sample character: the shape figure (animgen/sample.py, moved there from dev/tools/shape_character.py, which now
  imports it), opened in a session of its own per view under <data>/pose_editor/<side|34>, with its rig written
  from the shapes, so every frame is drawn exactly as generation draws it (rig render, no AI, under a second).
- Editing (PoseEditorWindow, same layout as the skeleton dialog): dragging a joint turns the body part from it
  around its parent joint (bones keep their length); dragging a hip moves the whole body; Shift moves one joint
  alone. After each drag the backend fits the frame to the sample (lengths kept, at most 1.25x; a shorter bone
  stays shorter, as pointing at the viewer; feet on the ground in grounded frames) and draws it. Per frame: feet
  on the ground / in the air, near/far hand open, and in the 3/4 view the layer order (far arm or far leg in front,
  near arm behind). "Chép khung trước", "Đặt lại khung này", the previous frame shown faint, 8 fps playback.
- Start: a view (side or 3/4, default the loaded character's) and an existing animation or standing; "Bắt đầu
  lại". Editing a saved pose keeps its view.
- Saving: a name (up to 40 characters) and the weapon grip (never / optional / always); the weapon direction for
  the hand-anchor file follows the near forearm. Saved as a motion template, checked like the built-in ones, in
  <data>/poses/u_<8 hex>.json (outside the app, so updates keep it). Its rest pose is the sample's bind pose with
  every bone at full length, so a frame fitted to the sample comes back unchanged (tested both views); another
  character takes the bone directions with its own lengths.
- In the app: step 3 has "Tạo tư thế mới…", "Sửa tư thế…" and "Xoá tư thế" (the last two for the owner's own
  poses); the list shows a saved pose only for characters seen from its view, after the built-in ones, and
  generates it like any other. Server: GET /poses, POST /poses/start|open|preview|save|delete.
- Not done: no keyboard way to move joints (the skeleton dialog's joint list was not copied); no undo beyond
  "Đặt lại khung này"; no per-frame weapon direction handle.

## Step-2 pictures zoomed in (owner, 2026-10-31) — owner choices Confirmed

The owner asked for a zoomed-in view of the character rig; chosen by the owner: the step-2 picture (and the open
hand next to it), in a separate window. ImageViewerWindow opens from a click on either picture or its
"Phóng to…" button: the mouse wheel zooms around the cursor (10 % to 1600 %), dragging pans, buttons "Vừa
khung" (fits the drawn part of the picture, not the whole square canvas) / 100 % / 200 % / 400 %; from 200 % the
pixels are drawn sharp (nearest neighbour) so the edges of each part can be judged.

## Not done on purpose

- No free-form prompts, no batch generation, no automatic quality scoring, no Unity configuration files, no
  project library, no effects (§2 prohibited list).
- No pixel-art step (C3).
