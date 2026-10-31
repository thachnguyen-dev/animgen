# Handoff — local finalisation (2026-10-30, resumed)

Paused at the owner's request, then resumed the same day. This page says where the work stands, what was verified on the owner's PC, and what
comes next, so a later session (on the PC or in the cloud) can pick it up. Decisions are in `DECISIONS.md`.

## Owner's PC (verified)

- RTX 4060 Laptop GPU 8 GB, 15.8 GB RAM (about 4 GB free in normal use), Windows 11.
- Runtime and models installed by `scripts\setup_runtime.ps1` in `C:\AnimGen` (D: had only 15 GB free):
  Python 3.12 + venv with CUDA PyTorch, 7 model folders (14.85 GB), `models\MANIFEST.json`.
- The app builds; the demo-engine walkthrough passes (character, weapon dialog with grip/tip, preview, generate,
  export, backend ends with the app).

## State of the real engine

| Item | State |
|---|---|
| Loading and running on 8 GB | Works. UNet streamed leaf by leaf (`diffusers.hooks.apply_group_offloading`). |
| Speed | Each frame drawn at 768 (`DRAW_SIZE`); about 1 min per frame. 1024 spilled into shared memory (9 s/step). |
| Drawing method | **Changed** (see DECISIONS.md "Drawing method"): rough puppet repose (`animgen/puppet.py`) + partial repaint (`STRENGTH` 0.6), pose control only. Full Phase 0 run: Idle, Walk, A2, A5 (bow) 8/8 frames first try, 40–55 s per frame. Weak: thin pale lines, small drift. |
| Old method (from noise, pose + inpaint control) | Rejected: the right panel stayed an empty flat field for poses unlike the reference (`dev/test_art/result_old_method_blank.png`); Idle/Walk drifted in outfit and style. |
| Facing | DWPose detects the test character as facing left; it is mirrored in and out as designed. |
| Second figures / props | `clean_alpha` now removes regions away from the target skeleton. |
| Licences | Verified; not a selection criterion (personal use). |

## Test art (temporary, made by Claude — not the owner's)

`dev/test_art/`: `test_hero.png` (Animagine + ToonOut, faces left), `test_sword.png` (grip 70,35 / tip 548,35),
`test_bow_1..3.png` (grip 210,410 / tips 183,40 · 165,50 · 149,60). The owner's images in `D:\Game\ainm\images` show
characters holding weapons and cannot be used as weapon inputs; the owner has no separated art yet and asked for a
recommendation (separate vs combined) after testing.

## Done in the resumed session

1. Full Phase 0 run with the new method: Idle, Walk, A2, A5 all 8/8 frames first try (DECISIONS.md).
2. Line-weight A/B: IP-Adapter 0.8 is the default now.
3. Separate vs combined input test: separate recommended to the owner (DECISIONS.md "Input images").
4. Skeleton editor (backend + app dialog).
5. Published to `C:\AnimGen`; moved-folder test on a throw-away copy passed (venv repaired, backend offline,
   skeleton dialog, backend ends with the app).
6. Real-engine walkthrough of the published app (UI Automation, my test art): DWPose pre-filled the skeleton dialog
   correctly and detected the facing; Generate Idle 8 frames in 342 s (model load ~20 s); export 256 px wrote 8 PNGs
   and the sprite sheet; the backend ended with the app.

## Next steps

1. **GPU, when the owner has separated art:** Phase 0 on it; judge the skeleton pre-fill (DWPose) on the owner's
   style, consistency and line weight; tune with `dev/tools/ab_run.py` + `dev/tools/ab_sheet.py`.
2. **Code only:** puppet improvements with tests (`tests/test_puppet.py`).

## Notes for a cloud session

No GPU there: the real engine, Phase 0 and all tuning need the owner's PC. Code-only work (step 3 of Next steps) and the test
suite (`backend/requirements-dev.txt`, `pytest`) run anywhere. Keep CRLF line endings and UTF-8 with BOM for
PowerShell scripts.

## Paused 2026-10-30 (second pause)

- Weapons are optional in every attack (a2/a5/a6 "optional"); armed attacks still grip; export writes `<name>_weapon.json`
  with per-frame hand anchors (x/y in exported pixels, angle, layer, bow stage), mirrored for left-facing characters.
  Built-in weapon compositing is kept. Tests: 102 passed before the pause (only an app note string changed after).
- Open: left/right limb accuracy (owner saw frames where the model swaps the character's left and right). Templates
  are correct (dev/tools/template_sheet.py). Next: `dev/tools/ab_run.py` on walk/run with variants base / far=0.5 /
  cn=1.0 / str=0.5, judged with `dev/tools/lr_sheet.py`.
- Western batch stopped at the pause: knight (old code), archer, archer34 done; mage and hero not run.
- The published C:/AnimGen backend is outdated (no tagger, puppet fixes, weapon anchors): republish with the app closed.

## Resumed 2026-10-30 (after the second pause)

- Weaponless app (d2b4210): grip choice, A7 cast, overhand A1a, 4-frame test runs. Depth guide and the cape-pocket
  fill: see DECISIONS.md "Left/right accuracy".
- Next: rerun `python -m animgen.phase0 --character dev/test_art/test_knight.png --anims a1a walk --frames 4` (was
  stopped for low memory) and check the punch load frame and the cape; then republish to C:/AnimGen with the app
  closed and check the "Tay nắm" checkbox colour in the running app.
## Owner test result and the layered cut-out direction (2026-10-30)

- Owner test of the diffusion-per-frame build: consistent frames, but face/hair drift, wrong legs, broken proportions.
  Owner inputs: one complete character image; closed fists always; keep overall face/hair; imperfections in secondary
  parts are acceptable.
- Prototype (e8423c3): animgen/rig.py + dev/tools/rig_proto.py. One AI rig image per character (spread pose), then
  every frame is layers moved onto the template skeleton. Knight and archer sheets: identical face, correct leg order,
  punch from behind the head, cast arm forward. See DECISIONS.md "Layered cut-out character".
- Next: owner decides on integration (rig image step reviewed/redrawn in the app, then instant frames). The phase0
  knight rerun above is moot if the rig method is adopted.
## Remedies built (2026-10-30, owner agreed all five)

- 6b445fd: head turns with the template; elbows/knees bend on skinned meshes; capes, skirts and long hair swing
  (cloth.py); open-palm hand image for A7 frames 4-6; optional AI finishing pass (finish.py) that keeps the rig's
  head, hands and outline. GPU runs: C:/AnimGen/venv/Scripts/python.exe (the dev venv has no torch).
- Next: integrate into the app: rig step (rig image + hand image, owner reviews/redraws), instant generation,
  "AI finishing" checkbox (off), hanging-part on/off in the skeleton editor; republish with the app closed.
## App integration (2026-10-31)

- Step 2 "Khung nhân vật" in the app (POST /rig, /rig/hanging; /generate takes `finish`). Real-engine check on the
  knight: rig 133 s, A7 8 frames 3 s, one finished frame 23 s. Published to C:/AnimGen for the owner to test.
- Then per-drawing redraw and owner editing (see DECISIONS.md): knight hand-only redraw 97 s (body kept), edit
  reload 2.7 s, A7 8/8 after the edit.
- Body-part layers and Photoshop editing (DECISIONS.md "Body-part layers"): parts.py, rig.build_rig_from_layers,
  session._make_layers / _import_psd, POST /rig/edit returns the PSD. New models sam2 and clothes-parsing (in
  scripts/download_models.py); psd-tools in both requirements files. Real runs: knight 380 s for the layers from
  an existing drawing, archer 328 s. Sheets in the session scratchpad only (test art).
- Every pose done (20 animations, DECISIONS.md "Every pose first"); shape test character
  (dev/tools/shape_character.py, dev/test_art/test_shapes.png) renders all of them to D:/Game/ainm/pose_test for the
  owner to tune poses.
- Part map flow (DECISIONS.md "Part map from the owner's own image"): POST /rig (part "parts"|"hand"),
  /rig/rebuild, /rig/map; RigInfo.Stage in the app; step 2 shows the map, "Vẽ phần bị che", "Tách lại",
  "Về bản đồ".
- Three-quarter pose set (DECISIONS.md "Three-quarter view pose set"): authoring34.py ->
  template_data/view34 and docs/templates/view34; POST /character/angle; step 1 "Đổi góc nhìn".
