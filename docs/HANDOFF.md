# Handoff — local finalisation paused (2026-10-30)

Paused at the owner's request. This page says where the work stands, what was verified on the owner's PC, and what
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
| Drawing method | **Changed** (see DECISIONS.md "Drawing method"): rough puppet repose (`animgen/puppet.py`) + partial repaint (`STRENGTH` 0.6), pose control only. Verified on **single-frame probes** only; no full 8-frame run yet. |
| Old method (from noise, pose + inpaint control) | Rejected: the right panel stayed an empty flat field for poses unlike the reference (`dev/test_art/result_old_method_blank.png`); Idle/Walk drifted in outfit and style. |
| Facing | DWPose detects the test character as facing left; it is mirrored in and out as designed. |
| Second figures / props | `clean_alpha` now removes regions away from the target skeleton. |
| Licences | Verified; not a selection criterion (personal use). |

## Test art (temporary, made by Claude — not the owner's)

`dev/test_art/`: `test_hero.png` (Animagine + ToonOut, faces left), `test_sword.png` (grip 70,35 / tip 548,35),
`test_bow_1..3.png` (grip 210,410 / tips 183,40 · 165,50 · 149,60). The owner's images in `D:\Game\ainm\images` show
characters holding weapons and cannot be used as weapon inputs; the owner has no separated art yet and asked for a
recommendation (separate vs combined) after testing.

## Next steps

1. **GPU (owner's PC only):** full run with the new method — `python -m animgen.phase0 --data "C:\AnimGen"
   --character dev\test_art\test_hero.png --sword … --bow …` (the arguments above). Judge frame consistency,
   hands, weapon placement; tune `STRENGTH`, `CONTROL_SCALE`, `IP_ADAPTER_SCALE`, steps (`dev/tools/probe.py` for
   single frames, `dev/tools/ab_run.py` for whole animations).
2. **GPU:** separate vs combined comparison for the owner (a character holding the sword: `dev/tools/make_temp_art.py
   … hero_sword`), then the recommendation.
3. **Code only (cloud is fine):** the skeleton editor decided in DECISIONS.md (joints over the character, DWPose
   pre-fill, confirm once per image, required before Generate); puppet improvements with tests (`tests/test_puppet.py`).
4. **Then:** publish (`scripts\publish.ps1 -Data C:\AnimGen`), offline test, moved-folder test; Phase 0 on the owner's
   art when it exists.

## Notes for a cloud session

No GPU there: the real engine, Phase 0 and all tuning need the owner's PC. Code-only work (step 3) and the test
suite (`backend/requirements-dev.txt`, `pytest`) run anywhere. Keep CRLF line endings and UTF-8 with BOM for
PowerShell scripts.
