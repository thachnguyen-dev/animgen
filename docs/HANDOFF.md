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
