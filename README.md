# AnimGen — local AI 2D animation generator

A Windows desktop tool that makes 8-frame, transparent, Unity-ready 2D character animations from one character
image, entirely offline, for the owner's own games. Design: [docs/DESIGN.md](docs/DESIGN.md).

**State (2026-10-30): cloud build done; not yet run on a GPU or compiled on Windows.**
What was decided during the build: [docs/DECISIONS.md](docs/DECISIONS.md).
What still needs testing on the owner's PC: [docs/PC_TESTING.md](docs/PC_TESTING.md).

## Layout

```
app/                     WPF app (.NET 9, C#), Vietnamese UI, no NuGet packages
  AnimGen/Services/      backend process (token, job object), HTTP client, moved-install repair
  AnimGen/ViewModels/    workflow of §11.2
  AnimGen/Views/         main window, weapon dialog
  AnimGen/Themes/        every colour, size and spacing value
backend/                 Python package "animgen" (the backend process)
  animgen/server.py      loopback HTTP API (token, tasks, progress, cancel)
  animgen/session.py     character, generation, retries, regenerate/compare, weapons, export
  animgen/inputs.py      input checks (§4.4)
  animgen/skeleton.py    18-point skeletons, fitting to the character, OpenPose drawing
  animgen/templates.py   template format and the animation list
  animgen/authoring.py   the 8 poses of every template, as joint angles (python -m animgen.authoring)
  animgen/template_data/ generated templates (JSON)
  animgen/weapons.py     weapon store, grip/tip points, placement
  animgen/frames.py      canvas placement, edge cleaning, canvas fit, validation
  animgen/export.py      PNG frames + sprite sheet
  animgen/engines/       demo engine (CPU) and the real engine (Diffusers, DWPose, ToonOut)
  animgen/phase0.py      Phase 0 measurement run
  tests/                 pytest suite (no GPU needed)
scripts/                 runtime setup, model download, publish, dev run
docs/templates/          contact sheets of the Proposed templates
```

## Run the tests (any OS, no GPU)

```
cd backend
pip install -r requirements-dev.txt
python -m pytest -q
```

## Run the backend alone (demo engine)

```
cd backend
ANIMGEN_TOKEN=secret python -m animgen --engine demo --data ../devdata
# prints READY <port>; send requests with header X-AnimGen-Token: secret (routes in animgen/server.py)
```

## Install on the owner's PC

1. `scripts\setup_runtime.ps1 -Data D:\AnimGen -Uv <uv.exe>` (with internet, once: Python, PyTorch, models).
2. `scripts\publish.ps1 -Data D:\AnimGen` (needs the .NET 9 SDK).
3. Start `D:\AnimGen\AnimGen.cmd`. `AnimGen (thu, khong AI).cmd` starts the demo engine.

The `D:\AnimGen` folder can be copied or moved as a whole; nothing needs the internet after step 1.

## Unity

Each animation exports `<character>_<animation>[_<weapon>_<skin>]_01..08.png` and `..._sheet.png` (one row of 8,
no padding). Import the sheet as Sprite Mode *Multiple* and slice with *Grid By Cell Count*, 1 row × 8 columns.
