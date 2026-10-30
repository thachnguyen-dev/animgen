"""Download the model folders into DATA\\models (run once, on the owner's PC, with internet).

Files are downloaded with ``local_dir`` into plain folders: Hugging Face cache symlinks fail on Windows (§14).
After this, the app runs offline (HF_HUB_OFFLINE=1). A MANIFEST.json records every file, its size and SHA-256.

Usage:  python download_models.py <DATA>\\models

Every repository and file name below is Proposed (§9.2). Read each model card before the first use and record
any licence restriction in docs/DECISIONS.md (§9.3).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = [
    # (folder, repo, patterns or files, licence note)
    ("animagine-xl-4.0", "cagliostrolab/animagine-xl-4.0",
     ["model_index.json", "scheduler/*", "text_encoder/*", "text_encoder_2/*", "tokenizer/*", "tokenizer_2/*",
      "unet/*", "vae/*"], "CreativeML Open RAIL++-M"),
    ("sdxl-vae-fp16-fix", "madebyollin/sdxl-vae-fp16-fix",
     ["config.json", "diffusion_pytorch_model.safetensors"], "MIT"),
    ("ip-adapter", "h94/IP-Adapter",
     ["sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors", "models/image_encoder/config.json",
      "models/image_encoder/model.safetensors"], "Apache-2.0 (verify)"),
    ("dwpose", "yzd-v/DWPose", ["yolox_l.onnx", "dw-ll_ucoco_384.onnx"], "Apache-2.0 (verify)"),
    ("birefnet", "ZhengPeng7/BiRefNet", ["*.json", "*.py", "model.safetensors"], "MIT"),
    ("toonout", "joelseytre/toonout", ["birefnet_finetuned_toonout.pth", "README.md"],
     "MIT; training data CC-BY 4.0 (verify repository and file name)"),
]

# ControlNet Union ProMax is two specially named files in the repository; they are renamed to the standard names.
CONTROLNET = ("controlnet-union-sdxl-promax", "xinsir/controlnet-union-sdxl-1.0",
              {"config_promax.json": "config.json",
               "diffusion_pytorch_model_promax.safetensors": "diffusion_pytorch_model.safetensors"},
              "Apache-2.0")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"models": []}

    for folder, repo, patterns, licence in MODELS:
        print(f"== {repo} -> {folder}", flush=True)
        snapshot_download(repo_id=repo, local_dir=root / folder, allow_patterns=patterns)
        manifest["models"].append({"folder": folder, "repo": repo, "licence": licence})

    folder, repo, files, licence = CONTROLNET
    print(f"== {repo} -> {folder}", flush=True)
    target = root / folder
    target.mkdir(parents=True, exist_ok=True)
    for src, dst in files.items():
        p = Path(hf_hub_download(repo_id=repo, filename=src, local_dir=target))
        shutil.move(str(p), target / dst)
    manifest["models"].append({"folder": folder, "repo": repo, "licence": licence})

    # Remove download bookkeeping folders, then record what is there.
    for cache in root.rglob(".cache"):
        shutil.rmtree(cache, ignore_errors=True)
    files_out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            files_out.append({"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256(p)})
    manifest["files"] = files_out
    total = sum(f["size"] for f in files_out)
    manifest["total_gb"] = round(total / 1e9, 2)
    with open(root / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"Done: {len(files_out)} files, {manifest['total_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
