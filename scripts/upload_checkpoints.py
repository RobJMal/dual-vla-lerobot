"""Upload trained dual-VLA checkpoints to HuggingFace Hub.

Discovers all subdirectories in OUTPUTS_DIR and uploads each one to a
separate HF repo named  <HF_USER>/dual-vla-<dir_name>.  This means
'spatial_dynamic', 'spatial_frozen', and 'spatial_disabled' each get
their own repo and never overwrite earlier variants.

Usage:
    uv run python scripts/upload_checkpoints.py
    uv run python scripts/upload_checkpoints.py --outputs_dir /path/to/outputs
"""

import argparse
import time
from pathlib import Path

from huggingface_hub import HfApi

HF_USER = "RobJMal"
DEFAULT_OUTPUTS_DIR = Path("/workspace/dual-vla-lerobot/outputs")

parser = argparse.ArgumentParser()
parser.add_argument("--outputs_dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
args = parser.parse_args()

OUTPUTS_DIR: Path = args.outputs_dir

if not OUTPUTS_DIR.exists():
    raise SystemExit(f"outputs_dir not found: {OUTPUTS_DIR}")

dirs = sorted(p for p in OUTPUTS_DIR.iterdir() if p.is_dir())
if not dirs:
    raise SystemExit(f"No subdirectories found in {OUTPUTS_DIR}")

print(f"Found {len(dirs)} run(s): {[d.name for d in dirs]}", flush=True)

api = HfApi()

for base in dirs:
    files = sorted(
        f for f in base.rglob("*")
        if f.is_file()
        and ".cache" not in str(f)
        and "optimizer_state" not in f.name
        and "training_state" not in str(f)
    )
    if not files:
        print(f"\nSkipping {base.name} — no files found", flush=True)
        continue

    repo_id = f"{HF_USER}/dual-vla-{base.name}"
    print(f"\n=== {base.name}: {len(files)} files → {repo_id} ===", flush=True)

    for i, fp in enumerate(files):
        rel = str(fp.relative_to(base))
        size_mb = fp.stat().st_size / 1024 / 1024
        print(f"[{i+1}/{len(files)}] {rel} ({size_mb:.1f} MB)", flush=True)

        for attempt in range(5):
            try:
                api.upload_file(
                    path_or_fileobj=str(fp),
                    path_in_repo=rel,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print("  ok", flush=True)
                break
            except Exception as e:
                print(f"  attempt {attempt+1}/5 failed: {e}", flush=True)
                time.sleep(15)

print("\nAll done.", flush=True)
