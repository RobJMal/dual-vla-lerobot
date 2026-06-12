"""Upload trained dual-VLA checkpoints to HuggingFace Hub.

Usage:
    uv run python scripts/upload_checkpoints.py
"""

from huggingface_hub import HfApi
from pathlib import Path
import time

HF_USER = "RobJMal"
VARIANTS = ["dynamic", "frozen", "disabled"]
OUTPUTS_DIR = Path("/workspace/lerobot/outputs")

api = HfApi()

for variant in VARIANTS:
    base = OUTPUTS_DIR / f"dual_vla_{variant}"
    if not base.exists():
        print(f"Skipping {variant} - not found at {base}")
        continue

    files = sorted([
        f for f in base.rglob("*")
        if f.is_file() and ".cache" not in str(f)
    ])
    repo_id = f"{HF_USER}/dual-vla-{variant}"
    print(f"\n=== {variant}: {len(files)} files → {repo_id} ===", flush=True)

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
