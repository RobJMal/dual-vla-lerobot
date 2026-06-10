#!/usr/bin/env bash
# Training commands for the Dual-System VLA (Diffusion Policy as System 1) ablation study.
#
# Usage:
#   bash scripts/train_dual_vla_diffusion.sh dynamic    # Full System 2 (GPU)
#   bash scripts/train_dual_vla_diffusion.sh frozen     # Frozen-initial ablation (GPU)
#   bash scripts/train_dual_vla_diffusion.sh disabled   # Disabled System 2 baseline (GPU)
#   bash scripts/train_dual_vla_diffusion.sh all        # All three variants sequentially

set -euo pipefail

MODE="${1:-all}"

COMMON="
  --policy.type=dual_vla_diffusion
  --policy.push_to_hub=false
  --dataset.repo_id=lerobot/libero_10
  --policy.device=cuda
  --batch_size=128
  --steps=200000
  --eval_freq=40000
  --save_freq=40000
  --log_freq=200
  --wandb.enable=true
  --wandb.project=dual-vla-lerobot
"

run_dynamic() {
  echo "=== dual_vla_diffusion: system2_mode=dynamic ==="
  uv run lerobot-train \
    $COMMON \
    --policy.system2_mode=dynamic \
    --policy.system2_update_freq=10 \
    --job_name=dual_vla_diffusion_dynamic \
    --output_dir=outputs/dual_vla_diffusion_dynamic
}

run_frozen() {
  echo "=== dual_vla_diffusion: system2_mode=frozen_initial ==="
  uv run lerobot-train \
    $COMMON \
    --policy.system2_mode=frozen_initial \
    --job_name=dual_vla_diffusion_frozen \
    --output_dir=outputs/dual_vla_diffusion_frozen
}

run_disabled() {
  echo "=== dual_vla_diffusion: system2_mode=disabled (pure Diffusion Policy baseline) ==="
  uv run lerobot-train \
    $COMMON \
    --policy.system2_mode=disabled \
    --job_name=dual_vla_diffusion_disabled \
    --output_dir=outputs/dual_vla_diffusion_disabled
}

case "$MODE" in
  dynamic)   run_dynamic ;;
  frozen)    run_frozen ;;
  disabled)  run_disabled ;;
  all)
    run_dynamic
    run_frozen
    run_disabled
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {dynamic|frozen|disabled|all}"
    exit 1
    ;;
esac
