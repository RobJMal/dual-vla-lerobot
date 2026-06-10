#!/usr/bin/env bash
# Training commands for the Dual-System VLA ablation study.
#
# Usage:
#   bash scripts/train_dual_vla.sh cpu_test     # Phase 1 – local CPU smoke test
#   bash scripts/train_dual_vla.sh dynamic       # Phase 2 – full System 2 (GPU)
#   bash scripts/train_dual_vla.sh frozen        # Phase 2 – frozen-initial ablation (GPU)
#   bash scripts/train_dual_vla.sh disabled      # Phase 2 – disabled System 2 baseline (GPU)
#   bash scripts/train_dual_vla.sh all           # Phase 2 – run all three GPU variants

set -euo pipefail

MODE="${1:-cpu_test}"

COMMON_CPU="
  --policy.type=dual_vla
  --policy.chunk_size=20
  --policy.n_action_steps=10
  --policy.dim_model=256
  --policy.push_to_hub=false
  --dataset.repo_id=lerobot/libero_10
"

COMMON_GPU="
  --policy.type=dual_vla
  --policy.chunk_size=20
  --policy.n_action_steps=10
  --policy.dim_model=256
  --policy.push_to_hub=false
  --dataset.repo_id=lerobot/libero_10
  --policy.device=cuda
  --batch_size=32
  --steps=100000
  --eval_freq=20000
  --save_freq=20000
  --log_freq=200
"

run_cpu_test() {
  echo "=== CPU smoke test (100 steps, no eval) ==="
  uv run lerobot-train \
    $COMMON_CPU \
    --policy.system2_mode=disabled \
    --policy.device=cpu \
    --batch_size=2 \
    --steps=100 \
    --eval_freq=0 \
    --save_freq=100 \
    --log_freq=10 \
    --output_dir=outputs/dual_vla_cpu_test
}

run_dynamic() {
  echo "=== Full dual-system: system2_mode=dynamic ==="
  uv run lerobot-train \
    $COMMON_GPU \
    --policy.system2_mode=dynamic \
    --policy.system2_update_freq=10 \
    --output_dir=outputs/dual_vla_dynamic
}

run_frozen() {
  echo "=== Ablation: system2_mode=frozen_initial ==="
  uv run lerobot-train \
    $COMMON_GPU \
    --policy.system2_mode=frozen_initial \
    --output_dir=outputs/dual_vla_frozen
}

run_disabled() {
  echo "=== Ablation: system2_mode=disabled (pure ACT baseline) ==="
  uv run lerobot-train \
    $COMMON_GPU \
    --policy.system2_mode=disabled \
    --output_dir=outputs/dual_vla_disabled
}

case "$MODE" in
  cpu_test)  run_cpu_test ;;
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
    echo "Usage: $0 {cpu_test|dynamic|frozen|disabled|all}"
    exit 1
    ;;
esac
