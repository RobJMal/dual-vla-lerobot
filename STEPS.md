# Dual-System VLA: Step-by-Step Progress Guide

Follow this document to trace every implementation step. Each section maps to a
commit/block of work you can verify independently.

---

## Architecture at a glance

```
                  ┌─────────────────────────────┐
  Task image ──►  │  System 2  (slow / semantic) │  Frozen CLIP ViT-B/16
  (every K steps) │  512-d → Linear → dim_model  │  (or zeros if disabled)
                  └──────────────┬──────────────┘
                                 │ task_context token
                  ┌──────────────▼──────────────┐
  Images ──────►  │  System 1  (fast / reactive) │  Trimmed ACT transformer
  Robot state ──► │  ResNet18 + Transformer      │  chunk_size=20, dim=256
                  │  → 20-step action chunk      │
                  └─────────────────────────────┘
```

**Ablation dial** (`system2_mode` config field):

| Mode            | System 2 behaviour                    |
|-----------------|---------------------------------------|
| `dynamic`       | Re-encodes image every 10 env steps   |
| `frozen_initial`| Encodes once at episode reset         |
| `disabled`      | Injects zero vector (CPU smoke test)  |

---

## Files created / modified

| File | Purpose |
|------|---------|
| `src/lerobot/policies/dual_vla/__init__.py` | Package marker |
| `src/lerobot/policies/dual_vla/configuration_dual_vla.py` | `DualVLAConfig` |
| `src/lerobot/policies/dual_vla/modeling_dual_vla.py` | `DualVLAPolicy` + `DualSystemVLA` |
| `src/lerobot/policies/dual_vla/processor_dual_vla.py` | Pre/post-processor factory |
| `src/lerobot/policies/factory.py` | Added `dual_vla` to all three dispatch functions |
| `scripts/train_dual_vla.sh` | Training commands for all 3 ablation variants |

---

## Phase 1 — CPU smoke test (do this locally)

### Step 1: Forward pass (no data download)

```bash
uv run python -c "
import torch
from lerobot.configs import FeatureType
from lerobot.configs.types import PolicyFeature
from lerobot.policies.dual_vla.configuration_dual_vla import DualVLAConfig
from lerobot.policies.dual_vla.modeling_dual_vla import DualVLAPolicy

# Minimal features: one image + state + action
cfg = DualVLAConfig(
    system2_mode='disabled',   # no CLIP download
    device='cpu',
    push_to_hub=False,
    chunk_size=20,
    n_action_steps=10,
)
cfg.input_features = {
    'observation.images.image': PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
    'observation.state':        PolicyFeature(type=FeatureType.STATE, shape=(9,)),
}
cfg.output_features = {
    'action': PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
}

policy = DualVLAPolicy(cfg)
print('Policy created. Parameters:', sum(p.numel() for p in policy.parameters())/1e6, 'M')

B = 2
batch = {
    'observation.images.image': torch.zeros(B, 3, 128, 128),
    'observation.state':        torch.zeros(B, 9),
    'action':                   torch.zeros(B, 20, 7),
    'action_is_pad':            torch.zeros(B, 20, dtype=torch.bool),
}
loss, info = policy.forward(batch)
print('Loss:', loss.item(), '| Info:', info)
print('Forward pass OK')
"
```

Expected output: loss value printed, no errors.

### Step 2: Smoke training run (100 steps, libero_10 dataset)

This downloads the `lerobot/libero_10` dataset (~1 GB) on first run.

```bash
uv run lerobot-train \
  --policy.type=dual_vla \
  --policy.system2_mode=disabled \
  --policy.chunk_size=20 \
  --policy.n_action_steps=10 \
  --policy.dim_model=256 \
  --policy.device=cpu \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=2 \
  --steps=100 \
  --eval_freq=0 \
  --save_freq=100 \
  --log_freq=10 \
  --output_dir=outputs/dual_vla_cpu_test
```

**Pass criteria:** loss appears and decreases over 100 steps. No CUDA/shape/device errors.

---

## Phase 2 — GPU cluster (run after Phase 1 passes)

### Step 3: Full training — 3 ablation variants

```bash
# Full system (System 2 active, re-encodes every 10 steps)
uv run lerobot-train \
  --policy.type=dual_vla \
  --policy.system2_mode=dynamic \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=32 --steps=100000 --eval_freq=20000 \
  --output_dir=outputs/dual_vla_dynamic

# Frozen baseline (System 2 sees only the first frame)
uv run lerobot-train \
  --policy.type=dual_vla \
  --policy.system2_mode=frozen_initial \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=32 --steps=100000 --eval_freq=20000 \
  --output_dir=outputs/dual_vla_frozen

# Disabled (System 2 injects zeros — pure ACT baseline)
uv run lerobot-train \
  --policy.type=dual_vla \
  --policy.system2_mode=disabled \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=lerobot/libero_10 \
  --batch_size=32 --steps=100000 --eval_freq=20000 \
  --output_dir=outputs/dual_vla_disabled
```

### Step 4: Evaluate each variant

```bash
for variant in dynamic frozen disabled; do
  uv run lerobot-eval \
    --policy.path=outputs/dual_vla_$variant \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.n_episodes=50
done
```

### Step 5: Perturbation recovery test (ablation metric)

Modify the eval loop (`scripts/run_baseline_eval.py`) to inject Gaussian noise
(`std=0.1`) into `observation.state` at step 50 of each episode, then measure
the fraction of episodes that still complete successfully. Run for all 3 variants.

---

## What each metric proves

| Metric | What it answers |
|--------|----------------|
| Success rate: dynamic > disabled | System 2 helps System 1 succeed |
| Success rate: dynamic > frozen | Dynamic re-planning matters, not just initial context |
| Recovery rate: dynamic > frozen > disabled | System 2 actively corrects course after perturbations |
| Steps to completion | System 2 improves efficiency (not just success) |
