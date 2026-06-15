# Dual-System VLA: Diffusion Policy Variant

Extends the ACT-based ablation study to use **Diffusion Policy as System 1**, with the same frozen CLIP System 2 conditioning. The goal is to show that the System 2 benefit generalizes across System 1 architectures — a stronger claim than a single-architecture ablation.

---

## Why Diffusion Policy

ACT predicts actions by regression (L1 loss). Diffusion Policy predicts actions by iterative denoising (DDPM/DDIM). They have fundamentally different inductive biases:

- ACT: deterministic, low-latency inference, good at smooth trajectories
- Diffusion: multi-modal action distributions, better at recovering from diverse states

If System 2 improves both, it suggests the benefit is semantic (task understanding) rather than architectural. That is the argument to make to interviewers.

---

## Architecture

### System 2 (unchanged from ACT variant)
Frozen CLIP ViT-B/16. Uses both encoders in the shared 512-d projection space:
```
vision_feat = clip.get_image_features(img)   # (B, 512)
text_feat   = clip.get_text_features(tokens) # (1, 512), cached once per episode
ctx         = (vision_feat + text_feat) / 2  # (B, 512)
```

Same three ablation modes:
- `dynamic` — re-encodes every 10 env steps
- `frozen_initial` — encodes once at episode reset
- `disabled` — zero vector (pure Diffusion Policy baseline)

### System 1: Diffusion Policy U-Net
The existing `DiffusionPolicy` conditions its U-Net via a `global_cond` vector:

```
global_cond = [robot_state, image_features(ResNet18), env_state]
              → flattened → FiLM-conditioned into U-Net
```

System 2 is injected by appending a projected CLIP embedding to this vector:

```
system2_proj = Linear(512, 256)  # frozen input, trained projection
global_cond  = [robot_state, image_features, system2_proj(clip_ctx)]
# clip_ctx = (vision_feat + text_feat) / 2, both 512-d from CLIPModel
```

No changes to the U-Net itself. The `global_cond_dim` parameter at U-Net init simply increases by 256.

### Key config differences from ACT variant

| Parameter | ACT variant | Diffusion variant |
|---|---|---|
| `horizon` | 20 (chunk_size) | 64 |
| `n_action_steps` | 10 | 32 |
| `n_obs_steps` | 1 | 2 |
| `down_dims` | — | (512, 1024, 2048) |
| `num_train_timesteps` | — | 100 |
| `noise_scheduler_type` | — | DDIM (faster eval) |
| `num_inference_steps` | — | 10 (DDIM, fast) |
| `optimizer_lr` | 1e-5 | 1e-4 |

Use DDIM with `num_inference_steps=10` instead of DDPM/100 — makes eval feasible at 32 denoising steps per action chunk. Quality loss is minimal.

---

## Implementation Plan

### New files

```
src/lerobot/policies/dual_vla_diffusion/
├── __init__.py
├── configuration_dual_vla_diffusion.py
└── modeling_dual_vla_diffusion.py
```

No new processor file needed — reuse `make_diffusion_pre_post_processors()` from `processor_diffusion.py` directly. Add a `elif isinstance(cfg, DualVLADiffusionConfig)` branch in `factory.py`.

### configuration_dual_vla_diffusion.py

Inherit from `DiffusionConfig`, add:

```python
@PreTrainedConfig.register_subclass("dual_vla_diffusion")
@dataclass
class DualVLADiffusionConfig(DiffusionConfig):
    system2_mode: str = "dynamic"
    system2_update_freq: int = 10
    clip_model_name: str = "openai/clip-vit-base-patch16"
    clip_embed_dim: int = 512  # CLIPModel projected dim, shared vision-text space
    system2_proj_dim: int = 256

    # Overrides for LIBERO
    horizon: int = 64
    n_action_steps: int = 32
    n_obs_steps: int = 2
    noise_scheduler_type: str = "DDIM"
    num_inference_steps: int = 10
    optimizer_lr: float = 1e-4
```

### modeling_dual_vla_diffusion.py

Subclass `DiffusionPolicy`, override three methods:

1. `__init__`: add `self.clip` (`CLIPModel`, frozen), `self.clip_tokenizer`, `self.system2_proj = Linear(512, system2_proj_dim)`, increase `global_cond_dim` by `system2_proj_dim` when constructing the U-Net.

2. `_prepare_global_conditioning(batch)`: call `encode_system2()`, project, append to `global_cond_feats` before the `torch.cat`.

3. `reset()`: clear `self._cached_system2` and `self._step_counter` (same pattern as ACT variant).

The `encode_system2()` logic is identical to the ACT variant — copy it directly.

---

## Training Script

Add to `scripts/train_dual_vla.sh` (or create `scripts/train_dual_vla_diffusion.sh`):

```bash
COMMON_DIFF="
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

# Full system
uv run lerobot-train $COMMON_DIFF \
  --policy.system2_mode=dynamic \
  --job_name=dual_vla_diff_dynamic \
  --output_dir=outputs/dual_vla_diff_dynamic

# Frozen ablation
uv run lerobot-train $COMMON_DIFF \
  --policy.system2_mode=frozen_initial \
  --job_name=dual_vla_diff_frozen \
  --output_dir=outputs/dual_vla_diff_frozen

# Disabled baseline (pure Diffusion Policy)
uv run lerobot-train $COMMON_DIFF \
  --policy.system2_mode=disabled \
  --job_name=dual_vla_diff_disabled \
  --output_dir=outputs/dual_vla_diff_disabled
```

Use `batch_size=128` (VRAM is underutilized at batch_size=32 — diffusion U-Net is larger and the A6000 can handle it).

---

## Resources

### GPU
- **RTX A6000 (48 GB)** — same pod as ACT runs, fine for this
- Minimum: RTX 3090 (24 GB) at batch_size=64
- The U-Net (down_dims=(512,1024,2048)) is ~50M params, larger than the ACT variant

### VRAM estimate
| Component | VRAM |
|---|---|
| U-Net (512/1024/2048) | ~1.5 GB |
| ResNet18 × 2 cameras × 2 obs steps | ~0.8 GB |
| Batch of 128 × horizon 64 | ~1.2 GB |
| CLIP (frozen, inference_mode) | ~0.3 GB |
| Optimizer states | ~2.5 GB |
| **Total** | **~6–7 GB** |

Well within A6000 limits. Could push `batch_size=256`.

### Training time estimate
- ~5–6 steps/sec at batch_size=128 (diffusion training is one U-Net forward per step, fast)
- 200K steps ÷ 5.5 steps/sec ≈ **10 hours**
- Run all 3 variants in parallel across 3 tmux sessions: wall-clock ~10 hours
- Cost on RunPod A6000 (~$0.80/hr): ~$8 for all 3 variants

### Eval time (slower than ACT)
- DDIM with 10 steps: ~0.15s per action chunk at inference
- 50 episodes × ~500 steps/episode ÷ 32 actions/chunk × 0.15s ≈ ~12 min per variant
- Total eval: ~40 min for all 3

---

## Expected Results and Story

| Variant | Hypothesis |
|---|---|
| `dynamic` (Diffusion + live CLIP) | Best success rate — multi-modal + semantic |
| `frozen_initial` (Diffusion + stale CLIP) | Middle — benefits from initial task context |
| `disabled` (pure Diffusion Policy) | Baseline — no semantic grounding |

Combined with the ACT ablation, the full story is:

> "System 2 (CLIP) improves task success across both regression-based (ACT) and generative (Diffusion) System 1 architectures on LIBERO-10, confirming the benefit is semantic rather than architecture-specific."

That is the argument that impresses interviewers.

---

## Timing Recommendation

Run this **after** the ACT ablation eval is complete and analyzed. Start when you have a free A6000 session and ~12 hours of pod time budgeted (10h training + 1h eval + buffer).
