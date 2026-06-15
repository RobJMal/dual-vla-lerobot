# Dual-System VLA: Step-by-Step Progress Guide

Follow this document to trace every implementation step. Each section maps to a
commit/block of work you can verify independently.

---

## Architecture

```
  Task image ──►  ┌─────────────────────────────────────────┐
  (every K steps) │  System 2  (slow / semantic)             │  Frozen CLIP ViT-B/16
  Task text  ──►  │  vision_feat = clip.get_image_features() │
  (once/episode)  │  text_feat  = clip.get_text_features()   │
                  │  ctx = (vision_feat + text_feat) / 2     │  → 512-d shared space
                  │  ctx_proj = Linear(512, dim_model)        │
                  └──────────────────┬──────────────────────┘
                                     │ task_context token
                  ┌──────────────────▼──────────────────────┐
  Images ──────►  │  System 1  (fast / reactive)             │  Trimmed ACT transformer
  Robot state ──► │  ResNet18 + Transformer                  │  chunk_size=20, dim=256
                  │  → 20-step action chunk                  │
                  └─────────────────────────────────────────┘
```

**Ablation dial** (`system2_mode` config field):

| Mode             | System 2 behaviour                                              |
|------------------|-----------------------------------------------------------------|
| `dynamic`        | Re-encodes image every 10 env steps; text cached once/episode  |
| `frozen_initial` | Encodes image+text once at episode reset, never refreshed      |
| `disabled`       | Injects zero vector — pure ACT baseline, no CLIP loaded        |

`frozen_initial` is the assignment's "static embedding of the initial instruction" baseline.
`disabled` is the "System 2 removed" counterfactual.

---

## Files created / modified

| File | Purpose |
|------|---------|
| `src/lerobot/policies/dual_vla/__init__.py` | Package marker |
| `src/lerobot/policies/dual_vla/configuration_dual_vla.py` | `DualVLAConfig` (`clip_embed_dim=512`) |
| `src/lerobot/policies/dual_vla/modeling_dual_vla.py` | `DualVLAPolicy` + `DualSystemVLA` |
| `src/lerobot/policies/dual_vla/processor_dual_vla.py` | Pre/post-processor factory |
| `src/lerobot/policies/dual_vla_diffusion/` | Diffusion Policy System 1 variant (separate ablation) |
| `src/lerobot/policies/factory.py` | Added `dual_vla` and `dual_vla_diffusion` to dispatch |
| `scripts/train_dual_vla.sh` | Training commands for all 3 ACT ablation variants |
| `scripts/train_dual_vla_diffusion.sh` | Training commands for all 3 Diffusion ablation variants |
| `docs/dual_vla_diffusion_plan.md` | Implementation plan for diffusion variant |

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

cfg = DualVLAConfig(
    system2_mode='disabled',   # no CLIP download on CPU test
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
print('Parameters:', sum(p.numel() for p in policy.parameters())/1e6, 'M')

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

### Step 2: Smoke training run (100 steps, libero_10 dataset)

```bash
bash scripts/train_dual_vla.sh cpu_test
```

**Pass criteria:** loss appears and decreases over 100 steps. No shape/device errors.

---

## Phase 2 — GPU cluster (ACT System 1)

### Step 3: Full training — 3 ablation variants

```bash
# Separate tmux sessions recommended so all three run in parallel.
tmux new -s train_dynamic  && bash scripts/train_dual_vla.sh dynamic
tmux new -s train_frozen   && bash scripts/train_dual_vla.sh frozen
tmux new -s train_disabled && bash scripts/train_dual_vla.sh disabled
```

Expected: ~7 hours per variant on RTX A6000 at ~4 steps/sec.
wandb runs appear as `dual_vla_dynamic`, `dual_vla_frozen`, `dual_vla_disabled`.

### Step 4: Evaluate each variant (50 episodes per task × 10 tasks)

```bash
for variant in dynamic frozen disabled; do
  MUJOCO_GL=egl uv run lerobot-eval \
    --policy.path=outputs/dual_vla_$variant \
    --policy.device=cuda \
    --env.type=libero \
    --env.task=libero_10 \
    --eval.n_episodes=50 \
    --eval.save_video=true \
    --output_dir=outputs/eval_$variant
done
```

**Important:** The eval loop must call `policy.set_task_description(env.task_description)`
after each `env.reset()` to activate the text encoder. Without this call, text encoding
silently falls back to vision-only.

Skeleton eval loop:
```python
obs, info = env.reset()
policy.reset()
policy.set_task_description(env.task_description)  # ← required for text encoding

while not done:
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```

### Step 5: Perturbation recovery test

Inject Gaussian noise (`std=0.1`) into `observation.state` at step 50 of each episode.
Measure fraction of episodes that still complete successfully. Run for all 3 variants.
This is the "Error Recovery" evidence the assignment requires.

### Step 6: Bottleneck analysis

The assignment asks: "Is failure from System 2 reasoning poorly, or System 1 failing to execute?"

**Method A — Oracle System 2:**
Run the `disabled` policy but replace the zero vector with the average dynamic context
from a successful `dynamic` rollout (frozen into a constant). If success rate improves,
System 2 quality is the bottleneck, not System 1 execution.

**Method B — Context trajectory plot:**
During a `dynamic` rollout, log `system2_ctx` (512-d vector) at each step.
PCA-project to 2D and plot over time. Compare trajectories between:
- Successful episodes (context drifts toward task-relevant region)
- Failed episodes (context stays in wrong region or oscillates)

If successful episodes show clear directional drift and failed ones don't, System 2 is guiding correctly but System 1 can't execute. If the context itself is wrong in failed episodes, System 2 is the bottleneck.

**Method C — Failure mode categorization (qualitative):**
Watch 10 failure videos from `dynamic` and label each failure as:
- "Wrong direction / wrong object" → System 2 gave bad context
- "Correct direction, didn't complete" → System 1 execution failure

### Step 7: System 2 output visualization

Required for submission ("output of System 2 if possible" in the video requirement).

Add a logging hook to `select_action` that saves `_cached_task_ctx` (the raw 512-d vector) at each step. Post-process:

```python
# During eval rollout, collect contexts
contexts = []  # list of (step, task_ctx.cpu().numpy())

# After rollout, visualize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
coords = pca.fit_transform(np.stack([c for _, c in contexts]))
plt.plot(coords[:, 0], coords[:, 1], alpha=0.7)
plt.scatter(coords[0, 0], coords[0, 1], marker='o', label='start')
plt.scatter(coords[-1, 0], coords[-1, 1], marker='*', label='end')
plt.title('System 2 context trajectory (PCA)')
```

Include one of these plots per task in the research report.

---

## Phase 3 — Diffusion Policy System 1 (separate GPU session)

Extends the ablation to a different System 1 architecture to prove the benefit generalizes.

```bash
tmux new -s diff_dynamic  && bash scripts/train_dual_vla_diffusion.sh dynamic
tmux new -s diff_frozen   && bash scripts/train_dual_vla_diffusion.sh frozen
tmux new -s diff_disabled && bash scripts/train_dual_vla_diffusion.sh disabled
```

Expected: ~10 hours per variant on RTX A6000.
wandb runs: `dual_vla_diffusion_dynamic`, `dual_vla_diffusion_frozen`, `dual_vla_diffusion_disabled`.

---

## Metrics and what they prove

| Metric | Assignment question answered |
|--------|------------------------------|
| Success rate: dynamic > disabled | "Counterfactual: removed" — System 2 improves System 1 |
| Success rate: dynamic > frozen | "Counterfactual: frozen/static" — dynamic re-planning matters |
| Steps to completion: dynamic < disabled | System 2 improves efficiency, not just binary success |
| Recovery rate: dynamic > frozen > disabled | "Error Recovery" — System 2 actively corrects errors |
| Bottleneck analysis (Method A/B/C above) | "Bottleneck Analysis" — identifies where to improve |
| Diffusion results match ACT pattern | Generalises across System 1 architectures (bonus) |

---

## Note on v1 training runs (current pod)

The three ACT variants currently training on the pod (`dual_vla_dynamic`, `dual_vla_frozen`,
`dual_vla_disabled`) use vision-only CLIP (`CLIPVisionModel`, 768-d). These are **v1 runs**.

The code was subsequently fixed to use both vision and text encoders (`CLIPModel`, 512-d).
The ablation comparison within v1 runs is still valid — all three variants used the same
vision-only CLIP consistently. But they do not satisfy the full assignment spec (text encoder
not used during training).

**Options:**
- Use v1 runs for the report with a note, and retrain v2 if time allows.
- After v1 finishes, delete `outputs/dual_vla_*` and retrain with the fixed code.
- v2 runs only require `set_task_description()` to be wired into the eval loop.

---

## Submission checklist

- [ ] GitHub repo: clean, reproducible code (this repo)
- [ ] `README.md`: setup, training, eval instructions — **CREATE THIS**
- [ ] Rollout videos: one per task variant (saved by `--eval.save_video=true`)
- [ ] System 2 output visualization: PCA trajectory plot (Step 7 above)
- [ ] Research report (max 3 pages PDF):
  - [ ] Architecture description and System 2→System 1 interface
  - [ ] Ablation table (success rate, steps, recovery rate)
  - [ ] Bottleneck analysis result
  - [ ] System 2 trajectory plots

### Research report outline (3 pages)

**Page 1 — Architecture**
- Motivation: why dual-system for long-horizon manipulation
- System 2: frozen CLIP ViT-B/16, (vision+text)/2 → 512-d, projected to dim_model
- System 1: trimmed ACT (dim=256, 3 encoder layers) conditioned on System 2 token
- System 2→System 1 interface: context token prepended to ACT encoder sequence
- Training: only System 1 + system2_proj trained; CLIP frozen; single multi-task model

**Page 2 — Experiments**
- Dataset: lerobot/libero_10 (10 diverse long-horizon tasks, up to 520 steps)
- Ablation table (3 variants × per-task success rate + aggregate)
- Perturbation recovery results
- Key result: dynamic > frozen > disabled on success rate and recovery

**Page 3 — Analysis**
- Bottleneck analysis (oracle test or context trajectory)
- System 2 visualization (PCA trajectory plots)
- Limitations and future work (retrain with full v2 spec if time allows)
