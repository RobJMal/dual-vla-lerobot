# Dual-System VLA on LIBERO

This repo is forked from [LeRobot](https://github.com/huggingface/lerobot) and extends it with a custom `dual_vla` policy — a frozen CLIP (System 2) conditioning a trimmed ACT transformer (System 1) — evaluated on the LIBERO-Spatial benchmark.

---

## Setup

These instructions assume you're running on an Ubuntu 24.04 system with a GPU. Run everything from `/workspace/dual-vla-lerobot/`.

**1. Clone and install**

```bash
git clone https://github.com/RobJMal/lerobot.git /workspace/dual-vla-lerobot
cd /workspace/dual-vla-lerobot
uv sync --locked --extra libero
```

> `--extra libero` is required. Without it, `import libero` will fail at eval time even if you manually `pip install` it — `uv run` re-syncs from the lockfile and wipes manually installed packages.

**2. Install system graphics libraries (required for headless GPU rendering)**

```bash
apt-get update && apt-get install -y libgl1-mesa-glx libgles2 libegl-dev
```

**3. Authenticate**

```bash
huggingface-cli login
wandb login
```

---

## Training

Three ablation variants, each in its own tmux session. Replace `RobJMal` with your W&B entity if different.

**disabled** — System 2 replaced with zeros (ACT baseline):
```bash
tmux new-session -d -s train_disabled \
  'MUJOCO_GL=egl uv run lerobot-train \
    --policy.type=dual_vla \
    --policy.system2_mode=disabled \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --batch_size=32 \
    --steps=100000 \
    --policy.device=cuda \
    --output_dir=outputs/spatial_disabled \
    --job_name=spatial_disabled \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=dual-vla-ablation \
    --wandb.entity=RobJMal \
    2>&1 | tee outputs/spatial_disabled.log; bash'
```

**dynamic** — CLIP re-encodes scene + task every 10 steps (full system):
```bash
tmux new-session -d -s train_dynamic \
  'MUJOCO_GL=egl uv run lerobot-train \
    --policy.type=dual_vla \
    --policy.system2_mode=dynamic \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --batch_size=32 \
    --steps=100000 \
    --policy.device=cuda \
    --output_dir=outputs/spatial_dynamic \
    --job_name=spatial_dynamic \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=dual-vla-ablation \
    --wandb.entity=RobJMal \
    2>&1 | tee outputs/spatial_dynamic.log; bash'
```

**frozen_initial** — CLIP encodes once at episode reset, held fixed:
```bash
tmux new-session -d -s train_frozen \
  'MUJOCO_GL=egl uv run lerobot-train \
    --policy.type=dual_vla \
    --policy.system2_mode=frozen_initial \
    --dataset.repo_id=lerobot/libero_spatial_image \
    --batch_size=32 \
    --steps=100000 \
    --policy.device=cuda \
    --output_dir=outputs/spatial_frozen \
    --job_name=spatial_frozen \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=dual-vla-ablation \
    --wandb.entity=RobJMal \
    2>&1 | tee outputs/spatial_frozen.log; bash'
```

Monitor a session: `tmux attach -t train_disabled`. A healthy run shows `l1_loss` dropping from ~0.6 to ~0.13 over 100k steps.

> `--policy.push_to_hub=false` is required when `--policy.repo_id` is not set — training errors on startup otherwise.

---

## Evaluation

Run after training completes. Each eval takes ~45 min on a modern GPU (200 episodes, 10 parallel envs).

**Important — read before running:**
- Always pass `--policy.type=dual_vla`. Without it draccus cannot parse the config and errors immediately.
- Always pass `--rename_map`. The eval environment names the wrist camera `image2` but the training dataset calls it `wrist_image`. Without this the eval crashes with a feature mismatch error.
- Use `--env.task=libero_spatial`, not `--env.task_suite` (that flag does not exist).
- `MUJOCO_GL=egl` is required for headless GPU rendering on Linux.

**disabled:**
```bash
tmux new-session -d -s eval_disabled \
  'MUJOCO_GL=egl uv run lerobot-eval \
    --policy.type=dual_vla \
    --policy.pretrained_path=outputs/spatial_disabled/checkpoints/100000/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.n_episodes=20 \
    --eval.batch_size=10 \
    --rename_map='"'"'{"observation.images.image2": "observation.images.wrist_image"}'"'"' \
    2>&1 | tee outputs/eval_spatial_disabled.log; bash'
```

**dynamic:**
```bash
tmux new-session -d -s eval_dynamic \
  'MUJOCO_GL=egl uv run lerobot-eval \
    --policy.type=dual_vla \
    --policy.pretrained_path=outputs/spatial_dynamic/checkpoints/100000/pretrained_model \
    --env.type=libero \
    --env.task=libero_spatial \
    --eval.n_episodes=20 \
    --eval.batch_size=10 \
    --rename_map='"'"'{"observation.images.image2": "observation.images.wrist_image"}'"'"' \
    2>&1 | tee outputs/eval_spatial_dynamic.log; bash'
```

Results are written to `outputs/eval/<date>/<timestamp>_libero_dual_vla/eval_info.json`. The `pc_success` field under `per_group.libero_spatial` gives the overall success rate.

---

## Downloading pretrained checkpoints

To skip training and run eval directly from uploaded checkpoints:

```bash
huggingface-cli download RobJMal/dual-vla-spatial_disabled \
  --local-dir outputs/spatial_disabled

huggingface-cli download RobJMal/dual-vla-spatial_dynamic \
  --local-dir outputs/spatial_dynamic
```

Then run the eval commands above pointing to the downloaded path.

---

## Results

| Variant | Overall | Task 2 | Task 6 | Task 3 |
|---------|--------:|-------:|-------:|-------:|
| Disabled | 0% | 0% | 0% | 0% |
| Dynamic | 9% | **60%** | 20% | 10% |

---

## Report

The full research report (architecture, interface design, ablation analysis) is at [media/deliverable/dual_system_report.pdf](media/deliverable/dual_system_report.pdf).

---

## Rollout Videos

Sample rollouts are in [media/deliverable/](media/deliverable/). Same episode numbers are used across variants for direct comparison.

### Dynamic (System 2 active)

**Task 2** — highest success rate (60%)
- [Episode 000](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-001_episode-000_libero-spatial-task-2.mp4)
- [Episode 002](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-002_episode-002_libero-spatial-task-2.mp4)
- [Episode 004](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-003_episode-004_libero-spatial-task-2.mp4)

**Task 3** — 10% success rate
- [Episode 000](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-004_episode-000_libero-spatial-task-3.mp4)
- [Episode 001](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-005_episode-001_libero-spatial-task-3.mp4) ✓ success
- [Episode 002](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-006_episode-002_libero-spatial-task-3.mp4)

**Task 6** — 20% success rate
- [Episode 004](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-007_episode-004_libero-spatial-task-6.mp4) ✓ success
- [Episode 005](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-008_episode-005_libero-spatial-task-6.mp4) ✓ success
- [Episode 000](media/deliverable/dual_vla_dynamic/dual-vla-dynamic-009_episode-000_libero-spatial-task-6.mp4)

### Disabled (System 2 = zeros, baseline)

**Task 2**
- [Episode 000](media/deliverable/dual_vla_disabled/dual-vla-disabled-001_episode-000_libero-spatial-task-2.mp4)
- [Episode 002](media/deliverable/dual_vla_disabled/dual-vla-disabled-002_episode-002_libero-spatial-task-2.mp4)
- [Episode 004](media/deliverable/dual_vla_disabled/dual-vla-disabled-003_episode-004_libero-spatial-task-2.mp4)

**Task 3**
- [Episode 000](media/deliverable/dual_vla_disabled/dual-vla-disabled-004_episode-000_libero-spatial-task-3.mp4)
- [Episode 001](media/deliverable/dual_vla_disabled/dual-vla-disabled-005_episode-001_libero-spatial-task-3.mp4)
- [Episode 002](media/deliverable/dual_vla_disabled/dual-vla-disabled-006_episode-002_libero-spatial-task-3.mp4)

**Task 6**
- [Episode 004](media/deliverable/dual_vla_disabled/dual-vla-disabled-007_episode-004_libero-spatial-task-6.mp4)
- [Episode 005](media/deliverable/dual_vla_disabled/dual-vla-disabled-008_episode-005_libero-spatial-task-6.mp4)
- [Episode 000](media/deliverable/dual_vla_disabled/dual-vla-disabled-009_episode-000_libero-spatial-task-6.mp4)

---

## Dataset

- **Training:** `lerobot/libero_spatial_image` (use this exact name — `lerobot/libero_spatial` does not exist on the Hub)
- **Eval environment:** LIBERO-Spatial (10 tasks, 20 episodes each)

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'libero'` | Run `uv sync --locked --extra libero`, not `pip install libero` |
| `libEGL.so.0 not found` | `apt-get install -y libgl1-mesa-glx libgles2 libegl-dev` |
| `mat1 and mat2 shapes cannot be multiplied` | Pod is running old code — run `git pull` then retry |
| `unrecognized arguments: --env.task_suite` | Use `--env.task=libero_spatial` instead |
| `Expected a dict with a 'type' key` | Add `--policy.type=dual_vla` to the eval command |
| `Missing features: ['observation.images.wrist_image']` | Add `--rename_map='{"observation.images.image2": "observation.images.wrist_image"}'` |
| `model.safetensors not found` | Check the full checkpoint path — HF downloads may nest subdirectories |
