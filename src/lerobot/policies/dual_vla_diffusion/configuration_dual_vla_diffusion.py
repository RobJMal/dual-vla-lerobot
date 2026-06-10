from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig


@PreTrainedConfig.register_subclass("dual_vla_diffusion")
@dataclass
class DualVLADiffusionConfig(DiffusionConfig):
    """Dual-System VLA with Diffusion Policy as System 1.

    System 2 (slow/semantic): frozen CLIP ViT-B/16 pooler output (768-d), projected to
    ``system2_proj_dim`` and appended to the diffusion U-Net's ``global_cond`` vector.

    System 2 modes:
      - ``"dynamic"``        — CLIP re-encodes every ``system2_update_freq`` env steps.
      - ``"frozen_initial"`` — CLIP encodes once at episode reset; context stays fixed.
      - ``"disabled"``       — zeros injected; plain Diffusion Policy baseline.
    """

    # ── System 2 ──────────────────────────────────────────────────────────────
    system2_mode: str = "dynamic"
    system2_update_freq: int = 10
    clip_model_name: str = "openai/clip-vit-base-patch16"
    clip_embed_dim: int = 512  # CLIPModel projected dim (get_image/text_features), shared vision-text space
    system2_proj_dim: int = 256

    # ── Overrides tuned for LIBERO long-horizon tasks ─────────────────────────
    n_obs_steps: int = 2
    horizon: int = 64
    n_action_steps: int = 32
    # DDIM with 10 steps makes eval feasible; quality loss vs DDPM/100 is minimal
    noise_scheduler_type: str = "DDIM"
    num_inference_steps: int = 10
    optimizer_lr: float = 1e-4

    def __post_init__(self):
        # Keep drop_n_last_frames consistent with the overridden horizon / n_action_steps.
        self.drop_n_last_frames = self.horizon - self.n_action_steps - self.n_obs_steps + 1
        super().__post_init__()
        if self.system2_mode not in ("dynamic", "frozen_initial", "disabled"):
            raise ValueError(
                f"system2_mode must be 'dynamic', 'frozen_initial', or 'disabled'. "
                f"Got '{self.system2_mode}'."
            )
