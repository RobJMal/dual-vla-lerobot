from dataclasses import dataclass, field

from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamWConfig


@PreTrainedConfig.register_subclass("dual_vla")
@dataclass
class DualVLAConfig(PreTrainedConfig):
    """Dual-System VLA: frozen CLIP (System 2) conditioning a trimmed ACT transformer (System 1).

    System 2 modes (set via ``system2_mode``):
      - ``"dynamic"``        — CLIP re-encodes the current image every ``system2_update_freq`` env steps.
      - ``"frozen_initial"`` — CLIP encodes once at episode reset; context stays fixed.
      - ``"disabled"``       — zeros injected; effectively a plain ACT baseline. Use for CPU smoke tests.

    System 1 is a trimmed ACT (dim_model=256, 3 encoder layers) conditioned on the System 2 context
    token prepended to the transformer encoder sequence.
    """

    # ── System 2 ──────────────────────────────────────────────────────────────
    system2_mode: str = "dynamic"
    system2_update_freq: int = 10
    clip_model_name: str = "openai/clip-vit-base-patch16"
    clip_embed_dim: int = 512  # fixed for ViT-B/16 pooler output

    # ── Observation / action structure ────────────────────────────────────────
    n_obs_steps: int = 1
    chunk_size: int = 20
    n_action_steps: int = 10

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # ── System 1 transformer ──────────────────────────────────────────────────
    dim_model: int = 256
    n_heads: int = 4
    dim_feedforward: int = 1024
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 3
    n_decoder_layers: int = 1
    pre_norm: bool = False
    dropout: float = 0.1

    # ── VAE ───────────────────────────────────────────────────────────────────
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 3

    # ── Vision backbone (System 1) ────────────────────────────────────────────
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # ── Loss ──────────────────────────────────────────────────────────────────
    kl_weight: float = 10.0

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer_lr: float = 1e-5
    optimizer_lr_backbone: float = 1e-5
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.system2_mode not in ("dynamic", "frozen_initial", "disabled"):
            raise ValueError(
                f"system2_mode must be 'dynamic', 'frozen_initial', or 'disabled'. Got '{self.system2_mode}'."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Only n_obs_steps=1 is supported. Got {self.n_obs_steps}.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("DualVLAConfig requires at least one image or the environment state as input.")

    @property
    def observation_delta_indices(self):
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
