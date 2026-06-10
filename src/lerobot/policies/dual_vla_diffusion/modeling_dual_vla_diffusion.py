"""Dual-System VLA with Diffusion Policy as System 1.

System 2 (slow/semantic): frozen CLIP ViT-B/16 appends a projected context vector to
the diffusion U-Net's global_cond, giving the denoiser a semantic grounding signal.

System 1 (fast/reactive): standard Diffusion Policy U-Net conditioned on
[robot_state, image_features, system2_ctx] via FiLM layers.
"""

from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import require_package

from ..diffusion.modeling_diffusion import (
    DiffusionConditionalUnet1d,
    DiffusionModel,
    DiffusionPolicy,
    _make_noise_scheduler,
)
from ..pretrained import PreTrainedPolicy
from ..utils import populate_queues
from .configuration_dual_vla_diffusion import DualVLADiffusionConfig


class DualVLADiffusionModel(DiffusionModel):
    """Diffusion U-Net conditioned on a frozen CLIP context vector (System 2)."""

    def __init__(self, config: DualVLADiffusionConfig):
        # Call DiffusionModel.__init__ to set up rgb_encoder, noise_scheduler, etc.
        # It creates a U-Net with the base global_cond_dim; we replace it below.
        super().__init__(config)

        # ── System 2: frozen CLIP vision encoder ──────────────────────────────
        if config.system2_mode != "disabled":
            from transformers import CLIPVisionModel

            self.clip = CLIPVisionModel.from_pretrained(config.clip_model_name)
            for p in self.clip.parameters():
                p.requires_grad_(False)
            self.register_buffer(
                "clip_pixel_mean",
                torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "clip_pixel_std",
                torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            )

        self.system2_proj = nn.Linear(config.clip_embed_dim, config.system2_proj_dim)
        self._system2_ctx: Tensor | None = None  # set by policy at inference time

        # ── Replace U-Net with extended global_cond_dim ───────────────────────
        global_cond_dim = config.robot_state_feature.shape[0]
        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                global_cond_dim += self.rgb_encoder[0].feature_dim * num_images
            else:
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]
        global_cond_dim += config.system2_proj_dim

        self.unet = DiffusionConditionalUnet1d(
            config, global_cond_dim=global_cond_dim * config.n_obs_steps
        )
        if config.compile_model:
            self.unet = torch.compile(self.unet, mode=config.compile_mode)

    @torch.no_grad()
    def encode_system2(self, img: Tensor) -> Tensor:
        """Return (B, clip_embed_dim) context from the primary camera image.

        For ``disabled`` mode returns zeros so the U-Net conditioning is inert.
        """
        if self.config.system2_mode == "disabled":
            return torch.zeros(img.shape[0], self.config.clip_embed_dim, device=img.device)
        img_resized = F.interpolate(img.float(), size=(224, 224), mode="bilinear", align_corners=False)
        img_resized = img_resized.clamp(0.0, 1.0)
        img_clip = (img_resized - self.clip_pixel_mean) / self.clip_pixel_std
        return self.clip(pixel_values=img_clip).pooler_output  # (B, clip_embed_dim)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode observations + System 2 context into the U-Net conditioning vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # System 2: use cached context at inference, or encode fresh during training.
        if self._system2_ctx is not None:
            system2_raw = self._system2_ctx  # (B, clip_embed_dim)
        else:
            # batch[OBS_IMAGES]: (B, n_obs_steps, n_cams, C, H, W) — take first step, first cam
            first_img = batch[OBS_IMAGES][:, 0, 0]
            system2_raw = self.encode_system2(first_img)  # (B, clip_embed_dim)

        system2_emb = self.system2_proj(system2_raw)  # (B, system2_proj_dim)
        # Repeat across obs steps so the feature dim is consistent with the others.
        system2_expanded = system2_emb.unsqueeze(1).expand(-1, n_obs_steps, -1)
        global_cond_feats.append(system2_expanded)

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


class DualVLADiffusionPolicy(DiffusionPolicy):
    """Diffusion Policy conditioned on a frozen CLIP System 2 context."""

    config_class = DualVLADiffusionConfig
    name = "dual_vla_diffusion"

    def __init__(self, config: DualVLADiffusionConfig, **kwargs):
        require_package("diffusers", extra="diffusion")
        # Bypass DiffusionPolicy.__init__ — we use DualVLADiffusionModel instead of DiffusionModel.
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.diffusion = DualVLADiffusionModel(config)
        self._step_count = 0
        self._cached_system2_ctx: Tensor | None = None
        self.reset()

    def reset(self):
        """Clear observation/action queues and System 2 cache. Call on env.reset()."""
        super().reset()  # DiffusionPolicy.reset() rebuilds self._queues
        self._step_count = 0
        self._cached_system2_ctx = None

    def _should_refresh_system2(self) -> bool:
        if self.config.system2_mode == "disabled":
            return False
        if self.config.system2_mode == "frozen_initial":
            return self._cached_system2_ctx is None
        return self._step_count % self.config.system2_update_freq == 0

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            if self._should_refresh_system2() and OBS_IMAGES in batch:
                img = batch[OBS_IMAGES]
                if img.ndim == 5:  # (B, n_cams, C, H, W) — take first camera
                    img = img[:, 0]
                self._cached_system2_ctx = self.diffusion.encode_system2(img)

            # Thread the cached context into the model for this prediction.
            self.diffusion._system2_ctx = self._cached_system2_ctx
            actions = self.predict_action_chunk(batch, noise=noise)
            self.diffusion._system2_ctx = None  # clear so training forward is unaffected

            self._queues[ACTION].extend(actions.transpose(0, 1))

        self._step_count += 1
        return self._queues[ACTION].popleft()
