"""Dual-System VLA policy.

System 2 (slow/semantic): frozen CLIP ViT-B/16 → 512-d pooled embedding.
System 1 (fast/reactive): trimmed ACT transformer conditioned on the System 2 token.

The System 2 context token is prepended to the ACT encoder's token sequence, giving
the transformer a global semantic signal alongside the local image feature patches.
"""

from collections import deque
from itertools import chain

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.modeling_act import (
    ACTDecoder,
    ACTEncoder,
    ACTSinusoidalPositionEmbedding2d,
    create_sinusoidal_pos_embedding,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

from ..pretrained import PreTrainedPolicy
from .configuration_dual_vla import DualVLAConfig


class DualVLAPolicy(PreTrainedPolicy):
    """Dual-System VLA: CLIP-conditioned trimmed ACT for long-horizon LIBERO tasks."""

    config_class = DualVLAConfig
    name = "dual_vla"

    def __init__(self, config: DualVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model = DualSystemVLA(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """Call at every episode reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)
        self._step_count = 0
        self._cached_task_ctx: Tensor | None = None

    def _should_refresh_system2(self) -> bool:
        if self.config.system2_mode == "disabled":
            return False
        if self.config.system2_mode == "frozen_initial":
            return self._cached_task_ctx is None
        # "dynamic": refresh every K steps
        return self._step_count % self.config.system2_update_freq == 0

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()

        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        if len(self._action_queue) == 0:
            if self._should_refresh_system2():
                self._cached_task_ctx = self.model.encode_system2(
                    batch[OBS_IMAGES][0], text=batch.get("task")
                )

            actions = self.model(batch, precomputed_ctx=self._cached_task_ctx)[0]
            actions = actions[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        self._step_count += 1
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        self.eval()
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        return self.model(batch)[0]

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training forward — computes L1 + KL loss."""
        batch = dict(batch)
        batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        abs_err = F.l1_loss(batch[ACTION], actions_hat, reduction="none")
        valid_mask = ~batch["action_is_pad"].unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        l1_loss = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae and mu_hat is not None:
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp()))
                .sum(-1)
                .mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class DualSystemVLA(nn.Module):
    """Core neural network for the Dual-System VLA policy.

    Token sequence fed to the ACT encoder:
        [latent, system2_ctx, (robot_state), (env_state), <image patches> ...]

    System 2 context = CLIP ViT-B/16 pooler output (512-d), projected to dim_model.
    When system2_mode='disabled' the context is an all-zero token (no CLIP weights loaded).
    """

    def __init__(self, config: DualVLAConfig):
        super().__init__()
        self.config = config

        # ── System 2: frozen CLIP encoder (vision + text) ─────────────────────
        if config.system2_mode != "disabled":
            from transformers import CLIPModel, CLIPTokenizerFast

            self.clip = CLIPModel.from_pretrained(config.clip_model_name)
            for p in self.clip.parameters():
                p.requires_grad_(False)
            self.clip_tokenizer = CLIPTokenizerFast.from_pretrained(config.clip_model_name)
            # CLIP canonical normalisation (applied to [0,1]-range images)
            self.register_buffer(
                "clip_pixel_mean",
                torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
            )
            self.register_buffer(
                "clip_pixel_std",
                torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
            )

        self.system2_proj = nn.Linear(config.clip_embed_dim, config.dim_model)

        # ── System 1: ResNet18 backbone ───────────────────────────────────────
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=config.pretrained_backbone_weights,
            norm_layer=FrozenBatchNorm2d,
        )
        self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
        backbone_out_channels = backbone_model.fc.in_features  # 512 for ResNet18

        # ── System 1: VAE encoder ─────────────────────────────────────────────
        if config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                config.action_feature.shape[0], config.dim_model
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            n_vae_tokens = 1 + config.chunk_size + (1 if config.robot_state_feature else 0)
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(n_vae_tokens, config.dim_model).unsqueeze(0),
            )

        # ── System 1: main transformer ────────────────────────────────────────
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Projection layers for 1-D encoder tokens
        if config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                config.robot_state_feature.shape[0], config.dim_model
            )
        if config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        self.encoder_img_feat_input_proj = nn.Conv2d(backbone_out_channels, config.dim_model, kernel_size=1)
        self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Positional embeddings for 1-D tokens: latent + system2 + optional state tokens
        n_1d_tokens = 2  # latent + system2_ctx
        if config.robot_state_feature:
            n_1d_tokens += 1
        if config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── System 2 ──────────────────────────────────────────────────────────────

    def encode_system2(self, img: Tensor, text: list[str] | None = None) -> Tensor:
        """Return a (B, clip_embed_dim) task context vector.

        When ``text`` is provided (list of B task-description strings), encodes both
        image and text with CLIP and returns the L2-normalised average, giving the
        model a semantic task signal. Falls back to image-only when text is absent.
        """
        if self.config.system2_mode == "disabled":
            return torch.zeros(img.shape[0], self.config.clip_embed_dim, device=img.device)

        # Resize to CLIP's expected resolution and apply CLIP normalisation
        img_resized = F.interpolate(img.float(), size=(224, 224), mode="bilinear", align_corners=False)
        img_clip = (img_resized.clamp(0.0, 1.0) - self.clip_pixel_mean) / self.clip_pixel_std

        with torch.no_grad():
            # Call sub-models directly to guarantee plain tensor outputs regardless of
            # the transformers version (get_image/text_features return type varies).
            img_features = self.clip.visual_projection(
                self.clip.vision_model(pixel_values=img_clip).pooler_output
            )  # (B, 512)

            if text is not None:
                tokens = self.clip_tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True, max_length=77
                ).to(img.device)
                text_features = self.clip.text_projection(
                    self.clip.text_model(**tokens).pooler_output
                )  # (B, 512)
                img_norm = img_features / img_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                txt_norm = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                return (img_norm + txt_norm) / 2

        return img_features

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        batch: dict[str, Tensor],
        precomputed_ctx: Tensor | None = None,
    ) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        """
        Args:
            batch: Must include OBS_IMAGES (list of (B,C,H,W) tensors), and optionally
                   OBS_STATE, OBS_ENV_STATE, ACTION, action_is_pad.
            precomputed_ctx: Optional cached (B, clip_embed_dim) tensor from a prior
                             encode_system2() call (used during inference to avoid re-encoding).
        Returns:
            actions: (B, chunk_size, action_dim)
            (mu, log_sigma_x2): VAE latent parameters, or (None, None) when VAE is off / inference.
        """
        device = batch[OBS_IMAGES][0].device
        batch_size = batch[OBS_IMAGES][0].shape[0]

        # ── System 2 ──────────────────────────────────────────────────────────
        if precomputed_ctx is not None:
            task_ctx = precomputed_ctx
        else:
            task_ctx = self.encode_system2(batch[OBS_IMAGES][0], text=batch.get("task"))  # (B, clip_embed_dim)
        ctx_token = self.system2_proj(task_ctx)  # (B, dim_model)

        # ── VAE encoder (training only) ───────────────────────────────────────
        if self.config.use_vae and self.training and ACTION in batch:
            mu, log_sigma_x2, latent_sample = self._vae_encode(batch, batch_size, device)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros(batch_size, self.config.latent_dim, device=device)

        # ── Build encoder token sequence ──────────────────────────────────────
        # 1-D tokens: [latent, system2_ctx, (robot_state), (env_state)]
        encoder_in_tokens = [
            self.encoder_latent_input_proj(latent_sample),  # (B, dim_model)
            ctx_token,  # (B, dim_model)
        ]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        if self.config.robot_state_feature and OBS_STATE in batch:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        if self.config.env_state_feature and OBS_ENV_STATE in batch:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # Camera feature patch tokens (2-D positional embeddings)
        for img in batch[OBS_IMAGES]:
            cam_feat = self.backbone(img)["feature_map"]  # (B, 512, H', W')
            cam_pos = self.encoder_cam_feat_pos_embed(cam_feat).to(dtype=cam_feat.dtype)
            cam_feat = self.encoder_img_feat_input_proj(cam_feat)  # (B, dim_model, H', W')
            cam_feat = einops.rearrange(cam_feat, "b c h w -> (h w) b c")
            cam_pos = einops.rearrange(cam_pos, "b c h w -> (h w) b c")
            encoder_in_tokens.extend(list(cam_feat))
            encoder_in_pos_embed.extend(list(cam_pos))

        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)  # (seq, B, dim_model)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)  # (seq, 1, dim_model)

        # ── Transformer forward ───────────────────────────────────────────────
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)  # (B, chunk_size, dim_model)
        actions = self.action_head(decoder_out)  # (B, chunk_size, action_dim)

        return actions, (mu, log_sigma_x2)

    def _vae_encode(
        self, batch: dict[str, Tensor], batch_size: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        """VAE encoder: encodes action sequence + robot state → latent (mu, log_sigma_x2, sample)."""
        cls_embed = einops.repeat(
            self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
        )  # (B, 1, dim_model)

        action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, chunk_size, dim_model)

        if self.config.robot_state_feature and OBS_STATE in batch:
            robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)
            vae_input = torch.cat([cls_embed, robot_state_embed, action_embed], dim=1)
        else:
            vae_input = torch.cat([cls_embed, action_embed], dim=1)

        pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, seq, dim_model)

        # Key-padding mask: cls (and state) tokens are never padding; action tokens may be.
        n_prefix = 2 if (self.config.robot_state_feature and OBS_STATE in batch) else 1
        prefix_pad = torch.zeros(batch_size, n_prefix, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([prefix_pad, batch["action_is_pad"]], dim=1)

        cls_token_out = self.vae_encoder(
            vae_input.permute(1, 0, 2),
            pos_embed=pos_embed.permute(1, 0, 2),
            key_padding_mask=key_padding_mask,
        )[0]  # (B, dim_model)

        latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
        mu = latent_pdf_params[:, : self.config.latent_dim]
        log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
        latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)

        return mu, log_sigma_x2, latent_sample
