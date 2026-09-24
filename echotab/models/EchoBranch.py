import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import DictConfig

from echotab.models.EchoPrime import EchoPrime
from echotab.util.echoprime import VIEW_TYPES, NUM_VIEWS, _TEMPORAL_STRIDE, _SPATIAL_PATCH


def init_echo_encoder(cfg: DictConfig) -> EchoPrime:

    echo_encoder = EchoPrime()

    checkpoint_path = cfg.train.encoder.echo.checkpoint_path
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        msg = echo_encoder.model.load_state_dict(checkpoint, strict=False)
        print(f'Successfully loaded EchoPrime encoder checkpoint: {checkpoint_path}')
        print(msg)
    else:
        print('The EchoPrime echo encoder is randomly initialized!')

    num_blocks = len(echo_encoder.model.blocks)
    freeze_first_n_layers = cfg.train.encoder.echo.freeze_first_n_layers
    if freeze_first_n_layers > num_blocks:
        raise ValueError(f"Echo encoder: freeze_first_n_layers ({freeze_first_n_layers}) exceeds number of blocks ({num_blocks})")

    if freeze_first_n_layers > 0:
        for layer_name in ['conv_proj', 'pos_encoding']:
            for param in getattr(echo_encoder.model, layer_name).parameters():
                param.requires_grad = False
        for name, param in echo_encoder.model.named_parameters():
            if 'blocks.' in name and int(name.split('.')[1]) < freeze_first_n_layers:
                param.requires_grad = False
        if freeze_first_n_layers == num_blocks:
            for layer_name in ['norm', 'head']:
                for param in getattr(echo_encoder.model, layer_name).parameters():
                    param.requires_grad = False

    return echo_encoder


class EchoBranch(nn.Module):
    """
    Encodes a multi-view echo study into a fixed (B, num_views * K, ECHO_DIM) token sequence.

    Inputs:
        study: dict[view_type_str -> (B, T, H, W, C)]
               Only view types that are present in the batch appear in the dict.
        view_present_mask: optional dict[view_type_str -> (B,) bool]
               Per-sample presence flag within a view-type tensor. If absent for a view
               type, all B samples are assumed to have that view.

    For each canonical view (VIEW_TYPES, fixed order):
        Present -> EchoPrime local tokens (B, 392, D)
                   -> all local tokens kept                 [preserve_local_tokens=True]  (K=392)
                   -> masked mean-pool over foreground      [use_bg_removal=True]         (K=1)
                   -> mean-pool                             [otherwise]                   (K=1)
                   + learnable view-type embedding          [use_learnable_view_emb=True]
        Absent  -> zeros (always masked out in the TabDecoder cross-attention)

    With use_bg_removal=True, patches whose mean normalised intensity is
    <= cfg.dataset.echo.background_threshold are treated as background. With
    preserve_local_tokens=True they are flagged in the key padding mask rather than
    removed, so the per-view token count stays uniform.

    Outputs:
        echo_tokens:        (B, num_views * K, ECHO_DIM)
        view_presence_mask: (B, num_views) bool, True where the view is present
        key_padding_mask:   (B, num_views * K) bool, True where the token is ignored in cross-attn
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.use_bg_removal = cfg.model.echo.use_bg_removal
        self.preserve_local_tokens = cfg.model.echo.preserve_local_tokens
        self.background_threshold = cfg.dataset.echo.background_threshold
        self.freeze_first_n_layers = cfg.train.encoder.echo.freeze_first_n_layers

        self.echo_encoder = init_echo_encoder(cfg)
        self.echo_dim = self.echo_encoder.embed_dim

        if self.preserve_local_tokens:
            n_t = cfg.model.echo.num_frames // _TEMPORAL_STRIDE
            n_hw = cfg.model.echo.img_size // _SPATIAL_PATCH
            self.K = n_t * n_hw * n_hw
        else:
            self.K = 1

        self.use_learnable_view_emb = cfg.model.echo.use_learnable_view_emb
        if self.use_learnable_view_emb:
            self.view_type_emb = nn.Embedding(NUM_VIEWS, self.echo_dim)
            nn.init.trunc_normal_(self.view_type_emb.weight, std=0.02)

    # ------------------------------------------------------------------
    def _compute_background_mask(self, video: torch.Tensor) -> torch.Tensor:
        """
        Background mask over the EchoPrime patch grid (temporal stride 2, 32x32 patches).

        Patches whose mean EchoPrime-normalised intensity is <= background_threshold are
        background. (Black, raw ~0, maps to ~-0.61 after EchoPrime normalisation.)

        Args:
            video: (B, T, H, W, C) normalised float values.

        Returns:
            (B, N_PATCHES) bool, True = background (ignored in cross-attention).
        """
        B, T, H, W, C = video.shape
        n_t, n_h, n_w = T // _TEMPORAL_STRIDE, H // _SPATIAL_PATCH, W // _SPATIAL_PATCH
        assert n_t * _TEMPORAL_STRIDE == T and n_h * _SPATIAL_PATCH == H and n_w * _SPATIAL_PATCH == W, \
            f"Video shape {(T, H, W)} must be divisible by temporal stride {_TEMPORAL_STRIDE} " \
            f"and spatial patch size {_SPATIAL_PATCH}"

        v = video.float().reshape(
            B, n_t, _TEMPORAL_STRIDE, n_h, _SPATIAL_PATCH, n_w, _SPATIAL_PATCH, C
        )
        patch_means = v.mean(dim=(2, 4, 6, 7)).reshape(B, n_t * n_h * n_w)  # (B, N_PATCHES)
        bg_mask = patch_means <= self.background_threshold

        # Ensure at least one patch is unmasked per sample
        all_masked = bg_mask.all(dim=1)
        if all_masked.any():
            bg_mask[all_masked, 0] = False

        return bg_mask

    # ------------------------------------------------------------------
    def forward(
        self,
        study: Dict[str, torch.Tensor],
        view_present_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            study:             view_type_str -> (B, T, H, W, C)
            view_present_mask: view_type_str -> (B,) bool, True = sample has this view.

        Returns:
            echo_tokens:        (B, num_views * K, ECHO_DIM)
            view_presence_mask: (B, num_views) bool
            key_padding_mask:   (B, num_views * K) bool, True = ignore in cross-attn.
        """
        if not study:
            raise ValueError("study dict is empty; cannot infer batch size / device")

        first_video = next(iter(study.values()))
        B, device = first_video.shape[0], first_video.device
        encoder_trainable = self.freeze_first_n_layers < len(self.echo_encoder.model.blocks)

        all_view_tokens: list[torch.Tensor] = []
        presence_flags: list[torch.Tensor] = []
        all_view_masks: list[torch.Tensor] = []  # only used when preserve_local_tokens=True

        for vi, view_type in enumerate(VIEW_TYPES):
            if view_type in study:
                video = study[view_type]  # (B, T, H, W, C)
                if view_present_mask and view_type in view_present_mask:
                    is_present = view_present_mask[view_type].to(device)
                else:
                    is_present = torch.ones(B, dtype=torch.bool, device=device)
            else:
                video = None
                is_present = torch.zeros(B, dtype=torch.bool, device=device)

            presence_flags.append(is_present)

            view_tokens = torch.empty(B, self.K, self.echo_dim, device=device)
            view_mask = torch.zeros(B, self.K, dtype=torch.bool, device=device)  # True = ignore

            # ---- Present samples -----------------------------------------
            n_present = int(is_present.sum().item())
            if n_present > 0 and video is not None:
                pres_video = video[is_present]  # (n_present, T, H, W, C)
                pres_video_t = rearrange(pres_video.float(), "B T H W C -> B C T H W")
                with torch.set_grad_enabled(encoder_trainable and torch.is_grad_enabled()):
                    local_tokens = self.echo_encoder(pres_video_t)["local_tokens"]  # (n_present, 392, D)

                bg_mask = self._compute_background_mask(pres_video) if self.use_bg_removal else None
                if self.preserve_local_tokens:
                    tokens_p = local_tokens  # (n_present, K, D)
                    if bg_mask is not None:
                        view_mask[is_present] = bg_mask
                elif bg_mask is not None:
                    # Masked mean-pool over the foreground patches
                    fg = ~bg_mask  # (n_present, N_PATCHES)
                    masked = local_tokens * fg.unsqueeze(-1).float()
                    tokens_p = masked.sum(dim=1, keepdim=True) / fg.float().sum(dim=1, keepdim=True).unsqueeze(-1)
                else:
                    tokens_p = local_tokens.mean(dim=1, keepdim=True)  # (n_present, 1, D)

                if self.use_learnable_view_emb:
                    tokens_p = tokens_p + self.view_type_emb(torch.tensor(vi, device=device))

                view_tokens[is_present] = tokens_p

            # ---- Absent samples ------------------------------------------
            is_absent = ~is_present
            if is_absent.any():
                view_tokens[is_absent] = 0.0
                view_mask[is_absent] = True

            all_view_tokens.append(view_tokens)
            all_view_masks.append(view_mask)

        echo_tokens = torch.cat(all_view_tokens, dim=1)          # (B, num_views * K, D)
        view_presence_out = torch.stack(presence_flags, dim=1)   # (B, num_views)
        key_padding_mask = torch.cat(all_view_masks, dim=1)      # (B, num_views * K)

        return echo_tokens, view_presence_out, key_padding_mask
