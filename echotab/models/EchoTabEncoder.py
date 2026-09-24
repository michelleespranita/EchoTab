import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from echotab.models.EchoBranch import EchoBranch
from echotab.models.TabDecoder import TabDecoder
from echotab.models.TabularBranch import TabularBranch
from echotab.util.init_stats import load_label_list


def build_projection(dim: int, num_layers: int) -> nn.Module:
    if num_layers == 0:
        return nn.Identity()
    if num_layers == 1:
        return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
    if num_layers == 2:
        return nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
    if num_layers == 3:
        return nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim),
        )
    raise ValueError(f"Unsupported number of projection layers: {num_layers}")


class EchoTabEncoder(nn.Module):
    """
    Maps a multi-view echo study to one token per measurement.

        EchoBranch      — per-view EchoPrime encoding -> (B, num_views*K, D) echo tokens
        echo_projection — LayerNorm + Linear on the echo tokens
        TabularBranch   — one learnable query per measurement -> (B, N, D)
        tab_projection  — LayerNorm + Linear on the queries
        TabDecoder      — queries cross-attend over the echo tokens -> (B, N, D)

    The measurement tokens only depend on the echo study (the queries are
    value-independent), so no tabular input is needed at inference.

    Each component is saved to / restored from its own checkpoint file (see
    COMPONENTS), so that the encoder can be rebuilt without the prediction head.
    """

    # attribute name -> checkpoint file name
    COMPONENTS = {
        'echo_branch': 'echo_branch.ckpt',
        'tab_branch': 'tab_branch.ckpt',
        'tab_decoder': 'tab_decoder.ckpt',
        'echo_projection': 'echo_projection.ckpt',
        'tab_projection': 'tab_projection.ckpt',
    }

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.labels = load_label_list(cfg)

        print('Initializing the echo branch...')
        self.echo_branch = EchoBranch(cfg)
        echo_dim = self.echo_branch.echo_dim

        print('Initializing the tabular branch...')
        self.tab_branch = TabularBranch(
            feature_names=self.labels,
            tabular_dim=cfg.model.tab.embed_dim,
            echo_dim=echo_dim,
        )

        print('Initializing the tabular decoder...')
        dec_cfg = cfg.model.tab_decoder
        self.tab_decoder = TabDecoder(
            echo_dim=echo_dim,
            num_heads=dec_cfg.num_heads,
            num_layers=dec_cfg.num_layers,
            dropout=dec_cfg.dropout,
            layerscale_init=dec_cfg.layerscale_init,
        )
        self.echo_projection = build_projection(echo_dim, dec_cfg.echo_projection.num_layers)
        self.tab_projection = build_projection(echo_dim, dec_cfg.tab_projection.num_layers)

    @property
    def n_features(self) -> int:
        return self.tab_branch.n_features

    def forward(
        self,
        study: Dict[str, torch.Tensor],
        view_present_mask: Optional[Dict[str, torch.Tensor]] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            study:             view_type -> (B, T, H, W, C)
            view_present_mask: view_type -> (B,) bool
            return_attn:       also return the per-layer cross-attention maps

        Returns:
            z_out:        (B, N_features, D), one token per measurement (label order)
            attn_weights: list[(B, N_features, num_views*K)] per TabDecoder layer; [] unless return_attn
        """
        echo_tokens, _, kpm = self.echo_branch(study, view_present_mask)  # (B, num_views*K, D)
        echo_tokens = self.echo_projection(echo_tokens)

        tab_queries = self.tab_projection(self.tab_branch(echo_tokens.shape[0]))  # (B, N, D)

        return self.tab_decoder(
            tab_queries, echo_tokens, key_padding_mask=kpm, return_attn=return_attn,
        )

    # ------------------------------------------------------------------
    def save_component_checkpoints(self, ckpt_dir: str, meta: dict) -> None:
        os.makedirs(ckpt_dir, exist_ok=True)
        for attr, filename in self.COMPONENTS.items():
            module = getattr(self, attr)
            if isinstance(module, nn.Identity):
                continue
            torch.save({"model": module.state_dict(), **meta}, os.path.join(ckpt_dir, filename))

    def load_component_checkpoints(self, ckpt_dir: str) -> None:
        for attr, filename in self.COMPONENTS.items():
            module = getattr(self, attr)
            if isinstance(module, nn.Identity):
                continue
            path = os.path.join(ckpt_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Required checkpoint {path} not found.")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            msg = module.load_state_dict(checkpoint["model"], strict=True)
            print(f"Successfully loaded {filename}: {msg}")
