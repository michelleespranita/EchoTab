from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class TabDecoderLayer(nn.Module):
    """
    One cross-attention + FFN block, pre-norm throughout.

    Q  = measurement query tokens  (echo_dim)
    KV = echo tokens               (echo_dim)

    The cross-attention output is added to the queries through a learnable per-channel
    layer scale (initialised to layerscale_init).
    """

    def __init__(
        self,
        echo_dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        layerscale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm_cross_q = nn.LayerNorm(echo_dim)
        self.norm_cross_kv = nn.LayerNorm(echo_dim)
        self.norm_ffn = nn.LayerNorm(echo_dim)

        # Learnable per-channel layer scale on the cross-attention residual branch.
        self.layerscale_init = nn.Parameter(layerscale_init * torch.ones(echo_dim))

        self.cross_attn = nn.MultiheadAttention(
            echo_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(echo_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, echo_dim),
        )

    def forward(
        self,
        queries: torch.Tensor,
        echo_kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        queries:          (B, N_features, echo_dim)
        echo_kv:          (B, num_views*K, echo_dim)
        key_padding_mask: (B, num_views*K) bool or None, True = ignore that key

        Returns (queries, attn_weights); attn_weights is (B, N_features, num_views*K) averaged
        over heads, or None unless return_attn.
        """
        q = self.norm_cross_q(queries)
        kv = self.norm_cross_kv(echo_kv)
        cross_out, attn_w = self.cross_attn(
            q, kv, kv,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
        )
        queries = queries + self.layerscale_init * cross_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries, attn_w


class TabDecoder(nn.Module):
    """
    Measurement queries cross-attend over the echo tokens of a study, producing one
    bottleneck token per measurement. All layers run at echo_dim.

    Returns:
        z_out:        (B, N_features, echo_dim)
        attn_weights: list of (B, N_features, num_views*K), one per layer; empty unless
                      return_attn=True.
    """

    def __init__(
        self,
        echo_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.0,
        layerscale_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TabDecoderLayer(echo_dim, num_heads, 4 * echo_dim, dropout, layerscale_init)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        tab_queries: torch.Tensor,
        echo_tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        tab_queries:      (B, N_features, echo_dim) - queries
        echo_tokens:      (B, num_views*K, echo_dim)       - keys / values
        key_padding_mask: (B, num_views*K) bool or None, True = ignore that key position
                          (absent views / background patches, flagged by EchoBranch)
        """
        attn_weights: List[torch.Tensor] = []
        for layer in self.layers:
            tab_queries, attn = layer(
                tab_queries, echo_tokens, key_padding_mask=key_padding_mask, return_attn=return_attn,
            )
            if return_attn:
                attn_weights.append(attn)
        return tab_queries, attn_weights
