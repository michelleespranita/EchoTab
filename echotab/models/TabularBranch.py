from typing import List, Optional

import torch
import torch.nn as nn


class RandomTabularTextEncoder(nn.Module):
    """Fully learnable per-measurement embeddings, one per phenotype (no text prior)."""

    def __init__(self, feature_names: List[str], tabular_dim: int) -> None:
        super().__init__()
        self.tabular_dim = tabular_dim
        self.n_features = len(feature_names)
        self.feature_names = list(feature_names)

        self.emb = nn.Embedding(self.n_features, tabular_dim)
        nn.init.trunc_normal_(self.emb.weight, std=0.02)

    def forward(self) -> torch.Tensor:
        """Returns (N_features, tabular_dim)."""
        return self.emb.weight


class TabularBranch(nn.Module):
    """
    Produces one query token per measurement for the TabDecoder.

    The queries depend only on *which* measurement is decoded, never on its value:
    each measurement owns a learnable embedding (RandomTabularTextEncoder), projected
    to echo_dim when tabular_dim differs. The same embedding serves as the
    conditioning vector of the partially-shared prediction head (see
    conditioning_emb()).

    Args:
        feature_names: list of N measurement names (defines the query order)
        tabular_dim:   dimension of the measurement embeddings
        echo_dim:      output (query) dimension, i.e. the echo token dimension
    """

    def __init__(self, feature_names: List[str], tabular_dim: int, echo_dim: int) -> None:
        super().__init__()
        self.tabular_dim = tabular_dim
        self.echo_dim = echo_dim
        self.n_features = len(feature_names)
        self.feature_names = list(feature_names)

        self.text_encoder = RandomTabularTextEncoder(feature_names, tabular_dim)

        self.out_proj: Optional[nn.Linear] = (
            nn.Linear(tabular_dim, echo_dim) if tabular_dim != echo_dim else None
        )

    def forward(self, batch_size: int) -> torch.Tensor:
        """Returns the measurement query tokens: (B, N_features, echo_dim)."""
        query_emb = self.text_encoder()                                   # (N, tabular_dim)
        tabular_tokens = query_emb.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, tabular_dim)
        if self.out_proj is not None:
            tabular_tokens = self.out_proj(tabular_tokens)                 # (B, N, echo_dim)
        return tabular_tokens

    def conditioning_emb(self) -> torch.Tensor:
        """Per-measurement embeddings (N_features, tabular_dim) that condition the
        partially-shared prediction head. No out_proj is applied."""
        return self.text_encoder()
