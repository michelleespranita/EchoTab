import torch
import torch.nn as nn

from .utils import build_head


class SharedHead(nn.Module):
    """A single MLP head applied identically to every measurement token: all
    measurements share the exact same weights, with no per-measurement conditioning."""

    def __init__(
        self,
        echo_dim: int,
        num_head_layers: int = 1,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = build_head(echo_dim, num_head_layers, head_dropout)

    def forward(self, feature_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_tokens: (B, N, echo_dim)
        Returns:
            (B, N)
        """
        return self.head(feature_tokens).squeeze(-1)
