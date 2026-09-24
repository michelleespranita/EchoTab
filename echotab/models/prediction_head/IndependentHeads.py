import torch
import torch.nn as nn

from .utils import build_head


class IndependentHeads(nn.Module):
    """One independent MLP head per measurement: head_i maps the i-th bottleneck token
    to the i-th prediction. No parameters are shared across measurements."""

    def __init__(
        self,
        echo_dim: int,
        num_features: int,
        num_head_layers: int = 1,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            build_head(echo_dim, num_head_layers, head_dropout)
            for _ in range(num_features)
        ])

    def forward(self, feature_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_tokens: (B, N, echo_dim)
        Returns:
            (B, N)
        """
        return torch.stack(
            [head(feature_tokens[:, i, :]) for i, head in enumerate(self.heads)],
            dim=1,
        ).squeeze(-1)
