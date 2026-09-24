import torch
import torch.nn as nn
import torchvision


class EchoPrime(nn.Module):
    """EchoPrime video encoder (MViTv2-S backbone).

    Returns the local (spatio-temporal patch) tokens of a clip. The projection head is
    kept only so that the official EchoPrime weights load without key mismatches.
    """

    embed_dim = 768

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.video.mvit_v2_s()
        self.model.head[-1] = nn.Linear(self.model.head[-1].in_features, 512)  # to be compatible with EchoPrime

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T, H, W) -> (B, 1 + N_patches, 768), CLS token first."""
        # patchify: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, T'H'W', embed_channels[0])
        x = self.model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        x = self.model.pos_encoding(x)

        thw = (self.model.pos_encoding.temporal_size,) + self.model.pos_encoding.spatial_size
        for block in self.model.blocks:
            x, thw = block(x, thw)
        x = self.model.norm(x)

        return x

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, C, T, H, W) -> {'local_tokens': (B, 392, 768)}."""
        x = self.forward_features(x)
        return {'local_tokens': x[:, 1:]}  # drop the CLS token
