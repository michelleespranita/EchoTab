import torch
import torch.nn as nn
import torchvision

from echotab.util.echoprime import NUM_VIEWS


class ViewClassifier(nn.Module):
    """EchoPrime view classifier (ConvNeXt-Base) over VIEW_TYPES."""

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        self.model = torchvision.models.convnext_base()
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, NUM_VIEWS)
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.eval()

    @torch.no_grad()
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (N, T, H, W, C), preprocessed with load_echoprime_clip -> (N, NUM_VIEWS) view probabilities."""
        first_frames = clips[:, 0].permute(0, 3, 1, 2)  # (N, C, H, W)
        device = next(self.parameters()).device
        return self.model(first_frames.to(device)).softmax(dim=-1)
