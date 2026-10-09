import os
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from echotab.models.EchoTabEncoder import EchoTabEncoder
from echotab.models.ViewClassifier import ViewClassifier
from echotab.models.prediction_head import PredictionHead
from echotab.util.echoprime import load_study


class EchoTabInference(nn.Module):
    """
    Pretrained EchoTab for inference: all clips of an echo study -> measurement tokens
    and measurement predictions.

        ViewClassifier  — assigns the clips to view types (see load_study)
        EchoTabEncoder  — echo study -> one token per measurement
        PredictionHead  — measurement token -> measurement value, in raw units

    The architecture is read from {ckpt_dir}/config.yaml, the weights from the
    per-component checkpoints in ckpt_dir.
    """

    def __init__(self, ckpt_dir: str, view_classifier_path: str) -> None:
        super().__init__()
        cfg = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
        # The echo encoder weights are restored from echo_branch.ckpt below.
        cfg.train.encoder.echo.checkpoint_path = None

        self.encoder = EchoTabEncoder(cfg)
        self.encoder.load_component_checkpoints(ckpt_dir)

        self.prediction_head = PredictionHead.from_config(
            cfg, echo_dim=self.encoder.echo_branch.echo_dim, num_features=self.encoder.n_features,
        )
        checkpoint = torch.load(os.path.join(ckpt_dir, 'prediction_head.ckpt'), map_location='cpu', weights_only=False)
        msg = self.prediction_head.load_state_dict(checkpoint['model'], strict=True)
        print(f'Successfully loaded prediction_head.ckpt: {msg}')

        self.view_classifier = ViewClassifier(view_classifier_path)

        self.labels = self.encoder.labels
        # View types the model was not trained on; clips assigned to them are not used.
        self.excluded_view_types = list(cfg.dataset.echo.get('excluded_view_types', None) or [])
        self.eval()

    @torch.no_grad()
    def predict(self, study: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            study: view_type -> (1, T, H, W, C), one clip per view type (load_echoprime_clip)

        Returns:
            tokens:      (1, N_measurements, D), in the order of self.labels
            predictions: (1, N_measurements), in raw units
        """
        device = next(self.parameters()).device
        tokens, _ = self.encoder({view: clip.to(device) for view, clip in study.items()})
        predictions = self.prediction_head(tokens, self.encoder.tab_branch.conditioning_emb())
        return tokens, predictions

    @torch.no_grad()
    def forward(self, paths: List[str]) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
        """
        Args:
            paths: all clips of one echo study

        Returns:
            tokens, predictions: see predict
            assignments:         one row per clip with columns path, view, confidence, selected
        """
        study, assignments = load_study(paths, self.view_classifier, self.excluded_view_types)
        return *self.predict(study), assignments
