import os
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from torchmetrics.functional import confusion_matrix
from tqdm import tqdm

from echotab.util.metrics import init_metrics_fn


class KNN:
    """Non-parametric probe on frozen features for ordinal classification.

    Uses a k-nearest-neighbour classifier (k=5).

    ``feature_fn(batch)`` must return {'features': (B, D) tensor, 'label': (B, num_classes) one-hot}.
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device = torch.device('cpu'),
        save_dir: Optional[str] = None,
    ):
        self.device = device
        self.num_classes = num_classes
        self.save_dir = save_dir

        self.model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
        self.metrics_fn = init_metrics_fn('ordinal_classification', device, num_classes)

    def _extract_features(self, dataloader, feature_fn: Callable[[dict], dict]):
        features, labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader), desc='KNN - feature extraction'):
            out = feature_fn(batch)
            features.append(out['features'].detach().float().cpu().numpy())
            labels.append(np.asarray(out['label']))
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    def fit(self, dataloader_train, feature_fn: Callable[[dict], dict]) -> None:
        features, labels = self._extract_features(dataloader_train, feature_fn)
        self.model.fit(features, np.argmax(labels, axis=-1))

    def evaluate(self, dataloader, feature_fn: Callable[[dict], dict], split_name: str) -> dict:
        features, labels = self._extract_features(dataloader, feature_fn)

        probs = torch.tensor(self.model.predict_proba(features), dtype=torch.float32)  # (n, num_classes)
        labels = torch.tensor(labels, dtype=torch.long)

        self.metrics_fn.reset()
        self.metrics_fn.update(probs, labels, pred_is_probs=True)
        preds, targets = probs.argmax(dim=-1), labels.argmax(dim=-1)
        cm = confusion_matrix(preds, targets, task='multiclass', num_classes=self.num_classes)
        metrics = self.metrics_fn.compute()
        self.metrics_fn.reset()

        print(f'Confusion matrix ({split_name}):\n{cm}')
        if self.save_dir is not None:
            torch.save(cm, os.path.join(self.save_dir, f'confmat_{split_name}.pt'))
            torch.save(preds, os.path.join(self.save_dir, f'preds_{split_name}.pt'))

        return metrics
