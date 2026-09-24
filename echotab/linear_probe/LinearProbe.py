import os
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from torchmetrics.functional import confusion_matrix
from tqdm import tqdm

from echotab.util.metrics import init_metrics_fn

CLASSIFICATION_TASKS = ('binary_classification', 'multiclass_classification', 'ordinal_classification')


class LinearProbe:
    """Non-parametric probe on frozen features.

    Classification tasks use a k-nearest-neighbour classifier (k=5); regression uses an
    ordinary least-squares linear regression (one per target).

    ``feature_fn(batch)`` must return {'features': (B, D) tensor, 'label': (B, ...)}.
    """

    def __init__(
        self,
        task_type: str = 'ordinal_classification',
        device: torch.device = torch.device('cpu'),
        num_classes: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        self.task_type = task_type
        self.device = device
        self.num_classes = num_classes
        self.save_dir = save_dir

        if task_type == 'regression':
            self.models = [LinearRegression() for _ in range(num_classes)]
        elif task_type in CLASSIFICATION_TASKS:
            self.model = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
        else:
            raise ValueError(f"Unsupported task_type: {task_type!r}")

        self.metrics_fn = init_metrics_fn(task_type, device, num_classes)

    def _extract_features(self, dataloader, feature_fn: Callable[[dict], dict]):
        features, labels = [], []
        for batch in tqdm(dataloader, total=len(dataloader), desc='Linear probe - feature extraction'):
            out = feature_fn(batch)
            features.append(out['features'].detach().float().cpu().numpy())
            labels.append(np.asarray(out['label']))
        return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

    def fit(self, dataloader_train, feature_fn: Callable[[dict], dict]) -> None:
        features, labels = self._extract_features(dataloader_train, feature_fn)

        if self.task_type == 'regression':
            labels = labels.reshape(len(labels), -1)
            for t, model in enumerate(self.models):
                valid = ~np.isnan(labels[:, t])
                if valid.any():
                    model.fit(features[valid], labels[valid, t])
        else:
            if self.task_type != 'binary_classification':
                labels = np.argmax(labels, axis=-1)
            self.model.fit(features, labels)

    def evaluate(self, dataloader, feature_fn: Callable[[dict], dict], split_name: str) -> dict:
        features, labels = self._extract_features(dataloader, feature_fn)

        if self.task_type == 'regression':
            labels = labels.reshape(len(labels), -1)
            per_target = []
            for t, model in enumerate(self.models):
                valid = ~np.isnan(labels[:, t])
                if not valid.any() or not hasattr(model, 'coef_'):
                    continue
                m = init_metrics_fn('regression', self.device, 1)
                m.update(
                    torch.tensor(model.predict(features[valid]), dtype=torch.float32),
                    torch.tensor(labels[valid, t], dtype=torch.float32),
                )
                per_target.append(m.compute())
            metrics = {}
            for key in (per_target[0] if per_target else {}):
                vals = [float(d[key]) for d in per_target if not np.isnan(float(d[key]))]
                metrics[key] = float(np.mean(vals)) if vals else float('nan')
            return metrics

        probs = torch.tensor(self.model.predict_proba(features), dtype=torch.float32)  # (n, num_classes)
        labels = torch.tensor(labels, dtype=torch.long)

        self.metrics_fn.reset()
        if self.task_type == 'binary_classification':
            probs = probs[:, 1]  # positive class
            self.metrics_fn.update(probs, labels, pred_is_probs=True)
            preds, targets = (probs > 0.5).long(), labels
            cm = confusion_matrix(preds, targets, task='binary')
        else:
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
