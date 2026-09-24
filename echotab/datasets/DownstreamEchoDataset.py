from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import lightning as L
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from echotab.datasets.EchoTabDataset import study_collate_fn
from echotab.util.echoprime import VIEW_TYPES, load_echoprime_clip


# Canonical severity ordering for ordinal categorical labels (e.g. "rv_size").
# Categories present in a target column are mapped to consecutive class indices
# following this order; absent categories are skipped.
ORDINAL_SEVERITY_CATEGORIES = [
    'Normal', 'Trace', 'Mild', 'Mild-Moderate', 'Moderate', 'Moderate-Severe', 'Severe',
]

CLASSIFICATION_TASKS = ('binary_classification', 'multiclass_classification', 'ordinal_classification')


class DownstreamEchoDataset(Dataset):
    """Study-level downstream echo dataset (one sample per echo study).

    The split CSV has one row per clip with columns echo_study_id, echo_view,
    echo_filepath and the target column(s); the label is read from the study's first row.

    __getitem__ returns {'study': {view_type: (T, H, W, C)}, 'label': ..., 'echo_study_id': ...}.
    Use study_collate_fn as the DataLoader collate_fn.

    Labels:
      classification — a categorical target column (ORDINAL_SEVERITY_CATEGORIES values),
                       one-hot encoded; the train split defines the label space, which
                       val/test inherit via canonical_categories.
      regression     — one or more numeric target columns.
    Samples with a missing label are dropped.
    """

    def __init__(
        self,
        cfg: DictConfig,
        mode: str,
        canonical_categories: Optional[list] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        task_cfg = cfg.downstream_task
        self.task_type = task_cfg.task_type
        target = task_cfg.target
        self.target = [target] if isinstance(target, str) else list(target)

        df = pd.read_csv(task_cfg.paths[f'data_{mode}']).reset_index(drop=True)
        self.excluded_view_types: List[str] = list(task_cfg.get("excluded_view_types", None) or [])
        first_positions = self._init_study_index(df)

        rows = df.loc[first_positions]
        if self.task_type == 'regression':
            self.labels = torch.tensor(rows[self.target].values.astype(float)).float()
            self.num_classes = len(self.target)
            keep = [i for i in range(len(self.labels)) if not torch.isnan(self.labels[i]).all()]
        elif self.task_type in CLASSIFICATION_TASKS:
            self.labels = self._categorical_to_onehot(rows[self.target[0]].tolist(), canonical_categories)
            keep = [i for i in range(len(self.labels)) if not torch.isnan(self.labels[i]).any()]
        else:
            raise ValueError(f"Unsupported task_type: {self.task_type!r}")

        self.study_ids = [self.study_ids[i] for i in keep]
        self.study_views = [self.study_views[i] for i in keep]
        self.labels = self.labels[keep]

        self._print_label_distribution()

    def _init_study_index(self, df: pd.DataFrame) -> List[int]:
        """Group clips by study, keep one clip (the first) per canonical view type and
        drop studies with no remaining views. Returns each study's first row index."""
        allowed_views = [v for v in VIEW_TYPES if v not in self.excluded_view_types]

        self.study_ids: List[str] = []
        self.study_views: List[Dict[str, str]] = []
        first_positions: List[int] = []
        for sid, rows in df.groupby('echo_study_id', sort=False):
            views = {}
            for view_type in allowed_views:
                view_rows = rows[rows['echo_view'] == view_type]
                if not view_rows.empty:
                    views[view_type] = view_rows.iloc[0]['echo_filepath']
            if not views:
                continue
            self.study_ids.append(sid)
            self.study_views.append(views)
            first_positions.append(rows.index[0])

        n_removed = df['echo_study_id'].nunique() - len(self.study_ids)
        if n_removed:
            print(f"Pre-filtered {n_removed} {self.mode} studies with no remaining views "
                  f"after excluding {self.excluded_view_types}")
        return first_positions

    def _categorical_to_onehot(self, values: list, canonical_categories: Optional[list]) -> torch.Tensor:
        """(n, num_classes) one-hot labels; rows with a missing or out-of-vocabulary
        category are NaN (and dropped afterwards)."""
        present = set(v for v in values if not pd.isna(v))
        unknown = present - set(ORDINAL_SEVERITY_CATEGORIES)
        if unknown:
            raise ValueError(
                f"Target column {self.target[0]!r} contains categories not in the canonical "
                f"severity ordering: {sorted(unknown)}. Expected a subset of {ORDINAL_SEVERITY_CATEGORIES}."
            )

        if canonical_categories is not None:
            self.categories = list(canonical_categories)
        else:
            self.categories = [c for c in ORDINAL_SEVERITY_CATEGORIES if c in present]
        category_to_index = {c: i for i, c in enumerate(self.categories)}
        self.num_classes = len(self.categories)

        labels = torch.full((len(values), self.num_classes), float('nan'))
        out_of_vocabulary = {}
        for i, value in enumerate(values):
            if pd.isna(value):
                continue
            if value not in category_to_index:
                out_of_vocabulary[value] = out_of_vocabulary.get(value, 0) + 1
                continue
            labels[i] = 0.0
            labels[i, category_to_index[value]] = 1.0

        if out_of_vocabulary:
            warnings.warn(
                f"[mode={self.mode} | target={self.target[0]!r}] dropping "
                f"{sum(out_of_vocabulary.values())} sample(s) whose category is not in the "
                f"train label space {self.categories}: {out_of_vocabulary}"
            )
        return labels

    def _print_label_distribution(self) -> None:
        lines = [f"[{self.mode} | target={self.target}] {len(self.study_ids)} studies"]
        if self.task_type in CLASSIFICATION_TASKS:
            classes, counts = np.unique(self.labels.argmax(dim=-1).numpy(), return_counts=True)
            for c, cnt in zip(classes, counts):
                lines.append(f'  {self.categories[c]}: {cnt} ({cnt / len(self.labels):.1%})')
        print('\n'.join(lines))

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> dict:
        task_cfg = self.cfg.downstream_task
        study = {
            view_type: load_echoprime_clip(
                filepath,
                sampling_rate=task_cfg.sampling_rate,
                num_frames=task_cfg.num_frames,
                img_size=task_cfg.img_size,
            )
            for view_type, filepath in self.study_views[idx].items()
        }
        if self.task_type == 'binary_classification':
            label = self.labels[idx].argmax(dim=-1)
        elif self.task_type == 'regression':
            label = self.labels[idx].squeeze()
        else:
            label = self.labels[idx]
        return {'study': study, 'label': label, 'echo_study_id': self.study_ids[idx]}


class DownstreamEchoDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.downstream_task.batch_size
        self.num_workers = cfg.downstream_task.num_workers
        self._canonical_categories = None

    def setup(self, stage: str) -> None:
        if stage in ('fit', 'knn'):
            self.dataset_train = DownstreamEchoDataset(self.cfg, 'train')
            self._canonical_categories = getattr(self.dataset_train, 'categories', None)
            self.dataset_val = DownstreamEchoDataset(self.cfg, 'val', self._canonical_categories)
            self.num_classes = self.dataset_train.num_classes
        elif stage == 'test':
            self.dataset_test = DownstreamEchoDataset(self.cfg, 'test', self._canonical_categories)

    def _make_loader(self, dataset: DownstreamEchoDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=study_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.dataset_train)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.dataset_val)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.dataset_test)
