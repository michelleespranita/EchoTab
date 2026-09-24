"""
Regression-metric computation, logging and result saving for EchoTab.

Concrete classes must set the following attributes before calling any method:

    self.labels                  : List[str]     — ordered phenotype names
    self.mean, self.std          : torch.Tensor  — (1, N), for de-normalisation
    self.eval_dropped_phenotypes : set           — phenotypes excluded from metrics
    self.batch_size              : int
    self.save_dir                : str
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from echotab.util.metrics import init_metrics_fn


class RegressionLoggingMixin:

    def _denormalize(self, preds: torch.Tensor, labels: torch.Tensor, label_masks: torch.Tensor):
        preds = preds * self.std + self.mean
        labels = torch.where(~label_masks, labels * self.std + self.mean, labels)
        return preds, labels

    def _compute_per_study_metrics(
        self,
        preds: torch.Tensor,        # (n_studies, N), normalised
        labels: torch.Tensor,       # (n_studies, N), normalised, NaN where missing
        label_masks: torch.Tensor,  # (n_studies, N) bool, True = missing
    ) -> Dict[str, Dict[str, Any]]:
        """Per-phenotype metrics in raw units; rare phenotypes are left out."""
        preds, labels = self._denormalize(preds, labels, label_masks)

        metrics_dict = {}
        for idx, label in enumerate(self.labels):
            if label in self.eval_dropped_phenotypes:
                continue
            metric_fn = init_metrics_fn('regression', torch.device('cpu'), 1)
            metric_fn.update(preds[:, idx], labels[:, idx], label_masks[:, idx])
            m = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metric_fn.compute().items()}
            m['num_samples'] = int((~label_masks[:, idx]).sum().item())

            median = torch.median(preds[:, idx])
            m['mad'] = torch.median(torch.abs(preds[:, idx] - median)).item()

            label_mean = torch.mean(labels[:, idx][~label_masks[:, idx]])
            m['nmae'] = (m['mae'] / label_mean).item()  # may be NaN

            metrics_dict[label] = m
        return metrics_dict

    def _log_regression_metrics(self, metrics_dict: Dict[str, Dict[str, Any]], prefix: str) -> None:
        r2s, maes, mses, mads, nmaes = [], [], [], [], []
        for label, m in metrics_dict.items():
            self.log(f'{prefix}/{label}_r2', float(m['r2']), batch_size=self.batch_size)
            r2s.append(float(m['r2']))
            maes.append(float(m['mae']))
            mses.append(float(m['mse']))
            mads.append(float(m['mad']))
            nmaes.append(float(m['nmae']))

        def _agg(fn, values):
            return fn(values) if values else float('nan')

        self.log(f'{prefix}/mean_r2',   _agg(np.nanmean, r2s),   batch_size=self.batch_size)
        self.log(f'{prefix}/median_r2', _agg(np.nanmedian, r2s), batch_size=self.batch_size)
        self.log(f'{prefix}/min_r2',    _agg(np.nanmin, r2s),    batch_size=self.batch_size)
        self.log(f'{prefix}/max_r2',    _agg(np.nanmax, r2s),    batch_size=self.batch_size)
        self.log(f'{prefix}/std_r2',    _agg(np.nanstd, r2s),    batch_size=self.batch_size)

        valid = [m for m in metrics_dict.values() if not np.isnan(float(m['r2']))]
        total = sum(m['num_samples'] for m in valid)
        weighted_mean_r2 = sum(m['r2'] * m['num_samples'] for m in valid) / total if total > 0 else float('nan')
        self.log(f'{prefix}/weighted_mean_r2', weighted_mean_r2, batch_size=self.batch_size)

        self.log(f'{prefix}/mean_mae',    _agg(np.nanmean, maes),    batch_size=self.batch_size)
        self.log(f'{prefix}/mean_mse',    _agg(np.nanmean, mses),    batch_size=self.batch_size)
        self.log(f'{prefix}/mean_mad',    _agg(np.nanmean, mads),    batch_size=self.batch_size)
        self.log(f'{prefix}/mean_nmae',   _agg(np.nanmean, nmaes),   batch_size=self.batch_size)
        self.log(f'{prefix}/median_nmae', _agg(np.nanmedian, nmaes), batch_size=self.batch_size)

    def _save_metrics_dataframe(self, metrics_dict: Dict[str, Dict[str, Any]], filename: str) -> None:
        df = pd.DataFrame.from_dict(metrics_dict, orient='index').reset_index().rename(columns={'index': 'phenotype'})
        df.to_csv(os.path.join(self.save_dir, filename), index=False)

    def _save_predictions_dataframe(
        self,
        study_ids: List[str],
        preds: torch.Tensor,
        labels: torch.Tensor,
        label_masks: torch.Tensor,
        filename: str,
    ) -> None:
        """One row per study with {phenotype}_label / _label_mask / _pred columns, raw units."""
        preds, labels = self._denormalize(preds, labels, label_masks)
        columns = {'echo_study_id': study_ids}
        for idx, label in enumerate(self.labels):
            columns[f'{label}_label'] = labels[:, idx].tolist()
            columns[f'{label}_label_mask'] = label_masks[:, idx].tolist()
            columns[f'{label}_pred'] = preds[:, idx].tolist()
        pd.DataFrame(columns).to_csv(os.path.join(self.save_dir, filename), index=False)
