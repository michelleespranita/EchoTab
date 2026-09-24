import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig


def load_label_list(cfg: DictConfig) -> List[str]:
    """Ordered list of phenotype (measurement) names the model is trained on.

    Read from the phenotype list CSV (``cfg.dataset.tab.label_list_path``), column
    ``cfg.dataset.paths.phenotype_col_name``. The CSV row order defines the order of the
    model's measurement queries and outputs.
    """
    label_df = pd.read_csv(cfg.dataset.tab.label_list_path)
    return label_df[cfg.dataset.paths.phenotype_col_name].drop_duplicates().tolist()


def _load_train_split(cfg: DictConfig, labels: List[str]) -> pd.DataFrame:
    """One row per study of the train split, restricted to ``labels``."""
    data_train = pd.read_csv(cfg.dataset.paths['data_train'])
    return data_train[['echo_study_id'] + list(labels)].drop_duplicates('echo_study_id')


def init_winsor_bounds(cfg: DictConfig, labels: List[str], verbose: bool = False) -> dict:
    """Per-label winsorization ``(lo, hi)`` bounds fit on the train split.

    Bounds come from the configured ``physical_bounds`` (falling back to the
    lower/upper percentile of the train column for any unset side) or, absent a
    physical bound, purely from those percentiles. Labels with fewer than
    ``min_eval_samples`` non-null train values are omitted (left unclipped).
    Returns {} when winsorization is disabled.
    """
    if not cfg.dataset.tab.winsorization.use:
        if verbose:
            print("No winsorization applied")
        return {}

    data_train = _load_train_split(cfg, labels)

    lower_p = cfg.dataset.tab.winsorization.lower_percentile
    upper_p = cfg.dataset.tab.winsorization.upper_percentile
    physical_bounds = (
        dict(cfg.dataset.tab.winsorization.physical_bounds)
        if cfg.dataset.tab.winsorization.physical_bounds
        else {}
    )

    bounds: dict = {}
    for col in labels:
        non_null = data_train[col].dropna()
        if len(non_null) < cfg.dataset.tab.min_eval_samples:
            if verbose:
                print(f"Skipping winsorization for {col}: only {len(non_null)} samples")
            continue
        if col in physical_bounds:
            lo, hi = physical_bounds[col]
            lo = lo if lo is not None else float(np.nanpercentile(non_null, lower_p))
            hi = hi if hi is not None else float(np.nanpercentile(non_null, upper_p))
        else:
            lo = float(np.nanpercentile(non_null, lower_p))
            hi = float(np.nanpercentile(non_null, upper_p))
        bounds[col] = (lo, hi)

        if verbose:
            n_clipped = int((non_null < lo).sum() + (non_null > hi).sum())
            print(f'Winsorize {col}: [{lo:.3f}, {hi:.3f}] | clipped {n_clipped}/{len(non_null)} ({100*n_clipped/len(non_null):.1f}%)')

    return bounds


def apply_winsor_bounds(df: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Set out-of-bounds values to NaN in-place using {label: (lo, hi)} bounds."""
    for col, (lo, hi) in bounds.items():
        if col in df.columns:
            out_of_bounds = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            df.loc[out_of_bounds, col] = np.nan
    return df


def init_mean_std(cfg: DictConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-target (or global) winsorized mean/std fit on the train split.

    Returns (means, stds), each of shape (1, N). Zeros / ones when normalization is
    disabled, so that ``pred * std + mean`` is always the raw-unit prediction.
    """
    labels = load_label_list(cfg)
    norm_cfg = cfg.dataset.tab.normalization

    if not norm_cfg.use:
        return torch.zeros(1, len(labels)), torch.ones(1, len(labels))

    data_train = _load_train_split(cfg, labels)
    apply_winsor_bounds(data_train, init_winsor_bounds(cfg, labels))

    if norm_cfg.normalize_per_target:
        means = torch.tensor([data_train[l].mean() for l in labels]).float().unsqueeze(0)
        stds = torch.tensor([data_train[l].std() for l in labels]).float().unsqueeze(0)
    else:
        all_vals = data_train[labels].to_numpy(dtype=np.float64).ravel()
        means = torch.tensor(np.nanmean(all_vals)).expand(len(labels)).float().unsqueeze(0)
        stds = torch.tensor(np.nanstd(all_vals, ddof=1)).expand(len(labels)).float().unsqueeze(0)
    return means, stds


def init_task_stds(cfg: DictConfig, verbose: bool = True) -> torch.Tensor:
    """Per-target winsorized standard deviation sigma_j, fit on the train split.

    Used as the weight 1/sigma_j^2 of the variance-weighted reconstruction loss, which
    is the raw-value substitute for per-target normalization: every measurement then
    contributes to the gradient on its own scale rather than in proportion to its
    units squared.

    Returns (N,) aligned to the label order.
    """
    labels = load_label_list(cfg)
    data_train = _load_train_split(cfg, labels)
    apply_winsor_bounds(data_train, init_winsor_bounds(cfg, labels))

    counts = [int(data_train[l].notna().sum()) for l in labels]
    stds = torch.tensor([data_train[l].std() for l in labels]).float()  # ddof=1

    # A label with no train values never enters the loss, so its weight is never read.
    for i, n in enumerate(counts):
        if n == 0:
            stds[i] = 1.0

    degenerate = [
        (l, s, n) for l, s, n in zip(labels, stds.tolist(), counts)
        if n > 0 and (not np.isfinite(s) or s <= 0.0)
    ]
    if degenerate:
        raise ValueError(
            "Cannot build 1/sigma^2 loss weights: these labels have train values but a "
            f"non-finite or zero std (label, std, n): {degenerate}. Remove them from the "
            "phenotype list."
        )

    if verbose:
        print(f'Per-task std for 1/sigma^2 loss weighting ({len(labels)} labels):')
        for l, s, n in sorted(zip(labels, stds.tolist(), counts), key=lambda t: -t[1]):
            print(f'  {l:<20} std={s:<12.4g} n={n}')

    return stds


def validate_var_per_task_cfg(cfg: DictConfig) -> None:
    """The variance-weighted loss is only defined for unnormalized (raw) targets:
    normalized targets already have unit variance, so it would double-correct."""
    if cfg.dataset.tab.normalization.use:
        raise ValueError(
            "train.l2_loss.use_var_per_task=true requires dataset.tab.normalization.use=false."
        )


def resolve_eval_dropped_phenotypes(cfg: DictConfig, labels: List[str]) -> set:
    """Phenotypes excluded from evaluation metrics because they are too rare to score.

    A phenotype is dropped when its study-level non-NaN sample count is below
    ``cfg.dataset.tab.min_eval_samples`` in any of the train / val / test splits.
    Returns an empty set when ``min_eval_samples`` is unset or <= 0.
    """
    n = cfg.dataset.tab.get('min_eval_samples', None)
    if not n or n <= 0:
        return set()

    dropped: set = set()
    for key in ('data_train', 'data_val', 'data_test'):
        path = cfg.dataset.paths.get(key, None)
        if not path or not os.path.exists(path):
            continue
        df = pd.read_csv(path).drop_duplicates('echo_study_id')
        for label in labels:
            count = int(df[label].notna().sum()) if label in df.columns else 0
            if count < n:
                dropped.add(label)
    return dropped
