"""Dataset for EchoTab pretraining.

Each sample is one echo study paired with the structured measurements of its report.
Measurements are winsorized (and optionally z-score normalised) with statistics fit
on the train split; missing values are NaN.

Echo views are returned as a per-view-type dict {view_type: (T, H, W, C)} so that
EchoBranch can process each canonical view independently. Use study_collate_fn as the
DataLoader collate_fn.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from echotab.util.echoprime import VIEW_TYPES, load_echoprime_clip
from echotab.util.init_stats import apply_winsor_bounds, init_mean_std, init_winsor_bounds, load_label_list


class EchoTabDataset(Dataset):
    """Study-level dataset for EchoTab pretraining.

    __getitem__ returns:
        echo_study_id : str
        study         : dict[view_type -> (T, H, W, C) float32], present views only
        tabular_values: (N,) float32, NaN where missing
    """

    def __init__(self, cfg: DictConfig, mode: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        self.data = pd.read_csv(cfg.dataset.paths[f"data_{mode}"])
        self.study_ids = self.data["echo_study_id"].unique().tolist()

        # study_id -> {view_type: [filepaths]}, precomputed once so __getitem__ never
        # scans the full DataFrame.
        self._study_views: Dict[str, Dict[str, List[str]]] = {
            study_id: {
                view_type: view_group["echo_filepath"].tolist()
                for view_type, view_group in study_group.groupby("echo_view")
            }
            for study_id, study_group in self.data.groupby("echo_study_id")
        }

        self.excluded_view_types: List[str] = list(cfg.dataset.echo.get("excluded_view_types", None) or [])
        if self.excluded_view_types:
            allowed_views = set(VIEW_TYPES) - set(self.excluded_view_types)
            view_groups = self.data.groupby("echo_study_id")["echo_view"].apply(set)
            n_before = len(self.study_ids)
            self.study_ids = [
                sid for sid in self.study_ids
                if view_groups.get(sid, set()) & allowed_views
            ]
            if n_before - len(self.study_ids):
                print(
                    f"Pre-filtered {n_before - len(self.study_ids)} {mode} studies with no "
                    f"remaining views after excluding {self.excluded_view_types}"
                )

        self.label_list: List[str] = load_label_list(cfg)
        self._build_tabular_data()

    def _build_tabular_data(self) -> None:
        """Winsorize (+ optionally z-score) the measurements with train-split statistics."""
        apply_winsor_bounds(
            self.data, init_winsor_bounds(self.cfg, self.label_list, verbose=True)
        )

        if self.cfg.dataset.tab.normalization.use:
            means, stds = init_mean_std(self.cfg)  # both (1, N)
            for i, col in enumerate(self.label_list):
                self.data[col] = (self.data[col] - float(means[0, i])) / float(stds[0, i])
        else:
            print("No label normalisation applied")

        self._tab_lookup = (
            self.data[["echo_study_id"] + self.label_list]
            .drop_duplicates("echo_study_id")
            .set_index("echo_study_id")
        )

    def __len__(self) -> int:
        return len(self.study_ids)

    def __getitem__(self, idx: int) -> dict:
        study_id = self.study_ids[idx]
        row = self._tab_lookup.loc[study_id, self.label_list]
        return {
            "echo_study_id": study_id,
            "study": self._load_study(study_id),
            "tabular_values": torch.tensor(row.to_numpy(dtype=np.float32), dtype=torch.float32),  # (N,), NaN = missing
        }

    def _load_study(self, study_id: str) -> Dict[str, torch.Tensor]:
        """Return {view_type: (T, H, W, C)} for every canonical view present in this study."""
        study_views = self._study_views.get(study_id, {})
        study: Dict[str, torch.Tensor] = {}
        for view_type in VIEW_TYPES:
            if view_type in self.excluded_view_types:
                continue
            filepaths = study_views.get(view_type)
            if not filepaths:
                continue
            study[view_type] = load_echoprime_clip(
                filepaths[self._pick_clip_idx(len(filepaths))],
                sampling_rate=self.cfg.dataset.echo.sampling_rate,
                num_frames=self.cfg.model.echo.num_frames,
                img_size=self.cfg.model.echo.img_size,
            )
        return study

    def _pick_clip_idx(self, n_clips: int) -> int:
        """Training samples a random clip per view type (a study's clips are spread
        across epochs); evaluation always takes the first one, for reproducibility."""
        if self.mode != "train" or n_clips == 1:
            return 0
        return int(torch.randint(n_clips, (1,)).item())


def study_collate_fn(batch: List[dict]) -> dict:
    """Merge study-level samples into one batch.

        study[view_type]            : (B, ...) — zero-filled for samples without that view
        view_present_mask[view_type]: (B,) bool — True where the sample has that view

    Only view types that appear in at least one sample enter the study dict. All other
    tensor-valued keys are stacked; the remaining keys are collected into lists.
    """
    all_views: List[str] = list(dict.fromkeys(v for sample in batch for v in sample["study"]))
    template = next(v for sample in batch for v in sample["study"].values())

    study: Dict[str, torch.Tensor] = {}
    view_present_mask: Dict[str, torch.Tensor] = {}
    for view in all_views:
        study[view] = torch.stack([
            s["study"][view] if view in s["study"] else torch.zeros_like(template)
            for s in batch
        ])
        view_present_mask[view] = torch.tensor([view in s["study"] for s in batch], dtype=torch.bool)

    collated = {"study": study, "view_present_mask": view_present_mask}
    for key in batch[0]:
        if key == "study":
            continue
        values = [s[key] for s in batch]
        collated[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else values
    return collated
