"""Measurement-token features of a pretrained EchoTab model, for linear probing.

The frozen EchoTabEncoder maps an echo study to one token per measurement. A study is
represented by the mean of a subset of these tokens:

  measurement_tokens     — the tokens of the measurements mapped to the target
                           diagnosis (``downstream_task.paths.diagnosis_measurement_map``)
  measurement_tokens_all — the tokens of all measurements
"""

import os
from typing import Callable, List

import torch
from omegaconf import DictConfig, OmegaConf

from echotab.models.EchoTabEncoder import EchoTabEncoder
from echotab.util.diagnosis_measurement_map import load_diagnosis_measurement_map

FEATURE_SOURCES = ('measurement_tokens', 'measurement_tokens_all')


def load_pretrained_encoder(cfg: DictConfig) -> EchoTabEncoder:
    """Rebuild the EchoTabEncoder of a pretraining run and restore its weights.

    The architecture is read from the run's own config.yaml (``{echotab_run_dir}/config.yaml``),
    the weights from its per-component checkpoints (``{echotab_run_dir}/checkpoints/*.ckpt``).
    Only the clip shape (num_frames / sampling_rate) is taken from the downstream task.
    """
    run_dir = cfg.echotab_run_dir
    pretrain_cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(pretrain_cfg_path):
        raise FileNotFoundError(
            f"Could not find the pretraining run's config.yaml at {pretrain_cfg_path}."
        )
    pretrain_cfg = OmegaConf.load(pretrain_cfg_path)

    OmegaConf.set_struct(pretrain_cfg, False)
    pretrain_cfg.model.echo.num_frames = cfg.downstream_task.num_frames
    pretrain_cfg.dataset.echo.sampling_rate = cfg.downstream_task.sampling_rate
    # The EchoPrime weights are restored from echo_branch.ckpt below.
    pretrain_cfg.train.encoder.echo.checkpoint_path = None
    # The phenotype list only provides the measurement names/order. It can be overridden,
    # e.g. when the path stored in the pretraining config does not exist on this machine.
    if cfg.get('label_list_path', None):
        pretrain_cfg.dataset.tab.label_list_path = cfg.label_list_path
    OmegaConf.set_struct(pretrain_cfg, True)

    encoder = EchoTabEncoder(pretrain_cfg)
    encoder.load_component_checkpoints(os.path.join(run_dir, "checkpoints"))
    return encoder


def build_feature_fn(cfg: DictConfig, encoder: EchoTabEncoder) -> Callable[[dict], dict]:
    """Returns feature_fn(batch) -> {'features': (B, D), 'label': ...}."""
    feature_source = cfg.downstream_task.feature_source
    if feature_source not in FEATURE_SOURCES:
        raise ValueError(f"downstream_task.feature_source must be one of {FEATURE_SOURCES}, got {feature_source!r}")

    if feature_source == 'measurement_tokens_all':
        names: List[str] = list(encoder.labels)
    else:
        diagnosis_map = load_diagnosis_measurement_map(cfg.downstream_task.paths.diagnosis_measurement_map)
        target = cfg.downstream_task.target
        if target not in diagnosis_map:
            raise ValueError(
                f"Diagnosis {target!r} not found in {cfg.downstream_task.paths.diagnosis_measurement_map}."
            )
        names = diagnosis_map[target]

    missing = [n for n in names if n not in encoder.labels]
    if missing:
        raise ValueError(f"Measurement(s) {missing} are not part of the pretrained model's measurements: {encoder.labels}")
    indices = torch.tensor([encoder.labels.index(n) for n in names])
    print(f"[{feature_source}] measurements={names} -> token indices={indices.tolist()}")

    def feature_fn(batch: dict) -> dict:
        device = next(encoder.parameters()).device
        study = {vt: v.to(device) for vt, v in batch['study'].items()}
        view_present_mask = {vt: m.to(device) for vt, m in batch['view_present_mask'].items()}

        z_out, _ = encoder(study, view_present_mask)                     # (B, N, D)
        features = z_out.index_select(dim=1, index=indices.to(device)).mean(dim=1)  # (B, D)
        return {'features': features, 'label': batch['label']}

    return feature_fn
