"""
EchoTab: pretraining that jointly encodes multi-view echocardiography studies and
reconstructs the structured measurements of the corresponding echo report.

Architecture:
  EchoTabEncoder
    EchoBranch     — per-view EchoPrime encoding -> num_views*K echo tokens
    TabularBranch  — one learnable query per measurement
    TabDecoder     — measurement queries cross-attend over the echo tokens
  PredictionHead   — one scalar prediction per measurement token

Training objective: MSE between predicted and observed measurements (missing values
are masked out), optionally weighted by 1/sigma_j^2 per measurement.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig, OmegaConf

from echotab.models.EchoTabEncoder import EchoTabEncoder
from echotab.models.RegressionLoggingMixin import RegressionLoggingMixin
from echotab.models.prediction_head import PredictionHead
from echotab.util.init_stats import (
    init_mean_std,
    init_task_stds,
    resolve_eval_dropped_phenotypes,
    validate_var_per_task_cfg,
)
from echotab.util.metrics import init_metrics_fn
from echotab.util.model import HookDrivenModelCheckpoint, get_grad_norm_
from echotab.util.optimizer import create_optimizer_and_scheduler


class EchoTab(RegressionLoggingMixin, L.LightningModule):

    def __init__(self, cfg: DictConfig, save_dir: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.save_dir = save_dir
        self.batch_size = cfg.dataset.batch_size

        self.encoder = EchoTabEncoder(cfg)
        self.labels = self.encoder.labels

        print('Initializing the prediction head...')
        head_cfg = cfg.model.prediction_head
        self.prediction_head = PredictionHead(
            head_type=head_cfg.head_type,
            echo_dim=self.encoder.echo_branch.echo_dim,
            num_features=self.encoder.n_features,
            num_head_layers=head_cfg.num_head_layers,
            head_dropout=head_cfg.head_dropout,
            cond_dim=cfg.model.tab.embed_dim,
            cond_proj_dim=head_cfg.cond_proj_dim,
            lora_rank=head_cfg.lora_rank,
            lora_alpha=head_cfg.lora_alpha,
            hyper_hidden=head_cfg.hyper_hidden,
            hypernet_size=head_cfg.hypernet_size,
        )

        # (1, N) stats used to report metrics / predictions in raw units.
        self.mean, self.std = init_mean_std(cfg)

        # Variance-weighted reconstruction loss: dividing each squared error by the
        # measurement's train variance lets every measurement contribute on its own
        # scale rather than in proportion to its units squared.
        self.use_var_per_task = cfg.train.l2_loss.use_var_per_task
        if self.use_var_per_task:
            validate_var_per_task_cfg(cfg)
            # Not persistent: derived from the train split and rebuilt on every load.
            self.register_buffer('target_var', init_task_stds(cfg).pow(2), persistent=False)

        # Phenotypes too rare to score reliably are excluded from the evaluation metrics.
        self.eval_dropped_phenotypes = resolve_eval_dropped_phenotypes(cfg, self.labels)
        if self.eval_dropped_phenotypes:
            print(f'[eval] Dropping {len(self.eval_dropped_phenotypes)} rare phenotypes from '
                  f'evaluation metrics: {sorted(self.eval_dropped_phenotypes)}')

        self.is_final_eval = False

    def setup(self, stage: str) -> None:
        self.train_metrics = {
            l: init_metrics_fn('regression', self.device, 1) for l in self.labels
        }

    # ------------------------------------------------------------------
    def forward(
        self,
        study: Dict[str, torch.Tensor],
        view_present_mask: Dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Predict every measurement from the echo study: (B, N), normalised units."""
        z_out, _ = self.encoder(study, view_present_mask)  # (B, N, D)
        cond_emb = (
            self.encoder.tab_branch.conditioning_emb()
            if self.prediction_head.needs_conditioning else None
        )
        return self.prediction_head(z_out, cond_emb)

    def compute_loss(self, pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Masked MSE over the observed measurements (targets are NaN where missing)."""
        valid = ~torch.isnan(targets)
        if not valid.any():
            return (pred * 0.0).sum()  # zero but differentiable
        if self.use_var_per_task:
            # sum_bj m_bj (pred - y)^2 / sigma_j^2  /  sum_bj m_bj
            sigma_sq = self.target_var.unsqueeze(0).expand_as(pred)
            diff = pred[valid] - targets[valid]
            return (diff.pow(2) / sigma_sq[valid]).mean()
        return F.mse_loss(pred[valid], targets[valid])

    def _shared_step(self, batch: dict):
        targets = batch["tabular_values"]  # (B, N), NaN where missing
        # Autocast is disabled by default (train.use_autocast=false), i.e. the forward
        # pass runs in fp32 even under the bf16-mixed trainer precision.
        with torch.autocast("cuda", enabled=self.cfg.train.use_autocast):
            pred = self.forward(batch["study"], batch.get("view_present_mask"))
            loss = self.compute_loss(pred, targets)
        return loss, pred.detach(), targets, torch.isnan(targets)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, pred, targets, label_mask = self._shared_step(batch)
        self.log("train_loss/loss", loss, on_step=True, on_epoch=True, logger=True,
                 batch_size=self.batch_size, prog_bar=True, sync_dist=True)

        for idx, label in enumerate(self.labels):
            self.train_metrics[label].update(pred[:, idx], targets[:, idx], label_mask[:, idx])
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        for name, module in (
            ("echo_branch", self.encoder.echo_branch),
            ("tab_branch", self.encoder.tab_branch),
            ("tab_decoder", self.encoder.tab_decoder),
            ("prediction_head", self.prediction_head),
        ):
            self.log(f"train_loss/grad_norm_{name}", get_grad_norm_(module.parameters(), norm_type=2.0),
                     on_step=True, on_epoch=True, logger=True, batch_size=self.batch_size, sync_dist=True)

    def on_train_epoch_end(self) -> None:
        r2s = []
        for label, metric_fn in self.train_metrics.items():
            r2 = metric_fn.compute()['r2']
            r2 = r2.mean().item() if isinstance(r2, torch.Tensor) else float(r2)
            self.log(f'train_metrics/{label}_r2', r2, batch_size=self.batch_size)
            r2s.append(r2)
            metric_fn.reset()
        self.log('train_metrics/mean_r2', np.nanmean(r2s) if r2s else float('nan'), batch_size=self.batch_size)

    # ------------------------------------------------------------------
    # Validation / test
    # ------------------------------------------------------------------
    def _reset_eval_buffers(self) -> None:
        self._eval_study_ids, self._eval_preds, self._eval_labels = [], [], []

    def _eval_step(self, batch: dict, loss_prefix: str) -> torch.Tensor:
        loss, pred, targets, _ = self._shared_step(batch)
        self.log(f"{loss_prefix}/loss", loss, on_step=False, on_epoch=True, logger=True,
                 batch_size=self.batch_size, prog_bar=True, sync_dist=True)
        self._eval_study_ids.extend(batch["echo_study_id"])
        self._eval_preds.append(pred.cpu())
        self._eval_labels.append(targets.cpu())
        return loss

    def _eval_epoch_end(self, metrics_prefix: str, file_prefix: str | None) -> None:
        preds = torch.cat(self._eval_preds, dim=0).float()
        labels = torch.cat(self._eval_labels, dim=0).float()
        label_masks = torch.isnan(labels)

        metrics_dict = self._compute_per_study_metrics(preds, labels, label_masks)
        self._log_regression_metrics(metrics_dict, prefix=metrics_prefix)

        if file_prefix is not None:
            self._save_metrics_dataframe(metrics_dict, f'metrics_{file_prefix}.csv')
            self._save_predictions_dataframe(
                self._eval_study_ids, preds, labels, label_masks, f'predictions_{file_prefix}.csv'
            )

    def on_validation_epoch_start(self) -> None:
        self._reset_eval_buffers()

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, loss_prefix="val_loss")

    def on_validation_epoch_end(self) -> None:
        if self.is_final_eval:
            self._eval_epoch_end('final_val_metrics', file_prefix='final_val')
        else:
            self._eval_epoch_end('val_metrics', file_prefix=None)

    def on_test_epoch_start(self) -> None:
        self._reset_eval_buffers()

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, loss_prefix="test_loss")

    def on_test_epoch_end(self) -> None:
        self._eval_epoch_end('final_test_metrics', file_prefix='final_test')

    # ------------------------------------------------------------------
    # Optimizers / callbacks
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        model_group_dict = {
            "echo_branch": self.encoder.echo_branch,
            "tab_branch": self.encoder.tab_branch,
            "tab_decoder": self.encoder.tab_decoder,
            "prediction_head": self.prediction_head,
        }
        if not isinstance(self.encoder.echo_projection, nn.Identity):
            model_group_dict["echo_projection"] = self.encoder.echo_projection
        if not isinstance(self.encoder.tab_projection, nn.Identity):
            model_group_dict["tab_projection"] = self.encoder.tab_projection

        optimizer, scheduler = create_optimizer_and_scheduler(
            model_group_dict=model_group_dict,
            optimizer_params=self.cfg.train.params,
            num_opt_steps_per_epoch=self.num_steps(),
        )
        if scheduler is not None:
            return [optimizer], [scheduler]
        return optimizer

    def configure_callbacks(self) -> Sequence[Callback] | Callback:
        callbacks = [
            HookDrivenModelCheckpoint(
                dirpath=os.path.join(self.save_dir, "checkpoints"),
                auto_insert_metric_name=True,
                monitor=self.cfg.metric_to_monitor,
                mode=self.cfg.metric_mode,
                save_top_k=self.cfg.save_top_k,
                every_n_epochs=self.cfg.check_model_every_n_epochs,
                save_on_train_epoch_end=False,
                save_weights_only=False,
                verbose=True,
                save_last=False,
            ),
        ]
        if self.cfg.train.params.early_stopping.use:
            callbacks.append(
                EarlyStopping(
                    monitor=self.cfg.metric_to_monitor,
                    mode=self.cfg.metric_mode,
                    patience=self.cfg.train.params.early_stopping.patience,
                )
            )
        return callbacks

    def num_steps(self) -> int:
        """Number of optimizer steps per epoch."""
        dataset_size = len(self.trainer.fit_loop._data_source.dataloader())
        num_devices = max(1, self.trainer.num_devices)
        return dataset_size // (self.trainer.accumulate_grad_batches * num_devices)

    # ------------------------------------------------------------------
    # Checkpoint saving
    # ------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Besides the Lightning checkpoint, save each component separately (used by
        the KNN probe, which only needs the encoder) and the run config."""
        ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        meta = {"cfg": self.cfg, "epoch": self.current_epoch}

        self.encoder.save_component_checkpoints(ckpt_dir, meta)
        torch.save(
            {"model": self.prediction_head.state_dict(), **meta},
            os.path.join(ckpt_dir, "prediction_head.ckpt"),
        )
        OmegaConf.save(self.cfg, os.path.join(self.save_dir, "config.yaml"))
