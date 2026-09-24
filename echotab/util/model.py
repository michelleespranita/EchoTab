from math import inf
import os

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(parameters)
    device = parameters[0].device if parameters else torch.device('cpu')
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0., device=device)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class HookDrivenModelCheckpoint(ModelCheckpoint):
    """
    Keep ModelCheckpoint's scheduling/monitoring behavior and save the usual
    Lightning checkpoint artifact, then delegate additional custom artifact
    writing (per-component checkpoints) to LightningModule.on_save_checkpoint().
    """

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        # All ranks must call super() so the barrier inside it is hit symmetrically.
        super()._save_checkpoint(trainer, filepath)

        if not trainer.is_global_zero:
            return

        checkpoint = {
            'epoch': trainer.current_epoch,
            'global_step': trainer.global_step,
        }

        if self.monitor is not None:
            monitor_value = trainer.callback_metrics.get(self.monitor)
            if isinstance(monitor_value, torch.Tensor):
                monitor_value = monitor_value.detach().cpu().item()
            checkpoint[self.monitor] = monitor_value

        trainer.lightning_module.on_save_checkpoint(checkpoint)
        self._last_global_step_saved = trainer.global_step

    def _remove_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        # We do custom saving in on_save_checkpoint and may not create the
        # callback-managed filepath at all.
        if os.path.exists(filepath):
            super()._remove_checkpoint(trainer, filepath)
