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
    Lightning checkpoint artifact, then write the per-component checkpoints once,
    on rank 0, via LightningModule.save_component_checkpoints().
    """

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        # All ranks must call super() so the barrier inside it is hit symmetrically.
        super()._save_checkpoint(trainer, filepath)

        if not trainer.is_global_zero:
            return

        trainer.lightning_module.save_component_checkpoints()
        self._last_global_step_saved = trainer.global_step

    def _remove_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        # Tolerate a callback-managed checkpoint file that no longer exists.
        if os.path.exists(filepath):
            super()._remove_checkpoint(trainer, filepath)
