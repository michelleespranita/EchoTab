import math
from typing import Dict, List, Union

import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineStepScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_factor=0.1, last_step=-1):
        """
        min_lr_factor: scalar multiplier (typically between 0.0 and 1.0) that defines how
        low the learning rate can go relative to the base_lr.

        During warmup:        LR increases linearly from min_lr_factor x base_lr to base_lr.
        During cosine decay:  LR decreases from base_lr to min_lr_factor x base_lr.
        After total_steps:    LR remains fixed at min_lr_factor x base_lr.
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch=last_step)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            lr_scale = self.min_lr_factor + (1 - self.min_lr_factor) * (self.last_epoch / self.warmup_steps)
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        elif self.last_epoch <= self.total_steps:
            cos_step = self.last_epoch - self.warmup_steps
            cos_steps = self.total_steps - self.warmup_steps
            cos_theta = min(max(cos_step / cos_steps, 0), 1)
            return [base_lr * (self.min_lr_factor + (1 - self.min_lr_factor) * 0.5 * (1 + math.cos(math.pi * cos_theta))) for base_lr in self.base_lrs]
        else:
            return [base_lr * self.min_lr_factor for base_lr in self.base_lrs]


def _exclude_from_wd(name: str) -> bool:
    return 'bias' in name or 'layerscale' in name


def _get_layer_id_for_vit(name: str, num_layers: int) -> int:
    """Layer id of an echo-encoder parameter, following BEiT
    (https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33)."""
    if 'cls_token' in name or 'pos_embed' in name or 'patch_embed' in name:
        return 0  # gets the smallest lr
    parts = name.split('.')
    if 'blocks' in parts:
        return int(parts[parts.index('blocks') + 1])
    return num_layers - 1  # gets the base lr


def _num_encoder_blocks(module: nn.Module) -> int:
    """Number of transformer blocks of the echo encoder held by an EchoBranch."""
    enc = getattr(module, 'echo_encoder', module)
    if hasattr(enc, 'blocks'):
        return len(enc.blocks)
    if hasattr(enc, 'model') and hasattr(enc.model, 'blocks'):
        return len(enc.model.blocks)
    return 1


def define_param_groups(
    model_group_dict: Dict[str, Union[nn.Module, List[nn.Module]]],
    lr_rates: Dict[str, float],
    weight_decay: float,
    layer_decay: float,
) -> list:
    """One param group per parameter.

    Every group key gets its own lr (``lr_rates[group_key]``, falling back to
    ``lr_rates['lr']``). Layer decay is applied only to the echo-encoder parameters of
    the 'echo_branch' group.
    """
    param_groups = []

    for group_key, models in model_group_dict.items():
        if not isinstance(models, list):
            models = [models]

        for model in models:
            encoder_num_layers = _num_encoder_blocks(model) if group_key == 'echo_branch' else 1
            lr = lr_rates.get(group_key, lr_rates['lr'])
            lrs = [lr * (layer_decay ** (encoder_num_layers - (i + 1))) for i in range(encoder_num_layers)]

            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if encoder_num_layers > 1 and 'echo_encoder' in name:
                    layer_id = _get_layer_id_for_vit(name, encoder_num_layers)
                else:
                    layer_id = encoder_num_layers - 1
                param_groups.append({
                    'name': f'{group_key}.{name}',
                    'weight_decay': 0.0 if _exclude_from_wd(name) else weight_decay,
                    'lr': lrs[layer_id],
                    'params': p,
                })

    return param_groups


def create_optimizer_and_scheduler(model_group_dict, optimizer_params, num_opt_steps_per_epoch):
    """AdamW over per-component param groups, plus an optional per-step warmup-cosine
    schedule (warmup_steps / total_steps are given in epochs)."""
    lr_rates = {'lr': optimizer_params['lr']}
    for group_key in model_group_dict:
        lr_name = f'lr_{group_key}'
        if lr_name in optimizer_params:
            lr_rates[group_key] = optimizer_params[lr_name]

    param_groups = define_param_groups(
        model_group_dict=model_group_dict,
        lr_rates=lr_rates,
        weight_decay=optimizer_params['weight_decay'],
        layer_decay=optimizer_params['layer_decay'],
    )
    optimizer = AdamW(param_groups, betas=tuple(optimizer_params['betas']))

    sched_cfg = optimizer_params['scheduler']['warmup_cosine']
    if not sched_cfg['use']:
        return optimizer, None

    scheduler = {
        'scheduler': WarmupCosineStepScheduler(
            optimizer,
            min_lr_factor=sched_cfg['min_lr_factor'],
            warmup_steps=num_opt_steps_per_epoch * sched_cfg['warmup_steps'],
            total_steps=num_opt_steps_per_epoch * sched_cfg['total_steps'],
        ),
        'name': 'warmup_cosine',
        'interval': 'step',
    }
    return optimizer, scheduler
