import os
import shutil

import hydra
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from echotab.util.misc import set_seed


def configure_experiment_name(cfg: DictConfig) -> str:
    suffix = cfg.suffix or ''

    if cfg.train.task == 'pretrain_echotab':
        bs = f'B{cfg.dataset.batch_size * cfg.dataset.accum_iter * cfg.num_gpus}'
        e = f'e{cfg.max_epochs}'
        echo = f'echof{cfg.train.encoder.echo.freeze_first_n_layers}'
        echo += 'loc' if cfg.model.echo.preserve_local_tokens else ''
        echo += f'bg{cfg.dataset.echo.background_threshold}' if cfg.model.echo.use_bg_removal else ''
        dec = cfg.model.tab_decoder
        decoder = f'dec{dec.num_layers}h{dec.num_heads}'
        head_cfg = cfg.model.prediction_head
        head = f'{head_cfg.head_type}'
        if head_cfg.head_type == 'partial_shared':
            head += f'{head_cfg.hypernet_size}-R{head_cfg.lora_rank}a{head_cfg.lora_alpha}'
        head += f'-h{head_cfg.num_head_layers}'
        p = cfg.train.params
        train = f'lrEcho{p.lr_echo_branch}_lr{p.lr}_wd{p.weight_decay}'
        return f'{bs}_{e}_{echo}_{decoder}_{head}_{train}_{cfg.dataset.dataset_name}{suffix}'

    if cfg.train.task == 'knn':
        run_name = os.path.basename(os.path.normpath(cfg.echotab_run_dir))
        return f'{run_name}_{cfg.downstream_task.feature_source}{suffix}'

    raise ValueError(f'Unknown task: {cfg.train.task}')


def determine_train_function(cfg: DictConfig):
    if cfg.train.task == 'pretrain_echotab':
        from echotab.train_functions.pretrain_echotab import pretrain_echotab
        return pretrain_echotab, 'EchoTab-Pretrain'

    if cfg.train.task == 'knn':
        from echotab.train_functions.knn import knn
        task = cfg.downstream_task
        return knn, f'EchoTab-KNN-{task.dataset.capitalize()}-{task.target}'

    raise ValueError(f'Unknown task: {cfg.train.task}')


def flatten_config(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten a nested config dict into dot-separated keys for wandb filtering."""
    items = {}
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_config(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


@hydra.main(config_path='configs', config_name='base_pretrain_echotab', version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    train_function, project_name = determine_train_function(cfg)
    experiment_name = cfg.experiment_name or configure_experiment_name(cfg)

    save_dir = os.path.join(cfg.save_dir, project_name, experiment_name)
    if os.path.exists(os.path.join(save_dir, 'checkpoints')):
        print(f'{save_dir} already exists, deleting existing checkpoints')
        shutil.rmtree(os.path.join(save_dir, 'checkpoints'))
    os.makedirs(save_dir, exist_ok=True)

    torch.set_float32_matmul_precision('medium')

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb_logger = WandbLogger(
        project=project_name,
        name=experiment_name,
        save_dir=save_dir,
        config=flatten_config(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)),
        log_model=False,
        resume='never',
    )

    print(OmegaConf.to_yaml(cfg))
    try:
        train_function(cfg, wandb_logger, save_dir, devices=cfg.num_gpus)
    finally:
        if wandb_logger._experiment is not None:
            wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
