import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig

from echotab.datasets.DownstreamEchoDataset import DownstreamEchoDataModule
from echotab.knn.KNN import KNN
from echotab.knn.feature_extraction import build_feature_fn, load_pretrained_encoder


def _flatten_metrics(metrics: dict, prefix: str) -> dict:
    out = {}
    for name, value in metrics.items():
        if torch.is_tensor(value) and value.dim() > 0:  # per-class breakdowns
            for i, v in enumerate(value):
                out[f'{prefix}/{name.replace("_per_output", "")}_output_{i}'] = float(v)
        else:
            out[f'{prefix}/{name}'] = float(value)
    return out


@torch.no_grad()
def knn(
    cfg: DictConfig,
    wandb_logger: WandbLogger,
    save_dir: str,
    devices: int = 1,
):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    datamodule = DownstreamEchoDataModule(cfg=cfg)
    datamodule.setup(stage='knn')

    encoder = load_pretrained_encoder(cfg).to(device).eval()
    feature_fn = build_feature_fn(cfg, encoder)

    probe = KNN(
        num_classes=datamodule.num_classes,
        device=torch.device('cpu'),
        save_dir=save_dir,
    )
    probe.fit(datamodule.train_dataloader(), feature_fn)

    results = {}
    if cfg.validate:
        results.update(_flatten_metrics(probe.evaluate(datamodule.val_dataloader(), feature_fn, 'val'), 'final_val'))
    if cfg.test:
        datamodule.setup(stage='test')
        results.update(_flatten_metrics(probe.evaluate(datamodule.test_dataloader(), feature_fn, 'test'), 'test'))

    for k, v in results.items():
        print(f'{k}: {v:.4f}')
    wandb_logger.experiment.log(results)
