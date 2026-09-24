from omegaconf import DictConfig

from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

from echotab.datasets.EchoTabDataModule import EchoTabDataModule
from echotab.models.EchoTab import EchoTab


def pretrain_echotab(
    cfg: DictConfig,
    wandb_logger: WandbLogger,
    save_dir: str,
    devices: int = 1,
):
    datamodule = EchoTabDataModule(cfg=cfg)
    model = EchoTab(cfg=cfg, save_dir=save_dir)

    trainer = Trainer(
        accelerator="gpu",
        devices=devices,
        strategy="ddp_find_unused_parameters_true" if devices > 1 else "auto",
        precision="bf16-mixed",
        logger=wandb_logger,
        max_epochs=cfg.max_epochs,
        accumulate_grad_batches=cfg.dataset.accum_iter,
        log_every_n_steps=cfg.log_every_n_steps,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        gradient_clip_val=cfg.train.gradient_clip_val,
        gradient_clip_algorithm=cfg.train.gradient_clip_algorithm,
        default_root_dir=save_dir,
        num_sanity_val_steps=0,
    )

    trainer.fit(model=model, datamodule=datamodule)

    # Final evaluation of the best checkpoint (saves metrics_*.csv / predictions_*.csv).
    model.is_final_eval = True
    trainer.validate(model=model, datamodule=datamodule, ckpt_path='best', weights_only=False)
    if cfg.test:
        trainer.test(model=model, datamodule=datamodule, ckpt_path='best', weights_only=False)
