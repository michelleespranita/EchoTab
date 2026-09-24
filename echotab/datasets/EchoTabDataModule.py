import lightning as L
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from echotab.datasets.EchoTabDataset import EchoTabDataset, study_collate_fn


class EchoTabDataModule(L.LightningDataModule):
    """LightningDataModule for EchoTab pretraining."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.dataset.batch_size
        self.num_workers = cfg.dataset.num_workers
        self.pin_memory = cfg.dataset.pin_memory

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.dataset_train = EchoTabDataset(self.cfg, mode="train")
            self.dataset_val = EchoTabDataset(self.cfg, mode="val")
        elif stage == "validate":
            self.dataset_val = EchoTabDataset(self.cfg, mode="val")
        elif stage == "test":
            self.dataset_test = EchoTabDataset(self.cfg, mode="test")

    def _make_loader(self, dataset: EchoTabDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=study_collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.dataset_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.dataset_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.dataset_test, shuffle=False)
