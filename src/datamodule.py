import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .const import LABEL_DICT, NUM_CLASSES
from .dataset import ReIDAttrDataset


class ReIDAttrDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int | None = None,
        num_workers: int = 12,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size if batch_size is not None else 32
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.num_persons = NUM_CLASSES
        self.attr_info = LABEL_DICT

    @property
    def transform(self) -> T.Compose:
        return T.Compose(
            [
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True
                ),
            ]
        )

    def setup(self, stage):
        match stage:
            case "fit":
                self.train = ReIDAttrDataset(
                    f"{self.data_dir}/train.csv",
                    root_dir=self.data_dir,
                    transform=self.transform,
                )
                self.val = ReIDAttrDataset(
                    f"{self.data_dir}/val.csv",
                    root_dir=self.data_dir,
                    transform=self.transform,
                )
            case "validate":
                self.val = ReIDAttrDataset(
                    f"{self.data_dir}/val.csv",
                    root_dir=self.data_dir,
                    transform=self.transform,
                )
            case "test":
                self.test = ReIDAttrDataset(
                    f"{self.data_dir}/test.csv",
                    root_dir=self.data_dir,
                    transform=self.transform,
                )
            case _:
                raise Exception("Not supported stage")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,  # 배치 크기 일관성을 위해
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        ...
