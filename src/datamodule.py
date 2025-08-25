import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .const import LABEL_DICT, NUM_CLASSES
from .dataset import ReIDAttrDataset


class ReIDAttrDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size

        self.num_persons = NUM_CLASSES
        self.attr_info = LABEL_DICT

    @property
    def transform(self) -> T.Compose:
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), inplace=True
                ),
                T.Resize((256, 128)),
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
            self.train, num_workers=12, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val, num_workers=12, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, num_workers=12, batch_size=self.batch_size)

    def on_exception(self, exception):
        # clean up state after the trainer faced an exception
        ...
