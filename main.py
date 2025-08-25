import lightning as L
from lightning.pytorch.callbacks import BatchSizeFinder, EarlyStopping, RichProgressBar
from rich import print

from src.const import ATTR_INFO, DATA_ROOT, NUM_CLASSES
from src.datamodule import ReIDAttrDataModule
from src.lightning_system import LightningSystem


def main():
    print(f"[INFO] Set Seed To {L.seed_everything(42)}")
    print(f"[INFO] {NUM_CLASSES} persons, {len(ATTR_INFO)} attributes")

    dm = ReIDAttrDataModule(DATA_ROOT)
    ls = LightningSystem(num_persons=NUM_CLASSES, attr_info=ATTR_INFO)

    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=5),
        BatchSizeFinder(),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        precision="16-mixed",
        callbacks=callbacks,
    )
    trainer.fit(ls, dm)


if __name__ == "__main__":
    main()
