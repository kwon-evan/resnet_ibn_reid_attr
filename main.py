import argparse
from argparse import Namespace

import lightning as L
from lightning.pytorch.callbacks import (
    BatchSizeFinder,
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from rich import print

from src.const import ATTR_INFO, DATA_ROOT, NUM_CLASSES
from src.datamodule import ReIDAttrDataModule
from src.lightning_system import LightningSystem


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="ResNet-IBN Re-ID + Attribute Recognition"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "validate", "test", "inference"],
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    return parser.parse_args()


def set_callbacks(args: Namespace) -> list[Callback]:
    callbacks = [
        RichProgressBar(),
    ]

    if args.mode == "train":
        callbacks += [
            EarlyStopping(monitor="val_loss", mode="min", patience=5),
            ModelCheckpoint(
                dirpath="checkpoints",
                monitor="val_loss",
                mode="min",
                filename="resnet-ibn-{epoch}-{val_loss:.2f}-{val_avg_accuracy:.2f}-{val_mAP:.2f}",
                save_top_k=3,
            ),
        ]

        if args.batch_size is None:
            callbacks.append(BatchSizeFinder())
            print("[INFO] Using BatchSizeFinder to determine optimal batch size")
        else:
            print(f"[INFO] Using batch size: {args.batch_size}")

    return callbacks


def main():
    args = parse_args()

    print(f"[INFO] Set Seed To {L.seed_everything(42)}")
    print(f"[INFO] {NUM_CLASSES} persons, {len(ATTR_INFO)} attributes")
    print(f"[INFO] Mode: {args.mode}")

    dm = ReIDAttrDataModule(DATA_ROOT, batch_size=args.batch_size)
    ls = LightningSystem(num_persons=NUM_CLASSES, attr_info=ATTR_INFO)

    callbacks = set_callbacks(args)
    trainer = L.Trainer(
        precision="16-mixed",
        callbacks=callbacks,
    )

    match args.mode:
        case "train":
            trainer.fit(ls, dm)
        case "validate":
            trainer.validate(ls, dm, ckpt_path=args.ckpt_path)
        case "test":
            trainer.test(ls, dm, ckpt_path=args.ckpt_path)
        case "inference":
            trainer.predict(ls, dm, ckpt_path=args.ckpt_path)


if __name__ == "__main__":
    main()
