import os
from typing import Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from .const import LABEL_DICT, NUM_CLASSES


class ReIDAttrDataset(Dataset):
    def __init__(self, csv_file, root_dir="", transform=None):
        self.df = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.transform = transform

    @property
    def num_persons(self):
        return NUM_CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> tuple[Image.Image, dict[str, Sequence]]:
        def _convert_multi_label_to_onehot(key, val):
            return LABEL_DICT[key].index(val)

        row = self.df.iloc[idx]

        # 이미지 경로
        img_path = os.path.join(self.root_dir, row["path"])
        img = Image.open(img_path.replace(".xml", ".png")).convert(
            "RGB"
        )  # XML → PNG 치환

        # 속성(Attribute) 라벨 (필요하면 tensor로 변환해서 리턴)
        # @ID에서 숫자 부분 추출 (H00001 -> 1)
        raw_id = int(row["@ID"][1:])
        person_id = raw_id - 1

        attributes = {
            "person_id": person_id,
            "is_male": row["is_male"],
            "age": row["age"],
            "tall": row["tall"],
            "hair_type": _convert_multi_label_to_onehot("hair_type", row["hair_type"]),
            "hair_color": _convert_multi_label_to_onehot(
                "hair_color", row["hair_color"]
            ),
            "upperclothes": _convert_multi_label_to_onehot(
                "upperclothes", row["upperclothes"]
            ),
            "upperclothes_color": _convert_multi_label_to_onehot(
                "upperclothes_color", row["upperclothes_color"]
            ),
            "lowerclothes": _convert_multi_label_to_onehot(
                "lowerclothes", row["lowerclothes"]
            ),
            "lowerclothes_color": _convert_multi_label_to_onehot(
                "lowerclothes_color", row["lowerclothes_color"]
            ),
        }

        if self.transform:
            img = self.transform(img)

        return img, attributes


if __name__ == "__main__":
    from rich import print
    from rich.progress import track

    ROOT_DIR = "/home/kwon/Datasets/Korean-ReID-Dataset"
    # dataset = ReIDCsvDataset(
    #     f"{ROOT_DIR}/train.csv",
    #     root_dir=ROOT_DIR,
    #     transform=None,
    # )
    # print(dataset.num_persons)

    train_dataset = ReIDAttrDataset(
        f"{ROOT_DIR}/train.csv",
        root_dir=ROOT_DIR,
        transform=None,
    )
    val_dataset = ReIDAttrDataset(
        f"{ROOT_DIR}/val.csv",
        root_dir=ROOT_DIR,
        transform=None,
    )
    test_dataset = ReIDAttrDataset(
        f"{ROOT_DIR}/test.csv",
        root_dir=ROOT_DIR,
        transform=None,
    )

    for img, attrs in track(train_dataset, description="Loading train dataset"):
        pass

    for img, attrs in track(val_dataset, description="Loading val dataset"):
        pass

    for img, attrs in track(test_dataset, description="Loading test dataset"):
        pass

    print(len(train_dataset), train_dataset.num_persons)
    print(len(val_dataset), val_dataset.num_persons)
    print(len(test_dataset), test_dataset.num_persons)
