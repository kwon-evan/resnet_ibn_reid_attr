DATA_ROOT = "/home/kwon/Datasets/Korean-ReID-Dataset/"

NUM_CLASSES = 1025

LABEL_DICT = {
    "is_male": [True, False],
    "hair_type": [
        "normal",
        "tied_hair",
        "straight",
        "permed",
        "long",
        "short",
        "sporting",
        "shaved",
    ],
    "hair_color": ["black", "brown", "yellow", "white"],
    "upperclothes": ["long_sleeve", "short_sleeve", "sleeveless"],
    "upperclothes_color": [
        "brown",
        "black",
        "navy",
        "white",
        "gray",
        "purple",
        "dark_gray",
        "green",
        "burgundy",
        "sky_blue",
        "blue",
        "pink",
        "blue_green",
        "yellow",
        "beige",
        "red",
        "navy_blue",
        "orange",
        "mint",
        "navy_green",
        "light_brown",
        "red_brown",
        "khaki",
        "yellow_green",
        "light_green",
        "soft_pink",
        "hot_pink",
        "olive",
        "ivory",
    ],
    "lowerclothes": ["long_pants", "short_pants", "long_skirt", "short_skirt", "dress"],
    "lowerclothes_color": [
        "black",
        "beige",
        "sky_blue",
        "blue",
        "navy",
        "white",
        "navy_green",
        "blue_green",
        "gray",
        "red_brown",
        "brown",
        "navy_blue",
        "dark_gray",
        "light_brown",
        "green",
        "red",
        "purple",
        "yellow",
        "pink",
        "khaki",
        "sky_blule",
        "orange",
    ],
}

ATTR_INFO = {
    a: ("multi", len(b)) if len(b) > 2 else ("binary", 2) for a, b in LABEL_DICT.items()
}

# 연속값 속성 추가 (regression)
ATTR_INFO.update({
    "age": ("regression", 1),
    "tall": ("regression", 1),
})

# Min-Max 정규화를 위한 범위값 (학습 데이터 기준)
REGRESSION_STATS = {
    "age": {"min": 17.0, "max": 65.0},
    "tall": {"min": 150.0, "max": 194.0},
}
