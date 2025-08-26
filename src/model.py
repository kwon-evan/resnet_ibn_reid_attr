import torch
import torch.nn as nn


class ResIBNReIDAttr(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_persons: int,
        attr_info: dict[str, tuple[str, int]],
        embed_dim: int = 512,
    ):
        """
        backbone: feature extractor (resnet)
        num_persons: Re-ID 클래스 개수
        attr_info: {attr_name: ("binary"/"multi", num_classes)}
        embed_dim: embedding 크기
        """
        super().__init__()
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # 마지막 FC 제거
        self.gap = nn.AdaptiveAvgPool2d(1)

        in_features = backbone.fc.in_features  # resnet 마지막 conv output dim

        # Embedding head (for Re-ID)
        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # ID classifier
        self.id_classifier = nn.Linear(embed_dim, num_persons)

        # Attribute classifiers
        self.attr_heads = nn.ModuleDict()
        for attr, (atype, ncls) in attr_info.items():
            if atype == "binary":
                self.attr_heads[attr] = nn.Linear(embed_dim, 1)
            elif atype == "multi":
                self.attr_heads[attr] = nn.Linear(embed_dim, ncls)
            elif atype == "regression":
                self.attr_heads[attr] = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # 1. Backbone feature
        feat_map = self.backbone(x)  # [B, C, H, W]
        pooled = self.gap(feat_map).flatten(1)  # [B, C]

        # 2. Embedding
        embedding = self.embedding_head(pooled)  # [B, D]

        # 3. ID classification
        logits_id = self.id_classifier(embedding)

        # 4. Attribute classification
        attr_logits = {}
        for attr, head in self.attr_heads.items():
            attr_logits[attr] = head(embedding)

        return embedding, logits_id, attr_logits


if __name__ == "__main__":
    backbone = torch.hub.load("XingangPan/IBN-Net", "resnet50_ibn_a", pretrained=True)
    attr_info = {
        "is_male": ("binary", 2),
        "hair_type": ("multi", 8),
        "hair_color": ("multi", 4),
        "upperclothes": ("multi", 3),
        "upperclothes_color": ("multi", 29),
        "lowerclothes": ("multi", 5),
        "lowerclothes_color": ("multi", 22),
    }
    num_persons = 1005
    model = ResIBNReIDAttr(backbone, num_persons, attr_info)
    print(model)

    x = torch.rand(2, 3, 256, 128)
    embedding, logits_id, attr_logits = model(x)
    print(embedding.shape, logits_id.shape)
    for attr, logits in attr_logits.items():
        print(attr, logits.shape)
