import lightning as L
import torch

from .loss import ReIDAttrLoss
from .model import ResIBNReIDAttr


class LightningSystem(L.LightningModule):
    def __init__(
        self,
        num_persons: int,
        attr_info: dict[str, tuple[str, int]],
        embed_dim: int = 512,
    ):
        super().__init__()

        backbone = torch.hub.load(
            "XingangPan/IBN-Net", "resnet50_ibn_a", pretrained=True
        )
        if not isinstance(backbone, torch.nn.Module):
            raise ValueError("backbone must be torch.nn.Module")

        self.model = ResIBNReIDAttr(
            backbone=backbone,
            num_persons=num_persons,
            attr_info=attr_info,
            embed_dim=embed_dim,
        )
        self.criterion = ReIDAttrLoss(num_classes=num_persons, attr_info=attr_info)

    def _generate_triplet_data(self, person_ids):
        """
        배치 내에서 triplet (anchor, positive, negative) 인덱스를 생성
        """
        batch_size = person_ids.size(0)
        triplet_indices = []

        for i in range(batch_size):
            anchor_id = person_ids[i]

            positive_mask = (person_ids == anchor_id) & (
                torch.arange(batch_size, device=person_ids.device) != i
            )
            positive_indices = torch.where(positive_mask)[0]
            negative_mask = person_ids != anchor_id
            negative_indices = torch.where(negative_mask)[0]

            if len(positive_indices) > 0 and len(negative_indices) > 0:
                pos_idx = positive_indices[
                    torch.randint(0, len(positive_indices), (1,))
                ].item()
                neg_idx = negative_indices[
                    torch.randint(0, len(negative_indices), (1,))
                ].item()
                triplet_indices.append((i, pos_idx, neg_idx))

        if len(triplet_indices) == 0:
            return None

        anchors = torch.tensor(
            [t[0] for t in triplet_indices], device=person_ids.device
        )
        positives = torch.tensor(
            [t[1] for t in triplet_indices], device=person_ids.device
        )
        negatives = torch.tensor(
            [t[2] for t in triplet_indices], device=person_ids.device
        )

        return (anchors, positives, negatives)

    def training_step(self, batch, batch_idx):
        img, attrs = batch

        embeddings, logits_id, attr_logits = self.model(img)

        person_ids = attrs["person_id"]
        triplet_data = self._generate_triplet_data(person_ids)

        loss_dict = self.criterion(
            embeddings, logits_id, triplet_data, attr_logits, attrs
        )
        loss = {f"train_{k}": v for k, v in loss_dict.items()}
        self.log_dict(loss, prog_bar=True)

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        img, attrs = batch

        embeddings, logits_id, attr_logits = self.model(img)

        person_ids = attrs["person_id"]
        triplet_data = self._generate_triplet_data(person_ids)

        loss_dict = self.criterion(
            embeddings, logits_id, triplet_data, attr_logits, attrs
        )
        loss = {f"val_{k}": v for k, v in loss_dict.items()}
        self.log_dict(loss, prog_bar=True)

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        img, attrs = batch

        embeddings, logits_id, attr_logits = self.model(img)

        person_ids = attrs["person_id"]
        triplet_data = self._generate_triplet_data(person_ids)

        loss_dict = self.criterion(
            embeddings, logits_id, triplet_data, attr_logits, attrs
        )
        loss = {f"test_{k}": v for k, v in loss_dict.items()}
        self.log_dict(loss, prog_bar=True)

        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
