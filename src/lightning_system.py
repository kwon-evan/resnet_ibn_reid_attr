import lightning as L
import torch

from .loss import ReIDAttrLoss
from .model import ResIBNReIDAttr
from .metrics import ReIDMetrics, PARMetrics


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

        # 메트릭 초기화
        self.reid_metrics = ReIDMetrics(num_classes=num_persons)
        self.par_metrics = PARMetrics(attr_info=attr_info)

        # Validation과 Test에서 embeddings와 IDs를 저장할 리스트
        self.val_embeddings = []
        self.val_person_ids = []
        self.test_embeddings = []
        self.test_person_ids = []

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
            embeddings, logits_id, triplet_data, attr_logits, attrs, is_train=False
        )
        loss = {f"val_{k}": v for k, v in loss_dict.items()}
        self.log_dict(loss, prog_bar=True)

        self.val_embeddings.append(embeddings.detach())
        self.val_person_ids.append(person_ids.detach())
        self.reid_metrics.update_id_accuracy(logits_id, person_ids)
        self.par_metrics.update(attr_logits, attrs)

        return loss_dict["loss"]

    def test_step(self, batch, batch_idx):
        img, attrs = batch

        embeddings, logits_id, attr_logits = self.model(img)

        person_ids = attrs["person_id"]
        triplet_data = self._generate_triplet_data(person_ids)

        loss_dict = self.criterion(
            embeddings, logits_id, triplet_data, attr_logits, attrs, is_train=False
        )
        loss = {f"test_{k}": v for k, v in loss_dict.items()}
        self.log_dict(loss, prog_bar=True)

        self.test_embeddings.append(embeddings.detach())
        self.test_person_ids.append(person_ids.detach())
        self.par_metrics.update(attr_logits, attrs)

        return loss_dict["loss"]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=2e-4, total_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_validation_epoch_end(self):
        """Validation epoch 종료 시 메트릭 계산 및 로깅"""

        # Person ID accuracy
        id_acc = self.reid_metrics.compute_id_accuracy()
        self.log("val_id_accuracy", id_acc, prog_bar=True)

        # Attribute metrics
        attr_metrics = self.par_metrics.compute()
        for metric_name, metric_value in attr_metrics.items():
            self.log(f"val_{metric_name}", metric_value, prog_bar=True)

        # CMC and mAP (validation embeddings)
        if len(self.val_embeddings) > 1:
            all_embeddings = torch.cat(self.val_embeddings, dim=0)
            all_person_ids = torch.cat(self.val_person_ids, dim=0)

            # memory saving
            max_samples = 5000
            if len(all_embeddings) > max_samples:
                indices = torch.randperm(len(all_embeddings))[:max_samples]
                all_embeddings = all_embeddings[indices]
                all_person_ids = all_person_ids[indices]

            # split Gallery and Query
            mid_idx = len(all_embeddings) // 2
            query_embeddings = all_embeddings[:mid_idx]
            query_ids = all_person_ids[:mid_idx]
            gallery_embeddings = all_embeddings[mid_idx:]
            gallery_ids = all_person_ids[mid_idx:]

            if len(query_embeddings) > 0 and len(gallery_embeddings) > 0:
                cmc_scores, map_score = self.reid_metrics.compute_cmc_map(
                    query_embeddings, query_ids, gallery_embeddings, gallery_ids
                )

                self.log("val_mAP", map_score, prog_bar=True)
                for k, score in cmc_scores.items():
                    self.log(f"val_CMC@{k}", score, prog_bar=True)

        # reset metrics
        self.reid_metrics.reset()
        self.par_metrics.reset()
        self.val_embeddings.clear()
        self.val_person_ids.clear()

    def on_test_epoch_end(self):
        """Test epoch 종료 시 메트릭 계산 및 로깅"""

        # Attribute metrics
        attr_metrics = self.par_metrics.compute()
        for metric_name, metric_value in attr_metrics.items():
            self.log(f"test_{metric_name}", metric_value, prog_bar=True)

        if len(self.test_embeddings) > 1:
            all_embeddings = torch.cat(self.test_embeddings, dim=0)
            all_person_ids = torch.cat(self.test_person_ids, dim=0)

            # for memory saving
            max_samples = 5000
            if len(all_embeddings) > max_samples:
                indices = torch.randperm(len(all_embeddings))[:max_samples]
                all_embeddings = all_embeddings[indices]
                all_person_ids = all_person_ids[indices]

            # calculate CMC, mAP
            mid_idx = len(all_embeddings) // 2
            query_embeddings = all_embeddings[:mid_idx]
            query_ids = all_person_ids[:mid_idx]
            gallery_embeddings = all_embeddings[mid_idx:]
            gallery_ids = all_person_ids[mid_idx:]

            if len(query_embeddings) > 0 and len(gallery_embeddings) > 0:
                cmc_scores, map_score = self.reid_metrics.compute_cmc_map(
                    query_embeddings, query_ids, gallery_embeddings, gallery_ids
                )

                self.log("test_mAP", map_score, prog_bar=True)
                for k, score in cmc_scores.items():
                    self.log(f"test_CMC@{k}", score, prog_bar=True)

        # reset metrics
        self.reid_metrics.reset()
        self.par_metrics.reset()
        self.test_embeddings.clear()
        self.test_person_ids.clear()
