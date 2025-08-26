import torch.nn as nn


class ReIDAttrLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        attr_info: dict[str, tuple[str, int]],
        margin: float = 0.3,
        alpha: float = 1.0,
        regression_weight: float = 1.0,
    ):
        """
        num_classes: Re-ID 클래스 개수 (사람 ID 수)
        attr_info: {attr_name: ("binary" or "multi", num_classes)}
        margin: triplet loss margin
        alpha: attribute loss 가중치
        regression_weight: regression loss 가중치 (MSE loss 스케일링)
        """
        super().__init__()
        self.id_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)

        # 속성별 Loss 저장
        self.attr_losses = {}
        for attr, (atype, ncls) in attr_info.items():
            if atype == "binary":
                self.attr_losses[attr] = nn.BCEWithLogitsLoss()
            elif atype == "multi":
                self.attr_losses[attr] = nn.CrossEntropyLoss()
            elif atype == "regression":
                self.attr_losses[attr] = nn.MSELoss()

        self.alpha = alpha
        self.regression_weight = regression_weight
        self.attr_info = attr_info  # attr_info 저장

    def forward(self, embeddings, logits_id, triplet_data, attr_logits, attr_labels, is_train=True):
        """
        embeddings: [B, D] feature vector
        logits_id: [B, num_classes] Re-ID classifier output
        triplet_data: (anchor, positive, negative) 인덱스 (optional)
        attr_logits: {attr_name: tensor}, 각 attr의 예측값
        attr_labels: {attr_name: tensor}, 각 attr의 정답 라벨
        is_train: 학습 모드인지 여부 (False면 id_loss 제외)
        """
        # 1. ID Loss
        id_labels = attr_labels["person_id"]
        loss_id = self.id_loss(logits_id, id_labels)

        # 2. Triplet Loss (선택적)
        loss_tri = 0.0
        if triplet_data is not None:
            a, p, n = triplet_data
            loss_tri = self.triplet_loss(embeddings[a], embeddings[p], embeddings[n])

        # 3. Attribute Loss
        loss_attr = 0.0
        for attr, logits in attr_logits.items():
            labels = attr_labels[attr]
            if attr in self.attr_losses:
                attr_type, _ = self.attr_info[attr]
                
                if isinstance(self.attr_losses[attr], nn.BCEWithLogitsLoss):
                    labels = labels.float().unsqueeze(1)  # binary → float
                    loss_attr += self.attr_losses[attr](logits, labels)
                elif isinstance(self.attr_losses[attr], nn.MSELoss):
                    labels = labels.float().unsqueeze(1)  # regression → float
                    # Regression loss에 가중치 적용
                    loss_attr += self.regression_weight * self.attr_losses[attr](logits, labels)
                else:
                    loss_attr += self.attr_losses[attr](logits, labels)

        # 최종 Loss (평가 모드에서는 id_loss 제외)
        if is_train:
            total_loss = loss_id + loss_tri + self.alpha * loss_attr
        else:
            total_loss = loss_tri + self.alpha * loss_attr
            
        return {
            "loss": total_loss,
            "id_loss": loss_id,
            "tri_loss": loss_tri,
            "attr_loss": loss_attr,
        }
