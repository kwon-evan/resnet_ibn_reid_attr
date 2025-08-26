import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torchmetrics.functional import accuracy


class ReIDMetrics:
    """Re-identification 작업을 위한 메트릭 클래스"""

    def __init__(self, num_classes: int, device: str = "cuda"):
        self.num_classes = num_classes
        self.device = device

        # Classification metrics for person ID
        self.id_accuracy = Accuracy(task="multiclass", num_classes=num_classes).to(
            device
        )

    def compute_cmc_map(
        self,
        embeddings: torch.Tensor,
        person_ids: torch.Tensor,
        gallery_embeddings: torch.Tensor,
        gallery_ids: torch.Tensor,
        k_values: list = [1, 5, 10, 20],
        batch_size: int = 100,
    ):
        """
        CMC (Cumulative Matching Characteristics)와 mAP (mean Average Precision) 계산
        메모리 효율적인 배치 처리로 구현

        Args:
            embeddings: Query embeddings [N, D]
            person_ids: Query person IDs [N]
            gallery_embeddings: Gallery embeddings [M, D]
            gallery_ids: Gallery person IDs [M]
            k_values: CMC를 계산할 k 값들
            batch_size: 배치 크기 (메모리 사용량 조절)
        """
        # CPU로 이동하여 메모리 절약
        embeddings = F.normalize(embeddings.cpu(), p=2, dim=1)
        gallery_embeddings = F.normalize(gallery_embeddings.cpu(), p=2, dim=1)
        person_ids = person_ids.cpu()
        gallery_ids = gallery_ids.cpu()

        cmc_scores = {k: 0.0 for k in k_values}
        ap_scores = []

        num_queries = len(embeddings)

        # 배치 단위로 처리
        for start_idx in range(0, num_queries, batch_size):
            end_idx = min(start_idx + batch_size, num_queries)
            batch_embeddings = embeddings[start_idx:end_idx]
            batch_person_ids = person_ids[start_idx:end_idx]

            # 배치별 similarity 계산
            similarity_batch = torch.mm(batch_embeddings, gallery_embeddings.t())

            # 각 query에 대해 gallery를 similarity 순으로 정렬
            _, indices_batch = torch.sort(similarity_batch, dim=1, descending=True)

            for i in range(len(batch_embeddings)):
                query_id = batch_person_ids[i]

                # Gallery에서 같은 ID를 가진 인덱스들 찾기
                positive_mask = gallery_ids == query_id
                positive_indices = torch.where(positive_mask)[0]

                if len(positive_indices) == 0:
                    continue

                # 정렬된 인덱스에서 positive의 위치 찾기
                sorted_gallery_ids = gallery_ids[indices_batch[i]]
                matches = sorted_gallery_ids == query_id

                # CMC 계산
                for k in k_values:
                    if matches[:k].any():
                        cmc_scores[k] += 1.0

                # AP 계산
                match_positions = torch.where(matches)[0]
                if len(match_positions) > 0:
                    precisions = []
                    for j, pos in enumerate(match_positions):
                        precision_at_pos = (j + 1) / (pos + 1)
                        precisions.append(precision_at_pos)
                    ap_scores.append(sum(precisions) / len(precisions))

        # 평균 계산
        cmc_scores = {k: score / num_queries for k, score in cmc_scores.items()}
        map_score = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0

        return cmc_scores, map_score

    def update_id_accuracy(self, logits: torch.Tensor, targets: torch.Tensor):
        """Person ID classification accuracy 업데이트"""
        # prevent device mismatch
        if logits.device != self.id_accuracy.device:
            self.id_accuracy = self.id_accuracy.to(logits.device)
        self.id_accuracy.update(logits, targets)

    def compute_id_accuracy(self):
        """Person ID classification accuracy 계산"""
        return self.id_accuracy.compute()

    def reset(self):
        """모든 메트릭 리셋"""
        self.id_accuracy.reset()


class PARMetrics:
    """Person Attribute Recognition 작업을 위한 메트릭 클래스"""

    def __init__(self, attr_info: dict[str, tuple[str, int]], device: str = "cuda"):
        self.attr_info = attr_info
        self.device = device

        # 각 속성별 메트릭 초기화
        self.binary_metrics = {}
        self.multi_metrics = {}
        self.regression_metrics = {}

        for attr_name, (attr_type, num_classes) in attr_info.items():
            if attr_type == "binary":
                self.binary_metrics[attr_name] = {
                    "accuracy": Accuracy(task="binary").to(device),
                    "precision": Precision(task="binary").to(device),
                    "recall": Recall(task="binary").to(device),
                    "f1": F1Score(task="binary").to(device),
                }
            elif attr_type == "multi":
                self.multi_metrics[attr_name] = {
                    "accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(
                        device
                    ),
                    "precision": Precision(
                        task="multiclass", num_classes=num_classes, average="macro"
                    ).to(device),
                    "recall": Recall(
                        task="multiclass", num_classes=num_classes, average="macro"
                    ).to(device),
                    "f1": F1Score(
                        task="multiclass", num_classes=num_classes, average="macro"
                    ).to(device),
                }
            elif attr_type == "regression":
                self.regression_metrics[attr_name] = {
                    "mae": [],  # Mean Absolute Error
                    "mse": [],  # Mean Squared Error
                }

    def update(
        self, attr_logits: dict[str, torch.Tensor], attr_labels: dict[str, torch.Tensor]
    ):
        """모든 속성 메트릭 업데이트"""
        for attr_name, logits in attr_logits.items():
            if attr_name not in attr_labels:
                continue

            labels = attr_labels[attr_name]
            attr_type, _ = self.attr_info[attr_name]

            if attr_type == "binary":
                # Binary classification: sigmoid + threshold
                preds = torch.sigmoid(logits).squeeze() > 0.5
                labels = labels.float()

                # DDP 환경에서 디바이스 불일치 방지
                for metric in self.binary_metrics[attr_name].values():
                    if logits.device != metric.device:
                        metric.to(logits.device)
                    metric.update(preds, labels)

            elif attr_type == "multi":
                # Multi-class classification: softmax + argmax
                preds = torch.argmax(logits, dim=1)

                # DDP 환경에서 디바이스 불일치 방지
                for metric in self.multi_metrics[attr_name].values():
                    if logits.device != metric.device:
                        metric.to(logits.device)
                    metric.update(preds, labels)

            elif attr_type == "regression":
                # Regression: direct prediction
                preds = logits.squeeze()
                labels = labels.float()

                mae = torch.abs(preds - labels).mean().item()
                mse = ((preds - labels) ** 2).mean().item()

                self.regression_metrics[attr_name]["mae"].append(mae)
                self.regression_metrics[attr_name]["mse"].append(mse)

    def compute(self):
        """모든 속성 메트릭 계산"""
        results = {}

        # Binary attributes
        for attr_name, metrics in self.binary_metrics.items():
            for metric_name, metric in metrics.items():
                results[f"{attr_name}_{metric_name}"] = metric.compute()

        # Multi-class attributes
        for attr_name, metrics in self.multi_metrics.items():
            for metric_name, metric in metrics.items():
                results[f"{attr_name}_{metric_name}"] = metric.compute()

        # Regression attributes
        for attr_name, metrics in self.regression_metrics.items():
            if metrics["mae"]:
                results[f"{attr_name}_mae"] = torch.tensor(metrics["mae"]).mean()
            if metrics["mse"]:
                results[f"{attr_name}_mse"] = torch.tensor(metrics["mse"]).mean()
                results[f"{attr_name}_rmse"] = torch.sqrt(
                    torch.tensor(metrics["mse"]).mean()
                )

        # 전체 평균 계산 (classification만)
        all_accuracies = [v for k, v in results.items() if k.endswith("_accuracy")]
        all_precisions = [v for k, v in results.items() if k.endswith("_precision")]
        all_recalls = [v for k, v in results.items() if k.endswith("_recall")]
        all_f1s = [v for k, v in results.items() if k.endswith("_f1")]

        if all_accuracies:
            results["avg_accuracy"] = torch.stack(all_accuracies).mean()
        if all_precisions:
            results["avg_precision"] = torch.stack(all_precisions).mean()
        if all_recalls:
            results["avg_recall"] = torch.stack(all_recalls).mean()
        if all_f1s:
            results["avg_f1"] = torch.stack(all_f1s).mean()

        return results

    def reset(self):
        """모든 메트릭 리셋"""
        for metrics in self.binary_metrics.values():
            for metric in metrics.values():
                metric.reset()

        for metrics in self.multi_metrics.values():
            for metric in metrics.values():
                metric.reset()

        for metrics in self.regression_metrics.values():
            metrics["mae"].clear()
            metrics["mse"].clear()


def compute_accuracy_from_logits(
    logits: torch.Tensor, targets: torch.Tensor, task: str = "multiclass"
):
    """로짓으로부터 accuracy 계산하는 헬퍼 함수"""
    if task == "binary":
        preds = torch.sigmoid(logits) > 0.5
        return accuracy(preds, targets.bool(), task="binary")
    elif task == "multiclass":
        preds = torch.argmax(logits, dim=1)
        return accuracy(preds, targets, task="multiclass", num_classes=logits.size(1))
    else:
        raise ValueError(f"Unsupported task: {task}")
