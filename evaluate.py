import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def _unwrap_batch(batch):
    if isinstance(batch, (tuple, list)):
        if len(batch) == 1:
            return batch[0], None
        return batch[0], batch[1]
    return batch, None


@torch.no_grad()
def compute_scores(model, loader, device):
    model.eval()

    scores = []
    labels = []

    for batch in loader:
        x, y = _unwrap_batch(batch)
        x = x.to(device)

        x_hat = model(x)
        rec_err = torch.mean((x - x_hat) ** 2, dim=(1, 2))
        if x.shape[1] >= 2:
            x_diff = x[:, 1:, :] - x[:, :-1, :]
            x_hat_diff = x_hat[:, 1:, :] - x_hat[:, :-1, :]
            deriv_err = torch.mean((x_diff - x_hat_diff) ** 2, dim=(1, 2))
            err = rec_err + 0.5 * deriv_err
        else:
            err = rec_err

        scores.append(err.cpu())
        if y is not None:
            labels.append(y.cpu())

    scores_array = torch.cat(scores).numpy() if scores else np.array([])
    labels_array = torch.cat(labels).numpy() if labels else None
    return scores_array, labels_array


def select_threshold(scores, percentile=95.0):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("Cannot select threshold from an empty score array.")
    if not np.isfinite(scores).all():
        raise ValueError("Cannot select a threshold from non-finite scores.")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in the interval (0, 100].")
    return float(np.percentile(scores, percentile))


def aggregate_scores_by_subject(scores, labels, metadata):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if len(scores) != len(labels) or len(scores) != len(metadata):
        raise ValueError("Scores, labels and metadata have inconsistent lengths.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")
    labels = labels.astype(np.int8, copy=False)

    grouped: dict[str, dict[str, list]] = {}
    for score, label, item in zip(scores, labels, metadata, strict=True):
        subject_id = item.get("subject_id")
        if not subject_id:
            raise ValueError("Every sample must include a subject_id.")
        group = grouped.setdefault(subject_id, {"scores": [], "labels": []})
        group["scores"].append(float(score))
        group["labels"].append(int(label))

    subject_scores = []
    subject_labels = []
    for subject_id in sorted(grouped):
        group = grouped[subject_id]
        unique_labels = set(group["labels"])
        if len(unique_labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels.")
        subject_scores.append(float(np.mean(group["scores"])))
        subject_labels.append(unique_labels.pop())

    return (
        np.asarray(subject_scores, dtype=np.float64),
        np.asarray(subject_labels, dtype=np.int8),
    )


def _validated_scores_and_labels(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.ndim != 1 or len(scores) != len(labels):
        raise ValueError("Scores and labels must be equally sized 1D arrays.")
    if len(scores) == 0:
        raise ValueError("Cannot evaluate empty score arrays.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN or infinite values.")
    if not np.isin(labels, [0, 1]).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both labels 0 and 1.")
    return scores, labels.astype(np.int8, copy=False)


def evaluate(scores, labels, threshold=None, title="EVALUATION"):
    scores, labels = _validated_scores_and_labels(scores, labels)
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    print(f"\n=== {title} ===")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"AP:      {ap:.4f}")

    metrics = {"roc_auc": float(roc), "average_precision": float(ap)}

    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")
        predictions = (scores >= threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

        print(f"Threshold: {threshold:.6f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"CM:        TN={tn} FP={fp} FN={fn} TP={tp}")

        metrics.update(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return metrics
