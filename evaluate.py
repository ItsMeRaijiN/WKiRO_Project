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
    if len(scores) == 0:
        raise ValueError("Cannot select threshold from an empty score array.")
    return float(np.percentile(scores, percentile))


def evaluate(scores, labels, threshold=None):
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    print("\n=== EVALUATION ===")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"AP:      {ap:.4f}")

    metrics = {"roc_auc": float(roc), "average_precision": float(ap)}

    if threshold is not None:
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