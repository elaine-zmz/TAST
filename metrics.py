from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def binary_metrics(
    labels: Iterable[float],
    predictions: Iterable[float],
    prefix: str,
) -> Dict[str, float]:
    y = np.asarray(list(labels), dtype=np.float64)
    p = np.asarray(list(predictions), dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(
            f"{prefix}: labels shape {y.shape} != predictions shape {p.shape}"
        )
    if y.size == 0:
        raise ValueError(f"{prefix}: empty evaluation arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError(f"{prefix}: non-finite labels or predictions")
    if np.unique(y).size < 2:
        raise ValueError(
            f"{prefix}: AUC is undefined because only one label class is present"
        )
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return {
        f"{prefix}_auc": float(roc_auc_score(y, p)),
        f"{prefix}_logloss": float(log_loss(y, p, labels=[0, 1])),
        f"{prefix}_positive_rate": float(y.mean()),
        f"{prefix}_prediction_mean": float(p.mean()),
        f"{prefix}_prediction_std": float(p.std()),
    }
