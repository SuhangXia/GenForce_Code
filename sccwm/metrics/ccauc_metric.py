from __future__ import annotations

import numpy as np


def _roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)
    return float(auc)


def compute_ccauc(positive_scores: np.ndarray, negative_scores: np.ndarray) -> dict[str, float]:
    pos = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    neg = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    if pos.size == 0 or neg.size == 0:
        return {"ccauc": 0.5}
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)], axis=0)
    scores = np.concatenate([pos, neg], axis=0)
    return {"ccauc": _roc_auc_binary(labels, scores)}
