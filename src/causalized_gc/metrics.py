"""Connectivity thresholding and evaluation shared by c-GC backends."""

from __future__ import annotations

from typing import Any

import numpy as np


Array = Any
ArrayModule = Any
_SHD = None


def connectivity_matrix(
    xp: ArrayModule,
    corr: Array,
    pval_corr: Array,
    inv_corr: Array,
    pval_inv_corr: Array,
    *,
    n_pasts: int,
    n_lags: int,
    simulation: bool,
    alpha: float,
    beta: float,
) -> Array:
    """Build a weighted connectivity matrix from significance masks."""

    sig_corr = corr * (pval_corr <= alpha)
    sig_inv = inv_corr * (pval_inv_corr <= beta)
    inferred = (sig_corr != 0) & (sig_inv != 0)

    n_neur = inferred.shape[1]
    blocks = [
        inferred[lag * n_neur : (lag + 1) * n_neur, :n_neur]
        for lag in range(n_pasts + 1)
    ]

    if simulation:
        conn = blocks[1] if len(blocks) > 1 else blocks[0]
        for lag in range(2, min(n_lags, len(blocks) - 1) + 1):
            conn = conn | blocks[lag]
    elif n_lags == 1:
        conn = (blocks[0] | blocks[1]) if len(blocks) > 1 else blocks[0]
    else:
        conn = blocks[0]
        for lag in range(1, min(n_lags, len(blocks) - 1) + 1):
            conn = conn | blocks[lag]

    return corr[:n_neur, :] * conn


def confusion_matrix(
    conn_mat: np.ndarray,
    truth: np.ndarray,
    *,
    simulation: bool = True,
) -> np.ndarray:
    """Return ``[[tp, fp], [fn, tn]]`` against ground truth."""

    target = truth.T if simulation else truth
    tp = int(np.sum((target != 0) & (conn_mat != 0)))
    fn = int(np.sum((target != 0) & (conn_mat == 0)))
    fp = int(np.sum((target == 0) & (conn_mat != 0)))
    tn = int(np.sum((target == 0) & (conn_mat == 0)))
    return np.array([[tp, fp], [fn, tn]])


def classification_metrics(confusion: np.ndarray) -> np.ndarray:
    """Return accuracy, precision, recall, FPR, balanced accuracy, and F1."""

    tp, fp, fn, tn = confusion.flatten()
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = (specificity + recall) / 2
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return np.array([accuracy, precision, recall, fpr, balanced_accuracy, f1])


def shd(
    truth: np.ndarray,
    inferred: np.ndarray,
    *,
    simulation: bool = True,
) -> float:
    """Compute Structural Hamming Distance with the optional ``cdt`` extra."""

    global _SHD
    if _SHD is None:
        try:
            from cdt.metrics import SHD as cdt_shd
        except Exception as exc:
            raise ImportError(
                "SHD metrics require the optional 'metrics' dependencies."
            ) from exc
        _SHD = cdt_shd
    target = truth.T if simulation else truth
    return _SHD(target=target, pred=inferred, double_for_anticausal=False)


def compare_with_ground_truth(
    truth: np.ndarray,
    inferred: np.ndarray,
    *,
    simulation: bool = True,
) -> np.ndarray:
    """Create an RGB true/false-positive/negative edge overlay."""

    target = np.asarray(truth).T if simulation else np.asarray(truth)
    predicted = np.asarray(inferred)
    if predicted.shape != target.shape:
        raise ValueError(
            f"inferred shape {predicted.shape} does not match truth shape {target.shape}"
        )
    ground_truth = target != 0
    predicted_edges = predicted != 0
    image = np.ones((*ground_truth.shape, 3), dtype=float)
    image[ground_truth & predicted_edges] = (0.0, 0.6, 0.0)
    image[~ground_truth & predicted_edges] = (0.8, 0.0, 0.0)
    image[ground_truth & ~predicted_edges] = (0.0, 0.0, 0.8)
    return image


__all__ = [
    "classification_metrics",
    "compare_with_ground_truth",
    "confusion_matrix",
    "connectivity_matrix",
    "shd",
]
