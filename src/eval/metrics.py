"""Evaluation metrics for deepfake audio detection.

compute_eer   : Equal Error Rate
compute_tdcf  : Normalized tandem detection cost function (ASVspoof 5 protocol)
"""
from __future__ import annotations

import numpy as np


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate.

    Parameters
    ----------
    scores : np.ndarray, shape [N]
        Higher score = more likely spoof/fake (1).
    labels : np.ndarray, shape [N]
        Binary labels: 1 = spoof/fake, 0 = bonafide/real.

    Returns
    -------
    float in [0, 1]
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()

    thresholds = np.unique(scores)

    n_pos = np.sum(labels == 1)  # spoof
    n_neg = np.sum(labels == 0)  # bonafide

    if n_pos == 0 or n_neg == 0:
        return 0.0

    best_eer = 1.0
    for thr in thresholds:
        # Predict spoof (1) when score >= threshold
        preds = (scores >= thr).astype(np.int32)
        # FAR: bonafide accepted as spoof
        fa = np.sum((preds == 1) & (labels == 0))
        # FRR: spoof rejected (classified as bonafide)
        fr = np.sum((preds == 0) & (labels == 1))
        far = fa / n_neg
        frr = fr / n_pos
        eer_candidate = (far + frr) / 2.0
        if abs(far - frr) < abs(far - frr) or eer_candidate < best_eer:
            # Track minimum |FAR - FRR| point
            best_eer = eer_candidate

    # More accurate: find crossing point
    far_list, frr_list = [], []
    for thr in np.sort(thresholds):
        preds = (scores >= thr).astype(np.int32)
        fa = np.sum((preds == 1) & (labels == 0))
        fr = np.sum((preds == 0) & (labels == 1))
        far_list.append(fa / n_neg)
        frr_list.append(fr / n_pos)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    diff = np.abs(far_arr - frr_arr)
    idx = np.argmin(diff)
    eer = (far_arr[idx] + frr_arr[idx]) / 2.0
    return float(np.clip(eer, 0.0, 1.0))


def compute_tdcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.05,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
) -> float:
    """Compute normalized tandem Detection Cost Function (t-DCF).

    Based on ASVspoof 5 protocol definition.

    Parameters
    ----------
    scores    : [N] float  — higher score → more likely spoof
    labels    : [N] int    — 1=spoof, 0=bonafide
    p_target  : prior probability of spoof (default 0.05)
    c_miss    : cost of missing a spoof (default 1.0)
    c_fa      : cost of false alarm (default 10.0)

    Returns
    -------
    float — normalized t-DCF in [0, ∞), lower is better
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int32).ravel()

    thresholds = np.unique(scores)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.0

    p_nontarget = 1.0 - p_target

    dcf_values = []
    for thr in thresholds:
        preds = (scores >= thr).astype(np.int32)
        fa = np.sum((preds == 1) & (labels == 0))
        fr = np.sum((preds == 0) & (labels == 1))
        pmiss = fr / n_pos
        pfa = fa / n_neg
        dcf = c_miss * pmiss * p_target + c_fa * pfa * p_nontarget
        dcf_values.append(dcf)

    # Normalize by minimum cost
    norm_factor = min(
        c_miss * p_target,
        c_fa * p_nontarget,
    )
    min_dcf = min(dcf_values)
    if norm_factor == 0:
        return 0.0
    return float(min_dcf / norm_factor)
