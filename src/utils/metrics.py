import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def compute_eer(y_true, y_score):
    """
    Calculate the Equal Error Rate (EER) and the corresponding threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    tpr_interp = interp1d(fpr, tpr, bounds_error=False, fill_value=(tpr[0], tpr[-1]))
    eer = brentq(lambda x: 1.0 - x - float(tpr_interp(x)), 0.0, 1.0)
    threshold_interp = interp1d(fpr, thresholds, bounds_error=False, fill_value=(thresholds[0], thresholds[-1]))
    threshold = float(threshold_interp(eer))
    return float(eer), threshold


def compute_t_dcf(
    y_true,
    y_score,
    prior_bonafide: float = 0.5,
    prior_spoof: float = 0.5,
    cost_miss: float = 1.0,
    cost_fa: float = 1.0,
):
    """
    Compute a normalized t-DCF-style detection cost over all thresholds.

    The repository does not yet include a separate ASV backend, so this uses a
    standalone countermeasure cost proxy: the minimum normalized risk over all
    score thresholds.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("y_true and y_score must have the same length")

    bona_mask = y_true == 0
    spoof_mask = y_true == 1

    if not bona_mask.any() or not spoof_mask.any():
        raise ValueError("Both bona fide (0) and spoof (1) samples are required")

    _, _, thresholds = roc_curve(y_true, y_score, pos_label=1)

    best_cost = np.inf
    best_threshold = float(thresholds[0])

    for threshold in thresholds:
        predict_spoof = y_score >= threshold
        # p_miss: spoof sample predicted bonafide (missed detection)
        p_miss = float(np.mean(~predict_spoof[spoof_mask]))
        # p_fa: bonafide sample predicted spoof (false alarm)
        p_fa = float(np.mean(predict_spoof[bona_mask]))

        cost = (prior_spoof * cost_miss * p_miss) + (prior_bonafide * cost_fa * p_fa)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(threshold)

    normalization = min(prior_bonafide * cost_miss, prior_spoof * cost_fa)
    normalization = normalization if normalization > 0 else 1.0

    return float(best_cost / normalization), best_threshold
