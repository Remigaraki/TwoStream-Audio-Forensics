import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

def compute_eer(y_true, y_score):
    """
    Calculates the Equal Error Rate (EER).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def compute_t_dcf(eer):
    """
    Placeholder for tandem Detection Cost Function (t-DCF).
    """
    return eer * 1.5 # simplified mock calculation
