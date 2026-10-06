"""Bootstrap confidence interval helper."""

import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve


def _bootstrap_ci(y_true, y_score, fn, n: int = 1000, seed: int = 0):
    """Return (point, lo, hi) with 95 % bootstrap CI.

    Parameters
    ----------
    y_true, y_score : array-like
    fn              : callable(y_true, y_score) -> scalar metric
    n               : number of bootstrap iterations
    seed            : RNG seed
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    point = fn(y_true, y_score)
    samples = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        try:
            samples[i] = fn(y_true[idx], y_score[idx])
        except Exception:
            samples[i] = np.nan
    valid = samples[~np.isnan(samples)]
    lo = float(np.percentile(valid, 2.5)) if len(valid) else np.nan
    hi = float(np.percentile(valid, 97.5)) if len(valid) else np.nan
    return float(point), lo, hi


def _bootstrap_roc_band(y_true, y_score, n: int = 1000, seed: int = 0,
                        grid_size: int = 101):
    """Bootstrap 95% CI band for ROC curve.

    Returns ``(fpr_grid, tpr_mean, tpr_lo, tpr_hi)`` where TPR is interpolated
    onto a common FPR grid so the bands have a well-defined envelope.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr_grid = np.linspace(0.0, 1.0, grid_size)
    tprs = np.full((n, grid_size), np.nan)

    for i in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        fpr_i, tpr_i, _ = roc_curve(yt, ys)
        tprs[i] = np.interp(fpr_grid, fpr_i, tpr_i, left=0.0, right=1.0)

    tpr_mean = np.nanmean(tprs, axis=0)
    tpr_lo = np.nanpercentile(tprs, 2.5, axis=0)
    tpr_hi = np.nanpercentile(tprs, 97.5, axis=0)
    return fpr_grid, tpr_mean, tpr_lo, tpr_hi


def _bootstrap_pr_band(y_true, y_score, n: int = 1000, seed: int = 0,
                       grid_size: int = 101):
    """Bootstrap 95% CI band for precision-recall curve.

    Returns ``(recall_grid, prec_mean, prec_lo, prec_hi)`` with precision
    interpolated onto a common recall grid.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    recall_grid = np.linspace(0.0, 1.0, grid_size)
    precs = np.full((n, grid_size), np.nan)

    for i in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        p_i, r_i, _ = precision_recall_curve(yt, ys)
        order = np.argsort(r_i)
        precs[i] = np.interp(recall_grid, r_i[order], p_i[order],
                             left=p_i[order][0], right=p_i[order][-1])

    prec_mean = np.nanmean(precs, axis=0)
    prec_lo = np.nanpercentile(precs, 2.5, axis=0)
    prec_hi = np.nanpercentile(precs, 97.5, axis=0)
    return recall_grid, prec_mean, prec_lo, prec_hi
