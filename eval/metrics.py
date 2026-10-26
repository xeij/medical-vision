"""Weighted ranking and calibration metrics, vectorised over bootstrap resamples.

Every metric takes labels `y` (n,), scores `s` (n,) and optional integer weights `w` of shape
(n,) or (B, n). With (B, n) weights it returns one value per row, which is how the patient-level
bootstrap is evaluated in one pass: resample patients, count how often each was drawn, give each
of its images that count as weight. Ties in `s` are handled the way scikit-learn does (mid-rank
for AUROC, one threshold per distinct score for AUPRC); kNN vote fractions tie constantly.
"""

import warnings

import numpy as np


def _prep(y, s, w):
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    if w is None:
        w = np.ones(len(y), dtype=np.int32)
    w = np.asarray(w)
    return y, s, np.atleast_2d(w), w.ndim == 1


def _groups(y, s, w):
    """Sort by score ascending and sum positive/negative weight per distinct score."""
    order = np.argsort(s, kind="stable")
    y, s, w = y[order], s[order], w[:, order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    gp = np.add.reduceat(w * y, starts, axis=1).astype(np.float64)
    gn = np.add.reduceat(w * ~y, starts, axis=1).astype(np.float64)
    return gp, gn


def _ratio(num, den, single):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return float(out[0]) if single else out


def _auroc(gp, gn, single):
    n_below = np.cumsum(gn, axis=1) - gn  # negative weight strictly below each score
    num = (gp * (n_below + 0.5 * gn)).sum(1)
    return _ratio(num, gp.sum(1) * gn.sum(1), single)


def _auprc(gp, gn, single):
    tp = np.cumsum(gp[:, ::-1], axis=1)  # descending thresholds
    fp = np.cumsum(gn[:, ::-1], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    return _ratio((prec * gp[:, ::-1]).sum(1), gp.sum(1), single)


def auroc(y, s, w=None):
    """Area under the ROC curve = P(score_pos > score_neg) + 0.5 P(tie), weighted."""
    y, s, w, single = _prep(y, s, w)
    return _auroc(*_groups(y, s, w), single)


def auprc(y, s, w=None):
    """Average precision (sum over distinct thresholds of delta-recall x precision), weighted."""
    y, s, w, single = _prep(y, s, w)
    return _auprc(*_groups(y, s, w), single)


def rank_metrics(y, s, w=None):
    """(auroc, auprc) from one sort; what the bootstrap loop calls."""
    y, s, w, single = _prep(y, s, w)
    gp, gn = _groups(y, s, w)
    return _auroc(gp, gn, single), _auprc(gp, gn, single)


def ece(y, p, w=None, bins=15):
    """Expected calibration error with equal-width bins: sum_b |mean conf - mean acc| x mass."""
    y, p, w, single = _prep(y, p, w)
    w = w.astype(np.float64)
    onehot = np.zeros((len(p), bins))
    onehot[np.arange(len(p)), np.minimum((p * bins).astype(int), bins - 1)] = 1.0
    mass, conf, acc = w @ onehot, (w * p) @ onehot, (w * y) @ onehot
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.where(mass > 0, np.abs(conf - acc) / mass, 0.0)
    return _ratio((mass * gap).sum(1), mass.sum(1), single)


def macro(values):
    """nan-mean over the first axis (labels); nan when no label is defined."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.asarray(values, dtype=np.float64), axis=0)


def patient_bootstrap_weights(patient_ids, n_boot, seed):
    """(n_boot, n_images) int32: how many times each image's patient was drawn when resampling
    patients with replacement. Fixed seed per test set so every run shares the same resamples."""
    _, inverse = np.unique(np.asarray(patient_ids), return_inverse=True)
    n_pat = inverse.max() + 1
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_pat, np.full(n_pat, 1.0 / n_pat), size=n_boot)
    return counts[:, inverse].astype(np.int32)
