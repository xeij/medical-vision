"""Linear probe: one L2-regularised logistic regression per label on standardised embeddings.
C is chosen on the validation split by macro AUROC over the pathology columns; when validation
is uninformative (no positives for any pathology) a fixed fallback C is used and flagged."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from eval.metrics import auroc, macro

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
FALLBACK_C = 1e-2


class Probe:
    def __init__(self, scaler, models, c, cs, c_fallback):
        self.scaler, self.models, self.c, self.cs = scaler, models, c, cs
        self.c_fallback = c_fallback

    def predict(self, x):
        xs = self.scaler.transform(x)
        cols = [
            m.predict_proba(xs)[:, 1] if isinstance(m, LogisticRegression) else np.full(len(xs), m)
            for m in self.models
        ]
        return np.stack(cols, axis=1).astype(np.float32)


def _fit_label(x, y, c):
    if y.min() == y.max():  # single class: constant prevalence (0.0 or 1.0)
        return float(y[0])
    return LogisticRegression(C=c, max_iter=2000).fit(x, y)


def fit_probe(x_tr, y_tr, x_va, y_va, cs=C_GRID, pathology_cols=None, fallback_c=FALLBACK_C):
    y_tr, y_va = np.asarray(y_tr), np.asarray(y_va)
    cols = list(range(y_tr.shape[1])) if pathology_cols is None else list(pathology_cols)
    scaler = StandardScaler().fit(x_tr)
    xs_tr = scaler.transform(x_tr)
    fits = {}
    for c in cs:
        models = [_fit_label(xs_tr, y_tr[:, j], c) for j in range(y_tr.shape[1])]
        probe = Probe(scaler, models, c, cs, False)
        p = probe.predict(x_va)
        fits[c] = (macro([auroc(y_va[:, j], p[:, j]) for j in cols]), probe)
    scored = {c: v for c, (v, _) in fits.items() if not np.isnan(v)}
    if scored:
        return fits[max(scored, key=scored.get)][1]
    probe = fits[fallback_c][1]
    probe.c_fallback = True
    return probe
