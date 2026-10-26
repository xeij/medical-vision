import numpy as np

from eval.metrics import auroc
from heads.knn import KNN
from heads.probe import fit_probe

rng = np.random.default_rng(0)


def _toy(n, d=8, seed=0):
    r = np.random.default_rng(seed)
    y = (r.random((n, 3)) < 0.4).astype(np.int8)
    y[:, 2] = 0  # a label with no positives
    x = r.normal(size=(n, d)).astype(np.float32)
    x[:, 0] += 3 * y[:, 0]  # label 0 separable along dim 0
    x[:, 1] += 3 * y[:, 1]
    return x, y


def test_knn_recovers_labels_and_clips_k():
    x, y = _toy(200)
    knn = KNN(x, y)
    s = knn.scores(x, ks=(1, 5, 500))
    assert (s[1] == y).all()  # nearest neighbour of a training point is itself
    assert s[5].shape == (200, 3) and s[500].shape == (200, 3)
    assert knn.k_eff(500) == 200 and knn.k_eff(5) == 5
    assert (s[500][:, 2] == 0).all()  # no positives in index -> zero votes
    # cosine: rescaling a query must not change its neighbours
    assert np.allclose(knn.scores(x[:5] * 7.0, ks=(5,))[5], s[5][:5])


def test_knn_scores_are_vote_fractions():
    x, y = _toy(50)
    s = KNN(x, y).scores(x, ks=(5,))[5]
    assert s.min() >= 0 and s.max() <= 1
    assert np.allclose(s * 5, np.round(s * 5))


def test_probe_separable_and_single_class():
    x_tr, y_tr = _toy(400, seed=1)
    x_va, y_va = _toy(100, seed=2)
    x_te, y_te = _toy(100, seed=3)
    probe = fit_probe(x_tr, y_tr, x_va, y_va, pathology_cols=[0, 1])
    p = probe.predict(x_te)
    assert p.shape == (100, 3) and p.min() >= 0 and p.max() <= 1
    assert auroc(y_te[:, 0], p[:, 0]) > 0.95 and auroc(y_te[:, 1], p[:, 1]) > 0.95
    assert (p[:, 2] == 0).all()  # single-class label -> constant prevalence, no crash
    assert probe.c in probe.cs and probe.c_fallback is False


def test_probe_falls_back_when_val_uninformative():
    x_tr, y_tr = _toy(60, seed=1)
    x_va, y_va = x_tr[:2], y_tr[:2] * 0  # no positives on val for any label
    probe = fit_probe(x_tr, y_tr, x_va, y_va, pathology_cols=[0, 1])
    assert probe.c_fallback is True


def test_knn_matches_brute_force():
    x, y = _toy(300, d=16)
    q = np.random.default_rng(5).normal(size=(40, 16)).astype(np.float32)
    s = KNN(x, y).scores(q, ks=(7,))[7]
    xn = x / np.linalg.norm(x, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    top = np.argsort(-(qn @ xn.T), axis=1)[:, :7]
    assert np.allclose(s, y[top].mean(1), atol=1e-6)
