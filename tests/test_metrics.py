import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from eval.metrics import auprc, auroc, ece, macro, patient_bootstrap_weights, rank_metrics

rng = np.random.default_rng(0)
N = 300
Y = rng.random(N) < 0.3
S_CONT = rng.random(N) + 0.5 * Y  # continuous scores
S_TIED = rng.integers(0, 6, N) / 5 + 0.1 * Y  # heavy ties, like kNN vote fractions
S_TIED = np.round(S_TIED, 1)
W = rng.integers(0, 4, (5, N))  # five integer resample-weight rows


@pytest.mark.parametrize("s", [S_CONT, S_TIED])
def test_auroc_matches_sklearn(s):
    assert auroc(Y, s) == pytest.approx(roc_auc_score(Y, s))
    out = auroc(Y, s, W)
    assert out.shape == (5,)
    for b in range(5):
        assert out[b] == pytest.approx(roc_auc_score(Y, s, sample_weight=W[b]))


@pytest.mark.parametrize("s", [S_CONT, S_TIED])
def test_auprc_matches_sklearn(s):
    assert auprc(Y, s) == pytest.approx(average_precision_score(Y, s))
    out = auprc(Y, s, W)
    for b in range(5):
        assert out[b] == pytest.approx(average_precision_score(Y, s, sample_weight=W[b]))


def test_degenerate_inputs():
    assert auroc(Y, np.full(N, 0.3)) == pytest.approx(0.5)  # all tied -> chance
    assert np.isnan(auroc(np.zeros(N, bool), S_CONT))  # no positives -> nan, not an error
    assert np.isnan(auprc(np.zeros(N, bool), S_CONT))
    w = np.array([[1] * N, [0] * N])
    assert np.isnan(auroc(Y, S_CONT, w)[1])  # empty resample -> nan


def test_ece_by_hand():
    p = np.array([0.15, 0.15, 0.85, 0.85])
    y = np.array([0, 1, 1, 1])
    # bin [0.1,0.2): conf .15 acc .5 -> .35 ; bin [0.8,0.9): conf .85 acc 1 -> .15 ; equal mass
    assert ece(y, p, bins=10) == pytest.approx(0.25)
    w = np.array([2, 0, 1, 1])
    # bin1: conf .15 acc 0 -> .15 (mass 2/4); bin2: .15 (mass 2/4)
    assert ece(y, p, w, bins=10) == pytest.approx(0.15)
    assert ece(y, p, np.stack([np.ones(4), w]), bins=10) == pytest.approx([0.25, 0.15])
    assert ece(np.array([1, 0]), np.array([1.0, 0.0])) == pytest.approx(0.0)  # p=1 in last bin


def test_macro_is_nan_safe():
    assert macro(np.array([0.8, np.nan, 0.6])) == pytest.approx(0.7)
    assert np.isnan(macro(np.array([np.nan, np.nan])))
    out = macro(np.array([[0.8, np.nan], [0.2, 0.4]]))  # (labels, B) -> (B,)
    assert out == pytest.approx([0.5, 0.4])


def test_patient_bootstrap_weights():
    patients = np.array(["a", "a", "a", "b", "c", "c"])
    w = patient_bootstrap_weights(patients, n_boot=50, seed=0)
    assert w.shape == (50, 6) and w.dtype == np.int32
    # every image of a patient gets that patient's draw count
    assert (w[:, 0] == w[:, 1]).all() and (w[:, 1] == w[:, 2]).all() and (w[:, 4] == w[:, 5]).all()
    # exactly n_patients draws per replicate
    assert ((w[:, 0] + w[:, 3] + w[:, 4]) == 3).all()
    assert (w == patient_bootstrap_weights(patients, 50, 0)).all()
    assert not (w == patient_bootstrap_weights(patients, 50, 1)).all()


def test_rank_metrics_matches_separate_calls():
    a, p = rank_metrics(Y, S_TIED, W)
    assert a == pytest.approx(auroc(Y, S_TIED, W)) and p == pytest.approx(auprc(Y, S_TIED, W))
    assert rank_metrics(Y, S_CONT) == pytest.approx((auroc(Y, S_CONT), auprc(Y, S_CONT)))
