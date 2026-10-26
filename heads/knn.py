"""Retrieval head: cosine kNN over a FAISS index of training embeddings, score = fraction of
the k nearest training images carrying each label (unweighted vote)."""

import os

# torch, scikit-learn and faiss-cpu wheels each bundle their own libomp on macOS; the second one
# to initialise aborts the process unless told the duplicate is expected. tests/test_heads.py
# checks faiss against brute force so a silently wrong search would be caught.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # noqa: E402
import numpy as np  # noqa: E402


def _unit(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class KNN:
    def __init__(self, x_train, y_train):
        x = _unit(x_train)
        self.index = faiss.IndexFlatIP(x.shape[1])
        self.index.add(x)
        self.y = np.asarray(y_train, dtype=np.float32)

    def k_eff(self, k):
        return min(k, self.index.ntotal)

    def scores(self, x_query, ks):
        """{k: (n_query, n_labels) vote fractions}; k is clipped to the index size."""
        kmax = self.k_eff(max(ks))
        _, idx = self.index.search(_unit(x_query), kmax)
        votes = np.cumsum(self.y[idx], axis=1)  # (n, kmax, L)
        return {k: votes[:, self.k_eff(k) - 1] / self.k_eff(k) for k in ks}
