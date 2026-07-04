"""Tests for the linear-chain CRF: gradient check, forward-backward, learning."""
import numpy as np
import pytest

from pgm.crf import LinearChainCRF


def make_toy_data(rng, n_seqs=20, T=6, n_features=5, n_labels=3):
    seqs = []
    for _ in range(n_seqs):
        X = (rng.random((T, n_features)) < 0.5).astype(float)
        y = rng.integers(0, n_labels, size=T)
        seqs.append((X, y))
    return seqs


def test_forward_backward_marginals_sum_to_one():
    crf = LinearChainCRF(n_features=4, n_labels=3, l2=0.0)
    rng = np.random.default_rng(0)
    crf.W = rng.standard_normal((4, 3))
    crf.Tr = rng.standard_normal((3, 3))
    X = (rng.random((5, 4)) < 0.5).astype(float)
    node = crf._node_scores(X, crf.W)
    logZ, node_marg, edge_marg = crf._forward_backward(node, crf.Tr)
    assert np.allclose(node_marg.sum(axis=1), 1.0)
    assert np.allclose(edge_marg.sum(axis=(1, 2)), 1.0)
    # edge marginals must be consistent with node marginals
    assert np.allclose(edge_marg.sum(axis=2), node_marg[:-1], atol=1e-8)
    assert np.allclose(edge_marg.sum(axis=1), node_marg[1:], atol=1e-8)


def test_analytic_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    crf = LinearChainCRF(n_features=4, n_labels=3, l2=0.7)
    seqs = make_toy_data(rng, n_seqs=5, T=5, n_features=4, n_labels=3)
    seqs = [(X, y.astype(int)) for X, y in seqs]
    theta = rng.standard_normal(crf.n_W + crf.n_T) * 0.3

    nll, grad = crf._nll_and_grad(theta, seqs)
    eps = 1e-6
    num_grad = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        fp, _ = crf._nll_and_grad(tp, seqs)
        fm, _ = crf._nll_and_grad(tm, seqs)
        num_grad[i] = (fp - fm) / (2 * eps)
    assert np.allclose(grad, num_grad, atol=1e-5), np.abs(grad - num_grad).max()


def test_crf_learns_separable_pattern():
    # Construct data where feature f == label perfectly: CRF must reach high acc.
    rng = np.random.default_rng(2)
    n_labels = 3
    seqs = []
    for _ in range(40):
        T = 8
        y = rng.integers(0, n_labels, size=T)
        X = np.zeros((T, n_labels))
        X[np.arange(T), y] = 1.0  # one informative feature per label
        seqs.append((X, y))
    crf = LinearChainCRF(n_features=n_labels, n_labels=n_labels, l2=0.01)
    crf.fit(seqs, max_iter=100)
    acc, _ = crf.score_sequences(seqs)
    assert acc > 0.95
    # objective should have decreased
    assert crf.history[-1] < crf.history[0]
