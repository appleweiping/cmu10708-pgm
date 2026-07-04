"""Tests for HMM inference and learning."""
import numpy as np
import pytest

from pgm.hmm import HMM, baum_welch, mle_supervised, sample_hmm


def toy_hmm():
    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.4, 0.6]])
    B = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
    return HMM(pi, A, B)


def brute_force_loglik(model, obs):
    """log P(x) by summing over all state paths (tiny sequences)."""
    K = model.K
    T = len(obs)
    import itertools

    total = 0.0
    for path in itertools.product(range(K), repeat=T):
        p = model.pi[path[0]] * model.B[path[0], obs[0]]
        for t in range(1, T):
            p *= model.A[path[t - 1], path[t]] * model.B[path[t], obs[t]]
        total += p
    return np.log(total)


def test_forward_matches_brute_force():
    model = toy_hmm()
    obs = [0, 2, 1, 1, 2]
    ll = model.loglikelihood(obs)
    assert ll == pytest.approx(brute_force_loglik(model, obs), abs=1e-10)


def test_forward_backward_consistency():
    model = toy_hmm()
    obs = [0, 2, 1, 0, 2, 1]
    gamma, xi, ll = model.forward_backward(obs)
    # gammas sum to 1 per timestep
    assert np.allclose(gamma.sum(axis=1), 1.0)
    # xi marginalizes down to gamma
    assert np.allclose(xi.sum(axis=2), gamma[:-1], atol=1e-10)
    assert ll == pytest.approx(model.loglikelihood(obs), abs=1e-10)


def test_viterbi_matches_brute_force():
    import itertools

    model = toy_hmm()
    obs = [0, 2, 1, 2, 0]
    path, logp = model.viterbi(obs)
    # brute-force best path
    best_lp = -np.inf
    best = None
    for cand in itertools.product(range(model.K), repeat=len(obs)):
        lp = np.log(model.pi[cand[0]]) + np.log(model.B[cand[0], obs[0]])
        for t in range(1, len(obs)):
            lp += np.log(model.A[cand[t - 1], cand[t]]) + np.log(
                model.B[cand[t], obs[t]]
            )
        if lp > best_lp:
            best_lp, best = lp, cand
    assert tuple(path) == best
    assert logp == pytest.approx(best_lp, abs=1e-8)


def test_baum_welch_monotonic_and_recovers():
    rng = np.random.default_rng(0)
    true = toy_hmm()
    seqs = [sample_hmm(true, 40, rng)[1] for _ in range(60)]
    model, hist = baum_welch(seqs, K=2, M=3, n_iter=100, tol=1e-6, seed=1)
    # log-likelihood is non-decreasing across EM iterations
    diffs = np.diff(hist)
    assert (diffs > -1e-6).all()
    # fitted model should not do worse than the truth on its own data (usually better)
    ll_fit = sum(model.loglikelihood(o) for o in seqs)
    ll_true = sum(true.loglikelihood(o) for o in seqs)
    assert ll_fit >= ll_true - 1.0


def test_supervised_mle_recovers_params():
    rng = np.random.default_rng(2)
    true = toy_hmm()
    states, obs = [], []
    for _ in range(200):
        s, o = sample_hmm(true, 50, rng)
        states.append(s)
        obs.append(o)
    est = mle_supervised(states, obs, K=2, M=3, alpha=0.0)
    # transition rows should be close to truth with this much data
    assert np.abs(est.A - true.A).max() < 0.05
    assert np.abs(est.B - true.B).max() < 0.05
