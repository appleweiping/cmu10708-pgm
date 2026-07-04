"""Tests for Gibbs sampling and mean-field VI on the Ising model."""
import itertools

import numpy as np
import pytest

from pgm.ising import IsingGrid, gibbs_sample, mean_field


def exact_marginals(model):
    """Exact P(x_i=+1) by enumerating all 2^(HW) states (tiny grids only)."""
    H, W = model.H, model.W
    n = H * W
    logits = []
    states = []
    for bits in itertools.product([-1, 1], repeat=n):
        x = np.array(bits, dtype=np.float64).reshape(H, W)
        states.append(x)
        logits.append(-model.energy(x))
    logits = np.array(logits)
    logits -= logits.max()
    w = np.exp(logits)
    w /= w.sum()
    marg = np.zeros((H, W))
    for x, p in zip(states, w):
        marg += p * (x > 0)
    return marg


def test_gibbs_matches_exact_small_grid():
    model = IsingGrid(H=3, W=3, J=0.4, h=0.2)
    exact = exact_marginals(model)
    marg, trace = gibbs_sample(model, n_sweeps=20000, burn_in=2000, seed=0)
    # Monte Carlo estimate should be within a few percent of exact
    assert np.abs(marg - exact).max() < 0.03


def test_mean_field_elbo_monotonic():
    model = IsingGrid(H=8, W=8, J=0.3, h=0.1)
    marg, elbo = mean_field(model, max_iters=200, tol=1e-9, damping=0.5)
    diffs = np.diff(elbo)
    # ELBO should increase (allow tiny numerical wiggle)
    assert (diffs > -1e-6).all()
    assert np.all((marg >= 0) & (marg <= 1))


def test_mean_field_close_to_exact_weak_coupling():
    # Mean field is accurate in the weak-coupling regime
    model = IsingGrid(H=3, W=3, J=0.15, h=0.3)
    exact = exact_marginals(model)
    marg, _ = mean_field(model, max_iters=500, tol=1e-10, damping=0.5)
    assert np.abs(marg - exact).max() < 0.03


def test_gibbs_and_mean_field_agree_weak_coupling():
    model = IsingGrid(H=4, W=4, J=0.1, h=0.2)
    gm, _ = gibbs_sample(model, n_sweeps=20000, burn_in=2000, seed=3)
    mf, _ = mean_field(model, max_iters=500, tol=1e-10)
    assert np.abs(gm - mf).max() < 0.05
