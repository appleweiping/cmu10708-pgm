"""Tests for exact inference: VE against brute force, BP against VE."""
import itertools

import numpy as np
import pytest

from pgm.factor import Factor, factor_product
from pgm.inference import FactorGraph, belief_propagation, variable_elimination


def brute_force_marginal(factors, query, card):
    """Exhaustive marginal by enumerating the full joint (small models only)."""
    all_vars = list(card.keys())
    q = Factor(query, np.zeros([card[v] for v in query]))
    for assign in itertools.product(*[range(card[v]) for v in all_vars]):
        a = dict(zip(all_vars, assign))
        val = 1.0
        for f in factors:
            val *= f.get_value(a)
        idx = tuple(a[v] for v in query)
        q.table[idx] += val
    return q.normalize()


def make_chain(rng, n=4, k=3):
    """A random chain MRF over variables 0..n-1 with cardinality k."""
    card = {i: k for i in range(n)}
    factors = [Factor([0], rng.random(k) + 0.1)]
    for i in range(n - 1):
        factors.append(Factor([i, i + 1], rng.random((k, k)) + 0.1))
    return factors, card


def test_ve_matches_brute_force_chain():
    rng = np.random.default_rng(1)
    factors, card = make_chain(rng, n=4, k=3)
    for q in range(4):
        ve = variable_elimination(factors, [q]).normalize()
        bf = brute_force_marginal(factors, [q], card)
        assert np.allclose(ve.table, bf.table, atol=1e-10)


def test_ve_with_evidence():
    rng = np.random.default_rng(2)
    factors, card = make_chain(rng, n=4, k=2)
    ve = variable_elimination(factors, [3], evidence={0: 1}).normalize()
    # brute force with evidence
    reduced = [f.reduce({0: 1}) for f in factors]
    card2 = {i: card[i] for i in range(1, 4)}
    bf = brute_force_marginal(reduced, [3], card2)
    assert np.allclose(ve.table, bf.table, atol=1e-10)


def test_ve_partition_function():
    rng = np.random.default_rng(3)
    factors, card = make_chain(rng, n=3, k=2)
    # full enumeration of Z
    all_vars = list(card.keys())
    Z = 0.0
    for assign in itertools.product(*[range(card[v]) for v in all_vars]):
        a = dict(zip(all_vars, assign))
        Z += float(np.prod([f.get_value(a) for f in factors]))
    ve = variable_elimination(factors, [])
    assert ve.partition() == pytest.approx(Z, rel=1e-9)


def test_bp_matches_ve_on_tree():
    rng = np.random.default_rng(4)
    # star tree: center 0 connected to leaves 1,2,3
    k = 3
    factors = [Factor([0], rng.random(k) + 0.1)]
    for leaf in (1, 2, 3):
        factors.append(Factor([0, leaf], rng.random((k, k)) + 0.1))
    beliefs = belief_propagation(factors, max_iters=50)
    for v in (0, 1, 2, 3):
        ve = variable_elimination(factors, [v]).normalize()
        assert np.allclose(beliefs[v], ve.table, atol=1e-8), v


def test_loopy_bp_reasonable_on_small_loop():
    rng = np.random.default_rng(5)
    # 3-cycle: edges (0,1),(1,2),(0,2)
    k = 2
    factors = [
        Factor([0, 1], rng.random((k, k)) + 0.5),
        Factor([1, 2], rng.random((k, k)) + 0.5),
        Factor([0, 2], rng.random((k, k)) + 0.5),
    ]
    beliefs = belief_propagation(factors, max_iters=500, tol=1e-10, damping=0.5)
    for v in (0, 1, 2):
        ve = variable_elimination(factors, [v]).normalize()
        # loopy BP is approximate but should be close on this weakly-coupled loop
        assert np.abs(beliefs[v] - ve.table).max() < 0.05, v
