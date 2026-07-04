"""Unit tests for discrete factor operations."""
import numpy as np
import pytest

from pgm.factor import Factor, factor_product


def test_product_matches_koller_example():
    # Koller & Friedman Fig 4.3 factor product example (shapes and a few values)
    phi1 = Factor(["A", "B"], np.array([[0.5, 0.8], [0.1, 0.0], [0.3, 0.9]]))
    phi2 = Factor(["B", "C"], np.array([[0.5, 0.7], [0.1, 0.2]]))
    prod = phi1.product(phi2)
    assert set(prod.scope) == {"A", "B", "C"}
    # value at A=0,B=0,C=0 = 0.5 * 0.5 = 0.25
    assert prod.get_value({"A": 0, "B": 0, "C": 0}) == pytest.approx(0.25)
    # A=2,B=1,C=1 = 0.9 * 0.2 = 0.18
    assert prod.get_value({"A": 2, "B": 1, "C": 1}) == pytest.approx(0.18)


def test_product_commutes():
    rng = np.random.default_rng(0)
    a = Factor(["X", "Y"], rng.random((2, 3)))
    b = Factor(["Y", "Z"], rng.random((3, 2)))
    p1 = a.product(b)
    p2 = b.product(a)
    # reorder p2 to p1's scope and compare
    perm = [p2.scope.index(v) for v in p1.scope]
    assert np.allclose(p1.table, np.transpose(p2.table, perm))


def test_marginalize_sum_and_max():
    f = Factor(["X", "Y"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    s = f.marginalize(["Y"], mode="sum")
    assert s.scope == ["X"]
    assert np.allclose(s.table, [3.0, 7.0])
    m = f.marginalize(["Y"], mode="max")
    assert np.allclose(m.table, [2.0, 4.0])


def test_reduce_and_normalize():
    f = Factor(["X", "Y"], np.array([[1.0, 3.0], [2.0, 2.0]]))
    r = f.reduce({"X": 1})
    assert r.scope == ["Y"]
    assert np.allclose(r.table, [2.0, 2.0])
    n = f.normalize()
    assert n.table.sum() == pytest.approx(1.0)


def test_cardinality_mismatch_raises():
    a = Factor(["X"], np.array([1.0, 2.0]))
    b = Factor(["X"], np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        a.product(b)
