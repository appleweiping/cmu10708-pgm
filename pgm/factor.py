"""Discrete factors over categorical random variables.

A :class:`Factor` is a nonnegative function phi(x_{scope}) represented as a dense
table.  Variables are identified by hashable names; ``card[v]`` is the number of
states of variable ``v``.  The table is stored as an ``np.ndarray`` whose axes are
ordered exactly as ``scope``.

This is the standard representation used for exact inference in Koller & Friedman
(the 10-708 textbook): factor product, marginalization (sum/max), reduction
(conditioning), and normalization are all implemented here.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class Factor:
    """A discrete factor phi over a set of variables (its *scope*).

    Parameters
    ----------
    scope:
        Ordered variable names. ``table.shape[i]`` is the cardinality of
        ``scope[i]``.
    table:
        Nonnegative array of factor values, one axis per variable in ``scope``.
    """

    __slots__ = ("scope", "table")

    def __init__(self, scope: Sequence, table: np.ndarray):
        scope = list(scope)
        table = np.asarray(table, dtype=np.float64)
        if len(scope) != table.ndim:
            raise ValueError(
                f"scope has {len(scope)} vars but table has {table.ndim} dims"
            )
        if len(set(scope)) != len(scope):
            raise ValueError(f"duplicate variables in scope: {scope}")
        self.scope: List = scope
        self.table: np.ndarray = table

    # -- basic accessors -------------------------------------------------
    @property
    def card(self) -> Dict:
        """Mapping variable -> cardinality."""
        return {v: self.table.shape[i] for i, v in enumerate(self.scope)}

    def copy(self) -> "Factor":
        return Factor(list(self.scope), self.table.copy())

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Factor(scope={self.scope}, shape={self.table.shape})"

    # -- core operations -------------------------------------------------
    def _align_to(self, variables: Sequence) -> np.ndarray:
        """Return this factor's table broadcast onto ``variables`` axis order.

        ``variables`` must be a superset of ``self.scope``.  The returned array
        has one axis per variable in ``variables`` (size-1 for absent vars),
        ready to broadcast-multiply against another aligned factor.
        """
        pos = {v: i for i, v in enumerate(variables)}
        # move each of our axes to the slot given by ``variables``
        shape = [1] * len(variables)
        perm = []
        for i, v in enumerate(self.scope):
            shape[pos[v]] = self.table.shape[i]
            perm.append(pos[v])
        # build the transposed table then reshape to full-rank broadcastable shape
        # order axes of self.table so they land in increasing target position
        order = np.argsort(perm)
        transposed = np.transpose(self.table, axes=order)
        return transposed.reshape(shape)

    def product(self, other: "Factor") -> "Factor":
        """Factor product ``self * other`` (Koller & Friedman, Def. 4.2)."""
        # union of scopes, self's order first then new vars from other
        union: List = list(self.scope)
        for v in other.scope:
            if v not in self.card or v in other.card:
                if v not in union:
                    union.append(v)
        a = self._align_to(union)
        b = other._align_to(union)
        # sanity: overlapping cardinalities must match
        for v in set(self.scope) & set(other.scope):
            if self.card[v] != other.card[v]:
                raise ValueError(f"cardinality mismatch for {v!r}")
        return Factor(union, a * b)

    def marginalize(self, variables: Iterable, *, mode: str = "sum") -> "Factor":
        """Sum-out (or max-out) ``variables`` from the scope.

        ``mode='sum'`` gives marginalization; ``mode='max'`` gives the max-marginal
        used by the max-product / Viterbi style algorithms.
        """
        variables = set(variables)
        keep = [v for v in self.scope if v not in variables]
        axes = tuple(i for i, v in enumerate(self.scope) if v in variables)
        if not axes:
            return self.copy()
        if mode == "sum":
            new_table = self.table.sum(axis=axes)
        elif mode == "max":
            new_table = self.table.max(axis=axes)
        else:  # pragma: no cover
            raise ValueError(f"unknown mode {mode!r}")
        return Factor(keep, new_table)

    def reduce(self, evidence: Dict) -> "Factor":
        """Condition on ``evidence`` (a dict var->state); slice those axes out."""
        scope = list(self.scope)
        table = self.table
        for v, val in evidence.items():
            if v not in scope:
                continue
            ax = scope.index(v)
            table = np.take(table, val, axis=ax)
            scope.pop(ax)
        return Factor(scope, table)

    def normalize(self) -> "Factor":
        """Return a factor whose entries sum to 1 (a distribution)."""
        z = self.table.sum()
        if z == 0:
            raise ZeroDivisionError("cannot normalize an all-zero factor")
        return Factor(list(self.scope), self.table / z)

    def partition(self) -> float:
        """Sum of all table entries (the un-normalized partition function)."""
        return float(self.table.sum())

    def get_value(self, assignment: Dict) -> float:
        """Look up phi at a full assignment of the scope."""
        idx = tuple(assignment[v] for v in self.scope)
        return float(self.table[idx])


def factor_product(factors: Sequence[Factor]) -> Factor:
    """Product of a list of factors (empty -> scalar 1 factor)."""
    if not factors:
        return Factor([], np.array(1.0))
    out = factors[0].copy()
    for f in factors[1:]:
        out = out.product(f)
    return out
