"""Exact inference in discrete graphical models.

Two classic algorithms from CMU 10-708 / Koller & Friedman:

* :func:`variable_elimination` -- sum-product VE with a configurable elimination
  order, computing a marginal (or conditional) and the partition function.
* :class:`FactorGraph` + belief propagation -- the sum-product message-passing
  algorithm.  On a tree it is exact and converges in one sweep; on a loopy graph
  it runs *loopy BP* to a fixed point.

Belief propagation marginals are checked against variable elimination in the
tests, which is the standard correctness oracle.
"""
from __future__ import annotations

from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from .factor import Factor, factor_product


# --------------------------------------------------------------------------- #
# Variable elimination
# --------------------------------------------------------------------------- #
def variable_elimination(
    factors: Sequence[Factor],
    query: Sequence[Hashable],
    *,
    evidence: Optional[Dict] = None,
    elim_order: Optional[Sequence[Hashable]] = None,
    mode: str = "sum",
) -> Factor:
    """Run sum-product (or max-product) variable elimination.

    Parameters
    ----------
    factors:
        The factors defining the unnormalized distribution.
    query:
        Variables to keep (the ones we want a marginal over).
    evidence:
        Optional dict var->state to condition on before elimination.
    elim_order:
        Order in which to eliminate non-query variables.  If ``None`` a
        min-neighbors (min-degree) heuristic order is used.
    mode:
        ``'sum'`` for marginals, ``'max'`` for max-marginals.

    Returns
    -------
    Factor over ``query``.  For ``mode='sum'`` this is the *unnormalized* marginal
    whose sum equals the partition function of the (reduced) model; call
    ``.normalize()`` for a distribution.
    """
    evidence = evidence or {}
    working = [f.reduce(evidence) for f in factors]

    query = list(query)
    all_vars: List = []
    for f in working:
        for v in f.scope:
            if v not in all_vars:
                all_vars.append(v)
    to_eliminate = [v for v in all_vars if v not in query]

    if elim_order is None:
        elim_order = _min_degree_order(working, to_eliminate)
    else:
        elim_order = [v for v in elim_order if v in to_eliminate]
        missing = [v for v in to_eliminate if v not in elim_order]
        elim_order = list(elim_order) + missing

    for v in elim_order:
        involved = [f for f in working if v in f.scope]
        rest = [f for f in working if v not in f.scope]
        prod = factor_product(involved)
        summed = prod.marginalize([v], mode=mode)
        working = rest + [summed]

    result = factor_product(working)
    # reorder to the requested query order for determinism
    if result.scope and result.scope != query:
        result = _reorder(result, [q for q in query if q in result.scope])
    return result


def _reorder(factor: Factor, order: Sequence[Hashable]) -> Factor:
    perm = [factor.scope.index(v) for v in order]
    return Factor(list(order), np.transpose(factor.table, axes=perm))


def _min_degree_order(factors: Sequence[Factor], candidates: Sequence[Hashable]):
    """Greedy min-degree elimination ordering over an induced graph."""
    # build adjacency among candidates+others from factor scopes
    adj: Dict = {}
    for f in factors:
        for v in f.scope:
            adj.setdefault(v, set())
        for i, u in enumerate(f.scope):
            for w in f.scope[i + 1 :]:
                adj[u].add(w)
                adj[w].add(u)
    remaining = list(candidates)
    order: List = []
    while remaining:
        # pick candidate with fewest current neighbors
        v = min(remaining, key=lambda x: len(adj.get(x, set())))
        order.append(v)
        neigh = adj.get(v, set())
        # connect neighbors (moralize / fill-in)
        for a in neigh:
            for b in neigh:
                if a != b:
                    adj[a].add(b)
        for a in list(neigh):
            adj[a].discard(v)
        adj.pop(v, None)
        remaining.remove(v)
    return order


# --------------------------------------------------------------------------- #
# Factor graph + (loopy) belief propagation
# --------------------------------------------------------------------------- #
class FactorGraph:
    """Bipartite factor graph for sum-product belief propagation."""

    def __init__(self, factors: Sequence[Factor]):
        self.factors: List[Factor] = [f.copy() for f in factors]
        # variable -> cardinality, variable -> incident factor indices
        self.card: Dict = {}
        self.var_factors: Dict[Hashable, List[int]] = {}
        for fi, f in enumerate(self.factors):
            for v in f.scope:
                if v in self.card and self.card[v] != f.card[v]:
                    raise ValueError(f"cardinality mismatch for {v!r}")
                self.card[v] = f.card[v]
                self.var_factors.setdefault(v, []).append(fi)

    @property
    def variables(self) -> List:
        return list(self.card.keys())

    def run_bp(
        self,
        *,
        max_iters: int = 100,
        tol: float = 1e-8,
        damping: float = 0.0,
    ) -> Tuple[Dict, int]:
        """Run sum-product BP; return (variable beliefs, #iterations run).

        Messages are stored in log space would be safer, but for the small,
        well-scaled models here we normalize each message to sum to 1 every
        step, which keeps everything numerically stable in linear space.
        """
        # msg_v2f[(v, fi)] and msg_f2v[(fi, v)] are length-card[v] vectors
        msg_v2f: Dict = {}
        msg_f2v: Dict = {}
        for fi, f in enumerate(self.factors):
            for v in f.scope:
                msg_v2f[(v, fi)] = np.ones(self.card[v]) / self.card[v]
                msg_f2v[(fi, v)] = np.ones(self.card[v]) / self.card[v]

        iters = 0
        for it in range(max_iters):
            iters = it + 1
            max_delta = 0.0

            # 1) variable -> factor: product of incoming factor msgs except target
            new_v2f: Dict = {}
            for v in self.variables:
                incident = self.var_factors[v]
                for fi in incident:
                    m = np.ones(self.card[v])
                    for fj in incident:
                        if fj != fi:
                            m = m * msg_f2v[(fj, v)]
                    s = m.sum()
                    m = m / s if s > 0 else np.ones(self.card[v]) / self.card[v]
                    new_v2f[(v, fi)] = m
            msg_v2f = new_v2f

            # 2) factor -> variable: multiply factor by incoming var msgs, marginalize
            for fi, f in enumerate(self.factors):
                for v in f.scope:
                    prod = f.copy()
                    for u in f.scope:
                        if u != v:
                            vec = Factor([u], msg_v2f[(u, fi)])
                            prod = prod.product(vec)
                    marg = prod.marginalize(
                        [u for u in prod.scope if u != v], mode="sum"
                    )
                    # marg scope is [v]; align to var state order
                    m = _reorder(marg, [v]).table if marg.scope else marg.table
                    s = m.sum()
                    m = m / s if s > 0 else np.ones(self.card[v]) / self.card[v]
                    if damping > 0:
                        m = damping * msg_f2v[(fi, v)] + (1 - damping) * m
                        m = m / m.sum()
                    delta = float(np.abs(m - msg_f2v[(fi, v)]).max())
                    max_delta = max(max_delta, delta)
                    msg_f2v[(fi, v)] = m

            if max_delta < tol:
                break

        beliefs = self._beliefs(msg_f2v)
        return beliefs, iters

    def _beliefs(self, msg_f2v: Dict) -> Dict:
        beliefs: Dict = {}
        for v in self.variables:
            b = np.ones(self.card[v])
            for fi in self.var_factors[v]:
                b = b * msg_f2v[(fi, v)]
            s = b.sum()
            beliefs[v] = b / s if s > 0 else np.ones(self.card[v]) / self.card[v]
        return beliefs


def belief_propagation(factors: Sequence[Factor], **kwargs) -> Dict:
    """Convenience wrapper: return the single-variable beliefs of a factor set."""
    fg = FactorGraph(factors)
    beliefs, _ = fg.run_bp(**kwargs)
    return beliefs
