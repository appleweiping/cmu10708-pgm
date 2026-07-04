"""Linear-chain Conditional Random Fields for sequence labelling.

This is the coding core of the 10-708 "general-graph CRF" homework, restricted to
the linear-chain case that underlies most sequence-tagging applications.  We model

    P(y | x) = (1/Z(x)) * exp( sum_t [ U(y_t, x, t) + T(y_{t-1}, y_t) ] )

with

* node (emission) potentials from binary features phi(x, t) via a weight matrix W
  of shape ``(n_features, n_labels)``, and
* a full transition matrix ``Tr`` of shape ``(n_labels, n_labels)``.

The partition function ``Z(x)`` and marginals are computed with the
forward-backward algorithm in log space; training maximises the (L2-regularised)
conditional log-likelihood with SciPy's L-BFGS.  Decoding uses Viterbi.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


class SparseSeq:
    """A sequence of tokens each described by a set of active binary features.

    ``tokens[t]`` is an integer array of feature indices that are 1 for token
    ``t`` (all other features are 0).  This is the natural representation for the
    sparse, high-dimensional one-hot features used in NLP tagging and lets the
    CRF avoid ever building the dense ``(T, n_features)`` matrix.
    """

    __slots__ = ("tokens", "T")

    def __init__(self, tokens: Sequence[np.ndarray]):
        self.tokens = [np.asarray(t, dtype=np.intp) for t in tokens]
        self.T = len(self.tokens)

    def __len__(self) -> int:
        return self.T


def _as_features(X):
    """Coerce input to either a dense float matrix or pass a SparseSeq through."""
    if isinstance(X, SparseSeq):
        return X
    return np.asarray(X, dtype=np.float64)


def _lse(a: np.ndarray, axis: int):
    """Fast, numerically-stable log-sum-exp along one axis.

    Equivalent to ``scipy.special.logsumexp`` but without its per-call validation
    overhead -- this lives in the CRF's innermost training loop, where it is ~7x
    faster and makes NLP-scale L-BFGS training tractable on CPU.
    """
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass
class LinearChainCRF:
    """A linear-chain CRF with ``n_labels`` states and ``n_features`` node features.

    Parameters are packed into a single flat vector for the optimiser:
    ``theta = [W.ravel(), Tr.ravel()]``.
    """

    n_features: int
    n_labels: int
    l2: float = 1.0

    def __post_init__(self) -> None:
        self.n_W = self.n_features * self.n_labels
        self.n_T = self.n_labels * self.n_labels
        self.W = np.zeros((self.n_features, self.n_labels))
        self.Tr = np.zeros((self.n_labels, self.n_labels))

    # -- parameter (un)packing ------------------------------------------
    def _unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        W = theta[: self.n_W].reshape(self.n_features, self.n_labels)
        Tr = theta[self.n_W :].reshape(self.n_labels, self.n_labels)
        return W, Tr

    def _pack(self, W: np.ndarray, Tr: np.ndarray) -> np.ndarray:
        return np.concatenate([W.ravel(), Tr.ravel()])

    # -- potentials ------------------------------------------------------
    @staticmethod
    def _node_scores(X, W: np.ndarray) -> np.ndarray:
        """Log node potentials, shape ``(T, n_labels)``, for one sequence.

        ``X`` may be either a dense ``(T, n_features)`` 0/1 matrix or a
        :class:`SparseSeq` (a per-token list of active binary-feature indices).
        The sparse path avoids materialising the huge feature matrix and is what
        makes real NLP-scale training tractable on CPU.
        """
        if isinstance(X, SparseSeq):
            out = np.empty((X.T, W.shape[1]))
            for t, idx in enumerate(X.tokens):
                out[t] = W[idx].sum(axis=0)
            return out
        return X @ W

    @staticmethod
    def _accumulate_node_grad(X, diff: np.ndarray, gW: np.ndarray) -> None:
        """Add ``sum_t feat(x,t) outer (diff[t])`` into ``gW`` in place."""
        if isinstance(X, SparseSeq):
            for t, idx in enumerate(X.tokens):
                # scatter-add the label-vector diff[t] onto each active feature row
                np.add.at(gW, idx, diff[t])
        else:
            gW += X.T @ diff

    # -- forward-backward in log space ----------------------------------
    def _forward_backward(
        self, node: np.ndarray, Tr: np.ndarray
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Return ``(logZ, node_marginals, edge_marginals)`` for one sequence.

        ``node`` is ``(T, L)`` log node potentials; ``Tr`` is ``(L, L)`` log
        transition scores.  ``node_marginals`` is ``(T, L)`` = ``P(y_t | x)`` and
        ``edge_marginals`` is ``(T-1, L, L)`` = ``P(y_{t-1}, y_t | x)``.
        """
        T, L = node.shape
        log_alpha = np.zeros((T, L))
        log_alpha[0] = node[0]
        for t in range(1, T):
            # alpha[t, j] = node[t, j] + logsumexp_i(alpha[t-1, i] + Tr[i, j])
            log_alpha[t] = node[t] + _lse(log_alpha[t - 1][:, None] + Tr, axis=0)
        logZ = _lse(log_alpha[-1], axis=0)

        log_beta = np.zeros((T, L))
        for t in range(T - 2, -1, -1):
            log_beta[t] = _lse(
                Tr + node[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
            )

        node_marg = np.exp(log_alpha + log_beta - logZ)
        edge_marg = np.zeros((T - 1, L, L))
        for t in range(T - 1):
            m = (
                log_alpha[t][:, None]
                + Tr
                + node[t + 1][None, :]
                + log_beta[t + 1][None, :]
            )
            edge_marg[t] = np.exp(m - logZ)
        return float(logZ), node_marg, edge_marg

    # -- objective (neg log-likelihood + gradient) ----------------------
    def _nll_and_grad(
        self,
        theta: np.ndarray,
        seqs: Sequence[Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[float, np.ndarray]:
        W, Tr = self._unpack(theta)
        total_nll = 0.0
        gW = np.zeros_like(W)
        gT = np.zeros_like(Tr)

        for X, y in seqs:
            node = self._node_scores(X, W)  # (T, L)
            logZ, node_marg, edge_marg = self._forward_backward(node, Tr)

            # score of the gold path
            T = len(y)
            gold = node[np.arange(T), y].sum()
            gold += Tr[y[:-1], y[1:]].sum()
            total_nll += logZ - gold

            # gradient: expected - empirical features (for the NLL)
            onehot = np.zeros((T, self.n_labels))
            onehot[np.arange(T), y] = 1.0
            self._accumulate_node_grad(X, node_marg - onehot, gW)
            # transition grad
            emp_T = np.zeros_like(Tr)
            np.add.at(emp_T, (y[:-1], y[1:]), 1.0)
            gT += edge_marg.sum(axis=0) - emp_T

        # L2 regularisation on both parameter blocks
        total_nll += 0.5 * self.l2 * (np.sum(W ** 2) + np.sum(Tr ** 2))
        gW += self.l2 * W
        gT += self.l2 * Tr
        return total_nll, self._pack(gW, gT)

    # -- public API ------------------------------------------------------
    def fit(
        self,
        seqs: Sequence[Tuple[np.ndarray, np.ndarray]],
        *,
        max_iter: int = 200,
        verbose: bool = False,
    ) -> "LinearChainCRF":
        """Train by L-BFGS on the regularised conditional log-likelihood.

        Each ``X`` may be a dense 0/1 matrix or a :class:`SparseSeq`.
        """
        seqs = [(_as_features(X), np.asarray(y, dtype=int)) for X, y in seqs]
        theta0 = np.zeros(self.n_W + self.n_T)
        history: List[float] = []

        def fun(theta):
            nll, grad = self._nll_and_grad(theta, seqs)
            history.append(nll)
            return nll, grad

        options = {"maxiter": max_iter}
        if verbose:
            options["iprint"] = 1
        res = minimize(
            fun,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options=options,
        )
        self.W, self.Tr = self._unpack(res.x)
        self.opt_result = res
        self.history = history
        return self

    def predict(self, X) -> np.ndarray:
        """Viterbi decode the most likely label sequence for one sequence."""
        node = self._node_scores(_as_features(X), self.W)
        T, L = node.shape
        delta = np.zeros((T, L))
        psi = np.zeros((T, L), dtype=int)
        delta[0] = node[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + self.Tr  # (L_prev, L_cur)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(L)] + node[t]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def score_sequences(
        self, seqs: Sequence[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[float, float]:
        """Return ``(token_accuracy, mean_loglik_per_token)`` on labelled data."""
        correct = 0
        total = 0
        total_ll = 0.0
        for X, y in seqs:
            Xf = _as_features(X)
            y = np.asarray(y, dtype=int)
            pred = self.predict(Xf)
            correct += int((pred == y).sum())
            total += len(y)
            node = self._node_scores(Xf, self.W)
            logZ, _, _ = self._forward_backward(node, self.Tr)
            gold = node[np.arange(len(y)), y].sum() + self.Tr[y[:-1], y[1:]].sum()
            total_ll += gold - logZ
        return correct / total, total_ll / total
