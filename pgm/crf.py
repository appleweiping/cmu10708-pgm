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
from scipy.special import logsumexp


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
    def _node_scores(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Log node potentials, shape ``(T, n_labels)``, for one sequence.

        ``X`` is ``(T, n_features)`` (may be sparse-ish 0/1 but stored dense).
        """
        return X @ W

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
            log_alpha[t] = node[t] + logsumexp(
                log_alpha[t - 1][:, None] + Tr, axis=0
            )
        logZ = logsumexp(log_alpha[-1])

        log_beta = np.zeros((T, L))
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
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

            # gradient: empirical - expected features
            # node features: for each t, feature vector X[t] assigned to label y[t]
            # empirical node grad
            onehot = np.zeros((T, self.n_labels))
            onehot[np.arange(T), y] = 1.0
            gW += X.T @ (node_marg - onehot)  # expected - empirical (for NLL)
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
        """Train by L-BFGS on the regularised conditional log-likelihood."""
        seqs = [(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=int)) for X, y in seqs]
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decode the most likely label sequence for one sequence."""
        X = np.asarray(X, dtype=np.float64)
        node = self._node_scores(X, self.W)
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
            X = np.asarray(X, dtype=np.float64)
            y = np.asarray(y, dtype=int)
            pred = self.predict(X)
            correct += int((pred == y).sum())
            total += len(y)
            node = self._node_scores(X, self.W)
            logZ, _, _ = self._forward_backward(node, self.Tr)
            gold = node[np.arange(len(y)), y].sum() + self.Tr[y[:-1], y[1:]].sum()
            total_ll += gold - logZ
        return correct / total, total_ll / total
