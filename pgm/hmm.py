"""Hidden Markov Models: exact inference and parameter learning.

Implements the standard HMM algorithms taught in 10-708:

* forward-backward (the sum-product algorithm specialized to a chain) with
  scaling for numerical stability, giving filtered/smoothed posteriors and the
  data log-likelihood;
* Viterbi decoding (max-product) in log space;
* supervised MLE from labelled sequences;
* Baum-Welch (EM) for unsupervised parameter learning.

All distributions are discrete (categorical emissions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class HMM:
    """A discrete HMM with categorical emissions.

    Attributes
    ----------
    pi:  (K,) initial state distribution.
    A:   (K, K) transition matrix, ``A[i, j] = P(z_{t+1}=j | z_t=i)``.
    B:   (K, M) emission matrix, ``B[k, o] = P(x_t=o | z_t=k)``.
    """

    pi: np.ndarray
    A: np.ndarray
    B: np.ndarray

    def __post_init__(self) -> None:
        self.pi = np.asarray(self.pi, dtype=np.float64)
        self.A = np.asarray(self.A, dtype=np.float64)
        self.B = np.asarray(self.B, dtype=np.float64)
        self.K = self.A.shape[0]
        self.M = self.B.shape[1]

    # -- inference -------------------------------------------------------
    def forward_backward(
        self, obs: Sequence[int]
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Scaled forward-backward.

        Returns ``(gamma, xi, loglik)`` where

        * ``gamma[t, k] = P(z_t=k | x_{1:T})`` (smoothed posterior),
        * ``xi[t, i, j] = P(z_t=i, z_{t+1}=j | x_{1:T})`` for t=0..T-2,
        * ``loglik`` is ``log P(x_{1:T})``.
        """
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        K = self.K
        alpha = np.zeros((T, K))
        c = np.zeros(T)  # scaling factors

        # t = 0
        alpha[0] = self.pi * self.B[:, obs[0]]
        c[0] = alpha[0].sum()
        if c[0] == 0:
            raise ValueError("zero-probability observation at t=0")
        alpha[0] /= c[0]
        # recursion
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]
            c[t] = alpha[t].sum()
            if c[t] == 0:
                raise ValueError(f"zero-probability observation at t={t}")
            alpha[t] /= c[t]

        loglik = float(np.log(c).sum())

        # backward with the same scaling
        beta = np.zeros((T, K))
        beta[T - 1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]

        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            m = (
                alpha[t][:, None]
                * self.A
                * (self.B[:, obs[t + 1]] * beta[t + 1])[None, :]
            )
            xi[t] = m / (m.sum() if m.sum() > 0 else 1.0)
        return gamma, xi, loglik

    def loglikelihood(self, obs: Sequence[int]) -> float:
        """``log P(x_{1:T})`` via the forward pass only."""
        obs = np.asarray(obs, dtype=int)
        alpha = self.pi * self.B[:, obs[0]]
        ll = np.log(alpha.sum())
        alpha /= alpha.sum()
        for t in range(1, len(obs)):
            alpha = (alpha @ self.A) * self.B[:, obs[t]]
            s = alpha.sum()
            ll += np.log(s)
            alpha /= s
        return float(ll)

    def viterbi(self, obs: Sequence[int]) -> Tuple[np.ndarray, float]:
        """Most likely state path via max-product in log space.

        Returns ``(path, logprob)``.
        """
        obs = np.asarray(obs, dtype=int)
        T = len(obs)
        K = self.K
        eps = 1e-300
        log_pi = np.log(self.pi + eps)
        log_A = np.log(self.A + eps)
        log_B = np.log(self.B + eps)

        delta = np.zeros((T, K))
        psi = np.zeros((T, K), dtype=int)
        delta[0] = log_pi + log_B[:, obs[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + log_A  # (K, K): from i to j
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = scores[psi[t], np.arange(K)] + log_B[:, obs[t]]

        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        logprob = float(delta[T - 1, path[T - 1]])
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path, logprob


# --------------------------------------------------------------------------- #
# Parameter learning
# --------------------------------------------------------------------------- #
def mle_supervised(
    state_seqs: Sequence[Sequence[int]],
    obs_seqs: Sequence[Sequence[int]],
    K: int,
    M: int,
    *,
    alpha: float = 1.0,
) -> HMM:
    """Maximum-likelihood HMM from *labelled* sequences with Laplace smoothing.

    ``alpha`` is the pseudocount (add-``alpha`` smoothing); ``alpha=0`` gives the
    raw counts MLE.
    """
    pi = np.full(K, alpha)
    A = np.full((K, K), alpha)
    B = np.full((K, M), alpha)
    for states, obs in zip(state_seqs, obs_seqs):
        states = np.asarray(states, dtype=int)
        obs = np.asarray(obs, dtype=int)
        pi[states[0]] += 1
        for t in range(len(states)):
            B[states[t], obs[t]] += 1
            if t + 1 < len(states):
                A[states[t], states[t + 1]] += 1
    pi /= pi.sum()
    A /= A.sum(axis=1, keepdims=True)
    B /= B.sum(axis=1, keepdims=True)
    return HMM(pi, A, B)


def baum_welch(
    obs_seqs: Sequence[Sequence[int]],
    K: int,
    M: int,
    *,
    n_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
    init: Optional[HMM] = None,
    verbose: bool = False,
) -> Tuple[HMM, List[float]]:
    """Unsupervised EM (Baum-Welch) for HMM parameters.

    Returns the fitted :class:`HMM` and the list of total log-likelihoods, one
    per EM iteration (guaranteed non-decreasing up to numerical error).
    """
    rng = np.random.default_rng(seed)
    if init is None:
        pi = rng.dirichlet(np.ones(K))
        A = rng.dirichlet(np.ones(K), size=K)
        B = rng.dirichlet(np.ones(M), size=K)
        model = HMM(pi, A, B)
    else:
        model = HMM(init.pi.copy(), init.A.copy(), init.B.copy())

    history: List[float] = []
    prev_ll = -np.inf
    for it in range(n_iter):
        # E-step: accumulate expected counts across all sequences
        pi_acc = np.zeros(K)
        A_num = np.zeros((K, K))
        A_den = np.zeros(K)
        B_num = np.zeros((K, M))
        B_den = np.zeros(K)
        total_ll = 0.0

        for obs in obs_seqs:
            obs = np.asarray(obs, dtype=int)
            gamma, xi, ll = model.forward_backward(obs)
            total_ll += ll
            pi_acc += gamma[0]
            A_num += xi.sum(axis=0)
            A_den += gamma[:-1].sum(axis=0)
            for k in range(K):
                np.add.at(B_num[k], obs, gamma[:, k])
            B_den += gamma.sum(axis=0)

        history.append(total_ll)
        if verbose:
            print(f"  iter {it:3d}  loglik = {total_ll:.6f}")

        # M-step
        pi = pi_acc / pi_acc.sum()
        A = A_num / np.maximum(A_den[:, None], 1e-300)
        B = B_num / np.maximum(B_den[:, None], 1e-300)
        model = HMM(pi, A, B)

        if total_ll - prev_ll < tol and it > 0:
            break
        prev_ll = total_ll

    return model, history


def sample_hmm(
    model: HMM, T: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample a (states, observations) pair of length ``T`` from ``model``."""
    states = np.zeros(T, dtype=int)
    obs = np.zeros(T, dtype=int)
    states[0] = rng.choice(model.K, p=model.pi)
    obs[0] = rng.choice(model.M, p=model.B[states[0]])
    for t in range(1, T):
        states[t] = rng.choice(model.K, p=model.A[states[t - 1]])
        obs[t] = rng.choice(model.M, p=model.B[states[t]])
    return states, obs
