"""The 2-D Ising / grid MRF, with Gibbs sampling and mean-field VI.

The model is a pairwise Markov random field on an ``H x W`` grid of binary spins
``x_i in {-1, +1}`` with energy

    E(x) = - J * sum_{(i,j) in edges} x_i x_j  -  sum_i h_i x_i

and distribution ``P(x) = exp(-E(x)) / Z``.  ``J`` is the coupling strength and
``h_i`` a per-site external field.  This is the workhorse example for approximate
inference in 10-708:

* :func:`gibbs_sample` -- a Gibbs sampler (MCMC) drawing from ``P(x)``;
* :func:`mean_field` -- naive mean-field variational inference, minimizing the
  KL to a fully-factorized ``q`` and returning per-site marginals plus the ELBO.

For image denoising we use the standard formulation where the field
``h_i = eta * y_i`` ties each spin to a noisy observation ``y_i``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class IsingGrid:
    """Ising model on an ``H x W`` grid.

    Parameters
    ----------
    H, W: grid dimensions.
    J:    coupling strength (scalar, ferromagnetic if > 0).
    h:    external field, either a scalar or an ``(H, W)`` array.
    """

    H: int
    W: int
    J: float = 1.0
    h: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))

    def __post_init__(self) -> None:
        if np.isscalar(self.h) or np.asarray(self.h).size == 1:
            self.h = np.full((self.H, self.W), float(np.asarray(self.h).reshape(-1)[0]))
        else:
            self.h = np.asarray(self.h, dtype=np.float64)
            if self.h.shape != (self.H, self.W):
                raise ValueError("field h must be scalar or shape (H, W)")

    def neighbor_sum(self, x: np.ndarray) -> np.ndarray:
        """Sum of the 4-connected neighbours of every site (zero padding)."""
        s = np.zeros_like(x, dtype=np.float64)
        s[:-1, :] += x[1:, :]
        s[1:, :] += x[:-1, :]
        s[:, :-1] += x[:, 1:]
        s[:, 1:] += x[:, :-1]
        return s

    def energy(self, x: np.ndarray) -> float:
        """Total energy ``E(x)`` (each edge counted once)."""
        pair = 0.0
        pair += np.sum(x[:-1, :] * x[1:, :])
        pair += np.sum(x[:, :-1] * x[:, 1:])
        return float(-self.J * pair - np.sum(self.h * x))


def gibbs_sample(
    model: IsingGrid,
    *,
    n_sweeps: int = 500,
    burn_in: int = 100,
    seed: int = 0,
    x0: Optional[np.ndarray] = None,
    record_every: int = 1,
) -> Tuple[np.ndarray, List[float]]:
    """Gibbs sampling for the Ising grid (MCMC).

    Each sweep resamples every site once from its full conditional

        P(x_i = +1 | x_{-i}) = sigmoid(2 (J * neighbor_sum_i + h_i)).

    Returns ``(marginals, magnetization_trace)`` where ``marginals[i]`` estimates
    ``P(x_i = +1)`` from the post-burn-in samples and the trace records the mean
    spin per recorded sweep (useful for a convergence plot).
    """
    rng = np.random.default_rng(seed)
    H, W = model.H, model.W
    if x0 is None:
        x = rng.choice([-1, 1], size=(H, W)).astype(np.float64)
    else:
        x = x0.astype(np.float64).copy()

    plus_counts = np.zeros((H, W))
    n_recorded = 0
    mag_trace: List[float] = []

    # Precompute a checkerboard so we can update half the lattice at once.
    ii, jj = np.indices((H, W))
    black = (ii + jj) % 2 == 0
    white = ~black

    total = burn_in + n_sweeps
    for sweep in range(total):
        for mask in (black, white):
            nb = model.J * model.neighbor_sum(x) + model.h
            p_plus = 1.0 / (1.0 + np.exp(-2.0 * nb))
            draws = np.where(rng.random((H, W)) < p_plus, 1.0, -1.0)
            x = np.where(mask, draws, x)
        if sweep >= burn_in:
            plus_counts += x > 0
            n_recorded += 1
        if sweep % record_every == 0:
            mag_trace.append(float(x.mean()))

    marginals = plus_counts / max(n_recorded, 1)
    return marginals, mag_trace


def mean_field(
    model: IsingGrid,
    *,
    max_iters: int = 200,
    tol: float = 1e-6,
    damping: float = 0.5,
) -> Tuple[np.ndarray, List[float]]:
    """Naive mean-field variational inference for the Ising grid.

    Uses a fully-factorized ``q(x) = prod_i q_i(x_i)`` parameterised by
    ``mu_i = E_q[x_i] in (-1, 1)``.  The coordinate-ascent fixed-point update is

        mu_i <- tanh(J * sum_{j~i} mu_j + h_i),

    which is exactly the stationarity condition of the mean-field ELBO.  Returns
    ``(marginals_plus, elbo_history)`` where ``marginals_plus[i] = q(x_i=+1)`` and
    the ELBO is a lower bound on ``log Z`` that increases monotonically.
    """
    H, W = model.H, model.W
    mu = np.tanh(model.h)  # sensible init from the field alone

    def elbo(mu: np.ndarray) -> float:
        # E_q[-E(x)] : pairwise uses independence => E[x_i x_j] = mu_i mu_j
        pair = np.sum(mu[:-1, :] * mu[1:, :]) + np.sum(mu[:, :-1] * mu[:, 1:])
        e_neg_energy = model.J * pair + np.sum(model.h * mu)
        # entropy of q for spins in {-1,+1}: p = (1+mu)/2
        p = np.clip((1 + mu) / 2, 1e-12, 1 - 1e-12)
        ent = -np.sum(p * np.log(p) + (1 - p) * np.log(1 - p))
        return float(e_neg_energy + ent)

    history: List[float] = [elbo(mu)]
    for _ in range(max_iters):
        new_mu = np.tanh(model.J * model.neighbor_sum(mu) + model.h)
        new_mu = damping * mu + (1 - damping) * new_mu
        delta = float(np.abs(new_mu - mu).max())
        mu = new_mu
        history.append(elbo(mu))
        if delta < tol:
            break

    marginals_plus = (1 + mu) / 2
    return marginals_plus, history


def denoising_field(noisy: np.ndarray, eta: float) -> np.ndarray:
    """External field ``h_i = eta * y_i`` for image denoising from ``y in {-1,+1}``."""
    return eta * noisy.astype(np.float64)
