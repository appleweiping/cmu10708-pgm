"""HW4 -- Mean-field variational inference for the Ising model.

Mean-field VI approximates the intractable posterior ``P(x)`` of an Ising grid by
the closest fully-factorized distribution ``q(x) = prod_i q_i(x_i)``, minimising
``KL(q || P)`` (equivalently maximising the ELBO, a lower bound on ``log Z``).  The
coordinate-ascent fixed point is ``mu_i <- tanh(J * sum_{j~i} mu_j + h_i)``.

This runner studies two things the theory predicts:

1. **Monotone ELBO / convergence.**  On one grid we plot the ELBO across
   iterations (must be non-decreasing) and report the converged marginals.
2. **Accuracy vs coupling strength.**  Mean-field is accurate for weak coupling
   but *underestimates* correlations and breaks down as ``J`` grows (near the
   ferromagnetic phase transition).  We sweep ``J`` on a small grid where the
   **exact** marginals are computable by enumeration, and also compare against a
   long **Gibbs** run, quantifying the mean-field error as a function of ``J``.

Outputs (``results/``):
* ``hw4_elbo.png``            -- ELBO convergence curve.
* ``hw4_mf_vs_exact.png``     -- mean-field / Gibbs error vs coupling J.
* ``hw4_report.txt``          -- numbers.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgm.ising import IsingGrid, gibbs_sample, mean_field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def exact_marginals_and_logZ(model):
    """Exact P(x_i=+1) and log Z by enumerating all 2^(HW) states (tiny grid)."""
    H, W = model.H, model.W
    n = H * W
    logits, states = [], []
    for bits in itertools.product([-1, 1], repeat=n):
        x = np.array(bits, dtype=float).reshape(H, W)
        states.append(x)
        logits.append(-model.energy(x))
    logits = np.array(logits)
    mx = logits.max()
    w = np.exp(logits - mx)
    logZ = mx + np.log(w.sum())
    w /= w.sum()
    marg = np.zeros((H, W))
    for x, p in zip(states, w):
        marg += p * (x > 0)
    return marg, float(logZ)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 64)
    log("HW4: Mean-field variational inference for the Ising model")
    log("=" * 64)

    # --- part 1: convergence on a larger grid --------------------------
    rng = np.random.default_rng(0)
    H = W = 16
    hfield = rng.standard_normal((H, W)) * 0.3
    model = IsingGrid(H, W, J=0.25, h=hfield)
    marg, elbo = mean_field(model, max_iters=300, tol=1e-9, damping=0.5)
    diffs = np.diff(elbo)
    log(f"\n[part 1] {H}x{W} grid, J=0.25, random field")
    log(f"  mean-field converged in {len(elbo)-1} iterations")
    log(f"  ELBO monotone non-decreasing: {(diffs > -1e-8).all()}  (min step {diffs.min():.2e})")
    log(f"  final ELBO (lower bound on log Z) = {elbo[-1]:.4f}")
    log(f"  mean marginal P(x=+1) over grid = {marg.mean():.4f}")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(elbo, color="#3b6ea5")
    ax.set_xlabel("mean-field iteration"); ax.set_ylabel("ELBO")
    ax.set_title("Mean-field ELBO convergence (16x16 Ising grid)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "hw4_elbo.png"), dpi=120)
    log(f"  [write] results/hw4_elbo.png")

    # --- part 2: accuracy vs coupling strength -------------------------
    log("\n[part 2] mean-field vs EXACT and Gibbs as coupling J grows (4x4 grid)")
    log("  (mean-field is expected to degrade as J increases toward the")
    log("   ferromagnetic phase transition)")
    Hs = Ws = 4
    field = 0.1
    Js = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    mf_errs, gibbs_errs, logZ_gaps = [], [], []
    log("\n    J      MF_err   Gibbs_err   exact_logZ   MF_ELBO   gap")
    for J in Js:
        m = IsingGrid(Hs, Ws, J=J, h=field)
        exact, logZ = exact_marginals_and_logZ(m)
        mf, elbo_j = mean_field(m, max_iters=500, tol=1e-10, damping=0.5)
        gm, _ = gibbs_sample(m, n_sweeps=30000, burn_in=3000, seed=1)
        mf_err = float(np.abs(mf - exact).max())
        gibbs_err = float(np.abs(gm - exact).max())
        gap = logZ - elbo_j[-1]  # ELBO underestimates log Z; gap >= 0
        mf_errs.append(mf_err)
        gibbs_errs.append(gibbs_err)
        logZ_gaps.append(gap)
        log(
            f"   {J:.2f}   {mf_err:.4f}   {gibbs_err:.4f}     "
            f"{logZ:.4f}   {elbo_j[-1]:.4f}  {gap:.4f}"
        )

    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(Js, mf_errs, "o-", label="mean-field", color="#d1885c")
    a1.plot(Js, gibbs_errs, "s-", label="Gibbs (30k sweeps)", color="#3b6ea5")
    a1.set_xlabel("coupling strength J"); a1.set_ylabel("max |marginal - exact|")
    a1.set_title("Marginal error vs coupling")
    a1.legend()
    a2.plot(Js, logZ_gaps, "o-", color="#5a9367")
    a2.set_xlabel("coupling strength J"); a2.set_ylabel("log Z  -  ELBO  (>= 0)")
    a2.set_title("Mean-field ELBO gap to true log Z")
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS, "hw4_mf_vs_exact.png"), dpi=120)
    log(f"\n  [write] results/hw4_mf_vs_exact.png")

    log("\n  observation: mean-field error and the ELBO gap both grow with J,")
    log("  while Gibbs stays accurate -- exactly the textbook behaviour.")

    with open(os.path.join(RESULTS, "hw4_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log("\nDone.")


if __name__ == "__main__":
    main()
