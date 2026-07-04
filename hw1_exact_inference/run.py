"""HW1 -- Exact inference on the classic ASIA Bayesian network.

We build the well-known ASIA network (Lauritzen & Spiegelhalter, 1988), a
staple example in graphical-model courses, as a set of discrete factors (CPDs),
then answer probabilistic queries two ways:

1. **Variable elimination** (sum-product), with and without evidence.
2. **Belief propagation** on the corresponding factor graph.

Because the moralized ASIA graph is loopy, we compare BP marginals against the
exact VE marginals to see how close loopy BP gets, and we verify VE itself
against exhaustive enumeration of the joint (8 binary variables = 256 states).

Outputs (written to ``results/``):
* ``hw1_marginals.csv``   -- prior and posterior marginals from VE.
* ``hw1_ve_vs_bp.png``    -- exact VE vs loopy BP marginals.
* ``hw1_report.txt``      -- a text summary of the queries and numbers.
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

from pgm.factor import Factor
from pgm.inference import FactorGraph, variable_elimination

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Variable index: 0 = False (no), 1 = True (yes)
VARS = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"]


def cpd(scope, table):
    return Factor(scope, np.array(table, dtype=float))


def build_asia():
    """Return the list of CPD factors of the ASIA network."""
    factors = []
    # P(asia): visited Asia?  (rare)
    factors.append(cpd(["asia"], [0.99, 0.01]))
    # P(tub | asia): tuberculosis
    factors.append(cpd(["asia", "tub"], [[0.99, 0.01], [0.95, 0.05]]))
    # P(smoke)
    factors.append(cpd(["smoke"], [0.5, 0.5]))
    # P(lung | smoke): lung cancer
    factors.append(cpd(["smoke", "lung"], [[0.99, 0.01], [0.9, 0.1]]))
    # P(bronc | smoke): bronchitis
    factors.append(cpd(["smoke", "bronc"], [[0.7, 0.3], [0.4, 0.6]]))
    # P(either | lung, tub): tuberculosis-or-cancer (logical OR)
    either = np.zeros((2, 2, 2))  # [lung, tub, either]
    for l in (0, 1):
        for t in (0, 1):
            e = 1 if (l or t) else 0
            either[l, t, e] = 1.0
    factors.append(cpd(["lung", "tub", "either"], either))
    # P(xray | either)
    factors.append(cpd(["either", "xray"], [[0.95, 0.05], [0.02, 0.98]]))
    # P(dysp | bronc, either): dyspnoea (shortness of breath)
    dysp = np.array(
        [
            [[0.9, 0.1], [0.2, 0.8]],  # bronc=0: either=0, either=1
            [[0.2, 0.8], [0.1, 0.9]],  # bronc=1
        ]
    )
    factors.append(cpd(["bronc", "either", "dysp"], dysp))
    return factors


def exhaustive_marginal(factors, var):
    """Exact P(var) by enumerating all 2^8 joint states -- the ground truth."""
    total = np.zeros(2)
    for bits in itertools.product([0, 1], repeat=len(VARS)):
        a = dict(zip(VARS, bits))
        p = 1.0
        for f in factors:
            p *= f.get_value(a)
        total[a[var]] += p
    return total / total.sum()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    factors = build_asia()
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 64)
    log("HW1: Exact inference on the ASIA Bayesian network")
    log("=" * 64)

    # --- sanity: VE marginals must match exhaustive enumeration ---------
    log("\n[check] variable elimination vs exhaustive enumeration (prior):")
    max_err = 0.0
    priors = {}
    for v in VARS:
        ve = variable_elimination(factors, [v]).normalize().table
        bf = exhaustive_marginal(factors, v)
        err = float(np.abs(ve - bf).max())
        max_err = max(max_err, err)
        priors[v] = ve
        log(f"  P({v:8s}=yes) = {ve[1]:.5f}   (|VE - brute| = {err:.2e})")
    log(f"  max |VE - exhaustive| over all variables = {max_err:.2e}")
    assert max_err < 1e-10, "VE disagrees with exhaustive enumeration!"

    # --- diagnostic query: patient has dyspnoea and a positive x-ray ----
    evidence = {"dysp": 1, "xray": 1}
    log(f"\n[query] posteriors given evidence {evidence}:")
    posteriors = {}
    for v in ("tub", "lung", "bronc", "smoke", "asia"):
        post = variable_elimination(factors, [v], evidence=evidence).normalize().table
        posteriors[v] = post
        log(f"  P({v:8s}=yes | evidence) = {post[1]:.5f}   (prior {priors[v][1]:.5f})")

    # --- belief propagation on the (loopy) factor graph ----------------
    log("\n[compare] loopy belief propagation vs exact VE (prior marginals):")
    fg = FactorGraph(factors)
    beliefs, iters = fg.run_bp(max_iters=200, tol=1e-10, damping=0.3)
    log(f"  loopy BP ran {iters} iterations")
    bp_err = {}
    for v in VARS:
        b = beliefs[v]
        err = float(np.abs(b - priors[v]).max())
        bp_err[v] = err
        log(f"  P({v:8s}=yes): VE={priors[v][1]:.5f}  BP={b[1]:.5f}  |diff|={err:.2e}")
    log(f"  max |BP - VE| = {max(bp_err.values()):.2e}")

    # --- write marginals CSV -------------------------------------------
    csv_path = os.path.join(RESULTS, "hw1_marginals.csv")
    with open(csv_path, "w") as fh:
        fh.write("variable,prior_yes,posterior_yes_given_evidence,bp_yes\n")
        for v in VARS:
            post = posteriors.get(v, [np.nan, np.nan])[1]
            fh.write(f"{v},{priors[v][1]:.6f},{post:.6f},{beliefs[v][1]:.6f}\n")
    log(f"\n[write] {csv_path}")

    # --- figure: VE vs BP ----------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(VARS))
    ve_vals = [priors[v][1] for v in VARS]
    bp_vals = [beliefs[v][1] for v in VARS]
    ax.bar(x - 0.2, ve_vals, width=0.4, label="exact (VE)", color="#3b6ea5")
    ax.bar(x + 0.2, bp_vals, width=0.4, label="loopy BP", color="#d1885c")
    ax.set_xticks(x)
    ax.set_xticklabels(VARS, rotation=30, ha="right")
    ax.set_ylabel("P(variable = yes)")
    ax.set_title("ASIA network: exact VE vs loopy belief propagation")
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(RESULTS, "hw1_ve_vs_bp.png")
    fig.savefig(fig_path, dpi=120)
    log(f"[write] {fig_path}")

    with open(os.path.join(RESULTS, "hw1_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log("\nDone.")


if __name__ == "__main__":
    main()
