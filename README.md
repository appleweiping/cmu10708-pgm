# CMU 10-708 — Probabilistic Graphical Models (coding solutions)

> A correct, from-scratch NumPy implementation of the coding cores of
> **10-708 — Probabilistic Graphical Models** (Carnegie Mellon University), part of a
> [csdiy.wiki](https://csdiy.wiki/) full-catalog build. Exact and approximate
> inference, parameter learning, and structured prediction — with **measured**
> results on real datasets.

![status](https://img.shields.io/badge/status-complete-brightgreen)
![language](https://img.shields.io/badge/python-3.11-informational)
![license](https://img.shields.io/badge/license-MIT-blue)

## Overview

10-708 is CMU's graduate course on probabilistic graphical models (representation,
exact/approximate inference, and learning). This repo re-implements the
*algorithmic* content of its programming homeworks as a single small library,
`pgm/`, and drives it with five self-contained homework runners that produce real
numbers and figures. Everything is NumPy/SciPy, CPU-only, and validated by a unit
test suite that checks each algorithm against a brute-force or finite-difference
oracle.

The five coding themes mirror the topics of the CMU offerings
([Spring-2019, Xing](https://sailinglab.github.io/pgm-spring-2019/) and
[Spring-2021, Gormley](http://www.cs.cmu.edu/~mgormley/courses/10708/)): exact
inference (variable elimination, belief propagation), HMMs, MCMC/Gibbs sampling,
mean-field variational inference, and CRFs.

## Results (measured on CPU, `OMP_NUM_THREADS=3`)

| HW | Coding topic | Task / dataset | Result (measured) |
|----|--------------|----------------|-------------------|
| 1 | Exact inference (VE + BP) | ASIA Bayesian network | VE = exhaustive enumeration to **1e-16**; loopy BP within **4e-3** of exact |
| 2a | HMM + Viterbi (supervised MLE) | Brown corpus POS tagging | **94.0%** token accuracy (32374/34441) on held-out test |
| 2b | Baum-Welch (EM) | synthetic HMM | monotone log-likelihood; recovers A, B to **0.031** |
| 3 | Gibbs sampling (MCMC) | binary image denoising (15% noise) | error **15.4% → 1.27%**; 580 pixels corrected |
| 4 | Mean-field variational inference | Ising model | error **7e-4** (weak) → **0.19** (near phase transition); Gibbs stays <0.008 |
| 5 | Linear-chain CRF (L-BFGS) | Brown corpus POS tagging | **95.4%** token accuracy (**+2.11 pp** vs HMM baseline on same split) |

Selected figures (in [`results/`](results/)):

- `hw1_ve_vs_bp.png` — exact VE vs loopy BP marginals on ASIA.
- `hw2_pos_confusion.png`, `hw2_em_loglik.png` — POS confusion matrix; EM curve.
- `hw3_denoise.png` — clean / noisy / Gibbs / mean-field denoising panels.
- `hw4_mf_vs_exact.png` — mean-field & Gibbs error vs coupling strength.
- `hw5_crf_training.png` — CRF L-BFGS objective curve.

## Implemented assignments

- [x] **HW1 — Exact inference.** Discrete factors + factor algebra; variable
  elimination (min-degree ordering, evidence, partition function); sum-product
  belief propagation on factor graphs (exact on trees, loopy BP otherwise).
- [x] **HW2 — Hidden Markov Models.** Scaled forward-backward, Viterbi (log-space
  max-product), supervised MLE, and Baum-Welch (EM).
- [x] **HW3 — MCMC.** Gibbs sampling for the Ising model, applied to binary image
  denoising with a red-black update schedule and a mixing diagnostic.
- [x] **HW4 — Variational inference.** Naive mean-field for the Ising model; ELBO
  as a `log Z` lower bound; accuracy-vs-coupling study against exact & Gibbs.
- [x] **HW5 — Structured prediction.** Linear-chain CRF with forward-backward in
  log space, L2-regularised conditional log-likelihood, L-BFGS training, sparse
  features, and Viterbi decoding; benchmarked against the HMM baseline.

Concise correct notes for the underlying theory are in [`NOTES.md`](NOTES.md).

## Project structure

```
cmu10708-pgm/
├── pgm/                    # the library
│   ├── factor.py           # discrete factors + algebra
│   ├── inference.py        # variable elimination, (loopy) belief propagation
│   ├── hmm.py              # forward-backward, Viterbi, MLE, Baum-Welch
│   ├── ising.py            # Ising model, Gibbs sampling, mean-field VI
│   └── crf.py              # linear-chain CRF (forward-backward, L-BFGS)
├── hw1_exact_inference/    # per-homework runners (produce results/)
├── hw2_hmm/
├── hw3_mcmc/
├── hw4_variational/
├── hw5_crf/
├── tests/                  # 23 unit tests (correctness oracles)
├── results/                # measured outputs + figures (committed)
├── NOTES.md                # theory solution notes
└── run_all.py
```

## How to run

```bash
# Python 3.11. Reuse the shared csdiy venv, or:
python -m pip install -r requirements.txt

# run the unit tests (VE vs brute force, BP vs VE, HMM vs enumeration,
# Viterbi vs brute force, Gibbs vs exact, MF ELBO monotonicity,
# CRF analytic gradient vs finite differences, sparse == dense):
python -m pytest tests/ -q

# reproduce every homework's numbers and figures (HW2 & HW5 download the
# Brown corpus via NLTK on first run):
python run_all.py

# or run one homework at a time:
python hw1_exact_inference/run.py
python hw2_hmm/run.py
python hw3_mcmc/run.py
python hw4_variational/run.py
python hw5_crf/run.py
```

## Verification

Every core algorithm is checked against an independent oracle:

- **Variable elimination** vs exhaustive enumeration of the joint (ASIA: agreement
  to 1e-16; random chains: to 1e-10).
- **Belief propagation** vs VE (exact on trees; loopy BP within tolerance).
- **HMM forward** and **Viterbi** vs brute-force enumeration over all state paths.
- **Baum-Welch** log-likelihood is monotone non-decreasing (verified numerically).
- **Gibbs sampling** vs exact enumeration of small Ising grids.
- **Mean-field** ELBO is monotone and, at weak coupling, close to exact.
- **CRF** analytic gradient vs central finite differences (atol 1e-5); the sparse
  feature path is asserted bit-for-bit equal to the dense one.

`results/` holds the actual run logs (`*_report.txt`), CSVs, and figures. The
per-homework `run.py` scripts re-assert the key oracle checks at runtime, so a
regression fails loudly.

## Tech stack

Python 3.11, NumPy, SciPy (L-BFGS, `logsumexp`), Matplotlib, NLTK (Brown corpus),
pytest. CPU-only; no GPU required.

## Key ideas / what I learned

- Exact inference is *one* algorithm (sum-product) viewed two ways: eliminating
  variables one at a time (VE) or passing messages (BP); on trees they agree
  exactly, and loopy BP is the same fixed-point iteration on cyclic graphs.
- The HMM is a chain graphical model: forward-backward *is* sum-product, Viterbi
  *is* max-product, and Baum-Welch *is* EM with those as its E-step.
- MCMC (Gibbs) and variational inference are the two great families of approximate
  inference — Gibbs trades compute for asymptotic exactness, mean-field trades
  accuracy for a fast deterministic bound, and each wins in a different regime
  (HW4 makes the trade-off quantitative around the Ising phase transition).
- Discriminative structured models (CRFs) beat generative ones (HMMs) on tagging
  precisely because they can pack in overlapping, non-independent features of the
  input.

## Credits & license

Based on the assignments of **CMU 10-708 Probabilistic Graphical Models** by
Eric P. Xing and Matt Gormley (Carnegie Mellon University). This repository is an
independent educational reimplementation; all course materials, datasets, and
specifications belong to their original authors. The Brown corpus is distributed
with NLTK. Original code in this repo is released under the [MIT License](LICENSE).
