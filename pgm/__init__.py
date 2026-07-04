"""pgm: a small, correct probabilistic graphical models library.

Implements the coding cores of CMU 10-708 (Probabilistic Graphical Models):

- Discrete factors and factor operations (``factor``)
- Exact inference: variable elimination and belief propagation (``inference``)
- Hidden Markov Models: forward-backward, Viterbi, Baum-Welch/EM (``hmm``)
- MCMC: Gibbs sampling for the Ising model (``ising``)
- Mean-field variational inference for the Ising model (``ising``)
- Linear-chain CRFs with the forward-backward algorithm (``crf``)

Everything is NumPy-only (CRF training also uses SciPy's L-BFGS), works on CPU,
and is exercised by unit tests in ``tests/`` and the per-homework runners.
"""

from .factor import Factor

__all__ = ["Factor"]
