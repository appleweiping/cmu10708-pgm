# Solution notes (theory)

Concise correct notes for the theory that underpins each coding assignment in this
repo. These are the derivations the code implements; they are not a substitute for
the official 10-708 written homeworks (which we do not redistribute).

## 1. Exact inference

**Factors and the joint.** A discrete graphical model factorizes an unnormalized
distribution as `P̃(x) = ∏_c φ_c(x_c)`, with `P(x) = P̃(x)/Z` and
`Z = ∑_x ∏_c φ_c(x_c)`. Bayesian networks use conditional probability tables
(CPDs) as factors; Markov random fields use arbitrary nonnegative potentials.

**Variable elimination.** To compute a marginal `P(X_q)` we push sums inside the
product and eliminate the other variables one at a time:
`∑_{x_i} ∏_{c: i∈c} φ_c = ψ`, replacing all factors mentioning `X_i` by a single
new factor `ψ` over their union minus `X_i`. The cost is exponential in the
*induced width* of the elimination order; finding the optimal order is NP-hard, so
we use a min-degree (greedy) heuristic. Summing over **all** variables yields `Z`.

**Sum-product belief propagation.** On a tree, message passing computes all
marginals exactly in two sweeps. Messages:

- variable → factor: `m_{i→c}(x_i) = ∏_{c'∈N(i)\c} m_{c'→i}(x_i)`
- factor → variable: `m_{c→i}(x_i) = ∑_{x_c\i} φ_c(x_c) ∏_{j∈c\i} m_{j→c}(x_j)`

Beliefs are `b_i(x_i) ∝ ∏_{c∈N(i)} m_{c→i}(x_i)`. On a **loopy** graph the same
updates iterated to a fixed point give *loopy BP*, an approximation whose fixed
points are stationary points of the Bethe free energy. Our ASIA experiment shows
loopy BP matches exact VE to ~1e-3 on that (weakly deterministic) network.

## 2. HMMs

An HMM has latent states `z_{1:T}` and observations `x_{1:T}` with
`P(z,x) = π_{z_1} ∏_t A_{z_{t-1},z_t} ∏_t B_{z_t,x_t}`.

**Forward-backward** is sum-product specialized to the chain:
`α_t(k)=P(x_{1:t},z_t=k)`, `β_t(k)=P(x_{t+1:T}|z_t=k)`, giving
`γ_t(k)=P(z_t=k|x)∝α_t(k)β_t(k)` and pairwise posteriors `ξ_t(i,j)`. We rescale
`α` each step and accumulate `log P(x) = ∑_t log c_t` for numerical stability.

**Viterbi** replaces sums with maxima (max-product) in log space to recover the
single most probable path `argmax_z P(z|x)`.

**Learning.** With labelled states, the MLE is just normalized counts (we add
Laplace smoothing). Unlabelled, **Baum-Welch** is EM: the E-step computes `γ, ξ`;
the M-step sets `π ∝ γ_1`, `A_{ij} ∝ ∑_t ξ_t(i,j)`, `B_{ko} ∝ ∑_{t:x_t=o} γ_t(k)`.
EM monotonically increases `log P(x)` (verified empirically).

## 3. MCMC — Gibbs sampling

For the Ising grid `P(x) ∝ exp(J ∑_{i~j} x_i x_j + ∑_i h_i x_i)` with
`x_i ∈ {−1,+1}`, the full conditional of one spin depends only on its neighbours:
`P(x_i=+1 | x_{−i}) = σ(2(J ∑_{j~i} x_j + h_i))`. Gibbs sampling cycles through
sites resampling each from its conditional; the chain's stationary distribution is
`P`. We use a checkerboard (red-black) schedule so half the lattice updates in
parallel, discard burn-in, and estimate marginals by sample averages. For image
denoising the field `h_i = η·y_i` ties spins to the noisy observation `y`.

## 4. Mean-field variational inference

Approximate `P` by a factorized `q(x)=∏_i q_i(x_i)` minimizing `KL(q‖P)`,
equivalently maximizing the ELBO `L(q) = E_q[log P̃(x)] + H(q) ≤ log Z`. For the
Ising model with `μ_i = E_q[x_i]`, coordinate ascent gives the fixed-point update
`μ_i ← tanh(J ∑_{j~i} μ_j + h_i)`. Mean-field ignores posterior correlations, so
it is accurate at weak coupling but **underestimates** correlations and its ELBO
gap grows near the ferromagnetic phase transition — reproduced in HW4.

## 5. Linear-chain CRF

A CRF models `P(y|x) = (1/Z(x)) exp(∑_t U(y_t,x,t) + ∑_t T(y_{t-1},y_t))`
discriminatively, allowing rich overlapping features of `x`. `Z(x)` and marginals
come from forward-backward over the label chain (log space). The conditional
log-likelihood is concave; its gradient is *empirical − expected* feature counts,
so training reduces to L-BFGS with L2 regularization. Because it can exploit
features the generative HMM cannot (suffixes, capitalization, word shape), the CRF
outperforms the HMM on POS tagging on identical data.
