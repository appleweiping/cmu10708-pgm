"""HW3 -- MCMC (Gibbs sampling) for binary image denoising.

Classic application of Gibbs sampling in graphical models: recover a clean
binary image from a noisy observation using an Ising prior.  We model each pixel
as a spin ``x_i in {-1,+1}`` with the pairwise energy

    E(x) = -J * sum_{(i,j)} x_i x_j  -  sum_i (eta * y_i) x_i,

where ``y`` is the noisy image and ``J`` encourages neighbouring pixels to agree.
The posterior over clean images is exactly an Ising model, and **Gibbs sampling**
draws from it; the posterior-mean spin at each pixel gives the denoised image.

We use a real, deterministic binary source image (a rendered glyph), corrupt it
with independent bit-flip noise, run the sampler, and measure how many pixel
errors are corrected.  For comparison we also denoise with **mean-field VI**
(HW4's method) so the two approximate-inference schemes can be contrasted.

Outputs (``results/``):
* ``hw3_denoise.png``     -- clean / noisy / Gibbs / mean-field panels.
* ``hw3_mag_trace.png``   -- Markov-chain magnetization trace (mixing diagnostic).
* ``hw3_report.txt``      -- error rates and settings.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgm.ising import IsingGrid, denoising_field, gibbs_sample, mean_field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def make_source_image(H=64, W=64):
    """A deterministic binary image with large connected regions.

    Thick shapes (a filled disk, a filled rectangle, a diagonal bar) are exactly
    the regime where the Ising smoothness prior helps: neighbouring pixels within
    a region should agree, and the prior suppresses isolated flipped pixels.
    """
    img = -np.ones((H, W))
    yy, xx = np.indices((H, W))
    # filled disk (top-left)
    cy, cx, r = H * 0.33, W * 0.33, min(H, W) * 0.22
    img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] = 1
    # filled rectangle (bottom-right)
    img[int(H * 0.55) : int(H * 0.85), int(W * 0.55) : int(W * 0.9)] = 1
    # thick diagonal bar
    for d in range(-2, 3):
        rr = np.arange(H)
        cc = (rr * 0.7).astype(int) + d + int(W * 0.05)
        ok = (cc >= 0) & (cc < W)
        img[rr[ok], cc[ok]] = 1
    return img


def error_rate(a, b):
    return float(np.mean(a != b))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 64)
    log("HW3: Gibbs sampling (MCMC) for binary image denoising")
    log("=" * 64)

    rng = np.random.default_rng(0)
    clean = make_source_image(64, 64)
    noise_level = 0.15
    flip = rng.random(clean.shape) < noise_level
    noisy = clean.copy()
    noisy[flip] *= -1
    log(f"  image {clean.shape}, injected bit-flip noise p = {noise_level}")
    log(f"  noisy image pixel error rate = {error_rate(noisy, clean):.4f}")

    J = 1.0
    eta = 1.0
    field = denoising_field(noisy, eta)
    model = IsingGrid(H=clean.shape[0], W=clean.shape[1], J=J, h=field)
    log(f"  Ising prior: J = {J}, observation coupling eta = {eta}")

    # --- Gibbs sampling ------------------------------------------------
    marg_plus, mag_trace = gibbs_sample(
        model, n_sweeps=400, burn_in=100, seed=1, x0=noisy.copy()
    )
    gibbs_img = np.where(marg_plus > 0.5, 1.0, -1.0)
    err_gibbs = error_rate(gibbs_img, clean)
    log(f"\n  [Gibbs] 400 sweeps (100 burn-in)")
    log(f"  [Gibbs] denoised pixel error rate = {err_gibbs:.4f}")
    log(f"  [Gibbs] corrected {int((error_rate(noisy,clean)-err_gibbs)*clean.size)} pixel errors")

    # --- Mean-field VI for comparison ----------------------------------
    mf_plus, elbo = mean_field(model, max_iters=200, tol=1e-7, damping=0.5)
    mf_img = np.where(mf_plus > 0.5, 1.0, -1.0)
    err_mf = error_rate(mf_img, clean)
    log(f"\n  [MeanField] {len(elbo)-1} iterations, final ELBO = {elbo[-1]:.2f}")
    log(f"  [MeanField] denoised pixel error rate = {err_mf:.4f}")

    # --- panels figure -------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, img, title in zip(
        axes,
        [clean, noisy, gibbs_img, mf_img],
        [
            "clean source",
            f"noisy (err {error_rate(noisy,clean):.3f})",
            f"Gibbs (err {err_gibbs:.3f})",
            f"mean-field (err {err_mf:.3f})",
        ],
    ):
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Binary image denoising with an Ising prior", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "hw3_denoise.png"), dpi=120, bbox_inches="tight")
    log(f"\n  [write] results/hw3_denoise.png")

    # --- mixing diagnostic ---------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    ax2.plot(mag_trace, color="#3b6ea5", lw=1)
    ax2.axvline(100, ls="--", color="#c44", label="end of burn-in")
    ax2.set_xlabel("Gibbs sweep"); ax2.set_ylabel("mean spin (magnetization)")
    ax2.set_title("Gibbs sampler magnetization trace (mixing diagnostic)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(RESULTS, "hw3_mag_trace.png"), dpi=120)
    log(f"  [write] results/hw3_mag_trace.png")

    with open(os.path.join(RESULTS, "hw3_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"\nDone. Gibbs error {err_gibbs:.4f}, mean-field error {err_mf:.4f}")


if __name__ == "__main__":
    main()
