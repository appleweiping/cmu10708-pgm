"""HW2 -- Hidden Markov Models for part-of-speech tagging (Brown corpus).

This exercises the full HMM toolkit on a **real** labelled dataset, the Brown
corpus with the universal 12-tag POS tagset (downloaded via NLTK at runtime):

1. **Supervised MLE.**  We estimate ``pi, A, B`` from the training tag/word
   sequences with add-alpha smoothing and evaluate Viterbi decoding accuracy on
   held-out sentences.  This is exact maximum-likelihood learning for a fully
   observed HMM.
2. **Unsupervised Baum-Welch (EM).**  On a synthetic HMM with known parameters we
   show that EM increases the data log-likelihood monotonically and recovers the
   generating distribution up to label permutation -- the standard sanity check
   for an EM implementation.

Outputs (``results/``):
* ``hw2_pos_report.txt``     -- tagging accuracy, per-tag breakdown, examples.
* ``hw2_em_loglik.png``      -- Baum-Welch log-likelihood curve.
* ``hw2_em_report.txt``      -- EM recovery numbers.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pgm.hmm import HMM, baum_welch, mle_supervised, sample_hmm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")


def load_brown(max_sents=8000, min_count=3):
    """Load Brown tagged sentences with the universal tagset."""
    import nltk

    nltk.data.path.insert(0, os.path.join(DATA, "nltk_data"))
    from nltk.corpus import brown

    sents = brown.tagged_sents(tagset="universal")[:max_sents]
    # build vocab; rare words -> <UNK>
    wc = Counter()
    for s in sents:
        for w, _ in s:
            wc[w.lower()] += 1
    vocab = {"<UNK>": 0}
    for w, c in wc.items():
        if c >= min_count:
            vocab[w] = len(vocab)
    tags = sorted({t for s in sents for _, t in s})
    tag2i = {t: i for i, t in enumerate(tags)}

    def enc(sent):
        xs = [vocab.get(w.lower(), 0) for w, _ in sent]
        ys = [tag2i[t] for _, t in sent]
        return np.array(ys), np.array(xs)

    encoded = [enc(s) for s in sents if len(s) > 0]
    return encoded, vocab, tags, tag2i


def run_pos():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 64)
    log("HW2a: Supervised HMM POS tagging on the Brown corpus")
    log("=" * 64)

    encoded, vocab, tags, tag2i = load_brown()
    rng = np.random.default_rng(0)
    rng.shuffle(encoded)
    n_train = int(0.8 * len(encoded))
    train = encoded[:n_train]
    test = encoded[n_train:]
    K, M = len(tags), len(vocab)
    log(f"  sentences: {len(encoded)}  (train {len(train)}, test {len(test)})")
    log(f"  states (POS tags): {K}   vocabulary (incl <UNK>): {M}")

    state_seqs = [y for y, x in train]
    obs_seqs = [x for y, x in train]
    model = mle_supervised(state_seqs, obs_seqs, K, M, alpha=0.1)

    # Viterbi accuracy on test
    correct = total = 0
    confusion = np.zeros((K, K), dtype=int)
    for y, x in test:
        pred, _ = model.viterbi(x)
        correct += int((pred == y).sum())
        total += len(y)
        for yt, pt in zip(y, pred):
            confusion[yt, pt] += 1
    acc = correct / total
    log(f"\n  token-level Viterbi accuracy on held-out test = {acc:.4f}")
    log(f"  ({correct}/{total} tokens correct)")

    # baseline: most-frequent-tag-per-word
    log("\n  per-tag recall (Viterbi):")
    for i, t in enumerate(tags):
        denom = confusion[i].sum()
        rec = confusion[i, i] / denom if denom else 0.0
        log(f"    {t:6s}: recall {rec:.3f}  (support {denom})")

    # show a decoded example
    inv_tag = {i: t for t, i in tag2i.items()}
    inv_vocab = {i: w for w, i in vocab.items()}
    y, x = test[0]
    pred, _ = model.viterbi(x)
    log("\n  example decoded sentence (word / gold / pred):")
    for j in range(min(12, len(x))):
        mark = "" if pred[j] == y[j] else "  <-- wrong"
        log(f"    {inv_vocab[x[j]]:14s} {inv_tag[y[j]]:6s} {inv_tag[pred[j]]:6s}{mark}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(confusion / confusion.sum(axis=1, keepdims=True).clip(1), cmap="viridis")
    ax.set_xticks(range(K)); ax.set_xticklabels(tags, rotation=90, fontsize=7)
    ax.set_yticks(range(K)); ax.set_yticklabels(tags, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("gold")
    ax.set_title(f"Brown POS tagging confusion (row-normalised)\nViterbi acc = {acc:.4f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "hw2_pos_confusion.png"), dpi=120)
    log(f"\n  [write] results/hw2_pos_confusion.png")

    with open(os.path.join(RESULTS, "hw2_pos_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return acc


def run_em():
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("\n" + "=" * 64)
    log("HW2b: Unsupervised parameter learning with Baum-Welch (EM)")
    log("=" * 64)

    pi = np.array([0.6, 0.4])
    A = np.array([[0.7, 0.3], [0.2, 0.8]])
    B = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    true = HMM(pi, A, B)
    rng = np.random.default_rng(7)
    seqs = [sample_hmm(true, 50, rng)[1] for _ in range(200)]

    model, hist = baum_welch(seqs, K=2, M=3, n_iter=200, tol=1e-6, seed=3, verbose=False)
    log(f"  generated 200 sequences of length 50 from a known 2-state / 3-symbol HMM")
    log(f"  EM ran {len(hist)} iterations")
    log(f"  initial total loglik = {hist[0]:.2f}")
    log(f"  final   total loglik = {hist[-1]:.2f}")
    diffs = np.diff(hist)
    log(f"  monotonic increase: {(diffs > -1e-6).all()}  (min step {diffs.min():.2e})")
    ll_true = sum(true.loglikelihood(o) for o in seqs)
    log(f"  loglik under TRUE params = {ll_true:.2f}")

    # align learned states to true states by best B-match (permutation)
    from itertools import permutations

    best_perm, best_err = None, np.inf
    for perm in permutations(range(2)):
        err = np.abs(model.B[list(perm)] - true.B).sum()
        if err < best_err:
            best_err, best_perm = err, perm
    Bp = model.B[list(best_perm)]
    Ap = model.A[np.ix_(best_perm, best_perm)]
    log("\n  recovered emission matrix B (rows=state, aligned to truth):")
    log(f"    true      : {np.round(true.B, 3).tolist()}")
    log(f"    learned   : {np.round(Bp, 3).tolist()}")
    log(f"    max |B_true - B_learned| = {np.abs(Bp - true.B).max():.3f}")
    log(f"    max |A_true - A_learned| = {np.abs(Ap - true.A).max():.3f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist, marker="o", ms=3, color="#3b6ea5")
    ax.axhline(ll_true, ls="--", color="#888", label=f"loglik @ true params ({ll_true:.0f})")
    ax.set_xlabel("EM iteration"); ax.set_ylabel("total data log-likelihood")
    ax.set_title("Baum-Welch (EM) convergence on synthetic HMM")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "hw2_em_loglik.png"), dpi=120)
    log(f"\n  [write] results/hw2_em_loglik.png")

    with open(os.path.join(RESULTS, "hw2_em_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    acc = run_pos()
    run_em()
    print(f"\nDone. POS Viterbi accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
