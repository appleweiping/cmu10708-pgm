"""HW5 -- Linear-chain CRF for sequence labelling (POS tagging on Brown).

The 10-708 CRF homework asks you to build a CRF module and train it by
maximising the conditional log-likelihood.  Here we apply our linear-chain CRF
to **the same Brown POS-tagging task used for the HMM in HW2**, so the two
models are directly comparable on identical train/test splits.

Unlike the generative HMM (which only sees the word identity), the CRF uses rich,
overlapping **discriminative features** of the observation -- word identity,
lowercased form, prefixes/suffixes, capitalization, hyphenation, digits, and
word-shape -- which is exactly why CRFs typically outperform HMMs for tagging.

We train with L-BFGS on the L2-regularised conditional log-likelihood (gradients
from forward-backward, verified against finite differences in the tests) and
report Viterbi token accuracy, contrasting it with the HMM baseline.

Outputs (``results/``):
* ``hw5_crf_report.txt``   -- accuracy, HMM-vs-CRF comparison, examples.
* ``hw5_crf_training.png`` -- L-BFGS objective (neg log-likelihood) curve.
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

from pgm.crf import LinearChainCRF, SparseSeq
from pgm.hmm import mle_supervised

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")


def word_shape(w):
    s = []
    for ch in w:
        if ch.isupper():
            s.append("X")
        elif ch.islower():
            s.append("x")
        elif ch.isdigit():
            s.append("d")
        else:
            s.append("-")
    # collapse runs
    out = []
    for c in s:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def word_features(w):
    """Return a list of string feature keys for a token."""
    lw = w.lower()
    feats = [
        f"word={lw}",
        f"suf3={lw[-3:]}",
        f"suf2={lw[-2:]}",
        f"pre3={lw[:3]}",
        f"shape={word_shape(w)}",
    ]
    feats.append("cap=1" if w[:1].isupper() else "cap=0")
    feats.append("hasdigit=1" if any(c.isdigit() for c in w) else "hasdigit=0")
    feats.append("hyphen=1" if "-" in w else "hyphen=0")
    feats.append("BIAS")
    return feats


def load_brown_words(max_sents=6000):
    import nltk

    nltk.data.path.insert(0, os.path.join(DATA, "nltk_data"))
    from nltk.corpus import brown

    sents = brown.tagged_sents(tagset="universal")[:max_sents]
    return [[(w, t) for w, t in s if len(w) > 0] for s in sents if len(s) > 0]


def build_feature_index(train_sents, min_count=2):
    fc = Counter()
    for s in train_sents:
        for w, _ in s:
            for f in word_features(w):
                fc[f] += 1
    feat2i = {}
    for f, c in fc.items():
        if c >= min_count or f in ("BIAS",):
            feat2i[f] = len(feat2i)
    return feat2i


def encode(sents, feat2i, tag2i):
    """Encode each sentence as a (SparseSeq, label array) pair.

    The sparse representation stores, per token, only the indices of the active
    binary features -- avoiding the dense (T x n_features) matrix entirely.
    """
    out = []
    for s in sents:
        tokens = []
        y = np.zeros(len(s), dtype=int)
        for t, (w, tag) in enumerate(s):
            idx = [feat2i[f] for f in word_features(w) if f in feat2i]
            tokens.append(np.array(idx, dtype=np.intp))
            y[t] = tag2i[tag]
        out.append((SparseSeq(tokens), y))
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 64)
    log("HW5: Linear-chain CRF for POS tagging (Brown corpus)")
    log("=" * 64)

    sents = load_brown_words(max_sents=4000)
    rng = np.random.default_rng(0)
    rng.shuffle(sents)
    n_train = int(0.8 * len(sents))
    train_sents, test_sents = sents[:n_train], sents[n_train:]

    tags = sorted({t for s in sents for _, t in s})
    tag2i = {t: i for i, t in enumerate(tags)}
    inv_tag = {i: t for t, i in tag2i.items()}
    feat2i = build_feature_index(train_sents, min_count=2)
    log(f"  sentences: {len(sents)} (train {len(train_sents)}, test {len(test_sents)})")
    log(f"  labels: {len(tags)}   CRF features: {len(feat2i)}")

    train = encode(train_sents, feat2i, tag2i)
    test = encode(test_sents, feat2i, tag2i)

    # --- train CRF ------------------------------------------------------
    crf = LinearChainCRF(n_features=len(feat2i), n_labels=len(tags), l2=1.0)
    log("\n  training CRF with L-BFGS (max 60 iters)...")
    crf.fit(train, max_iter=60, verbose=False)
    log(f"  L-BFGS finished: {crf.opt_result.nit} iters, "
        f"success={crf.opt_result.success}")
    log(f"  objective (neg cond. loglik) start={crf.history[0]:.1f} "
        f"end={crf.history[-1]:.1f}")

    crf_acc, crf_ll = crf.score_sequences(test)
    log(f"\n  [CRF] token accuracy on test = {crf_acc:.4f}")
    log(f"  [CRF] mean conditional log-likelihood / token = {crf_ll:.4f}")

    # --- HMM baseline on the SAME split (word-identity only) -----------
    # build word vocab for the HMM
    wc = Counter(w.lower() for s in train_sents for w, _ in s)
    vocab = {"<UNK>": 0}
    for w, c in wc.items():
        if c >= 2:
            vocab[w] = len(vocab)

    def enc_hmm(s):
        xs = np.array([vocab.get(w.lower(), 0) for w, _ in s])
        ys = np.array([tag2i[t] for _, t in s])
        return ys, xs

    hmm_train = [enc_hmm(s) for s in train_sents]
    hmm_test = [enc_hmm(s) for s in test_sents]
    hmm = mle_supervised(
        [y for y, x in hmm_train], [x for y, x in hmm_train],
        K=len(tags), M=len(vocab), alpha=0.1,
    )
    correct = total = 0
    for y, x in hmm_test:
        pred, _ = hmm.viterbi(x)
        correct += int((pred == y).sum())
        total += len(y)
    hmm_acc = correct / total
    log(f"\n  [HMM baseline] token accuracy on same split = {hmm_acc:.4f}")
    log(f"  ==> CRF improves over HMM by {100*(crf_acc-hmm_acc):+.2f} percentage points")

    # --- example ---------------------------------------------------------
    X, y = test[0]
    pred = crf.predict(X)
    log("\n  example CRF decode (gold / pred over first 12 tokens):")
    words = [w for w, _ in test_sents[0]]
    for j in range(min(12, len(y))):
        mark = "" if pred[j] == y[j] else "  <-- wrong"
        log(f"    {words[j]:14s} {inv_tag[y[j]]:6s} {inv_tag[pred[j]]:6s}{mark}")

    # --- training curve --------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(crf.history, color="#3b6ea5")
    ax.set_xlabel("L-BFGS function evaluation")
    ax.set_ylabel("regularised negative conditional log-likelihood")
    ax.set_title(f"CRF training (final token acc {crf_acc:.4f})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "hw5_crf_training.png"), dpi=120)
    log(f"\n  [write] results/hw5_crf_training.png")

    with open(os.path.join(RESULTS, "hw5_crf_report.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"\nDone. CRF acc {crf_acc:.4f} vs HMM acc {hmm_acc:.4f}")


if __name__ == "__main__":
    main()
