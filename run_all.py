"""Run every homework end-to-end and (re)generate all artefacts in results/.

Usage:
    python run_all.py

Each homework is self-contained; this just calls them in order.  HW2 and HW5
download the Brown corpus via NLTK on first run.
"""
import runpy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNERS = [
    "hw1_exact_inference/run.py",
    "hw2_hmm/run.py",
    "hw3_mcmc/run.py",
    "hw4_variational/run.py",
    "hw5_crf/run.py",
]


def main():
    for r in RUNNERS:
        path = os.path.join(HERE, r)
        print("\n" + "#" * 70)
        print("# RUNNING", r)
        print("#" * 70)
        runpy.run_path(path, run_name="__main__")


if __name__ == "__main__":
    main()
