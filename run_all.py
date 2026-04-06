"""
Master script: run all three experiments sequentially.

Usage:
    conda activate qng-baseline
    python run_all.py
"""

import matplotlib
matplotlib.use("Agg")

print("=" * 70)
print("  QNG EUCLIDEAN BASELINE -- Running all experiments")
print("=" * 70)

from experiments.function_fitting import run_1d, run_2d
from experiments.vqe import run as run_vqe
from experiments.classification import run as run_cls

run_1d()
run_2d()
run_vqe()
run_cls()

print("\n" + "=" * 70)
print("  ALL EXPERIMENTS COMPLETE")
print("  Results saved in results/")
print("  Plots  saved in results/plots/")
print("=" * 70)
