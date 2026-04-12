"""
Master script: run all three experiments sequentially.

Usage:
    conda activate qng-baseline
    python run_all.py
"""

import os
import matplotlib
matplotlib.use("Agg")

from src.metrics import make_run_dir

RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
run_dir, plots_dir = make_run_dir(RESULTS_BASE)

print("=" * 70)
print("  QNG EUCLIDEAN BASELINE -- Running all experiments")
print(f"  Run directory: {run_dir}")
print("=" * 70)

from experiments.function_fitting import run_1d, run_2d
from experiments.vqe import run as run_vqe
from experiments.classification import run as run_cls

run_1d(results_dir=run_dir)
run_2d(results_dir=run_dir)
run_vqe(results_dir=run_dir)
run_cls(results_dir=run_dir)

print("\n" + "=" * 70)
print("  ALL EXPERIMENTS COMPLETE")
print(f"  Results saved in {run_dir}")
print(f"  Plots  saved in {plots_dir}")
print("=" * 70)
