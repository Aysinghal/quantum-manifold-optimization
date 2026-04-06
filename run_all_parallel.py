"""
Parallel runner: executes all 80 experiment jobs across multiple CPU cores.

Each (experiment, optimizer, seed) combination runs as an independent worker
process. Results are collected, aggregated, and saved/plotted identically to
the sequential run_all.py.

Usage:
    python run_all_parallel.py               # use all cores
    python run_all_parallel.py --workers 16  # use 16 cores
"""

import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")

import numpy as np
from pennylane import numpy as pnp

from src.models import (
    make_regression_circuit_1d,
    make_regression_circuit_2d,
    make_vqe_circuit,
    make_classification_circuit,
    init_params_regression,
    init_params_vqe,
    exact_ground_energy,
)
from src.training import train_with_data, train_vqe
from src.metrics import aggregate_seeds, save_results, accuracy
from src.visualization import convergence_plot, resource_plot, final_loss_bar


# ── Shared config (mirrors the individual experiment modules) ────────────────

SEEDS = [0, 1, 2, 3, 4]

OPTIMIZERS = {
    "GD":        {"lr": 0.1},
    "Adam":      {"lr": 0.05},
    "QNG_block": {"lr": 0.01},
    "QNG_full":  {"lr": 0.01},
}

REG_N_QUBITS = 2
REG_N_LAYERS = 4
REG_N_STEPS  = 100

VQE_N_QUBITS = 4
VQE_N_LAYERS = 4
VQE_N_STEPS  = 150
VQE_J        = 1.0
VQE_H        = 1.0

CLS_N_DATA   = 100

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

TASK_LABELS = {
    "fit1d": "1D-sin",
    "fit2d": "2D-quad",
    "vqe":   "VQE",
    "cls":   "Classification",
}


# ── Worker functions (module-level for Windows spawn compatibility) ──────────

def _drop_params(result):
    """Drop the params tensor before returning to the main process."""
    return {k: v for k, v in result.items() if k != "params"}


def _run_fit1d(opt_name, lr, seed):
    x_raw = np.linspace(-np.pi, np.pi, 20)
    y_raw = np.sin(x_raw)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]

    circuit = make_regression_circuit_1d(REG_N_QUBITS, REG_N_LAYERS)
    params  = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=opt_name, lr=lr, n_steps=REG_N_STEPS,
        n_layers=REG_N_LAYERS, loss_type="mse", verbose=False,
    )
    return ("fit1d", opt_name, seed, _drop_params(result))


def _run_fit2d(opt_name, lr, seed):
    g = np.linspace(-1, 1, 8)
    x1, x2 = np.meshgrid(g, g)
    x_raw = np.stack([x1.ravel(), x2.ravel()], axis=1)
    y_raw = 0.5 * (x_raw[:, 0] ** 2 + x_raw[:, 1] ** 2)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]

    circuit = make_regression_circuit_2d(REG_N_QUBITS, REG_N_LAYERS)
    params  = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=opt_name, lr=lr, n_steps=REG_N_STEPS,
        n_layers=REG_N_LAYERS, loss_type="mse", verbose=False,
    )
    return ("fit2d", opt_name, seed, _drop_params(result))


def _run_vqe(opt_name, lr, seed):
    circuit, _H = make_vqe_circuit(VQE_N_QUBITS, VQE_N_LAYERS, VQE_J, VQE_H)
    params = init_params_vqe(VQE_N_QUBITS, VQE_N_LAYERS, seed)

    result = train_vqe(
        circuit, params,
        opt_name=opt_name, lr=lr, n_steps=VQE_N_STEPS,
        n_layers=VQE_N_LAYERS, verbose=False,
    )
    return ("vqe", opt_name, seed, _drop_params(result))


def _run_cls(opt_name, lr, seed):
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler

    X, y = make_moons(n_samples=CLS_N_DATA, noise=0.15, random_state=42)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    y = 2 * y - 1

    x_train = [pnp.array(xi, requires_grad=False) for xi in X]
    y_train = [pnp.array(float(yi), requires_grad=False) for yi in y]

    circuit = make_classification_circuit(REG_N_QUBITS, REG_N_LAYERS)
    params  = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=opt_name, lr=lr, n_steps=REG_N_STEPS,
        n_layers=REG_N_LAYERS, loss_type="hinge", verbose=False,
    )

    preds = [float(circuit(result["params"], x)) for x in x_train]
    result["final_accuracy"] = accuracy(preds, [float(yi) for yi in y])

    return ("cls", opt_name, seed, _drop_params(result))


# ── Main orchestration ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all QNG baseline experiments in parallel",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel worker processes (default: all CPU cores)",
    )
    args = parser.parse_args()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    n_workers = args.workers

    # Build job list: one entry per (experiment, optimizer, seed)
    worker_fns = [_run_fit1d, _run_fit2d, _run_vqe, _run_cls]
    jobs = []
    for opt_name, cfg in OPTIMIZERS.items():
        for seed in SEEDS:
            for fn in worker_fns:
                jobs.append((fn, opt_name, cfg["lr"], seed))

    total = len(jobs)
    print("=" * 70)
    print("  QNG EUCLIDEAN BASELINE -- Parallel runner")
    print(f"  Workers: {n_workers}  |  Total jobs: {total}")
    print("=" * 70)

    # Submit all jobs to the process pool
    t_start = time.perf_counter()
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {}
        for fn, opt_name, lr, seed in jobs:
            fut = pool.submit(fn, opt_name, lr, seed)
            future_map[fut] = (fn.__name__, opt_name, seed)

        done_count = 0
        for fut in as_completed(future_map):
            info = future_map[fut]
            done_count += 1
            try:
                task, opt_name, seed, result = fut.result()
                results.append((task, opt_name, seed, result))
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [{done_count:3d}/{total}]  "
                    f"{TASK_LABELS[task]:>15s} / {opt_name:<10s} / seed={seed}"
                    f"  ({elapsed:.1f}s)"
                )
            except Exception as exc:
                print(f"  [{done_count:3d}/{total}]  FAILED {info}: {exc}")

    total_time = time.perf_counter() - t_start
    print(f"\nAll {total} jobs finished in {total_time:.1f}s")

    # ── Group results by (task, optimizer, seed) ─────────────────────────
    grouped = defaultdict(lambda: defaultdict(dict))
    for task, opt_name, seed, result in results:
        grouped[task][opt_name][seed] = result

    # ── 1D Function Fitting ──────────────────────────────────────────────
    if "fit1d" in grouped:
        print("\nSaving 1D function fitting results...")
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["fit1d"].items()}
        save_results(agg, os.path.join(RESULTS_DIR, "function_fitting_1d.json"))
        convergence_plot(agg, title="1D Regression: sin(x)", log_y=True,
                         save_path=os.path.join(PLOTS_DIR, "fit1d_convergence.png"))
        resource_plot(agg, title="1D Regression: sin(x) (resource-normalised)", log_y=True,
                      save_path=os.path.join(PLOTS_DIR, "fit1d_resource.png"))
        final_loss_bar(agg, title="1D Regression: final MSE",
                       save_path=os.path.join(PLOTS_DIR, "fit1d_final.png"))

    # ── 2D Function Fitting ──────────────────────────────────────────────
    if "fit2d" in grouped:
        print("\nSaving 2D function fitting results...")
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["fit2d"].items()}
        save_results(agg, os.path.join(RESULTS_DIR, "function_fitting_2d.json"))
        convergence_plot(agg, title="2D Regression: (x1\u00b2+x2\u00b2)/2", log_y=True,
                         save_path=os.path.join(PLOTS_DIR, "fit2d_convergence.png"))
        resource_plot(agg, title="2D Regression (resource-normalised)", log_y=True,
                      save_path=os.path.join(PLOTS_DIR, "fit2d_resource.png"))
        final_loss_bar(agg, title="2D Regression: final MSE",
                       save_path=os.path.join(PLOTS_DIR, "fit2d_final.png"))

    # ── VQE ──────────────────────────────────────────────────────────────
    if "vqe" in grouped:
        print("\nSaving VQE results...")
        E_exact = exact_ground_energy(VQE_N_QUBITS, VQE_J, VQE_H)
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["vqe"].items()}
        save_results(
            {"config": {"n_qubits": VQE_N_QUBITS, "n_layers": VQE_N_LAYERS,
                         "J": VQE_J, "h": VQE_H, "E_exact": E_exact},
             **agg},
            os.path.join(RESULTS_DIR, "vqe.json"),
        )
        convergence_plot(agg, title=f"VQE Ising {VQE_N_QUBITS}q  (E*={E_exact:.3f})",
                         ylabel="Energy \u27e8H\u27e9",
                         save_path=os.path.join(PLOTS_DIR, "vqe_convergence.png"))
        resource_plot(agg, title="VQE (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=os.path.join(PLOTS_DIR, "vqe_resource.png"))
        final_loss_bar(agg, title="VQE: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=os.path.join(PLOTS_DIR, "vqe_final.png"))

    # ── Classification ───────────────────────────────────────────────────
    if "cls" in grouped:
        print("\nSaving classification results...")
        agg = {}
        for opt, seed_dict in grouped["cls"].items():
            agg_opt = aggregate_seeds(seed_dict)
            accs = [seed_dict[s]["final_accuracy"] for s in SEEDS]
            agg_opt["final_acc_mean"] = float(np.mean(accs))
            agg_opt["final_acc_std"]  = float(np.std(accs))
            agg[opt] = agg_opt
        save_results(agg, os.path.join(RESULTS_DIR, "classification.json"))
        convergence_plot(agg, title="Classification (make_moons)", log_y=True,
                         save_path=os.path.join(PLOTS_DIR, "cls_convergence.png"))
        resource_plot(agg, title="Classification (resource-normalised)", log_y=True,
                      save_path=os.path.join(PLOTS_DIR, "cls_resource.png"))
        final_loss_bar(agg, title="Classification: final hinge loss",
                       save_path=os.path.join(PLOTS_DIR, "cls_final.png"))

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Total wall time: {total_time:.1f}s")
    print("  Results saved in results/")
    print("  Plots  saved in results/plots/")
    print("=" * 70)


if __name__ == "__main__":
    main()
