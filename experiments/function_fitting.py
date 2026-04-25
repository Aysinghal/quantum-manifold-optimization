"""
Experiment: Function Fitting (Regression)
  - 1D: f(x) = sin(x) on [-pi, pi]
  - 2D: f(x1, x2) = (x1^2 + x2^2) / 2 on [-1, 1]^2

Compares GD, Adam, QNG (block-diag), QNG (full) across 5 seeds.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pennylane import numpy as pnp

from src.models import (
    make_regression_circuit_1d,
    make_regression_circuit_2d,
    init_params_regression,
)
from src.training import train_with_data
from src.metrics import aggregate_seeds, save_results, make_run_dir
from src.visualization import convergence_plot, resource_plot, final_loss_bar, task_plot_path

# ── Config ──────────────────────────────────────────────────────────────────
N_QUBITS = 2
N_LAYERS = 4
N_STEPS = 100
SEEDS = [0, 1, 2, 3, 4]

OPTIMIZERS = {
    "GD":        {"lr": 0.1},
    "Adam":      {"lr": 0.05},
    "QNG_block": {"lr": 0.05},
    "QNG_full":  {"lr": 0.05},
}

RESULTS_BASE = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Helpers ─────────────────────────────────────────────────────────────────

def run_task(task_name, circuit, x_train, y_train, loss_type="mse"):
    """Run all optimizers x all seeds for one sub-task."""
    all_agg = {}

    for opt_name, cfg in OPTIMIZERS.items():
        print(f"\n[{task_name}] optimizer={opt_name}  lr={cfg['lr']}")
        seed_results = {}

        for seed in SEEDS:
            print(f"  seed={seed}")
            params = init_params_regression(N_QUBITS, N_LAYERS, seed)
            result = train_with_data(
                circuit, params,
                x_train, y_train,
                opt_name=opt_name,
                lr=cfg["lr"],
                n_steps=N_STEPS,
                n_layers=N_LAYERS,
                loss_type=loss_type,
                verbose=True,
            )
            seed_results[seed] = result

        all_agg[opt_name] = aggregate_seeds(seed_results)

    return all_agg


# ── 1D Regression: sin(x) ──────────────────────────────────────────────────

def run_1d(results_dir=None):
    if results_dir is None:
        results_dir, plots_dir = make_run_dir(RESULTS_BASE)
    else:
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    print("=" * 60)
    print("TASK: 1D Function Fitting  f(x) = sin(x)")
    print("=" * 60)

    x_train_raw = np.linspace(-np.pi, np.pi, 20)
    y_train_raw = np.sin(x_train_raw)

    x_train = [pnp.array(x, requires_grad=False) for x in x_train_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_train_raw]

    circuit = make_regression_circuit_1d(N_QUBITS, N_LAYERS)
    agg = run_task("1D-sin", circuit, x_train, y_train)

    save_results(agg, os.path.join(results_dir, "function_fitting_1d.json"))
    convergence_plot(agg, title="1D Regression: sin(x)", log_y=True,
                     save_path=task_plot_path(plots_dir, "function_fitting_1d", "convergence"))
    resource_plot(agg, title="1D Regression: sin(x) (resource-normalised)", log_y=True,
                  save_path=task_plot_path(plots_dir, "function_fitting_1d", "resource"))
    final_loss_bar(agg, title="1D Regression: final MSE",
                   save_path=task_plot_path(plots_dir, "function_fitting_1d", "final"))
    return agg


# ── 2D Regression: (x1^2+x2^2)/2 ──────────────────────────────────────────

def run_2d(results_dir=None):
    if results_dir is None:
        results_dir, plots_dir = make_run_dir(RESULTS_BASE)
    else:
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("TASK: 2D Function Fitting  f(x1,x2) = (x1^2+x2^2)/2")
    print("=" * 60)

    g = np.linspace(-1, 1, 8)
    x1, x2 = np.meshgrid(g, g)
    x_raw = np.stack([x1.ravel(), x2.ravel()], axis=1)
    y_raw = 0.5 * (x_raw[:, 0] ** 2 + x_raw[:, 1] ** 2)

    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]

    circuit = make_regression_circuit_2d(N_QUBITS, N_LAYERS)
    agg = run_task("2D-quad", circuit, x_train, y_train)

    save_results(agg, os.path.join(results_dir, "function_fitting_2d.json"))
    convergence_plot(agg, title="2D Regression: (x1²+x2²)/2", log_y=True,
                     save_path=task_plot_path(plots_dir, "function_fitting_2d", "convergence"))
    resource_plot(agg, title="2D Regression (resource-normalised)", log_y=True,
                  save_path=task_plot_path(plots_dir, "function_fitting_2d", "resource"))
    final_loss_bar(agg, title="2D Regression: final MSE",
                   save_path=task_plot_path(plots_dir, "function_fitting_2d", "final"))
    return agg


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    run_1d()
    run_2d()
    print("\nDone. Results and plots saved to results/")
