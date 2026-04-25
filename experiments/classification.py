"""
Experiment: Binary Classification on make_moons
  2 qubits, 2D input, 100 training points

Compares GD, Adam, QNG (block-diag), QNG (full) across 5 seeds.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pennylane import numpy as pnp
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler

from src.models import make_classification_circuit, init_params_regression
from src.training import train_with_data
from src.metrics import aggregate_seeds, save_results, accuracy, make_run_dir
from src.visualization import convergence_plot, resource_plot, final_loss_bar, task_plot_path

# ── Config ──────────────────────────────────────────────────────────────────
N_QUBITS = 2
N_LAYERS = 4
N_STEPS = 100
N_DATA = 100
SEEDS = [0, 1, 2, 3, 4]

OPTIMIZERS = {
    "GD":        {"lr": 0.1},
    "Adam":      {"lr": 0.05},
    "QNG_block": {"lr": 0.05},
    "QNG_full":  {"lr": 0.05},
}

RESULTS_BASE = os.path.join(os.path.dirname(__file__), "..", "results")


def make_dataset(n_samples=100, noise=0.15, random_state=42):
    """Generate make_moons dataset with labels in {-1, +1}, scaled to [-1, 1]."""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    y = 2 * y - 1  # {0,1} -> {-1,+1}
    return X, y


def run(results_dir=None):
    if results_dir is None:
        results_dir, plots_dir = make_run_dir(RESULTS_BASE)
    else:
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    print("=" * 60)
    print("TASK: Classification  (make_moons, 2 qubits)")
    print("=" * 60)

    X, y = make_dataset(n_samples=N_DATA)
    x_train = [pnp.array(xi, requires_grad=False) for xi in X]
    y_train = [pnp.array(float(yi), requires_grad=False) for yi in y]

    circuit = make_classification_circuit(N_QUBITS, N_LAYERS)
    all_agg = {}

    for opt_name, cfg in OPTIMIZERS.items():
        print(f"\noptimizer={opt_name}  lr={cfg['lr']}")
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
                loss_type="hinge",
                verbose=True,
            )
            preds = [float(circuit(result["params"], x)) for x in x_train]
            result["final_accuracy"] = accuracy(preds, [float(yi) for yi in y])
            seed_results[seed] = result

        agg = aggregate_seeds(seed_results)
        accs = [seed_results[s]["final_accuracy"] for s in SEEDS]
        agg["final_acc_mean"] = float(np.mean(accs))
        agg["final_acc_std"] = float(np.std(accs))
        all_agg[opt_name] = agg

    save_results(all_agg, os.path.join(results_dir, "classification.json"))
    convergence_plot(all_agg, title="Classification (make_moons)", log_y=True,
                     save_path=task_plot_path(plots_dir, "classification", "convergence"))
    resource_plot(all_agg, title="Classification (resource-normalised)", log_y=True,
                  save_path=task_plot_path(plots_dir, "classification", "resource"))
    final_loss_bar(all_agg, title="Classification: final hinge loss",
                   save_path=task_plot_path(plots_dir, "classification", "final"))
    return all_agg


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    run()
    print("\nDone. Results and plots saved to results/")
