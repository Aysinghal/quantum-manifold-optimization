"""
Experiment: VQE on the Transverse-Field Ising Model
  H = -J * sum(Z_i Z_{i+1}) - h * sum(X_i)    (open boundary)
  4 qubits, J=1, h=1

Compares GD, Adam, QNG (block-diag), QNG (full) across 5 seeds.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pennylane import numpy as pnp

from src.models import make_vqe_circuit, init_params_vqe, exact_ground_energy
from src.training import train_vqe
from src.metrics import aggregate_seeds, save_results, make_run_dir
from src.visualization import convergence_plot, resource_plot, final_loss_bar, task_plot_path

# ── Config ──────────────────────────────────────────────────────────────────
N_QUBITS = 11
N_LAYERS = 4
N_STEPS = 150
SEEDS = [0, 1, 2, 3, 4]
J, H_FIELD = 1.0, 1.0

OPTIMIZERS = {
    "GD":              {"lr": 0.1},
    "Adam":            {"lr": 0.05},
    "QNG_block":       {"lr": 0.05},
    "QNG_block_lr01":  {"lr": 0.1},
    "QNG_block_lr02":  {"lr": 0.2},
    "QNG_full":        {"lr": 0.01},
}


def _canonical_opt(opt_name):
    """Map optimizer-variant names back to the base optimizer name that
    train_vqe knows how to dispatch on (mirrors run_all_parallel._canonical_opt)."""
    if opt_name.startswith("QNG_block"):
        return "QNG_block"
    if opt_name.startswith("QNG_full"):
        return "QNG_full"
    return opt_name

RESULTS_BASE = os.path.join(os.path.dirname(__file__), "..", "results")


def run(results_dir=None, shots=None):
    if results_dir is None:
        results_dir, plots_dir = make_run_dir(RESULTS_BASE)
    else:
        plots_dir = os.path.join(results_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

    E_exact = exact_ground_energy(N_QUBITS, J, H_FIELD)
    diff_method_default = "parameter-shift" if shots else "adjoint"
    print("=" * 60)
    print(f"TASK: VQE  Ising model  {N_QUBITS} qubits")
    print(f"Shots: {shots or 'None'} (diff_method={diff_method_default} for GD/Adam; "
          f"parameter-shift for QNG)")
    print(f"Exact ground-state energy: {E_exact:.6f}")
    print("=" * 60)

    all_agg = {}

    for opt_name, cfg in OPTIMIZERS.items():
        print(f"\noptimizer={opt_name}  lr={cfg['lr']}")
        seed_results = {}

        # QNG optimizers need parameter-shift regardless of shots setting
        # (qml.metric_tensor tapes aren't compatible with lightning adjoint).
        dm = "parameter-shift" if opt_name.startswith("QNG") else None
        circuit, H = make_vqe_circuit(
            N_QUBITS, N_LAYERS, J, H_FIELD, shots=shots, diff_method=dm,
        )

        for seed in SEEDS:
            print(f"  seed={seed}")
            params = init_params_vqe(N_QUBITS, N_LAYERS, seed)
            result = train_vqe(
                circuit, params,
                opt_name=_canonical_opt(opt_name),
                lr=cfg["lr"],
                n_steps=N_STEPS,
                n_layers=N_LAYERS,
                verbose=True,
            )
            seed_results[seed] = result

        all_agg[opt_name] = aggregate_seeds(seed_results)

    save_results(
        {"config": {"n_qubits": N_QUBITS, "n_layers": N_LAYERS,
                     "J": J, "h": H_FIELD, "E_exact": E_exact,
                     "shots": shots,
                     "diff_method_gd_adam": diff_method_default,
                     "diff_method_qng": "parameter-shift"},
         **all_agg},
        os.path.join(results_dir, "vqe.json"),
    )

    convergence_plot(all_agg, title=f"VQE Ising {N_QUBITS}q  (E*={E_exact:.3f})",
                     ylabel="Energy ⟨H⟩",
                     save_path=task_plot_path(plots_dir, "vqe", "convergence"))
    resource_plot(all_agg, title="VQE (resource-normalised)",
                  ylabel="Energy ⟨H⟩",
                  save_path=task_plot_path(plots_dir, "vqe", "resource"))
    final_loss_bar(all_agg, title="VQE: final energy",
                   ylabel="Energy ⟨H⟩",
                   save_path=task_plot_path(plots_dir, "vqe", "final"))
    return all_agg


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    run()
    print("\nDone. Results and plots saved to results/")
