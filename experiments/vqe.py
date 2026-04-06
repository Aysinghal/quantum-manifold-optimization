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
from src.metrics import aggregate_seeds, save_results
from src.visualization import convergence_plot, resource_plot, final_loss_bar

# ── Config ──────────────────────────────────────────────────────────────────
N_QUBITS = 4
N_LAYERS = 4
N_STEPS = 150
SEEDS = [0, 1, 2, 3, 4]
J, H_FIELD = 1.0, 1.0

OPTIMIZERS = {
    "GD":        {"lr": 0.1},
    "Adam":      {"lr": 0.05},
    "QNG_block": {"lr": 0.01},
    "QNG_full":  {"lr": 0.01},
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def run():
    E_exact = exact_ground_energy(N_QUBITS, J, H_FIELD)
    print("=" * 60)
    print(f"TASK: VQE  Ising model  {N_QUBITS} qubits")
    print(f"Exact ground-state energy: {E_exact:.6f}")
    print("=" * 60)

    circuit, H = make_vqe_circuit(N_QUBITS, N_LAYERS, J, H_FIELD)
    all_agg = {}

    for opt_name, cfg in OPTIMIZERS.items():
        print(f"\noptimizer={opt_name}  lr={cfg['lr']}")
        seed_results = {}

        for seed in SEEDS:
            print(f"  seed={seed}")
            params = init_params_vqe(N_QUBITS, N_LAYERS, seed)
            result = train_vqe(
                circuit, params,
                opt_name=opt_name,
                lr=cfg["lr"],
                n_steps=N_STEPS,
                n_layers=N_LAYERS,
                verbose=True,
            )
            seed_results[seed] = result

        all_agg[opt_name] = aggregate_seeds(seed_results)

    # Save + plot
    save_results(
        {"config": {"n_qubits": N_QUBITS, "n_layers": N_LAYERS,
                     "J": J, "h": H_FIELD, "E_exact": E_exact},
         **all_agg},
        os.path.join(RESULTS_DIR, "vqe.json"),
    )

    convergence_plot(all_agg, title=f"VQE Ising {N_QUBITS}q  (E*={E_exact:.3f})",
                     ylabel="Energy ⟨H⟩",
                     save_path=os.path.join(PLOTS_DIR, "vqe_convergence.png"))
    resource_plot(all_agg, title="VQE (resource-normalised)",
                  ylabel="Energy ⟨H⟩",
                  save_path=os.path.join(PLOTS_DIR, "vqe_resource.png"))
    final_loss_bar(all_agg, title="VQE: final energy",
                   ylabel="Energy ⟨H⟩",
                   save_path=os.path.join(PLOTS_DIR, "vqe_final.png"))
    return all_agg


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")

    run()
    print("\nDone. Results and plots saved to results/")
