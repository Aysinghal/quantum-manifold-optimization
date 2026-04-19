"""
Smoke test: time a few optimization steps for EVERY task (fit1d, fit2d, cls,
vqe) under both analytic mode (adjoint, or parameter-shift for QNG) and
shot-noise mode (parameter-shift, shots=1000).

Runs WARMUP_STEPS (untimed) + TIMED_STEPS (timed) per (task, optimizer, mode),
extrapolates to the task's production step count, and reports the slowest
projected single-seed job (which governs Slurm wall time, since seeds run
in parallel).

Intended to be run once before submitting the Slurm job.
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pennylane.numpy as pnp

from src.models import (
    make_regression_circuit_1d,
    make_regression_circuit_2d,
    make_classification_circuit,
    make_vqe_circuit,
    init_params_regression,
    init_params_vqe,
)
from src.training import train_with_data, train_vqe


SEED = 0
WARMUP_STEPS = 1
TIMED_STEPS = 2

# Mirror the production constants in run_all_parallel.py
REG_N_QUBITS, REG_N_LAYERS, REG_N_STEPS = 2, 4, 100
CLS_N_DATA = 100
VQE_N_QUBITS, VQE_N_LAYERS, VQE_N_STEPS = 11, 4, 150
VQE_J, VQE_H = 1.0, 1.0

OPTIMIZERS = {
    "GD":        0.1,
    "Adam":      0.05,
    "QNG_block": 0.05,
    "QNG_full":  0.05,
}

MODES = [
    ("analytic",   None),
    ("shots=1000", 1000),
]


def fmt_seconds(s):
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{int(m):02d}m"


def _dm_for(opt_name):
    """QNG needs parameter-shift because metric_tensor tapes aren't
    compatible with lightning.qubit's adjoint. GD/Adam get adjoint free."""
    return "parameter-shift" if opt_name.startswith("QNG") else None


def build_task(task, shots, opt_name):
    """Return (step_fn, prod_steps) where step_fn() runs one optimization step
    starting from a freshly initialized state (so we can time it in isolation)."""
    dm = _dm_for(opt_name)

    if task == "fit1d":
        x_raw = np.linspace(-np.pi, np.pi, 20)
        y_raw = np.sin(x_raw)
        x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
        y_train = [pnp.array(y, requires_grad=False) for y in y_raw]
        circuit = make_regression_circuit_1d(
            REG_N_QUBITS, REG_N_LAYERS, shots=shots, diff_method=dm,
        )
        params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, SEED)

        def run(n_steps):
            train_with_data(circuit, params, x_train, y_train,
                            opt_name=opt_name, lr=OPTIMIZERS[opt_name],
                            n_steps=n_steps, n_layers=REG_N_LAYERS,
                            loss_type="mse", verbose=False)
        return run, REG_N_STEPS

    if task == "fit2d":
        g = np.linspace(-1, 1, 8)
        x1, x2 = np.meshgrid(g, g)
        x_raw = np.stack([x1.flatten(), x2.flatten()], axis=1)
        y_raw = 0.5 * (x_raw[:, 0] ** 2 + x_raw[:, 1] ** 2)
        x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
        y_train = [pnp.array(y, requires_grad=False) for y in y_raw]
        circuit = make_regression_circuit_2d(
            REG_N_QUBITS, REG_N_LAYERS, shots=shots, diff_method=dm,
        )
        params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, SEED)

        def run(n_steps):
            train_with_data(circuit, params, x_train, y_train,
                            opt_name=opt_name, lr=OPTIMIZERS[opt_name],
                            n_steps=n_steps, n_layers=REG_N_LAYERS,
                            loss_type="mse", verbose=False)
        return run, REG_N_STEPS

    if task == "cls":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=CLS_N_DATA, noise=0.15, random_state=42)
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = 2 * y - 1
        x_train = [pnp.array(xi, requires_grad=False) for xi in X]
        y_train = [pnp.array(yi, requires_grad=False) for yi in y]
        circuit = make_classification_circuit(
            REG_N_QUBITS, REG_N_LAYERS, shots=shots, diff_method=dm,
        )
        params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, SEED)

        def run(n_steps):
            train_with_data(circuit, params, x_train, y_train,
                            opt_name=opt_name, lr=OPTIMIZERS[opt_name],
                            n_steps=n_steps, n_layers=REG_N_LAYERS,
                            loss_type="hinge", verbose=False)
        return run, REG_N_STEPS

    if task == "vqe":
        circuit, _H = make_vqe_circuit(
            VQE_N_QUBITS, VQE_N_LAYERS, VQE_J, VQE_H,
            shots=shots, diff_method=dm,
        )
        params = init_params_vqe(VQE_N_QUBITS, VQE_N_LAYERS, SEED)

        def run(n_steps):
            train_vqe(circuit, params, opt_name,
                      lr=OPTIMIZERS[opt_name], n_steps=n_steps,
                      n_layers=VQE_N_LAYERS, verbose=False)
        return run, VQE_N_STEPS

    raise ValueError(task)


def time_one(task, opt_name, shots):
    run, prod_steps = build_task(task, shots, opt_name)
    run(WARMUP_STEPS)
    t0 = time.perf_counter()
    run(TIMED_STEPS)
    elapsed = time.perf_counter() - t0
    return elapsed / TIMED_STEPS, prod_steps


def main():
    print("=" * 82)
    print(f"  Smoke test across all 4 tasks  (warmup={WARMUP_STEPS}, timed={TIMED_STEPS})")
    print("=" * 82)
    print(f"  Config: fit1d=2q/20pts/100steps  fit2d=2q/64pts/100steps")
    print(f"          cls=2q/100pts/100steps   vqe=11q/Hexpval/150steps")
    print("=" * 82)
    header = (f"  {'task':<6s} {'optimizer':<10s} {'mode':<11s} "
              f"{'sec/step':>10s} {'full run':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    rows = []
    for task in ("fit1d", "fit2d", "cls", "vqe"):
        for mode_label, shots in MODES:
            for opt_name in OPTIMIZERS:
                try:
                    per_step, prod_steps = time_one(task, opt_name, shots)
                    total = per_step * prod_steps
                    print(f"  {task:<6s} {opt_name:<10s} {mode_label:<11s} "
                          f"{per_step:>9.3f}s {fmt_seconds(total):>10s}",
                          flush=True)
                    rows.append((task, opt_name, mode_label, per_step, total))
                except Exception as e:
                    print(f"  {task:<6s} {opt_name:<10s} {mode_label:<11s} "
                          f"FAILED: {e}", flush=True)
                    rows.append((task, opt_name, mode_label, None, None))

    # Per-seed wall time = max across all jobs for a given shots mode
    # (each seed runs all 16 (task, opt) combos serially within one worker).
    # Actually: each worker runs ONE job, so per-seed wall-time = max over
    # (task, opt) for that mode.
    print()
    for mode_label, _ in MODES:
        mode_rows = [r for r in rows if r[2] == mode_label and r[4] is not None]
        if not mode_rows:
            continue
        worst = max(mode_rows, key=lambda r: r[4])
        total = worst[4]
        flag = "OK" if total < 10 * 3600 else "WARNING (>10h)"
        print(f"  [{mode_label}] slowest job: {worst[0]}/{worst[1]}  "
              f"= {fmt_seconds(total)}  [{flag}]")
    print()
    print("  Slurm wall time ≈ slowest single job (80 cores => all seeds/tasks")
    print("  run in parallel). 12h allocation has margin up to ~10h.")


if __name__ == "__main__":
    main()
