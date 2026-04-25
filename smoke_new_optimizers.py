"""
Smoke test for the three new QNG-family optimizers (MomentumQNG_block,
QNGAdam_v1_block, QNGAdam_v2_block).

Runs a handful of steps on each (task, optimizer) pair and checks:
  1. No exceptions / NaN / Inf in the loss trajectory.
  2. Final loss <= initial loss (training direction is sane).
  3. Each new optimizer produces a different trajectory from baseline QNG_block
     (i.e. the new code paths are actually engaged, not silently no-oping).

Designed to finish in a couple of minutes on a single compute node.
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
    make_stokes_hamiltonian,
    make_heisenberg_ring_hamiltonian,
    init_params_regression,
    init_params_vqe,
)
from src.training import train_with_data, train_vqe


SEED = 0
N_STEPS_SMALL = 5      # for cheap 2q tasks
N_STEPS_VQE   = 3      # 11q VQE step is expensive

# Smaller-than-production circuit sizes to keep wall time low.
REG_N_QUBITS, REG_N_LAYERS = 2, 2
VQE_N_QUBITS, VQE_N_LAYERS = 6, 2
VQE_J, VQE_H = 1.0, 1.0

# Tier 1 smoke sizes: scaled down from production but big enough to actually
# exercise the Hamiltonian / dataset wiring.
STOKES_N_QUBITS_S, STOKES_N_LAYERS_S = 4, 2
HEIS_N_QUBITS_S,   HEIS_N_LAYERS_S   = 4, 2

OPTIMIZERS = [
    ("QNG_block",          0.05),   # baseline reference
    ("MomentumQNG_block",  0.05),
    ("QNGAdam_v1_block",   0.05),
    ("QNGAdam_v2_block",   0.05),
]


def _check_trajectory(losses, label):
    arr = np.asarray(losses, dtype=float)
    if not np.all(np.isfinite(arr)):
        return f"FAIL ({label}: non-finite values in trajectory)"
    if arr[-1] > arr[0] + 1e-6:
        return f"WARN ({label}: loss increased over the run)"
    return "OK"


def _print_traj(name, losses, t_elapsed):
    losses_s = "  ".join(f"{v:+.6f}" for v in losses)
    print(f"  {name:<22s}  losses=[{losses_s}]  ({t_elapsed:.1f}s)")


def _run_data_task(name, build, opt_name, lr, n_steps, loss_type):
    circuit, x_train, y_train = build()
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, SEED)
    t0 = time.perf_counter()
    out = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=opt_name, lr=lr, n_steps=n_steps,
        n_layers=REG_N_LAYERS, loss_type=loss_type, verbose=False,
    )
    return out["losses"], time.perf_counter() - t0


def _build_fit1d():
    x_raw = np.linspace(-np.pi, np.pi, 12)
    y_raw = np.sin(x_raw)
    x = [pnp.array(v, requires_grad=False) for v in x_raw]
    y = [pnp.array(v, requires_grad=False) for v in y_raw]
    circuit = make_regression_circuit_1d(
        REG_N_QUBITS, REG_N_LAYERS, shots=None,
        diff_method="parameter-shift",  # required for QNG-family
    )
    return circuit, x, y


def _build_fit2d():
    g = np.linspace(-1, 1, 4)
    x1, x2 = np.meshgrid(g, g)
    x_raw = np.stack([x1.ravel(), x2.ravel()], axis=1)
    y_raw = 0.5 * (x_raw[:, 0] ** 2 + x_raw[:, 1] ** 2)
    x = [pnp.array(v, requires_grad=False) for v in x_raw]
    y = [pnp.array(v, requires_grad=False) for v in y_raw]
    circuit = make_regression_circuit_2d(
        REG_N_QUBITS, REG_N_LAYERS, shots=None,
        diff_method="parameter-shift",
    )
    return circuit, x, y


def _build_cls():
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler
    X, y = make_moons(n_samples=40, noise=0.15, random_state=42)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = 2 * y - 1
    x = [pnp.array(xi, requires_grad=False) for xi in X]
    yl = [pnp.array(float(yi), requires_grad=False) for yi in y]
    circuit = make_classification_circuit(
        REG_N_QUBITS, REG_N_LAYERS, shots=None,
        diff_method="parameter-shift",
    )
    return circuit, x, yl


def _run_vqe(opt_name, lr, n_steps):
    circuit, _H = make_vqe_circuit(
        VQE_N_QUBITS, VQE_N_LAYERS, VQE_J, VQE_H,
        shots=None, diff_method="parameter-shift",
    )
    params = init_params_vqe(VQE_N_QUBITS, VQE_N_LAYERS, SEED)
    t0 = time.perf_counter()
    out = train_vqe(
        circuit, params, opt_name=opt_name, lr=lr,
        n_steps=n_steps, n_layers=VQE_N_LAYERS, verbose=False,
    )
    return out["losses"], time.perf_counter() - t0


# ── Tier 1 smoke builders ──────────────────────────────────────────────────

def _build_multifreq1d():
    x_raw = np.linspace(-np.pi, np.pi, 12)
    y_raw = np.sin(x_raw) + 0.4 * np.sin(5 * x_raw) + 0.2 * np.sin(13 * x_raw)
    x = [pnp.array(v, requires_grad=False) for v in x_raw]
    y = [pnp.array(v, requires_grad=False) for v in y_raw]
    circuit = make_regression_circuit_1d(
        REG_N_QUBITS, REG_N_LAYERS, shots=None,
        diff_method="parameter-shift",
    )
    return circuit, x, y


def _build_cls_hard():
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler
    X, y = make_moons(n_samples=40, noise=0.40, random_state=42)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = 2 * y - 1
    x = [pnp.array(xi, requires_grad=False) for xi in X]
    yl = [pnp.array(float(yi), requires_grad=False) for yi in y]
    circuit = make_classification_circuit(
        REG_N_QUBITS, REG_N_LAYERS, shots=None,
        diff_method="parameter-shift",
    )
    return circuit, x, yl


def _run_vqe_stokes_smoke(opt_name, lr, n_steps):
    H = make_stokes_hamiltonian(STOKES_N_QUBITS_S)
    circuit, _ = make_vqe_circuit(
        STOKES_N_QUBITS_S, STOKES_N_LAYERS_S, hamiltonian=H,
        shots=None, diff_method="parameter-shift",
    )
    params = init_params_vqe(STOKES_N_QUBITS_S, STOKES_N_LAYERS_S, SEED)
    t0 = time.perf_counter()
    out = train_vqe(
        circuit, params, opt_name=opt_name, lr=lr,
        n_steps=n_steps, n_layers=STOKES_N_LAYERS_S, verbose=False,
    )
    return out["losses"], time.perf_counter() - t0


def _run_vqe_heis_ring_smoke(opt_name, lr, n_steps):
    H = make_heisenberg_ring_hamiltonian(HEIS_N_QUBITS_S, J=1.0, periodic=True)
    circuit, _ = make_vqe_circuit(
        HEIS_N_QUBITS_S, HEIS_N_LAYERS_S, hamiltonian=H,
        shots=None, diff_method="parameter-shift",
    )
    params = init_params_vqe(HEIS_N_QUBITS_S, HEIS_N_LAYERS_S, SEED)
    t0 = time.perf_counter()
    out = train_vqe(
        circuit, params, opt_name=opt_name, lr=lr,
        n_steps=n_steps, n_layers=HEIS_N_LAYERS_S, verbose=False,
    )
    return out["losses"], time.perf_counter() - t0


def main():
    print("=" * 72)
    print("  Smoke test: new QNG-family optimizers")
    print(f"  Reg circuits : {REG_N_QUBITS}q / {REG_N_LAYERS} layers / {N_STEPS_SMALL} steps")
    print(f"  VQE          : {VQE_N_QUBITS}q / {VQE_N_LAYERS} layers / {N_STEPS_VQE} steps")
    print("=" * 72)

    tasks = [
        # Tier 0 (existing baseline tasks).
        ("fit1d",          lambda opt, lr: _run_data_task("fit1d",  _build_fit1d, opt, lr, N_STEPS_SMALL, "mse")),
        ("fit2d",          lambda opt, lr: _run_data_task("fit2d",  _build_fit2d, opt, lr, N_STEPS_SMALL, "mse")),
        ("cls",            lambda opt, lr: _run_data_task("cls",    _build_cls,   opt, lr, N_STEPS_SMALL, "hinge")),
        ("vqe",            lambda opt, lr: _run_vqe(opt, lr, N_STEPS_VQE)),
        # Tier 1 (new rigged tasks).
        ("fit_multifreq1d", lambda opt, lr: _run_data_task("fit_multifreq1d", _build_multifreq1d, opt, lr, N_STEPS_SMALL, "mse")),
        ("cls_moons_hard",  lambda opt, lr: _run_data_task("cls_moons_hard",  _build_cls_hard,    opt, lr, N_STEPS_SMALL, "hinge")),
        ("vqe_stokes",      lambda opt, lr: _run_vqe_stokes_smoke(opt, lr, N_STEPS_VQE)),
        ("vqe_heis_ring",   lambda opt, lr: _run_vqe_heis_ring_smoke(opt, lr, N_STEPS_VQE)),
    ]

    summary = []
    for task_name, runner in tasks:
        print(f"\n[{task_name}]")
        baseline_losses = None
        for opt_name, lr in OPTIMIZERS:
            try:
                losses, dt = runner(opt_name, lr)
                _print_traj(opt_name, losses, dt)
                status = _check_trajectory(losses, f"{task_name}/{opt_name}")
                if opt_name == "QNG_block":
                    baseline_losses = np.asarray(losses, dtype=float)
                else:
                    diff = float(np.max(np.abs(np.asarray(losses) - baseline_losses)))
                    if diff < 1e-9:
                        status += f"  [WARN: trajectory identical to QNG_block (max|diff|={diff:.2e})]"
                    else:
                        status += f"  [diff vs QNG_block: max={diff:.4f}]"
                summary.append((task_name, opt_name, status))
            except Exception as exc:
                msg = f"FAIL ({type(exc).__name__}: {exc})"
                print(f"  {opt_name:<22s}  {msg}")
                summary.append((task_name, opt_name, msg))

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for task_name, opt_name, status in summary:
        print(f"  {task_name:<6s} / {opt_name:<22s}  {status}")
    n_fail = sum(1 for _, _, s in summary if s.startswith("FAIL"))
    print("=" * 72)
    print(f"  {len(summary) - n_fail}/{len(summary)} runs OK, {n_fail} failed")
    print("=" * 72)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
