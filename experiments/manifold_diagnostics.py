"""
Manifold diagnostics: per-step ||theta|| trajectories.

Phase 1 (torus) needs to verify that flat baselines drift while R-QNG-torus
stays bounded. Phase 2 (sphere) needs to verify that the three sphere
optimizers actually pin ||theta|| at the init radius
(R = 2*pi*sqrt(d/3) by default; otherwise the geodesic retraction is
buggy or the init is wrong).

This script trains the optimizer set for the chosen task with
``track_theta_norm=True`` and emits a 2-panel plot:

  - Panel A: ||theta|| vs step (the mechanistic evidence).
  - Panel B: loss vs step (the convergence trace, for context).

Default task is ``vqe_stokes_overparam_long`` (Phase 1 drift check). For
Phase 2 use ``--task vqe_overparam_heis``. ``--phase`` selects which
optimizer tier to run (default ``torus`` for back-compat with prior runs).

Usage:
    python experiments/manifold_diagnostics.py
    python experiments/manifold_diagnostics.py --task fit_high_periodic --seed 0
    python experiments/manifold_diagnostics.py --task vqe_overparam_heis --phase sphere
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pennylane import numpy as pnp

from src.models import (
    make_vqe_circuit, make_regression_circuit_1d,
    make_stokes_hamiltonian, make_heisenberg_ring_hamiltonian,
    init_params_vqe, init_params_regression,
)
from src.training import train_vqe, train_with_data
from src.visualization import COLORS, LABELS


# Optimizer tiers (mirror run_all_parallel.OPT_TIERS).
OPTIMIZER_TIERS = {
    "torus": ["Adam", "QNG_block", "QNGAdam_v1_block", "RQNG_torus_block"],
    "sphere": [
        "Adam", "QNG_block", "QNGAdam_v1_block",
        "RQNG_torus_block",
        "RQNG_sphere_block", "ProjQNG_sphere", "ProjAdam_sphere",
    ],
}
LR = 0.05


def _diff_method_for(opt_name):
    """QNG-family (including R-QNG and ProjQNG_sphere) needs parameter-shift
    for metric_tensor; everything else uses adjoint."""
    if (opt_name.startswith("QNG")
            or opt_name.startswith("MomentumQNG")
            or opt_name.startswith("RQNG")
            or opt_name == "ProjQNG_sphere"):
        return "parameter-shift"
    return "adjoint"


def _make_progress_cb(opt_name, n_steps, every=10):
    """Per-10-step heartbeat printed to stdout. Each line is prefixed with the
    optimizer name so parallel runs interleave readably. Prints step 0, then
    every `every` steps, then a final tick at the last step."""
    t0 = time.perf_counter()
    last_printed = {"step": -1}

    def cb(step, total, loss):
        # Print at step 0, every `every` steps, and at the very last step.
        is_last = (step == total - 1)
        if step % every != 0 and not is_last:
            return
        if step == last_printed["step"]:
            return
        last_printed["step"] = step
        elapsed = time.perf_counter() - t0
        eta = (elapsed / max(step, 1)) * (n_steps - step) if step > 0 else 0.0
        print(f"  [{opt_name:<20s}] step {step:>4d}/{n_steps}  "
              f"loss={loss:.6f}  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
              flush=True)

    return cb


def _run_one_vqe_stokes_overparam_long(opt_name, seed, n_steps):
    """Same config as run_all_parallel._run_vqe_stokes_overparam_long but
    shorter (so the diagnostics script is fast) and with theta-norm tracking."""
    n_qubits, n_layers = 8, 8
    H = make_stokes_hamiltonian(n_qubits)
    circuit, _ = make_vqe_circuit(
        n_qubits, n_layers, hamiltonian=H,
        shots=None, diff_method=_diff_method_for(opt_name),
    )
    params = init_params_vqe(n_qubits, n_layers, seed)
    return train_vqe(
        circuit, params, opt_name=opt_name, lr=LR, n_steps=n_steps,
        n_layers=n_layers, verbose=False, track_theta_norm=True,
        progress_cb=_make_progress_cb(opt_name, n_steps),
    )


def _run_one_fit_high_periodic(opt_name, seed, n_steps):
    """Same config as run_all_parallel._run_fit_high_periodic, theta-tracking."""
    n_qubits, n_layers = 2, 4
    x_raw = np.linspace(-3 * np.pi, 3 * np.pi, 60)
    y_raw = np.sin(7.0 * x_raw)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]
    circuit = make_regression_circuit_1d(
        n_qubits, n_layers, shots=None, diff_method=_diff_method_for(opt_name),
    )
    params = init_params_regression(n_qubits, n_layers, seed)
    return train_with_data(
        circuit, params, x_train, y_train,
        opt_name=opt_name, lr=LR, n_steps=n_steps, n_layers=n_layers,
        loss_type="mse", verbose=False, track_theta_norm=True,
        progress_cb=_make_progress_cb(opt_name, n_steps),
    )


def _run_one_vqe_overparam_heis(opt_name, seed, n_steps):
    """Same config as run_all_parallel._run_vqe_overparam_heis, theta-tracking.

    Phase 2 sphere stress test: 4q periodic Heisenberg ring with 8 layers
    so the parameter dimension d ~> 96 sits well above the Hilbert-space
    dimension 16. The interesting question is whether sphere optimizers
    (forced to ||theta||=1) can compete with flat optimizers that range
    freely.
    """
    n_qubits, n_layers = 4, 8
    H = make_heisenberg_ring_hamiltonian(n_qubits, J=1.0, periodic=True)
    circuit, _ = make_vqe_circuit(
        n_qubits, n_layers, hamiltonian=H,
        shots=None, diff_method=_diff_method_for(opt_name),
    )
    params = init_params_vqe(n_qubits, n_layers, seed)
    return train_vqe(
        circuit, params, opt_name=opt_name, lr=LR, n_steps=n_steps,
        n_layers=n_layers, verbose=False, track_theta_norm=True,
        progress_cb=_make_progress_cb(opt_name, n_steps),
    )


TASKS = {
    "vqe_stokes_overparam_long": _run_one_vqe_stokes_overparam_long,
    "fit_high_periodic":         _run_one_fit_high_periodic,
    "vqe_overparam_heis":        _run_one_vqe_overparam_heis,
}


def _worker(task_name, opt_name, seed, n_steps):
    """Module-level worker fn (must be top-level so ProcessPoolExecutor can
    pickle it for `fork`/`spawn`). Returns the data the main process plots,
    not the params themselves -- params can be large and we never read them."""
    runner = TASKS[task_name]
    result = runner(opt_name, seed, n_steps)
    return opt_name, {
        "losses":      result["losses"],
        "theta_norms": result["theta_norms"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="vqe_stokes_overparam_long",
                   choices=list(TASKS.keys()),
                   help="Which task to diagnose. Phase 1: "
                        "vqe_stokes_overparam_long, fit_high_periodic. "
                        "Phase 2: vqe_overparam_heis.")
    p.add_argument("--phase", default="torus",
                   choices=list(OPTIMIZER_TIERS.keys()),
                   help="Which optimizer set to run. 'torus' = 4 Phase-1 "
                        "optimizers (back-compat with prior runs). 'sphere' "
                        "= 7 optimizers including the three Phase-2 sphere "
                        "variants.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=80,
                   help="Step count for the diagnostic run (smaller than the "
                        "production run; we just want to see the drift).")
    p.add_argument("--out", default=None,
                   help="Output path for the plot. Defaults to "
                        "results/manifold_diagnostics_<task>_<phase>_seed<seed>.png "
                        "in the workspace root.")
    p.add_argument("--workers", type=int, default=1,
                   help="Run optimizers in parallel across this many worker "
                        "processes. With 7 sphere-tier optimizers, any value "
                        ">= 7 gives the full speedup. Default 1 keeps the run "
                        "sequential and deterministic in stdout.")
    args = p.parse_args()

    optimizers = OPTIMIZER_TIERS[args.phase]

    print(f"Running manifold diagnostics on task={args.task} phase={args.phase} "
          f"seed={args.seed} n_steps={args.n_steps}  workers={args.workers}")
    print(f"  optimizers ({len(optimizers)}): {', '.join(optimizers)}")
    t0 = time.perf_counter()
    histories = {}
    if args.workers <= 1:
        for opt_name in optimizers:
            print(f"  ... {opt_name}")
            opt_name, hist = _worker(args.task, opt_name, args.seed, args.n_steps)
            histories[opt_name] = hist
    else:
        # Parallel branch: dispatch one process per optimizer. With N opts and
        # >=N workers, the total wall is bounded by the slowest single opt
        # (typically a QNG variant since metric_tensor + linear-solve is the
        # bottleneck). Each subprocess builds its own QNode -- circuits are
        # not picklable, so they cannot be shared across the pool.
        n = min(args.workers, len(optimizers))
        print(f"  Dispatching {len(optimizers)} optimizers across {n} workers")
        with ProcessPoolExecutor(max_workers=n) as pool:
            future_map = {
                pool.submit(_worker, args.task, opt_name, args.seed, args.n_steps): opt_name
                for opt_name in optimizers
            }
            for fut in as_completed(future_map):
                opt_name, hist = fut.result()
                histories[opt_name] = hist
                elapsed = time.perf_counter() - t0
                print(f"  done  {opt_name:<20s}  final loss={hist['losses'][-1]:.6f}  "
                      f"final ||theta||={hist['theta_norms'][-1]:.4f}  "
                      f"@ {elapsed:.1f}s")
    print(f"Total wall: {time.perf_counter() - t0:.1f}s")

    fig, (ax_t, ax_l) = plt.subplots(1, 2, figsize=(13, 4.5))

    for opt_name in optimizers:
        h = histories[opt_name]
        c = COLORS.get(opt_name)
        l = LABELS.get(opt_name, opt_name)
        steps = np.arange(1, len(h["theta_norms"]) + 1)
        ax_t.plot(steps, h["theta_norms"], color=c, label=l)
        ax_l.plot(steps, h["losses"],      color=c, label=l)

    ax_t.set_title(f"||theta|| trajectory  [{args.task}, {args.phase}, seed={args.seed}]")
    ax_t.set_xlabel("Step")
    ax_t.set_ylabel("||theta||")
    ax_t.grid(True, alpha=0.3)
    ax_t.legend(fontsize=8)

    ax_l.set_title(f"Loss trajectory  [{args.task}, {args.phase}, seed={args.seed}]")
    ax_l.set_xlabel("Step")
    ax_l.set_ylabel("Loss")
    ax_l.set_yscale("log")
    ax_l.grid(True, alpha=0.3)
    ax_l.legend(fontsize=8)

    fig.tight_layout()

    if args.out is None:
        results_root = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_root, exist_ok=True)
        out = os.path.join(
            results_root,
            f"manifold_diagnostics_{args.task}_{args.phase}_seed{args.seed}.png",
        )
    else:
        out = args.out
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    fig.savefig(out, dpi=150)
    print(f"Saved diagnostics plot -> {out}")

    # Print a final-state summary so the operator can sanity-check at a glance.
    # For sphere optimizers, ||theta|| MUST be 1.0 +/- 1e-6 -- if it isn't,
    # the geodesic retraction or the manifold init is buggy.
    print("\nFinal state per optimizer:")
    print(f"  {'optimizer':<20s}  {'final ||theta||':>15s}  {'final loss':>15s}")
    for opt_name in optimizers:
        h = histories[opt_name]
        norm = h["theta_norms"][-1]
        norm0 = h["theta_norms"][0]
        flag = ""
        if "sphere" in opt_name and abs(norm - norm0) > 1e-4 * max(1.0, norm0):
            flag = (f"  <-- WARNING: sphere optimizer drifted off init shell "
                    f"(R0={norm0:.4f})")
        print(f"  {opt_name:<20s}  {norm:>15.6f}  {h['losses'][-1]:>15.6f}{flag}")


if __name__ == "__main__":
    main()
