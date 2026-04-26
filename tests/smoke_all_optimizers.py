"""
Unified regression smoke test for every optimizer in the registry.

Replaces the older ``smoke_new_optimizers.py`` (QNG-family-only) and
``smoke_sphere.py`` (sphere-only) with a single suite that iterates over
*every* name in ``run_all_parallel.OPTIMIZERS`` and asserts the manifold
invariants the optimizer is contractually obligated to satisfy.

Per-optimizer checks
--------------------
For every optimizer (any manifold):
  1. Trajectory is finite (no NaN / Inf in losses or theta_norms).
  2. Loss is non-increasing across the run (final <= initial + LOSS_TOL),
     allowing for a tiny numerical slack.

Plus, dispatched on ``manifold_for(opt_name)``:
  - Sphere optimizers: ``||theta||`` must stay pinned at the init radius
    within ``NORM_TOL`` (the sphere is implemented via geodesic retract,
    so any drift is a bug in the retract / projection code).
  - Torus / Euclidean optimizers: ``||theta||`` must be finite. Magnitude
    drift is allowed (and is exactly the diagnostic signal we want to
    record in the runner's theta_trajectory_plot).

Task: VQE on a 4-qubit periodic Heisenberg ring with 2 layers and 5 steps.
Cheapest realistic Hamiltonian-driven task we have; covers the full
analytic + parameter-shift codepath without burning wall time.

Usage:
    python tests/smoke_all_optimizers.py
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Make the repo root importable when running this script directly from
# anywhere -- tests/ is intentionally not a package, so a sys.path tweak
# is the simplest cross-shell solution.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.models import (
    make_heisenberg_ring_hamiltonian,
    make_vqe_circuit,
    init_params_vqe,
)
from src.training import train_vqe
from src.manifolds import manifold_for, Sphere, Torus
from run_all_parallel import OPTIMIZERS, _canonical_opt


SEED      = 0
N_QUBITS  = 4
N_LAYERS  = 2
N_STEPS   = 5

NORM_TOL  = 1e-6   # sphere optimizers: max drift off the init shell
LOSS_TOL  = 1e-6   # tolerated loss increase (numerical slack)


def _diff_method_for(opt_name):
    """Adam-family optimizers can use the cheaper analytic adjoint method;
    anything that calls qml.metric_tensor must use parameter-shift because
    lightning.qubit's adjoint pipeline doesn't tape-rewrite metric_tensor."""
    if opt_name in ("GD", "Adam", "ProjAdam_sphere"):
        return "adjoint"
    return "parameter-shift"


def _manifold_kind(opt_name):
    m = manifold_for(opt_name)
    if isinstance(m, Sphere):
        return "sphere"
    if isinstance(m, Torus):
        return "torus"
    return "euclidean"


def _run_one(opt_name, lr):
    """Run train_vqe on the canonical implementation behind opt_name.

    OPTIMIZERS contains learning-rate-pinned aliases (e.g. QNG_block_lr01
    is QNG_block at lr=0.1) which the runner resolves via _canonical_opt
    before dispatch; the training layer's EVAL_COST / _QNG_FAMILY tables
    only know canonical names. Mirroring that here lets the smoke exercise
    every registry entry without KeyError-ing on the aliases.
    """
    canonical = _canonical_opt(opt_name)
    H = make_heisenberg_ring_hamiltonian(N_QUBITS, J=1.0, periodic=True)
    circuit, _ = make_vqe_circuit(
        N_QUBITS, N_LAYERS, hamiltonian=H,
        shots=None, diff_method=_diff_method_for(canonical),
    )
    params = init_params_vqe(N_QUBITS, N_LAYERS, SEED)
    t0 = time.perf_counter()
    out = train_vqe(
        circuit, params, opt_name=canonical, lr=lr,
        n_steps=N_STEPS, n_layers=N_LAYERS, verbose=False,
    )
    return out, time.perf_counter() - t0


def _check(opt_name, out):
    """Return (problems, summary_dict) for one optimizer's run."""
    losses = np.asarray(out["losses"], dtype=float)
    norms  = np.asarray(out["theta_norms"], dtype=float)
    kind   = _manifold_kind(_canonical_opt(opt_name))

    problems = []

    if not np.all(np.isfinite(losses)):
        problems.append("non-finite losses (NaN/Inf in trajectory)")
    if not np.all(np.isfinite(norms)):
        problems.append("non-finite theta_norms (NaN/Inf in trajectory)")

    # Loss must not blow up. We allow tiny slack because parameter-shift
    # is finite-precision and 5-step runs see legitimate per-step jitter.
    if losses.size >= 2 and (losses[-1] > losses[0] + LOSS_TOL):
        problems.append(
            f"loss increased over the run: initial={losses[0]:.6f} "
            f"final={losses[-1]:.6f} (delta={losses[-1] - losses[0]:+.3e})"
        )

    # Sphere invariant: norm pinned at init radius.
    R0 = float(norms[0]) if norms.size else float("nan")
    max_dev = float(np.max(np.abs(norms - R0))) if norms.size else float("inf")
    if kind == "sphere" and max_dev > NORM_TOL:
        problems.append(
            f"sphere drift: max|||theta|| - R0| = {max_dev:.3e} "
            f"(R0={R0:.4f}, tol {NORM_TOL:.0e}); retract bug?"
        )

    summary = {
        "kind":     kind,
        "loss_i":   float(losses[0]) if losses.size else float("nan"),
        "loss_f":   float(losses[-1]) if losses.size else float("nan"),
        "R0":       R0,
        "max_dev":  max_dev,
    }
    return problems, summary


def main():
    print("=" * 78)
    print("  Smoke test: every optimizer in the registry on vqe_heis_ring")
    print(f"  Circuit  : {N_QUBITS}q x {N_LAYERS}L  ({N_STEPS} steps)")
    print(f"  Seed     : {SEED}")
    print(f"  Opts     : {len(OPTIMIZERS)} -- {', '.join(OPTIMIZERS)}")
    print("=" * 78)

    rows = []
    n_fail = 0
    for opt_name, cfg in OPTIMIZERS.items():
        lr = cfg["lr"]
        kind = _manifold_kind(_canonical_opt(opt_name))
        print(f"\n[{opt_name}]  ({kind}, lr={lr})")
        try:
            out, dt = _run_one(opt_name, lr)
        except Exception as exc:
            n_fail += 1
            msg = f"FAIL ({type(exc).__name__}: {exc})"
            print(f"  {msg}")
            rows.append((opt_name, kind, "FAIL", msg, 0.0))
            continue

        problems, s = _check(opt_name, out)
        print(f"  init loss = {s['loss_i']:+.6f}   final loss = {s['loss_f']:+.6f}"
              f"   delta = {s['loss_f'] - s['loss_i']:+.3e}")
        print(f"  ||theta||_init = {s['R0']:.4f}   max|||theta||-R0| = {s['max_dev']:.3e}")
        print(f"  wall = {dt:.2f}s")

        if problems:
            n_fail += 1
            print("  RESULT: FAIL")
            for p in problems:
                print(f"    - {p}")
            status = "FAIL"
            note = "; ".join(problems)
        else:
            print("  RESULT: OK")
            status = "OK"
            note = ""
        rows.append((opt_name, kind, status, note, dt))

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    width = max(len(name) for name, *_ in rows)
    for name, kind, status, note, dt in rows:
        suffix = f"  -- {note}" if note else ""
        print(f"  {name:<{width}}  ({kind:<9}) {status:<4} {dt:5.2f}s{suffix}")
    n_total = len(rows)
    print("=" * 78)
    print(f"  {n_total - n_fail}/{n_total} optimizers passed  ({n_fail} failed)")
    print("=" * 78)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
