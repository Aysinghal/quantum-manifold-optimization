"""
Phase 2 smoke test: three sphere-aware optimizers on vqe_heis_ring.

Verifies the minimum invariants the plan calls out before any single-task
diagnostic or full sweep runs:

  1. No NaN / Inf in the loss trajectory.
  2. ||theta|| stays at its initial value (within 1e-6) for ALL three
     sphere optimizers across all tracked steps. The Sphere manifold uses
     an implicit radius read from the init point, defaulted to the
     "natural shell" R = 2*pi*sqrt(d/3) by init_on_manifold.
  3. Loss decreases (final < initial). The sphere doesn't have to *win* --
     it just needs to be sane optimization on a constrained surface.

Designed to finish in well under a minute. Cheapest task (4q x 4L
Heisenberg ring, 30 steps).

Usage:
    python smoke_sphere.py
"""

import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from src.models import (
    make_heisenberg_ring_hamiltonian,
    make_vqe_circuit,
    init_params_vqe,
)
from src.training import train_vqe


SEED      = 0
N_QUBITS  = 4
N_LAYERS  = 4
N_STEPS   = 30
LR        = 0.05

NORM_TOL  = 1e-6   # invariant: sphere optimizers must pin |theta| at its init value
LOSS_TOL  = 1e-6   # loss must decrease by at least this much

SPHERE_OPTS = ["RQNG_sphere_block", "ProjQNG_sphere", "ProjAdam_sphere"]


def _diff_method_for(opt_name):
    """ProjAdam_sphere can use adjoint; QNG-side sphere variants need
    parameter-shift because qml.metric_tensor is involved."""
    if opt_name == "ProjAdam_sphere":
        return "adjoint"
    return "parameter-shift"


def _run_one(opt_name):
    H = make_heisenberg_ring_hamiltonian(N_QUBITS, J=1.0, periodic=True)
    circuit, _ = make_vqe_circuit(
        N_QUBITS, N_LAYERS, hamiltonian=H,
        shots=None, diff_method=_diff_method_for(opt_name),
    )
    params = init_params_vqe(N_QUBITS, N_LAYERS, SEED)
    t0 = time.perf_counter()
    out = train_vqe(
        circuit, params, opt_name=opt_name, lr=LR, n_steps=N_STEPS,
        n_layers=N_LAYERS, verbose=False, track_theta_norm=True,
    )
    return out, time.perf_counter() - t0


def _check(opt_name, out):
    losses = np.asarray(out["losses"], dtype=float)
    norms  = np.asarray(out["theta_norms"], dtype=float)

    problems = []

    if not np.all(np.isfinite(losses)):
        problems.append(f"non-finite losses (any NaN/Inf? "
                        f"{np.any(~np.isfinite(losses))})")
    if not np.all(np.isfinite(norms)):
        problems.append("non-finite theta norms")

    if norms.size:
        R0 = float(norms[0])
        max_norm_dev = float(np.max(np.abs(norms - R0)))
    else:
        R0 = float("nan")
        max_norm_dev = float("inf")
    if max_norm_dev > NORM_TOL:
        problems.append(
            f"||theta|| drifted off the sphere: max|norm - R0| = {max_norm_dev:.3e} "
            f"(R0={R0:.4f}, tol {NORM_TOL:.0e})"
        )

    if losses.size >= 2:
        delta = losses[0] - losses[-1]
        if delta < LOSS_TOL:
            problems.append(
                f"loss did not decrease: initial={losses[0]:.6f} "
                f"final={losses[-1]:.6f} (delta={delta:+.3e})"
            )

    return problems, max_norm_dev, losses, R0


def main():
    print("=" * 72)
    print("  Phase 2 smoke test: sphere optimizers on vqe_heis_ring")
    print(f"  Circuit : {N_QUBITS}q x {N_LAYERS}L  ({N_STEPS} steps, lr={LR})")
    print(f"  Seed    : {SEED}")
    print(f"  Tasks   : {', '.join(SPHERE_OPTS)}")
    print("=" * 72)

    n_fail = 0
    for opt in SPHERE_OPTS:
        print(f"\n[{opt}]")
        try:
            out, dt = _run_one(opt)
        except Exception as exc:
            print(f"  FAIL ({type(exc).__name__}: {exc})")
            n_fail += 1
            continue

        problems, max_dev, losses, R0 = _check(opt, out)
        print(f"  init loss = {losses[0]:.6f}")
        print(f"  final loss= {losses[-1]:.6f}   (delta = {losses[0] - losses[-1]:+.6f})")
        print(f"  ||theta||_init   = {R0:.4f}")
        print(f"  max|||theta||-R0| = {max_dev:.3e}   (tol {NORM_TOL:.0e})")
        print(f"  wall              = {dt:.2f}s")

        if problems:
            n_fail += 1
            print("  RESULT: FAIL")
            for p in problems:
                print(f"    - {p}")
        else:
            print("  RESULT: OK")

    print("\n" + "=" * 72)
    print(f"  {len(SPHERE_OPTS) - n_fail}/{len(SPHERE_OPTS)} sphere optimizers passed")
    print("=" * 72)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
