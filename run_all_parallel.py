"""
Parallel runner: executes all 80 experiment jobs across multiple CPU cores.

Each (experiment, optimizer, seed) combination runs as an independent worker
process. Results are collected, aggregated, and saved/plotted; for each task
the runner emits four plots (convergence, resource, final, theta) so that
single-seed single-task invocations double as the manifold diagnostic that
used to live in experiments/manifold_diagnostics.py.

Usage:
    python run_all_parallel.py               # use all cores
    python run_all_parallel.py --workers 16  # use 16 cores
"""

import argparse
import heapq
import json
import multiprocessing
import os
import platform
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import numpy as np
from pennylane import numpy as pnp

from src.models import (
    make_regression_circuit_1d,
    make_regression_circuit_2d,
    make_vqe_circuit,
    make_classification_circuit,
    make_stokes_hamiltonian,
    make_heisenberg_ring_hamiltonian,
    make_sk_hamiltonian,
    make_xxz_hamiltonian,
    init_params_regression,
    init_params_vqe,
    exact_ground_energy,
    exact_ground_energy_from_h,
)
from src.training import train_with_data, train_vqe
from src.metrics import aggregate_seeds, save_results, accuracy, make_run_dir
from src.visualization import (
    convergence_plot, resource_plot, final_loss_bar,
    theta_trajectory_plot, task_plot_path,
)


# ── Shared config (mirrors the individual experiment modules) ────────────────

DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def _detect_available_cpus():
    """Number of cores actually available to this process.

    Priority: SLURM_CPUS_PER_TASK (the cores Slurm allocated to *this* job)
    > os.sched_getaffinity (cgroup/cpuset honoured by the kernel) >
    os.cpu_count (number of cores on the box; pessimistic only on shared
    nodes, which is exactly the case where the prior two checks help).
    """
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus:
        try:
            return max(1, int(cpus))
        except ValueError:
            pass
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)

OPTIMIZERS = {
    "GD":              {"lr": 0.1},
    "Adam":            {"lr": 0.05},
    "QNG_block":       {"lr": 0.05},
    "QNG_block_lr01":  {"lr": 0.1},
    "QNG_block_lr02":  {"lr": 0.2},
    "MomentumQNG_block": {"lr": 0.05},
    "QNGAdam_v1_block":  {"lr": 0.05},
    "QNGAdam_v2_block":  {"lr": 0.05},
    # QNG_full is in the registry but excluded from the default opt tier
    # because its O(d^2) metric tensor cost dominates wall-time. Reference
    # it from a tier (e.g. OPT_TIERS["full_qng_sweep"]) to opt in.
    "QNG_full":        {"lr": 0.01},
    # Phase 1 R-QNG: same lr as QNG_block since the algorithm is identical
    # except for the mod-2pi retraction (see src/manifolds.py::Torus).
    "RQNG_torus_block": {"lr": 0.05},
    # Phase 2 sphere optimizers. RQNG_sphere_block / ProjQNG_sphere reuse
    # the QNG_block lr; ProjAdam_sphere reuses the Adam lr. lr*||v|| is the
    # geodesic arc length traversed per step -- with lr=0.05 and unit
    # tangent vectors this is ~3 degrees per step, which is conservative.
    "RQNG_sphere_block": {"lr": 0.05},
    "ProjQNG_sphere":    {"lr": 0.05},
    "ProjAdam_sphere":   {"lr": 0.05},
}

# LPT scheduling priorities. Higher = scheduled earlier so the longest-running
# jobs grab workers first and short jobs pack into the gaps. Numbers are
# relative; tweak by adding/replacing entries when introducing a new optimizer
# or task in the future.
OPT_PRIORITY = {
    "QNG_full":          100,
    "QNG_block":          50,
    "QNG_block_lr01":     50,
    "QNG_block_lr02":     50,
    "MomentumQNG_block":  50,
    "QNGAdam_v1_block":   50,
    "QNGAdam_v2_block":   50,
    "RQNG_torus_block":   50,
    "RQNG_sphere_block":  50,
    "ProjQNG_sphere":     50,
    "ProjAdam_sphere":    10,
    "Adam":               10,
    "GD":                 10,
}
# TASK_PRIORITY weights are proportional to the measured QNG-family mean
# wall-time per job in results/2026-04-24_030537 (seconds, divided by ~45 s).
# QNG variants dominate the wall-clock, so calibrating against their runtime
# lets LPT schedule the real giants (cls, cls_moons_hard, fit2d) in the first
# wave instead of letting them drag the tail. Recalibrate this table when
# adding new tasks/optimizers, or when problem sizes (n_qubits, n_layers,
# n_steps, n_data) change enough to shift the per-task ranking.
TASK_PRIORITY = {
    # Tier 0: original baseline tasks.
    "cls":              20,   # ~15m08s -- bottleneck
    "fit2d":            14,   # ~10m18s
    "vqe":               4,   # ~3m00s  (11q TFIM)
    "fit1d":             4,   # ~2m55s
    # Tier 1: new "rigged" tasks. Cls-moons-hard is the other 15-min giant;
    # the two VQE-shaped ones (6q and 4q) are actually cheap.
    "cls_moons_hard":   20,   # ~15m08s -- bottleneck
    "fit_multifreq1d":   5,   # ~3m55s
    "vqe_stokes":        2,   # ~1m15s  (6q x 6L)
    "vqe_heis_ring":     1,   # ~0m20s  (4q x 4L)
    # Phase 1 (manifold_torus) -- initial guesses, recalibrate after first
    # full run from the timing-breakdown table.
    "vqe_stokes_overparam_long":  10,  # 8q x 8L x 300 steps -- biggest VQE
    "fit_high_periodic":           4,  # 2q x 4L x 200 steps
    "vqe_sk_spinglass":            6,  # 6q x 4L x 150 steps + shots
    "vqe_xxz":                     8,  # 8q x 4L x 150 steps + shots
    # Phase 2 (manifold_sphere) -- only NEW task; the rest of the tier
    # reuses Phase 1 task funcs and inherits their priorities above.
    "vqe_overparam_heis":          3,  # 4q x 8L x 150 steps -- small-d, dense L
}


# Task groupings drive the --tasks CLI flag. Each key is a purpose-based
# label; the value is the list of task names that worker_specs knows how to
# dispatch. The flag also accepts individual task names directly, so a tier
# is just an optional convenience grouping. Add new tier entries (e.g.
# "manifold_torus") as new worker funcs land per the R-QNG plan.
TASK_TIERS = {
    "sanity":        ["fit1d", "fit2d", "vqe", "cls"],
    "qng_advantage": ["vqe_stokes", "vqe_heis_ring", "fit_multifreq1d", "cls_moons_hard"],
    # Phase 1 of the R-QNG plan: 7 tasks chosen so that R-QNG-torus has a
    # plausible mechanism to help. Two carry over from qng_advantage as
    # "sanity that R-QNG is at least competitive on existing benchmarks";
    # two are deliberately rigged (long Stokes overparam, high-frequency
    # periodic target); three are hard real-world physics Hamiltonians.
    "manifold_torus": [
        "vqe_heis_ring",
        "fit_multifreq1d",
        "vqe_stokes_overparam_long",
        "fit_high_periodic",
        "vqe",
        "vqe_sk_spinglass",
        "vqe_xxz",
    ],
    # Phase 2 of the R-QNG plan: same 7 tasks as Phase 1 plus a single
    # overparameterized Heisenberg ring designed to stress the sphere
    # constraint. The sphere-aware optimizers run alongside Phase 1's
    # baselines + R-QNG-torus, so we can read both the Phase-1 null and
    # the Phase-2 result off one run.
    "manifold_sphere": [
        "vqe_heis_ring",
        "fit_multifreq1d",
        "vqe_stokes_overparam_long",
        "fit_high_periodic",
        "vqe",
        "vqe_sk_spinglass",
        "vqe_xxz",
        "vqe_overparam_heis",
    ],
}

# Optimizer groupings drive the --optimizers CLI flag. OPTIMIZERS above is
# the *registry* of every defined optimizer + its lr; OPT_TIERS picks which
# subset of registry entries actually gets scheduled per run. The flag also
# accepts individual optimizer names directly, so tiers are an optional
# convenience grouping. Reference any name that exists in OPTIMIZERS.
OPT_TIERS = {
    "flat_only":      ["GD", "Adam"],
    "qng_full_sweep": ["GD", "Adam", "QNG_block", "QNG_block_lr01",
                       "QNG_block_lr02", "MomentumQNG_block",
                       "QNGAdam_v1_block", "QNGAdam_v2_block"],
    # Opt-in tier for the QNG_full O(d^2) metric tensor variant. Kept
    # separate so the default benchmark isn't bottlenecked by it.
    "full_qng_sweep": ["GD", "Adam", "QNG_block", "QNG_block_lr01",
                       "QNG_block_lr02", "MomentumQNG_block",
                       "QNGAdam_v1_block", "QNGAdam_v2_block", "QNG_full"],
    # Phase 1 of the R-QNG plan: 4 optimizers -- 3 flat baselines (one
    # vanilla, one QNG, one current best) + R-QNG on the torus. Smaller than
    # qng_full_sweep on purpose; the QNG variant zoo is no longer the story.
    "torus_concept":  ["Adam", "QNG_block", "QNGAdam_v1_block", "RQNG_torus_block"],
    # Phase 2 of the R-QNG plan: 7 optimizers. Three flat baselines + the
    # Phase-1 torus result + three sphere-aware optimizers. With ProjAdam
    # and ProjQNG also living on the sphere, the sphere comparison is fully
    # apples-to-apples (everyone searches the same restricted state space;
    # only the update rule differs).
    "sphere_concept": [
        "Adam", "QNG_block", "QNGAdam_v1_block",
        "RQNG_torus_block",
        "RQNG_sphere_block", "ProjQNG_sphere", "ProjAdam_sphere",
    ],
}


def _canonical_opt(opt_name):
    """Map optimizer-variant names back to the base optimizer the train_*
    functions know how to dispatch on (e.g. 'QNG_block_lr01' -> 'QNG_block').

    The QNG-family variants MomentumQNG_block / QNGAdam_v{1,2}_block are
    distinct training paths and pass through unchanged. Phase 1/2 R-QNG
    variants (RQNG_torus_block, RQNG_sphere_block, ProjQNG_sphere,
    ProjAdam_sphere) also pass through unchanged because training.py
    dispatches per-name on EVAL_COST / _QNG_FAMILY / manifold_for().
    """
    if opt_name in (
        "MomentumQNG_block", "QNGAdam_v1_block", "QNGAdam_v2_block",
        "RQNG_torus_block", "RQNG_sphere_block",
        "ProjQNG_sphere", "ProjAdam_sphere",
    ):
        return opt_name
    if opt_name.startswith("QNG_block"):
        return "QNG_block"
    if opt_name.startswith("QNG_full"):
        return "QNG_full"
    return opt_name


def job_priority(task, opt_name):
    """LPT score for a (task, optimizer) pair."""
    return OPT_PRIORITY[opt_name] * TASK_PRIORITY[task]


def _resolve_names(flag_value, tier_registry, individual_set, axis_name):
    """Resolve a comma-separated CLI list against a tier registry and an
    individual-name set, supporting four input shapes (and any mix of them):

      1. Single tier name             e.g. 'torus_concept'
      2. Comma-separated tier names   e.g. 'sanity,qng_advantage'
      3. Single individual name       e.g. 'Adam'
      4. Comma-separated individuals  e.g. 'Adam,QNG_block'
      5. Any mix of the above         e.g. 'torus_concept,QNG_full'

    Each token is independently looked up: tier names expand to their list,
    individual names pass through as-is. Returns (tier_keys, flat_items)
    where flat_items preserves first-seen order across the union and dedupes.
    Tokens that match neither raise SystemExit with a helpful diagnostic.
    """
    tokens = [k.strip() for k in flag_value.split(",") if k.strip()]
    seen, out, tier_keys = set(), [], []
    for tok in tokens:
        if tok in tier_registry:
            tier_keys.append(tok)
            for item in tier_registry[tok]:
                if item not in seen:
                    seen.add(item)
                    out.append(item)
        elif tok in individual_set:
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
        else:
            raise SystemExit(
                f"Unknown {axis_name}: {tok!r}. "
                f"Defined tiers: {sorted(tier_registry.keys())}. "
                f"Defined individuals: {sorted(individual_set)}."
            )
    return tier_keys, out


def _parse_seeds(flag_value):
    """Parse the --seeds flag as either a single int or a comma-separated
    list of ints. ``None`` (flag not provided) returns the default."""
    if flag_value is None:
        return list(DEFAULT_SEEDS)
    try:
        return [int(s.strip()) for s in flag_value.split(",") if s.strip()]
    except ValueError as exc:
        raise SystemExit(f"--seeds must be int or comma-separated ints: {exc}")

REG_N_QUBITS = 2
REG_N_LAYERS = 4
REG_N_STEPS  = 100

VQE_N_QUBITS = 11
VQE_N_LAYERS = 4
VQE_N_STEPS  = 150
VQE_J        = 1.0
VQE_H        = 1.0

CLS_N_DATA   = 100

# Tier 1 task config.
STOKES_N_QUBITS  = 6     # circuit width; H only acts on wires 0,1 -> heavy redundancy
STOKES_N_LAYERS  = 6
STOKES_N_STEPS   = 100

HEIS_N_QUBITS    = 4
HEIS_N_LAYERS    = 4
HEIS_N_STEPS     = 100
HEIS_J           = 1.0   # antiferromagnetic, periodic boundary

CLS_HARD_NOISE   = 0.40  # vs the saturated 0.15 of the original cls task
CLS_HARD_N_DATA  = 100

RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

TASK_LABELS = {
    # Tier 0
    "fit1d": "1D-sin",
    "fit2d": "2D-quad",
    "vqe":   "VQE-TFIM-11q",
    "cls":   "Cls-moons-easy",
    # Tier 1
    "vqe_stokes":      "VQE-Stokes-6q",
    "vqe_heis_ring":   "VQE-Heis-ring-4q",
    "fit_multifreq1d": "1D-multifreq",
    "cls_moons_hard":  "Cls-moons-hard",
    # Phase 1 (manifold_torus tier)
    "vqe_stokes_overparam_long": "VQE-Stokes-overparam-8q",
    "fit_high_periodic":         "1D-high-periodic",
    "vqe_sk_spinglass":          "VQE-SK-spinglass-6q",
    "vqe_xxz":                   "VQE-XXZ-chain-8q",
    # Phase 2 (manifold_sphere tier)
    "vqe_overparam_heis":        "VQE-Heis-overparam-4q8L",
}


# ── Phase 1 (manifold_torus) task config ────────────────────────────────────
#
# Designed to *let R-QNG-torus help* (overparameterized + long horizons + high
# frequency content) while still spanning toy-sanity, rigged, and hard-physics
# regimes. See plan section 4 ("Phase 1 -- Torus only").

# vqe_stokes_overparam_long: 8q, 8L circuit on the same Z0Z1 Hamiltonian as
# the original Stokes task, but with 300 steps. Massive overparameterization
# is what makes ||theta|| drift large for flat optimizers; the long horizon
# is what gives the drift time to matter.
STOKES_LONG_N_QUBITS = 8
STOKES_LONG_N_LAYERS = 8
STOKES_LONG_N_STEPS  = 300

# fit_high_periodic: target sin(7x) on [-3*pi, 3*pi] with 60 sample points.
# High-frequency target on a multiple-period domain -- gradients that wrap
# around 2*pi naturally favor torus updates.
FIT_HIGH_N_DATA  = 60
FIT_HIGH_N_STEPS = 200
FIT_HIGH_X_SPAN  = 3 * np.pi   # data lives in [-3*pi, 3*pi]

# vqe_sk_spinglass: SK Hamiltonian, 6q, 4L, 1000 shots, 150 steps.
SK_N_QUBITS = 6
SK_N_LAYERS = 4
SK_N_STEPS  = 150
SK_SHOTS    = 1000
SK_SEED     = 42

# vqe_xxz: XXZ chain, 8q, 4L, 1000 shots, 150 steps.
XXZ_N_QUBITS = 8
XXZ_N_LAYERS = 4
XXZ_N_STEPS  = 150
XXZ_SHOTS    = 1000
XXZ_DELTA    = 1.0

# vqe_overparam_heis (Phase 2): 4q Heisenberg ring with 8 layers (vs 4 layers
# for the Phase-1 vqe_heis_ring task). Same Hamiltonian, twice the layer
# count -> ~2x parameters on the same 16-dim Hilbert space, classic
# overparameterized regime where the sphere constraint stresses the
# optimizers most. Analytic, 150 steps.
HEIS_OVERPARAM_N_QUBITS = 4
HEIS_OVERPARAM_N_LAYERS = 8
HEIS_OVERPARAM_N_STEPS  = 150
HEIS_OVERPARAM_J        = 1.0


def _multifreq1d_target(x):
    """Target function for fit_multifreq1d: y = sin x + 0.4 sin 5x + 0.2 sin 13x.

    Heterogeneous frequency content -> the high-frequency components produce
    much smaller gradients on certain encoding parameters than the dominant
    sin x term. This is the textbook setting where Adam's per-parameter
    adaptive learning rate (1/sqrt(v_t)) really pays off; combined with the
    coupled 2-qubit encoding, the QNG-Adam hybrids should win here.
    """
    return np.sin(x) + 0.4 * np.sin(5 * x) + 0.2 * np.sin(13 * x)


def _high_periodic_target(x):
    """Target function for fit_high_periodic: y = sin(7x).

    High-frequency single-tone target on a multi-period domain (x in
    [-3*pi, 3*pi]). The "natural" parameter angles for fitting this target
    span several full rotations; flat optimizers therefore have no upper
    bound on ||theta||, while torus-aware R-QNG keeps everything in
    [0, 2*pi)^d. Phase 1's clearest "torus should help" task.
    """
    return np.sin(7.0 * x)


# ── Worker functions (module-level for Windows spawn compatibility) ──────────

def _drop_params(result):
    """Drop the params tensor before returning to the main process."""
    return {k: v for k, v in result.items() if k != "params"}


def _diff_method_for(opt_name, shots):
    """QNG-family optimizers need parameter-shift regardless of shots, because
    qml.metric_tensor builds tapes that lightning.qubit's adjoint does
    not support. GD/Adam (incl. ProjAdam_sphere, which is Adam + retract)
    can use adjoint in analytic mode for a big speedup."""
    if (opt_name.startswith("QNG")
            or opt_name.startswith("MomentumQNG")
            or opt_name.startswith("RQNG")
            or opt_name == "ProjQNG_sphere"):
        return "parameter-shift"
    return None  # let _resolve_diff_method pick based on shots


# ── Bench builders ──────────────────────────────────────────────────────────
#
# tests/bench_smoke.py needs to invoke the same construction the workers above
# perform, but only for a handful of timed steps. Rather than duplicate the
# 13 worker bodies (drift risk every time we add a task), each builder
# below returns a closure ``run(n_steps)`` plus the production step count
# and layer count for that task. The bench:
#   1. calls run(WARMUP)        # untimed
#   2. times run(TIMED_STEPS)
#   3. extrapolates per_step * prod_steps
#
# Two design notes:
# * Bench builders intentionally use ``seed=0`` and skip the progress-dict
#   bookkeeping that the production workers do; bench is a single-process,
#   single-seed timing harness.
# * vqe_sk_spinglass and vqe_xxz hardcode their own shot counts in the
#   production path (1000 each), so the ``shots`` argument here is the
#   *effective* shot count -- the runner's --shots flag does not override
#   them. That means in the bench JSON the "analytic" and "shots=1000"
#   entries for those two tasks will be near-identical; that's correct.

def _bench_build_fit1d(opt_name, shots):
    x_raw = np.linspace(-np.pi, np.pi, 20)
    y_raw = np.sin(x_raw)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]
    circuit = make_regression_circuit_1d(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_with_data(circuit, params, x_train, y_train,
                        opt_name=_canonical_opt(opt_name), lr=lr,
                        n_steps=n_steps, n_layers=REG_N_LAYERS,
                        loss_type="mse", verbose=False)
    return run, REG_N_STEPS, REG_N_LAYERS


def _bench_build_fit2d(opt_name, shots):
    g = np.linspace(-1, 1, 8)
    x1, x2 = np.meshgrid(g, g)
    x_raw = np.stack([x1.ravel(), x2.ravel()], axis=1)
    y_raw = 0.5 * (x_raw[:, 0] ** 2 + x_raw[:, 1] ** 2)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]
    circuit = make_regression_circuit_2d(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_with_data(circuit, params, x_train, y_train,
                        opt_name=_canonical_opt(opt_name), lr=lr,
                        n_steps=n_steps, n_layers=REG_N_LAYERS,
                        loss_type="mse", verbose=False)
    return run, REG_N_STEPS, REG_N_LAYERS


def _bench_build_vqe(opt_name, shots):
    circuit, _ = make_vqe_circuit(
        VQE_N_QUBITS, VQE_N_LAYERS, VQE_J, VQE_H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(VQE_N_QUBITS, VQE_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_vqe(circuit, params, opt_name=_canonical_opt(opt_name),
                  lr=lr, n_steps=n_steps, n_layers=VQE_N_LAYERS, verbose=False)
    return run, VQE_N_STEPS, VQE_N_LAYERS


def _bench_build_cls(opt_name, shots):
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler
    X, y = make_moons(n_samples=CLS_N_DATA, noise=0.15, random_state=42)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = 2 * y - 1
    x_train = [pnp.array(xi, requires_grad=False) for xi in X]
    y_train = [pnp.array(float(yi), requires_grad=False) for yi in y]
    circuit = make_classification_circuit(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_with_data(circuit, params, x_train, y_train,
                        opt_name=_canonical_opt(opt_name), lr=lr,
                        n_steps=n_steps, n_layers=REG_N_LAYERS,
                        loss_type="hinge", verbose=False)
    return run, REG_N_STEPS, REG_N_LAYERS


def _bench_build_vqe_stokes(opt_name, shots):
    H = make_stokes_hamiltonian(STOKES_N_QUBITS)
    circuit, _ = make_vqe_circuit(
        STOKES_N_QUBITS, STOKES_N_LAYERS, hamiltonian=H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(STOKES_N_QUBITS, STOKES_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_vqe(circuit, params, opt_name=_canonical_opt(opt_name),
                  lr=lr, n_steps=n_steps, n_layers=STOKES_N_LAYERS, verbose=False)
    return run, STOKES_N_STEPS, STOKES_N_LAYERS


def _bench_build_vqe_heis_ring(opt_name, shots):
    H = make_heisenberg_ring_hamiltonian(HEIS_N_QUBITS, J=HEIS_J, periodic=True)
    circuit, _ = make_vqe_circuit(
        HEIS_N_QUBITS, HEIS_N_LAYERS, hamiltonian=H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(HEIS_N_QUBITS, HEIS_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_vqe(circuit, params, opt_name=_canonical_opt(opt_name),
                  lr=lr, n_steps=n_steps, n_layers=HEIS_N_LAYERS, verbose=False)
    return run, HEIS_N_STEPS, HEIS_N_LAYERS


def _bench_build_fit_multifreq1d(opt_name, shots):
    x_raw = np.linspace(-np.pi, np.pi, 30)
    y_raw = _multifreq1d_target(x_raw)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]
    circuit = make_regression_circuit_1d(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_with_data(circuit, params, x_train, y_train,
                        opt_name=_canonical_opt(opt_name), lr=lr,
                        n_steps=n_steps, n_layers=REG_N_LAYERS,
                        loss_type="mse", verbose=False)
    return run, REG_N_STEPS, REG_N_LAYERS


def _bench_build_cls_moons_hard(opt_name, shots):
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler
    X, y = make_moons(n_samples=CLS_HARD_N_DATA, noise=CLS_HARD_NOISE, random_state=42)
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)
    y = 2 * y - 1
    x_train = [pnp.array(xi, requires_grad=False) for xi in X]
    y_train = [pnp.array(float(yi), requires_grad=False) for yi in y]
    circuit = make_classification_circuit(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_with_data(circuit, params, x_train, y_train,
                        opt_name=_canonical_opt(opt_name), lr=lr,
                        n_steps=n_steps, n_layers=REG_N_LAYERS,
                        loss_type="hinge", verbose=False)
    return run, REG_N_STEPS, REG_N_LAYERS


def _bench_build_vqe_stokes_overparam_long(opt_name, shots):
    H = make_stokes_hamiltonian(STOKES_LONG_N_QUBITS)
    circuit, _ = make_vqe_circuit(
        STOKES_LONG_N_QUBITS, STOKES_LONG_N_LAYERS, hamiltonian=H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(STOKES_LONG_N_QUBITS, STOKES_LONG_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_vqe(circuit, params, opt_name=_canonical_opt(opt_name),
                  lr=lr, n_steps=n_steps, n_layers=STOKES_LONG_N_LAYERS,
                  verbose=False)
    return run, STOKES_LONG_N_STEPS, STOKES_LONG_N_LAYERS


def _bench_build_fit_high_periodic(opt_name, shots):
    x_raw = np.linspace(-FIT_HIGH_X_SPAN, FIT_HIGH_X_SPAN, FIT_HIGH_N_DATA)
    y_raw = _high_periodic_target(x_raw)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]
    circuit = make_regression_circuit_1d(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_with_data(circuit, params, x_train, y_train,
                        opt_name=_canonical_opt(opt_name), lr=lr,
                        n_steps=n_steps, n_layers=REG_N_LAYERS,
                        loss_type="mse", verbose=False)
    return run, FIT_HIGH_N_STEPS, REG_N_LAYERS


def _bench_build_vqe_sk_spinglass(opt_name, shots):
    # Production hardcodes 1000 shots regardless of CLI flag (see worker
    # docstring); mirror that here so the timing matches what runs.
    task_shots = SK_SHOTS
    H = make_sk_hamiltonian(SK_N_QUBITS, seed=SK_SEED)
    circuit, _ = make_vqe_circuit(
        SK_N_QUBITS, SK_N_LAYERS, hamiltonian=H,
        shots=task_shots, diff_method=_diff_method_for(opt_name, task_shots),
    )
    params = init_params_vqe(SK_N_QUBITS, SK_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_vqe(circuit, params, opt_name=_canonical_opt(opt_name),
                  lr=lr, n_steps=n_steps, n_layers=SK_N_LAYERS, verbose=False)
    return run, SK_N_STEPS, SK_N_LAYERS


def _bench_build_vqe_xxz(opt_name, shots):
    task_shots = XXZ_SHOTS  # see _bench_build_vqe_sk_spinglass
    H = make_xxz_hamiltonian(XXZ_N_QUBITS, delta=XXZ_DELTA)
    circuit, _ = make_vqe_circuit(
        XXZ_N_QUBITS, XXZ_N_LAYERS, hamiltonian=H,
        shots=task_shots, diff_method=_diff_method_for(opt_name, task_shots),
    )
    params = init_params_vqe(XXZ_N_QUBITS, XXZ_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_vqe(circuit, params, opt_name=_canonical_opt(opt_name),
                  lr=lr, n_steps=n_steps, n_layers=XXZ_N_LAYERS, verbose=False)
    return run, XXZ_N_STEPS, XXZ_N_LAYERS


def _bench_build_vqe_overparam_heis(opt_name, shots):
    H = make_heisenberg_ring_hamiltonian(
        HEIS_OVERPARAM_N_QUBITS, J=HEIS_OVERPARAM_J, periodic=True,
    )
    circuit, _ = make_vqe_circuit(
        HEIS_OVERPARAM_N_QUBITS, HEIS_OVERPARAM_N_LAYERS, hamiltonian=H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(HEIS_OVERPARAM_N_QUBITS, HEIS_OVERPARAM_N_LAYERS, 0)
    lr = OPTIMIZERS[opt_name]["lr"]
    def run(n_steps):
        train_vqe(circuit, params, opt_name=_canonical_opt(opt_name),
                  lr=lr, n_steps=n_steps,
                  n_layers=HEIS_OVERPARAM_N_LAYERS, verbose=False)
    return run, HEIS_OVERPARAM_N_STEPS, HEIS_OVERPARAM_N_LAYERS


# Lookup table: task_name -> bench builder. Mirrors the order of
# all_worker_specs in main(). bench_smoke iterates over these keys.
BENCH_BUILDERS = {
    "fit1d":                     _bench_build_fit1d,
    "fit2d":                     _bench_build_fit2d,
    "vqe":                       _bench_build_vqe,
    "cls":                       _bench_build_cls,
    "vqe_stokes":                _bench_build_vqe_stokes,
    "vqe_heis_ring":             _bench_build_vqe_heis_ring,
    "fit_multifreq1d":           _bench_build_fit_multifreq1d,
    "cls_moons_hard":            _bench_build_cls_moons_hard,
    "vqe_stokes_overparam_long": _bench_build_vqe_stokes_overparam_long,
    "fit_high_periodic":         _bench_build_fit_high_periodic,
    "vqe_sk_spinglass":          _bench_build_vqe_sk_spinglass,
    "vqe_xxz":                   _bench_build_vqe_xxz,
    "vqe_overparam_heis":        _bench_build_vqe_overparam_heis,
}


# Path of the bench JSON the heartbeat reads for ETA fallback.
BENCH_ESTIMATES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bench_smoke_estimates.json",
)


def load_bench_estimates(path=BENCH_ESTIMATES_PATH):
    """Load saved per-(task, opt, mode) timing estimates.

    Returns the parsed JSON dict, or None if the file is missing /
    malformed. Heartbeat treats a missing file as "no fallback available"
    and falls back to its prior behaviour (assume 0 load for unknowns).
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def lookup_bench_full_run(bench, task, opt_name, shots, steps_override=None):
    """Predicted full-run wall time (seconds) for one (task, opt, shots)
    combo, pulled from a loaded bench JSON.

    ``shots`` is the runner's CLI value (None for analytic, int for shots).
    ``steps_override`` is the global ``--steps N`` override; if set, the
    bench's stored ``per_step_s`` is rescaled to ``per_step_s * N`` instead
    of using the saved ``full_run_s`` (which assumed the production step
    count). Falls through to None if no entry exists, which the heartbeat
    treats as "no estimate available".
    """
    if not bench:
        return None
    mode_key = "analytic" if shots is None else f"shots={shots}"
    try:
        entry = bench["estimates"][task][opt_name][mode_key]
        if steps_override is not None:
            return float(entry["per_step_s"]) * int(steps_override)
        return float(entry["full_run_s"])
    except (KeyError, TypeError, ValueError):
        return None




def _run_fit1d(opt_name, lr, seed, progress, shots, n_steps_override=None):
    key = ("fit1d", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else REG_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    x_raw = np.linspace(-np.pi, np.pi, 20)
    y_raw = np.sin(x_raw)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]

    circuit = make_regression_circuit_1d(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params  = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=REG_N_LAYERS, loss_type="mse", verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps, "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("fit1d", opt_name, seed, _drop_params(result))


def _run_fit2d(opt_name, lr, seed, progress, shots, n_steps_override=None):
    key = ("fit2d", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else REG_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    g = np.linspace(-1, 1, 8)
    x1, x2 = np.meshgrid(g, g)
    x_raw = np.stack([x1.ravel(), x2.ravel()], axis=1)
    y_raw = 0.5 * (x_raw[:, 0] ** 2 + x_raw[:, 1] ** 2)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]

    circuit = make_regression_circuit_2d(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params  = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=REG_N_LAYERS, loss_type="mse", verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps, "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("fit2d", opt_name, seed, _drop_params(result))


def _run_vqe(opt_name, lr, seed, progress, shots, n_steps_override=None):
    key = ("vqe", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else VQE_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    circuit, _H = make_vqe_circuit(
        VQE_N_QUBITS, VQE_N_LAYERS, VQE_J, VQE_H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(VQE_N_QUBITS, VQE_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_vqe(
        circuit, params,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=VQE_N_LAYERS, verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps, "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("vqe", opt_name, seed, _drop_params(result))


def _run_cls(opt_name, lr, seed, progress, shots, n_steps_override=None):
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler

    key = ("cls", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else REG_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    X, y = make_moons(n_samples=CLS_N_DATA, noise=0.15, random_state=42)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    y = 2 * y - 1

    x_train = [pnp.array(xi, requires_grad=False) for xi in X]
    y_train = [pnp.array(float(yi), requires_grad=False) for yi in y]

    circuit = make_classification_circuit(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params  = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=REG_N_LAYERS, loss_type="hinge", verbose=False,
        progress_cb=_cb,
    )

    preds = [float(circuit(result["params"], x)) for x in x_train]
    result["final_accuracy"] = accuracy(preds, [float(yi) for yi in y])

    progress[key] = {"status": "done", "step": n_steps, "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("cls", opt_name, seed, _drop_params(result))


# ── Tier 1 workers ──────────────────────────────────────────────────────

def _run_vqe_stokes(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """A1: Stokes-style mini-VQE -- favors vanilla QNG.

    H = Z_0 (x) Z_1 only, on a 6q x 6L circuit. Massive parameter redundancy
    (~108 trainable params, only 2 wires actually contribute to <H>) gives
    the QNG metric tensor a lot of "down-weight the dead directions" work to
    do. Loss is bounded in [-1, 1], ground-state energy is exactly -1.
    """
    key = ("vqe_stokes", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else STOKES_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    H = make_stokes_hamiltonian(STOKES_N_QUBITS)
    circuit, _ = make_vqe_circuit(
        STOKES_N_QUBITS, STOKES_N_LAYERS, hamiltonian=H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(STOKES_N_QUBITS, STOKES_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_vqe(
        circuit, params,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=STOKES_N_LAYERS, verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps, "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("vqe_stokes", opt_name, seed, _drop_params(result))


def _run_vqe_heis_ring(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """B1: Frustrated 4q Heisenberg ring -- favors MomentumQNG.

    Antiferromagnetic Heisenberg on a periodic 4-site ring. Rugged-but-smooth
    landscape with multiple near-degenerate local minima -- bare QNG tends to
    plateau on this, momentum lets it coast through. Direct test of the
    Borysenko 2024 "momentum unsticks QNG" claim on a small frustrated system.
    """
    key = ("vqe_heis_ring", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else HEIS_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    H = make_heisenberg_ring_hamiltonian(HEIS_N_QUBITS, J=HEIS_J, periodic=True)
    circuit, _ = make_vqe_circuit(
        HEIS_N_QUBITS, HEIS_N_LAYERS, hamiltonian=H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(HEIS_N_QUBITS, HEIS_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_vqe(
        circuit, params,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=HEIS_N_LAYERS, verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps, "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("vqe_heis_ring", opt_name, seed, _drop_params(result))


def _run_fit_multifreq1d(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """C1: Multi-frequency 1D regression -- favors QNG-Adam hybrids.

    y = sin x + 0.4 sin 5x + 0.2 sin 13x, fit on 30 points in [-pi, pi].
    Mixed gradient magnitudes across parameters (the high-frequency components
    push tiny gradients on some encoding params); Adam's per-parameter adaptive
    LR shines, while the coupled 2-qubit encoding still gives QNG geometry
    something useful to do.
    """
    key = ("fit_multifreq1d", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else REG_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    x_raw = np.linspace(-np.pi, np.pi, 30)
    y_raw = _multifreq1d_target(x_raw)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]

    circuit = make_regression_circuit_1d(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=REG_N_LAYERS, loss_type="mse", verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps, "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("fit_multifreq1d", opt_name, seed, _drop_params(result))


def _run_cls_moons_hard(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """D1: High-noise two-moons classification -- favors Adam.

    make_moons with noise=0.40 (vs 0.15 for the original `cls`). The classes
    overlap heavily, so the loss landscape is genuinely rugged and the
    Fubini-Study geometry no longer aligns with anything useful. Pure
    rugged-optimization regime where Adam's momentum dominates.
    """
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler

    key = ("cls_moons_hard", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else REG_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    X, y = make_moons(n_samples=CLS_HARD_N_DATA, noise=CLS_HARD_NOISE, random_state=42)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X = scaler.fit_transform(X)
    y = 2 * y - 1

    x_train = [pnp.array(xi, requires_grad=False) for xi in X]
    y_train = [pnp.array(float(yi), requires_grad=False) for yi in y]

    circuit = make_classification_circuit(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=REG_N_LAYERS, loss_type="hinge", verbose=False,
        progress_cb=_cb,
    )

    preds = [float(circuit(result["params"], x)) for x in x_train]
    result["final_accuracy"] = accuracy(preds, [float(yi) for yi in y])

    progress[key] = {"status": "done", "step": n_steps, "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("cls_moons_hard", opt_name, seed, _drop_params(result))


# ── Phase 1 (manifold_torus tier) workers ───────────────────────────────

def _run_vqe_stokes_overparam_long(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """Phase 1: massively-overparameterized Stokes VQE on a long horizon.

    Same H = Z_0 (x) Z_1 as `vqe_stokes`, but the circuit is 8 qubits x 8
    layers (~192 trainable params, 2 active wires) and the run is 300 steps.
    The point is to give flat optimizers enough room AND enough time for
    ||theta|| to drift large, so that R-QNG-torus's mod-2*pi retraction has
    something to push back against.
    """
    key = ("vqe_stokes_overparam_long", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else STOKES_LONG_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    H = make_stokes_hamiltonian(STOKES_LONG_N_QUBITS)
    circuit, _ = make_vqe_circuit(
        STOKES_LONG_N_QUBITS, STOKES_LONG_N_LAYERS, hamiltonian=H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(STOKES_LONG_N_QUBITS, STOKES_LONG_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_vqe(
        circuit, params,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=STOKES_LONG_N_LAYERS, verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps,
                     "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("vqe_stokes_overparam_long", opt_name, seed, _drop_params(result))


def _run_fit_high_periodic(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """Phase 1: fit y = sin(7x) on x in [-3*pi, 3*pi], 60 points, 200 steps.

    High-frequency single-tone target spanning several full rotations of x.
    Encoding gates wrap around 2*pi naturally; flat-Adam tends to push the
    encoding parameters into ranges where momentum / metric estimation
    becomes unstable, while R-QNG-torus stays bounded by construction.
    """
    key = ("fit_high_periodic", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else FIT_HIGH_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    x_raw = np.linspace(-FIT_HIGH_X_SPAN, FIT_HIGH_X_SPAN, FIT_HIGH_N_DATA)
    y_raw = _high_periodic_target(x_raw)
    x_train = [pnp.array(x, requires_grad=False) for x in x_raw]
    y_train = [pnp.array(y, requires_grad=False) for y in y_raw]

    circuit = make_regression_circuit_1d(
        REG_N_QUBITS, REG_N_LAYERS,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_regression(REG_N_QUBITS, REG_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=REG_N_LAYERS, loss_type="mse", verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps,
                     "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("fit_high_periodic", opt_name, seed, _drop_params(result))


def _run_vqe_sk_spinglass(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """Phase 1 (hard bucket): Sherrington-Kirkpatrick spin glass.

    H = sum_{i<j} J_{ij} Z_i Z_j, J_{ij} ~ U(-1,1), 6q x 4L. Notoriously
    rugged frustrated landscape -- a "does R-QNG-torus survive on a real
    physics problem?" check, not a benchmark it is rigged to win.

    Hardcoded to 1000 shots regardless of the CLI --shots flag because shot
    noise is part of the task definition (the plan stresses the hard bucket
    under hardware-realistic noise). To run the analytic version, edit
    SK_SHOTS at the top of this file.
    """
    key = ("vqe_sk_spinglass", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else SK_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    task_shots = SK_SHOTS
    H = make_sk_hamiltonian(SK_N_QUBITS, seed=SK_SEED)
    circuit, _ = make_vqe_circuit(
        SK_N_QUBITS, SK_N_LAYERS, hamiltonian=H,
        shots=task_shots, diff_method=_diff_method_for(opt_name, task_shots),
    )
    params = init_params_vqe(SK_N_QUBITS, SK_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_vqe(
        circuit, params,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=SK_N_LAYERS, verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps,
                     "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("vqe_sk_spinglass", opt_name, seed, _drop_params(result))


def _run_vqe_xxz(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """Phase 1 (hard bucket): isotropic XXZ chain (delta = 1).

    H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) on an 8q open chain,
    4 layers, 1000 shots, 150 steps. Smooth condensed-matter Hamiltonian
    that's actually hard at this size -- second "real-physics" check that
    sits next to vqe_sk_spinglass.

    Hardcoded to 1000 shots; see _run_vqe_sk_spinglass for rationale.
    """
    key = ("vqe_xxz", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else XXZ_N_STEPS
    progress[key] = {"status": "running", "step": 0, "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    task_shots = XXZ_SHOTS
    H = make_xxz_hamiltonian(XXZ_N_QUBITS, delta=XXZ_DELTA)
    circuit, _ = make_vqe_circuit(
        XXZ_N_QUBITS, XXZ_N_LAYERS, hamiltonian=H,
        shots=task_shots, diff_method=_diff_method_for(opt_name, task_shots),
    )
    params = init_params_vqe(XXZ_N_QUBITS, XXZ_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_vqe(
        circuit, params,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=n_steps,
        n_layers=XXZ_N_LAYERS, verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps,
                     "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("vqe_xxz", opt_name, seed, _drop_params(result))


# ── Phase 2 (manifold_sphere tier) workers ──────────────────────────────

def _run_vqe_overparam_heis(opt_name, lr, seed, progress, shots, n_steps_override=None):
    """Phase 2: overparameterized Heisenberg ring -- the natural sphere
    stress test.

    Same H as `vqe_heis_ring` (4q periodic antiferromagnet) but 8 layers
    instead of 4, so the parameter space is ~2x larger while the Hilbert
    space stays at 16 dim. This is the regime where the sphere constraint
    is most informative: lots of redundant directions in parameter space,
    so forcing ||theta||=1 prunes capacity that the flat optimizers were
    burning. Analytic, 150 steps.
    """
    key = ("vqe_overparam_heis", opt_name, seed)
    t0 = time.perf_counter()
    n_steps = n_steps_override if n_steps_override is not None else HEIS_OVERPARAM_N_STEPS
    progress[key] = {"status": "running", "step": 0,
                     "total_steps": n_steps,
                     "loss": None, "elapsed": 0.0}

    H = make_heisenberg_ring_hamiltonian(
        HEIS_OVERPARAM_N_QUBITS, J=HEIS_OVERPARAM_J, periodic=True,
    )
    circuit, _ = make_vqe_circuit(
        HEIS_OVERPARAM_N_QUBITS, HEIS_OVERPARAM_N_LAYERS, hamiltonian=H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(HEIS_OVERPARAM_N_QUBITS, HEIS_OVERPARAM_N_LAYERS, seed)

    def _cb(step, n_steps_cb, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps_cb,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_vqe(
        circuit, params,
        opt_name=_canonical_opt(opt_name), lr=lr,
        n_steps=n_steps,
        n_layers=HEIS_OVERPARAM_N_LAYERS, verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": n_steps,
                     "total_steps": n_steps,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("vqe_overparam_heis", opt_name, seed, _drop_params(result))


# ── Heartbeat helper ─────────────────────────────────────────────────────

def _fmt_elapsed(seconds):
    """Format seconds into a compact m:ss or h:mm:ss string."""
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _heartbeat_loop(progress, jobs, total_jobs, t_start, stop_event,
                    n_workers, active_opts, interval=20, full_every=60,
                    bench=None, shots=None, steps_override=None):
    """Print a compact one-liner every `interval` seconds and a full breakdown
    every `full_every` seconds.

    The overall ETA is computed by simulating LPT placement of all remaining
    work (running + pending) onto `n_workers` workers. Each remaining job is
    assigned a predicted wall time from this priority chain:
      1. Median of completed (task, opt) wall times this run.
      2. Median of completed wall times for the same task (any opt).
      3. The bench_smoke estimate for (task, opt, mode) -- only if `bench`
         was supplied. This is the "fresh start" case where nothing has
         finished yet, so the on-the-fly history is empty.
      4. None -- the LPT simulator treats unknowns as zero load.
    For currently-running jobs that have logged at least one step, we use
    elapsed/step * (total_steps - step) instead of the predictions above.

    Runs as a daemon thread in the main process.
    """
    # Canonical (task, opt, seed) triples for everything that must run.
    all_keys = [(task, opt_name, seed)
                for (_fn, task, opt_name, _lr, seed) in jobs]

    # First tick prints the full breakdown so the operator sees the structure
    # immediately; subsequent ticks throttle to once per `full_every` seconds.
    last_full = -float("inf")

    while not stop_event.wait(interval):
        elapsed = time.perf_counter() - t_start
        snap = dict(progress)

        # ── Counts ──────────────────────────────────────────────────────
        n_done = sum(1 for v in snap.values() if v["status"] == "done")
        n_fail = sum(1 for v in snap.values() if v["status"] == "failed")
        n_run  = sum(1 for v in snap.values() if v["status"] == "running")
        n_pend = total_jobs - len(snap)

        # ── Wall-time history per (task, opt) from done jobs ────────────
        # Reconstructed each tick from the snapshot so memory stays bounded
        # by total job count.
        wall_history = defaultdict(list)
        per_task_done = defaultdict(int)
        per_task_walls = defaultdict(list)
        per_opt_done = defaultdict(int)
        per_opt_walls = defaultdict(list)
        for (task, opt, _seed), v in snap.items():
            if v["status"] == "done":
                wall_history[(task, opt)].append(v["elapsed"])
                per_task_done[task] += 1
                per_task_walls[task].append(v["elapsed"])
                per_opt_done[opt] += 1
                per_opt_walls[opt].append(v["elapsed"])

        def _predicted_duration(task, opt):
            if (task, opt) in wall_history:
                return float(np.median(wall_history[(task, opt)]))
            same_task = [w for (t, _), ws in wall_history.items()
                         if t == task for w in ws]
            if same_task:
                return float(np.median(same_task))
            # Bench fallback: lookup_bench_full_run returns the predicted
            # full-run wall for this (task, opt, shots-mode) tuple, or None
            # if bench is missing or the entry isn't there. Returning None
            # tells _lpt_simulate to treat this load as 0. When the runner
            # passed --steps N, we rescale by N so the ETA matches the
            # shorter run rather than the saved production-step total.
            return lookup_bench_full_run(bench, task, opt, shots, steps_override)

        def _running_remaining(task, opt, info):
            step = info["step"]
            tot  = info["total_steps"]
            el   = info["elapsed"]
            if step > 0 and tot:
                return (el / step) * (tot - step)
            pred = _predicted_duration(task, opt)
            if pred is not None:
                return max(pred - el, 0.0)
            return None

        def _lpt_simulate(running_loads, pending_durations, W):
            # Workers with "unknown" load are treated as 0 (best case) so a
            # missing prediction doesn't pin the estimate.
            loads = [r if r is not None else 0.0 for r in running_loads]
            while len(loads) < W:
                loads.append(0.0)
            heapq.heapify(loads)
            for d in sorted([d for d in pending_durations if d is not None],
                            reverse=True):
                smallest = heapq.heappop(loads)
                heapq.heappush(loads, smallest + d)
            return max(loads)

        # ── Identify running and pending jobs ───────────────────────────
        running = [(k, v) for k, v in snap.items() if v["status"] == "running"]
        snap_keys = set(snap.keys())
        pending_keys = [k for k in all_keys if k not in snap_keys]

        # ── Compute overall ETA via LPT simulation ──────────────────────
        running_loads = [_running_remaining(t, o, v)
                         for (t, o, _), v in running]
        pending_durs  = [_predicted_duration(t, o)
                         for (t, o, _) in pending_keys]
        if (not any(r is not None for r in running_loads)
                and not any(d is not None for d in pending_durs)):
            eta_str = "n/a"
        else:
            eta = _lpt_simulate(running_loads, pending_durs, n_workers)
            eta_str = f"~{_fmt_elapsed(eta)}"

        compact = (
            f"  [{_fmt_elapsed(elapsed)}] "
            f"Done {n_done}/{total_jobs} | Running {n_run} | "
            f"Pending {n_pend} | Failed {n_fail} | ETA {eta_str}"
        )

        full_due = (elapsed - last_full) >= full_every

        print(f"\n{'':─<70}")
        print(compact)

        if full_due:
            last_full = elapsed

            # ── Per-task block ──────────────────────────────────────────
            per_task_total = defaultdict(int)
            for (task, _opt, _seed) in all_keys:
                per_task_total[task] += 1
            per_task_running = defaultdict(int)
            per_task_pending = defaultdict(int)
            per_task_runinfo = defaultdict(list)
            for (task, opt, _seed), info in running:
                per_task_running[task] += 1
                per_task_runinfo[task].append((task, opt, info))
            for (task, _opt, _seed) in pending_keys:
                per_task_pending[task] += 1

            # Fair-share workers across tasks that still have work left.
            n_active_tasks = sum(
                1 for tk in per_task_total
                if per_task_done.get(tk, 0) < per_task_total[tk]
            )
            share_w = max(1, n_workers // max(1, n_active_tasks))

            print("\n  Per-task progress:")
            for task in TASK_LABELS:
                if per_task_total.get(task, 0) == 0:
                    continue
                label = TASK_LABELS[task]
                done_n = per_task_done.get(task, 0)
                tot_n  = per_task_total[task]
                run_n  = per_task_running.get(task, 0)
                pend_n = per_task_pending.get(task, 0)
                if done_n == tot_n:
                    avg = float(np.mean(per_task_walls[task])) if per_task_walls[task] else 0.0
                    print(
                        f"    {label:<18s}{done_n:>3d}/{tot_n:<3d} done"
                        f"  (avg {_fmt_elapsed(avg)}/job)"
                    )
                else:
                    task_run_loads = [
                        _running_remaining(t, o, info)
                        for (t, o, info) in per_task_runinfo[task]
                    ]
                    task_pend_durs = [
                        _predicted_duration(task, o)
                        for (t, o, _) in pending_keys if t == task
                    ]
                    if (any(r is not None for r in task_run_loads)
                            or any(d is not None for d in task_pend_durs)):
                        rem = _lpt_simulate(task_run_loads, task_pend_durs, share_w)
                        rem_s = _fmt_elapsed(rem)
                    else:
                        rem_s = "n/a"
                    print(
                        f"    {label:<18s}{done_n:>3d}/{tot_n:<3d} done  | "
                        f"running {run_n}, pending {pend_n}"
                        f"  (est rem {rem_s})"
                    )

            # ── Per-optimizer block ─────────────────────────────────────
            per_opt_total = defaultdict(int)
            for (_task, opt, _seed) in all_keys:
                per_opt_total[opt] += 1
            print("\n  Per-optimizer (mean job wall time across done jobs):")
            for opt in active_opts:
                if per_opt_total.get(opt, 0) == 0:
                    continue
                done_n = per_opt_done.get(opt, 0)
                tot_n  = per_opt_total[opt]
                if done_n > 0:
                    mean_s = _fmt_elapsed(float(np.mean(per_opt_walls[opt])))
                else:
                    mean_s = "n/a"
                print(
                    f"    {opt:<20s} mean={mean_s:<8s} done {done_n}/{tot_n}"
                )

            # ── Top-3 in-flight by predicted total wall ─────────────────
            if running:
                def _bottleneck_key(kv):
                    (task, opt, _seed), info = kv
                    pred = _predicted_duration(task, opt)
                    pred_v = pred if pred is not None else 0.0
                    return (-pred_v, -info["elapsed"])

                running.sort(key=_bottleneck_key)
                print("\n  Highest-impact in-flight (sorted by predicted total wall):")
                for (task, opt, seed), info in running[:3]:
                    step = info["step"]
                    tot  = info["total_steps"]
                    loss_s = f"{info['loss']:.4f}" if info["loss"] is not None else "..."
                    el_s = _fmt_elapsed(info["elapsed"])
                    pred = _predicted_duration(task, opt)
                    pred_s = _fmt_elapsed(pred) if pred is not None else "n/a"
                    label = TASK_LABELS.get(task, task)
                    print(
                        f"    {label} / {opt} / seed={seed}"
                        f"  step {step:>3d}/{tot}  loss={loss_s}"
                        f"  elapsed={el_s}  pred={pred_s}"
                    )

        print(f"{'':─<70}", flush=True)
        sys.stdout.flush()


# ── Main orchestration ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all QNG baseline experiments in parallel",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel worker processes. Default: auto-detected "
             "from SLURM_CPUS_PER_TASK / cgroup affinity / os.cpu_count() in "
             "that order. Pass an explicit value to override.",
    )
    parser.add_argument(
        "--shots", type=int, default=0,
        help="Measurement shots per circuit (0 = analytic noiseless). "
             "When >0, forces diff_method=parameter-shift (adjoint is incompatible with shots).",
    )
    parser.add_argument(
        "--tasks", default="sanity,qng_advantage",
        help="Comma-separated tokens, where each token is either a TASK_TIERS "
             "key (expands to its task list) or an individual task name. Mixing "
             "is allowed. e.g. 'sanity', 'sanity,qng_advantage', "
             "'vqe_stokes_overparam_long', 'manifold_torus,vqe_overparam_heis'. "
             "Default runs both currently-defined Tier 0/1 sanity tiers.",
    )
    parser.add_argument(
        "--optimizers", default="qng_full_sweep",
        help="Comma-separated tokens, where each token is either an OPT_TIERS "
             "key or an individual optimizer name from OPTIMIZERS. Mixing is "
             "allowed. e.g. 'flat_only', 'torus_concept', 'Adam,QNG_block', "
             "'torus_concept,QNG_full'. Default runs the QNG sweep w/o QNG_full.",
    )
    parser.add_argument(
        "--seeds", default=None,
        help=f"Comma-separated seed list (e.g. '0' or '0,1,2'). "
             f"Default: {DEFAULT_SEEDS}.",
    )
    parser.add_argument(
        "--title", default=None,
        help="Optional human-readable purpose label for this run. The slugified "
             "form is appended to the run-folder name (e.g. "
             "results/2026-04-25_204500__rqng_torus_diag), and the raw title "
             "plus full CLI is preserved in <run>/run_meta.json.",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Global step-count override. If set, every task uses this number "
             "of optimization steps instead of its production constant "
             "(REG_N_STEPS, VQE_N_STEPS, etc.). Useful for quick all-task "
             "smoke runs (e.g. --steps 10). Bench-derived ETAs in the "
             "heartbeat are rescaled accordingly.",
    )
    args = parser.parse_args()

    if args.steps is not None and args.steps < 1:
        raise SystemExit(f"--steps must be >= 1 (got {args.steps})")

    all_individual_tasks = {t for tasks in TASK_TIERS.values() for t in tasks}
    all_individual_opts = set(OPTIMIZERS.keys())
    active_task_tiers, active_tasks = _resolve_names(
        args.tasks, TASK_TIERS, all_individual_tasks, "task",
    )
    active_opt_tiers, active_opts = _resolve_names(
        args.optimizers, OPT_TIERS, all_individual_opts, "optimizer",
    )
    active_seeds = _parse_seeds(args.seeds)

    results_dir, plots_dir = make_run_dir(RESULTS_BASE, title=args.title)
    n_workers = args.workers if args.workers is not None else _detect_available_cpus()
    shots = args.shots if args.shots > 0 else None

    # Persist the human intent + reproducibility metadata next to the
    # results JSONs. Anything saved here is what we read months later when
    # the timestamp alone has stopped being meaningful.
    run_meta = {
        "title": args.title,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "cli": " ".join(sys.argv),
        "workers": n_workers,
        "workers_source": "explicit --workers" if args.workers is not None else "auto-detected",
        "shots": shots,
        "tasks_arg": args.tasks,
        "active_task_tiers": active_task_tiers,
        "active_tasks": active_tasks,
        "optimizers_arg": args.optimizers,
        "active_opt_tiers": active_opt_tiers,
        "active_opts": active_opts,
        "seeds": active_seeds,
        "steps_override": args.steps,
    }
    with open(os.path.join(results_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)
    # GD/Adam: adjoint (analytic) or parameter-shift (shots).
    # QNG_block/QNG_full: always parameter-shift, since qml.metric_tensor
    # tapes aren't compatible with lightning.qubit's adjoint implementation.
    diff_method_gd_adam = "parameter-shift" if shots else "adjoint"
    diff_method_qng = "parameter-shift"

    # Registry of every available task. Tier filtering picks the active subset.
    all_worker_specs = [
        (_run_fit1d,           "fit1d"),
        (_run_fit2d,           "fit2d"),
        (_run_vqe,             "vqe"),
        (_run_cls,             "cls"),
        (_run_vqe_stokes,      "vqe_stokes"),
        (_run_vqe_heis_ring,   "vqe_heis_ring"),
        (_run_fit_multifreq1d, "fit_multifreq1d"),
        (_run_cls_moons_hard,  "cls_moons_hard"),
        # Phase 1 (manifold_torus tier)
        (_run_vqe_stokes_overparam_long, "vqe_stokes_overparam_long"),
        (_run_fit_high_periodic,         "fit_high_periodic"),
        (_run_vqe_sk_spinglass,          "vqe_sk_spinglass"),
        (_run_vqe_xxz,                   "vqe_xxz"),
        # Phase 2 (manifold_sphere tier)
        (_run_vqe_overparam_heis,        "vqe_overparam_heis"),
    ]
    worker_specs = [(fn, task) for fn, task in all_worker_specs if task in active_tasks]
    jobs = []
    for seed in active_seeds:
        for fn, task in worker_specs:
            for opt_name in active_opts:
                cfg = OPTIMIZERS[opt_name]
                jobs.append((fn, task, opt_name, cfg["lr"], seed))

    # LPT heuristic: longest-running jobs first so workers grab them immediately
    # and shorter jobs naturally pack into the gaps as workers free up.
    jobs.sort(key=lambda j: -job_priority(j[1], j[2]))

    total = len(jobs)
    if shots is None:
        shots_banner = "None (analytic); GD/Adam=adjoint, QNG=parameter-shift"
    else:
        shots_banner = f"{shots} (hardware-realistic, parameter-shift)"
    print("=" * 70)
    print("  QNG EUCLIDEAN BASELINE -- Parallel runner")
    if args.title:
        print(f"  Title   : {args.title}")
    print(f"  Workers : {n_workers}  ({run_meta['workers_source']})")
    print(f"  Jobs    : {total}")
    print(f"  Tasks   : {active_task_tiers or '(individuals only)'} -> {', '.join(active_tasks)}")
    print(f"  Opts    : {active_opt_tiers or '(individuals only)'}  -> {', '.join(active_opts)}")
    print(f"  Seeds   : {active_seeds}")
    if args.steps is not None:
        print(f"  Steps   : {args.steps}  (global override)")
    print(f"  Shots   : {shots_banner}")
    print(f"  VQE     : {VQE_N_QUBITS} qubits, {VQE_N_LAYERS} layers")
    bench_estimates = load_bench_estimates()
    bench_status = (
        f"loaded ({len(bench_estimates.get('estimates', {}))} tasks)"
        if bench_estimates else "missing -- ETA will start as 'n/a'"
    )
    print(f"  Bench   : {bench_status}")
    print("=" * 70)
    sys.stdout.flush()

    manager = multiprocessing.Manager()
    progress = manager.dict()

    t_start = time.perf_counter()
    results = []
    n_failed = 0

    stop_heartbeat = threading.Event()
    hb_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(progress, jobs, total, t_start, stop_heartbeat,
              n_workers, active_opts),
        kwargs={"bench": bench_estimates, "shots": shots,
                "steps_override": args.steps},
        daemon=True,
    )
    hb_thread.start()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {}
        for fn, _task, opt_name, lr, seed in jobs:
            fut = pool.submit(fn, opt_name, lr, seed, progress, shots, args.steps)
            future_map[fut] = (fn.__name__, opt_name, seed)

        done_count = 0
        for fut in as_completed(future_map):
            fn_name, opt_name, seed = future_map[fut]
            done_count += 1
            try:
                task, opt_name, seed, result = fut.result()
                results.append((task, opt_name, seed, result))
                elapsed = time.perf_counter() - t_start
                final_loss = result["losses"][-1] if result.get("losses") else "?"
                wall = result["wall_times"][-1] if result.get("wall_times") else 0
                print(
                    f"  [{done_count:3d}/{total}] DONE  "
                    f"{TASK_LABELS[task]:>15s} / {opt_name:<10s} / seed={seed}"
                    f"  loss={final_loss:<10.6f}  job={_fmt_elapsed(wall)}"
                    f"  @{_fmt_elapsed(elapsed)}",
                    flush=True,
                )
            except Exception as exc:
                n_failed += 1
                elapsed = time.perf_counter() - t_start
                print(
                    f"  [{done_count:3d}/{total}] FAIL  "
                    f"{fn_name} / {opt_name} / seed={seed}"
                    f"  error={exc}  @{_fmt_elapsed(elapsed)}",
                    flush=True,
                )

    stop_heartbeat.set()
    hb_thread.join(timeout=2)

    total_time = time.perf_counter() - t_start

    # ── Per-experiment / per-optimizer timing breakdown ────────────────────
    print("\n" + "=" * 70)
    print("  TIMING BREAKDOWN")
    print("=" * 70)
    timing = defaultdict(lambda: defaultdict(list))
    for task, opt_name, seed, result in results:
        wall = result["wall_times"][-1] if result.get("wall_times") else 0
        timing[task][opt_name].append(wall)

    for task in TASK_LABELS:
        if task not in timing:
            continue
        label = TASK_LABELS[task]
        print(f"\n  {label}:")
        for opt in active_opts:
            if opt not in timing[task]:
                continue
            walls = timing[task][opt]
            mean_w = np.mean(walls)
            max_w = np.max(walls)
            print(f"    {opt:<10s}  mean={_fmt_elapsed(mean_w)}  max={_fmt_elapsed(max_w)}")

    print(f"\n  Total wall time : {_fmt_elapsed(total_time)}")
    print(f"  Succeeded       : {len(results)}/{total}")
    if n_failed:
        print(f"  Failed          : {n_failed}/{total}")
    print("=" * 70)

    # ── Group results by (task, optimizer, seed) ─────────────────────────
    grouped = defaultdict(lambda: defaultdict(dict))
    for task, opt_name, seed, result in results:
        grouped[task][opt_name][seed] = result

    # ── 1D Function Fitting ──────────────────────────────────────────────
    if "fit1d" in grouped:
        print("\nSaving 1D function fitting results...")
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["fit1d"].items()}
        save_results(
            {"config": {"n_qubits": REG_N_QUBITS, "n_layers": REG_N_LAYERS,
                         "n_steps": REG_N_STEPS,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "function_fitting_1d.json"),
        )
        convergence_plot(agg, title="1D Regression: sin(x)", log_y=True,
                         save_path=task_plot_path(plots_dir, "function_fitting_1d", "convergence"))
        resource_plot(agg, title="1D Regression: sin(x) (resource-normalised)", log_y=True,
                      save_path=task_plot_path(plots_dir, "function_fitting_1d", "resource"))
        final_loss_bar(agg, title="1D Regression: final MSE",
                       save_path=task_plot_path(plots_dir, "function_fitting_1d", "final"))
        theta_trajectory_plot(agg, title="1D Regression: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "function_fitting_1d", "theta"))

    # ── 2D Function Fitting ──────────────────────────────────────────────
    if "fit2d" in grouped:
        print("\nSaving 2D function fitting results...")
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["fit2d"].items()}
        save_results(
            {"config": {"n_qubits": REG_N_QUBITS, "n_layers": REG_N_LAYERS,
                         "n_steps": REG_N_STEPS,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "function_fitting_2d.json"),
        )
        convergence_plot(agg, title="2D Regression: (x1\u00b2+x2\u00b2)/2", log_y=True,
                         save_path=task_plot_path(plots_dir, "function_fitting_2d", "convergence"))
        resource_plot(agg, title="2D Regression (resource-normalised)", log_y=True,
                      save_path=task_plot_path(plots_dir, "function_fitting_2d", "resource"))
        final_loss_bar(agg, title="2D Regression: final MSE",
                       save_path=task_plot_path(plots_dir, "function_fitting_2d", "final"))
        theta_trajectory_plot(agg, title="2D Regression: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "function_fitting_2d", "theta"))

    # ── VQE ──────────────────────────────────────────────────────────────
    if "vqe" in grouped:
        print("\nSaving VQE results...")
        E_exact = exact_ground_energy(VQE_N_QUBITS, VQE_J, VQE_H)
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["vqe"].items()}
        save_results(
            {"config": {"n_qubits": VQE_N_QUBITS, "n_layers": VQE_N_LAYERS,
                         "n_steps": VQE_N_STEPS,
                         "J": VQE_J, "h": VQE_H, "E_exact": E_exact,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "vqe.json"),
        )
        convergence_plot(agg, title=f"VQE Ising {VQE_N_QUBITS}q  (E*={E_exact:.3f})",
                         ylabel="Energy \u27e8H\u27e9",
                         save_path=task_plot_path(plots_dir, "vqe", "convergence"))
        resource_plot(agg, title="VQE (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=task_plot_path(plots_dir, "vqe", "resource"))
        final_loss_bar(agg, title="VQE: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=task_plot_path(plots_dir, "vqe", "final"))
        theta_trajectory_plot(agg, title=f"VQE Ising {VQE_N_QUBITS}q: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "vqe", "theta"))

    # ── Classification ───────────────────────────────────────────────────
    if "cls" in grouped:
        print("\nSaving classification results...")
        agg = {}
        for opt, seed_dict in grouped["cls"].items():
            agg_opt = aggregate_seeds(seed_dict)
            accs = [seed_dict[s]["final_accuracy"] for s in active_seeds if s in seed_dict]
            agg_opt["final_acc_mean"] = float(np.mean(accs))
            agg_opt["final_acc_std"]  = float(np.std(accs))
            agg[opt] = agg_opt
        save_results(
            {"config": {"n_qubits": REG_N_QUBITS, "n_layers": REG_N_LAYERS,
                         "n_steps": REG_N_STEPS, "n_data": CLS_N_DATA,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "classification.json"),
        )
        convergence_plot(agg, title="Classification (make_moons)", log_y=True,
                         save_path=task_plot_path(plots_dir, "classification", "convergence"))
        resource_plot(agg, title="Classification (resource-normalised)", log_y=True,
                      save_path=task_plot_path(plots_dir, "classification", "resource"))
        final_loss_bar(agg, title="Classification: final hinge loss",
                       save_path=task_plot_path(plots_dir, "classification", "final"))
        theta_trajectory_plot(agg, title="Classification: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "classification", "theta"))

    # ── Tier 1: VQE Stokes ───────────────────────────────────────────────
    if "vqe_stokes" in grouped:
        print("\nSaving Tier 1 vqe_stokes results...")
        H = make_stokes_hamiltonian(STOKES_N_QUBITS)
        E_exact = exact_ground_energy_from_h(H)
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["vqe_stokes"].items()}
        save_results(
            {"config": {"n_qubits": STOKES_N_QUBITS, "n_layers": STOKES_N_LAYERS,
                         "n_steps": STOKES_N_STEPS,
                         "hamiltonian": "Z_0 (x) Z_1", "E_exact": E_exact,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "vqe_stokes.json"),
        )
        convergence_plot(agg, title=f"VQE Stokes Z0Z1 ({STOKES_N_QUBITS}q x {STOKES_N_LAYERS}L)  (E*={E_exact:.3f})",
                         ylabel="Energy \u27e8H\u27e9",
                         save_path=task_plot_path(plots_dir, "vqe_stokes", "convergence"))
        resource_plot(agg, title="VQE Stokes (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=task_plot_path(plots_dir, "vqe_stokes", "resource"))
        final_loss_bar(agg, title="VQE Stokes: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=task_plot_path(plots_dir, "vqe_stokes", "final"))
        theta_trajectory_plot(agg, title="VQE Stokes: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "vqe_stokes", "theta"))

    # ── Tier 1: VQE Heisenberg ring ──────────────────────────────────────
    if "vqe_heis_ring" in grouped:
        print("\nSaving Tier 1 vqe_heis_ring results...")
        H = make_heisenberg_ring_hamiltonian(HEIS_N_QUBITS, J=HEIS_J, periodic=True)
        E_exact = exact_ground_energy_from_h(H)
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["vqe_heis_ring"].items()}
        save_results(
            {"config": {"n_qubits": HEIS_N_QUBITS, "n_layers": HEIS_N_LAYERS,
                         "n_steps": HEIS_N_STEPS, "J": HEIS_J,
                         "hamiltonian": "Heisenberg ring (periodic)", "E_exact": E_exact,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "vqe_heis_ring.json"),
        )
        convergence_plot(agg, title=f"VQE Heisenberg ring ({HEIS_N_QUBITS}q periodic)  (E*={E_exact:.3f})",
                         ylabel="Energy \u27e8H\u27e9",
                         save_path=task_plot_path(plots_dir, "vqe_heis_ring", "convergence"))
        resource_plot(agg, title="VQE Heisenberg ring (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=task_plot_path(plots_dir, "vqe_heis_ring", "resource"))
        final_loss_bar(agg, title="VQE Heisenberg ring: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=task_plot_path(plots_dir, "vqe_heis_ring", "final"))
        theta_trajectory_plot(agg, title="VQE Heisenberg ring: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "vqe_heis_ring", "theta"))

    # ── Tier 1: multi-frequency 1D regression ────────────────────────────
    if "fit_multifreq1d" in grouped:
        print("\nSaving Tier 1 fit_multifreq1d results...")
        agg = {opt: aggregate_seeds(seeds) for opt, seeds in grouped["fit_multifreq1d"].items()}
        save_results(
            {"config": {"n_qubits": REG_N_QUBITS, "n_layers": REG_N_LAYERS,
                         "n_steps": REG_N_STEPS,
                         "target": "sin(x) + 0.4 sin(5x) + 0.2 sin(13x)",
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "fit_multifreq1d.json"),
        )
        convergence_plot(agg, title="1D Multi-frequency Regression: sin x + 0.4 sin 5x + 0.2 sin 13x",
                         log_y=True,
                         save_path=task_plot_path(plots_dir, "fit_multifreq1d", "convergence"))
        resource_plot(agg, title="1D Multi-frequency (resource-normalised)", log_y=True,
                      save_path=task_plot_path(plots_dir, "fit_multifreq1d", "resource"))
        final_loss_bar(agg, title="1D Multi-frequency: final MSE",
                       save_path=task_plot_path(plots_dir, "fit_multifreq1d", "final"))
        theta_trajectory_plot(agg, title="1D Multi-frequency: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "fit_multifreq1d", "theta"))

    # ── Tier 1: hard moons classification ────────────────────────────────
    if "cls_moons_hard" in grouped:
        print("\nSaving Tier 1 cls_moons_hard results...")
        agg = {}
        for opt, seed_dict in grouped["cls_moons_hard"].items():
            agg_opt = aggregate_seeds(seed_dict)
            accs = [seed_dict[s]["final_accuracy"] for s in active_seeds if s in seed_dict]
            agg_opt["final_acc_mean"] = float(np.mean(accs))
            agg_opt["final_acc_std"]  = float(np.std(accs))
            agg[opt] = agg_opt
        save_results(
            {"config": {"n_qubits": REG_N_QUBITS, "n_layers": REG_N_LAYERS,
                         "n_steps": REG_N_STEPS, "n_data": CLS_HARD_N_DATA,
                         "noise": CLS_HARD_NOISE,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "cls_moons_hard.json"),
        )
        convergence_plot(agg, title=f"Hard Classification (make_moons noise={CLS_HARD_NOISE})",
                         log_y=True,
                         save_path=task_plot_path(plots_dir, "cls_moons_hard", "convergence"))
        resource_plot(agg, title="Hard Classification (resource-normalised)", log_y=True,
                      save_path=task_plot_path(plots_dir, "cls_moons_hard", "resource"))
        final_loss_bar(agg, title="Hard Classification: final hinge loss",
                       save_path=task_plot_path(plots_dir, "cls_moons_hard", "final"))
        theta_trajectory_plot(agg, title="Hard Classification: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "cls_moons_hard", "theta"))

    # ── Phase 1: VQE Stokes overparameterized (long horizon) ─────────────
    if "vqe_stokes_overparam_long" in grouped:
        print("\nSaving Phase-1 vqe_stokes_overparam_long results...")
        H = make_stokes_hamiltonian(STOKES_LONG_N_QUBITS)
        E_exact = exact_ground_energy_from_h(H)
        agg = {opt: aggregate_seeds(seeds)
               for opt, seeds in grouped["vqe_stokes_overparam_long"].items()}
        save_results(
            {"config": {"n_qubits": STOKES_LONG_N_QUBITS, "n_layers": STOKES_LONG_N_LAYERS,
                         "n_steps": STOKES_LONG_N_STEPS,
                         "hamiltonian": "Z_0 (x) Z_1 (long-horizon, overparam)",
                         "E_exact": E_exact,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "vqe_stokes_overparam_long.json"),
        )
        convergence_plot(agg,
                         title=f"VQE Stokes-overparam ({STOKES_LONG_N_QUBITS}q x {STOKES_LONG_N_LAYERS}L)  (E*={E_exact:.3f})",
                         ylabel="Energy \u27e8H\u27e9",
                         save_path=task_plot_path(plots_dir, "vqe_stokes_overparam_long", "convergence"))
        resource_plot(agg, title="VQE Stokes-overparam (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=task_plot_path(plots_dir, "vqe_stokes_overparam_long", "resource"))
        final_loss_bar(agg, title="VQE Stokes-overparam: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=task_plot_path(plots_dir, "vqe_stokes_overparam_long", "final"))
        theta_trajectory_plot(
            agg, title="VQE Stokes-overparam: ||theta|| trajectory",
            save_path=task_plot_path(plots_dir, "vqe_stokes_overparam_long", "theta"),
        )

    # ── Phase 1: high-frequency 1D regression ────────────────────────────
    if "fit_high_periodic" in grouped:
        print("\nSaving Phase-1 fit_high_periodic results...")
        agg = {opt: aggregate_seeds(seeds)
               for opt, seeds in grouped["fit_high_periodic"].items()}
        save_results(
            {"config": {"n_qubits": REG_N_QUBITS, "n_layers": REG_N_LAYERS,
                         "n_steps": FIT_HIGH_N_STEPS,
                         "n_data": FIT_HIGH_N_DATA,
                         "x_span": float(FIT_HIGH_X_SPAN),
                         "target": "sin(7x) on [-3*pi, 3*pi]",
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "fit_high_periodic.json"),
        )
        convergence_plot(agg,
                         title="1D High-Periodic Regression: sin(7x) on [-3pi, 3pi]",
                         log_y=True,
                         save_path=task_plot_path(plots_dir, "fit_high_periodic", "convergence"))
        resource_plot(agg, title="1D High-Periodic (resource-normalised)", log_y=True,
                      save_path=task_plot_path(plots_dir, "fit_high_periodic", "resource"))
        final_loss_bar(agg, title="1D High-Periodic: final MSE",
                       save_path=task_plot_path(plots_dir, "fit_high_periodic", "final"))
        theta_trajectory_plot(agg, title="1D High-Periodic: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "fit_high_periodic", "theta"))

    # ── Phase 1: SK spin-glass VQE ───────────────────────────────────────
    if "vqe_sk_spinglass" in grouped:
        print("\nSaving Phase-1 vqe_sk_spinglass results...")
        H = make_sk_hamiltonian(SK_N_QUBITS, seed=SK_SEED)
        E_exact = exact_ground_energy_from_h(H)
        agg = {opt: aggregate_seeds(seeds)
               for opt, seeds in grouped["vqe_sk_spinglass"].items()}
        save_results(
            {"config": {"n_qubits": SK_N_QUBITS, "n_layers": SK_N_LAYERS,
                         "n_steps": SK_N_STEPS,
                         "hamiltonian_seed": SK_SEED,
                         "hamiltonian": "SK spin-glass: sum_{i<j} J_ij Z_i Z_j, J~U(-1,1)",
                         "E_exact": E_exact,
                         "shots": SK_SHOTS,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "vqe_sk_spinglass.json"),
        )
        convergence_plot(agg,
                         title=f"VQE SK spin-glass ({SK_N_QUBITS}q x {SK_N_LAYERS}L, {SK_SHOTS} shots)  (E*={E_exact:.3f})",
                         ylabel="Energy \u27e8H\u27e9",
                         save_path=task_plot_path(plots_dir, "vqe_sk_spinglass", "convergence"))
        resource_plot(agg, title="VQE SK spin-glass (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=task_plot_path(plots_dir, "vqe_sk_spinglass", "resource"))
        final_loss_bar(agg, title="VQE SK spin-glass: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=task_plot_path(plots_dir, "vqe_sk_spinglass", "final"))
        theta_trajectory_plot(agg, title="VQE SK spin-glass: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "vqe_sk_spinglass", "theta"))

    # ── Phase 1: XXZ chain VQE ───────────────────────────────────────────
    if "vqe_xxz" in grouped:
        print("\nSaving Phase-1 vqe_xxz results...")
        H = make_xxz_hamiltonian(XXZ_N_QUBITS, delta=XXZ_DELTA)
        E_exact = exact_ground_energy_from_h(H)
        agg = {opt: aggregate_seeds(seeds)
               for opt, seeds in grouped["vqe_xxz"].items()}
        save_results(
            {"config": {"n_qubits": XXZ_N_QUBITS, "n_layers": XXZ_N_LAYERS,
                         "n_steps": XXZ_N_STEPS,
                         "delta": XXZ_DELTA,
                         "hamiltonian": "XXZ chain (open boundary)",
                         "E_exact": E_exact,
                         "shots": XXZ_SHOTS,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "vqe_xxz.json"),
        )
        convergence_plot(agg,
                         title=f"VQE XXZ chain ({XXZ_N_QUBITS}q x {XXZ_N_LAYERS}L, delta={XXZ_DELTA}, {XXZ_SHOTS} shots)  (E*={E_exact:.3f})",
                         ylabel="Energy \u27e8H\u27e9",
                         save_path=task_plot_path(plots_dir, "vqe_xxz", "convergence"))
        resource_plot(agg, title="VQE XXZ chain (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=task_plot_path(plots_dir, "vqe_xxz", "resource"))
        final_loss_bar(agg, title="VQE XXZ chain: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=task_plot_path(plots_dir, "vqe_xxz", "final"))
        theta_trajectory_plot(agg, title="VQE XXZ chain: ||theta|| trajectory",
                              save_path=task_plot_path(plots_dir, "vqe_xxz", "theta"))

    # ── Phase 2: overparameterized Heisenberg ring ───────────────────────
    if "vqe_overparam_heis" in grouped:
        print("\nSaving Phase-2 vqe_overparam_heis results...")
        H = make_heisenberg_ring_hamiltonian(
            HEIS_OVERPARAM_N_QUBITS, J=HEIS_OVERPARAM_J, periodic=True,
        )
        E_exact = exact_ground_energy_from_h(H)
        agg = {opt: aggregate_seeds(seeds)
               for opt, seeds in grouped["vqe_overparam_heis"].items()}
        save_results(
            {"config": {"n_qubits": HEIS_OVERPARAM_N_QUBITS,
                         "n_layers": HEIS_OVERPARAM_N_LAYERS,
                         "n_steps": HEIS_OVERPARAM_N_STEPS,
                         "J": HEIS_OVERPARAM_J,
                         "hamiltonian": "Heisenberg ring (periodic, overparam)",
                         "E_exact": E_exact,
                         "shots": shots,
                         "diff_method_gd_adam": diff_method_gd_adam,
                         "diff_method_qng": diff_method_qng},
             **agg},
            os.path.join(results_dir, "vqe_overparam_heis.json"),
        )
        convergence_plot(agg,
                         title=f"VQE Heisenberg-overparam ({HEIS_OVERPARAM_N_QUBITS}q x {HEIS_OVERPARAM_N_LAYERS}L)  (E*={E_exact:.3f})",
                         ylabel="Energy \u27e8H\u27e9",
                         save_path=task_plot_path(plots_dir, "vqe_overparam_heis", "convergence"))
        resource_plot(agg, title="VQE Heisenberg-overparam (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=task_plot_path(plots_dir, "vqe_overparam_heis", "resource"))
        final_loss_bar(agg, title="VQE Heisenberg-overparam: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=task_plot_path(plots_dir, "vqe_overparam_heis", "final"))
        theta_trajectory_plot(
            agg, title="VQE Heisenberg-overparam: ||theta|| trajectory",
            save_path=task_plot_path(plots_dir, "vqe_overparam_heis", "theta"),
        )

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Total wall time: {_fmt_elapsed(total_time)}")
    print(f"  Results saved in {results_dir}")
    print(f"  Plots  saved in {plots_dir}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
