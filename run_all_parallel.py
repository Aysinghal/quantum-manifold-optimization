"""
Parallel runner: executes all 80 experiment jobs across multiple CPU cores.

Each (experiment, optimizer, seed) combination runs as an independent worker
process. Results are collected, aggregated, and saved/plotted identically to
the sequential run_all.py.

Usage:
    python run_all_parallel.py               # use all cores
    python run_all_parallel.py --workers 16  # use 16 cores
"""

import argparse
import multiprocessing
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")

import numpy as np
from pennylane import numpy as pnp

from src.models import (
    make_regression_circuit_1d,
    make_regression_circuit_2d,
    make_vqe_circuit,
    make_classification_circuit,
    init_params_regression,
    init_params_vqe,
    exact_ground_energy,
)
from src.training import train_with_data, train_vqe
from src.metrics import aggregate_seeds, save_results, accuracy, make_run_dir
from src.visualization import convergence_plot, resource_plot, final_loss_bar


# ── Shared config (mirrors the individual experiment modules) ────────────────

SEEDS = [0, 1, 2, 3, 4]

OPTIMIZERS = {
    "GD":              {"lr": 0.1},
    "Adam":            {"lr": 0.05},
    "QNG_block":       {"lr": 0.05},
    "QNG_block_lr01":  {"lr": 0.1},
    "QNG_block_lr02":  {"lr": 0.2},
    "QNG_full":        {"lr": 0.01},
}

# LPT scheduling priorities. Higher = scheduled earlier so the longest-running
# jobs grab workers first and short jobs pack into the gaps. Numbers are
# relative; tweak by adding/replacing entries when introducing a new optimizer
# or task in the future.
OPT_PRIORITY = {
    "QNG_full":        100,
    "QNG_block":        50,
    "QNG_block_lr01":   50,
    "QNG_block_lr02":   50,
    "Adam":             10,
    "GD":               10,
}
TASK_PRIORITY = {
    "vqe":   10,
    "fit2d":  3,
    "cls":    3,
    "fit1d":  1,
}


def _canonical_opt(opt_name):
    """Map optimizer-variant names back to the base optimizer the train_*
    functions know how to dispatch on (e.g. 'QNG_block_lr01' -> 'QNG_block').
    """
    if opt_name.startswith("QNG_block"):
        return "QNG_block"
    if opt_name.startswith("QNG_full"):
        return "QNG_full"
    return opt_name


def job_priority(task, opt_name):
    """LPT score for a (task, optimizer) pair."""
    return OPT_PRIORITY[opt_name] * TASK_PRIORITY[task]

REG_N_QUBITS = 2
REG_N_LAYERS = 4
REG_N_STEPS  = 100

VQE_N_QUBITS = 11
VQE_N_LAYERS = 4
VQE_N_STEPS  = 150
VQE_J        = 1.0
VQE_H        = 1.0

CLS_N_DATA   = 100

RESULTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

TASK_LABELS = {
    "fit1d": "1D-sin",
    "fit2d": "2D-quad",
    "vqe":   "VQE",
    "cls":   "Classification",
}


# ── Worker functions (module-level for Windows spawn compatibility) ──────────

def _drop_params(result):
    """Drop the params tensor before returning to the main process."""
    return {k: v for k, v in result.items() if k != "params"}


def _diff_method_for(opt_name, shots):
    """QNG optimizers need parameter-shift regardless of shots, because
    qml.metric_tensor builds tapes that lightning.qubit's adjoint does
    not support. GD/Adam can use adjoint in analytic mode for a big speedup."""
    if opt_name.startswith("QNG"):
        return "parameter-shift"
    return None  # let _resolve_diff_method pick based on shots


def _run_fit1d(opt_name, lr, seed, progress, shots):
    key = ("fit1d", opt_name, seed)
    t0 = time.perf_counter()
    progress[key] = {"status": "running", "step": 0, "total_steps": REG_N_STEPS,
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

    def _cb(step, n_steps, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=REG_N_STEPS,
        n_layers=REG_N_LAYERS, loss_type="mse", verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": REG_N_STEPS, "total_steps": REG_N_STEPS,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("fit1d", opt_name, seed, _drop_params(result))


def _run_fit2d(opt_name, lr, seed, progress, shots):
    key = ("fit2d", opt_name, seed)
    t0 = time.perf_counter()
    progress[key] = {"status": "running", "step": 0, "total_steps": REG_N_STEPS,
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

    def _cb(step, n_steps, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=REG_N_STEPS,
        n_layers=REG_N_LAYERS, loss_type="mse", verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": REG_N_STEPS, "total_steps": REG_N_STEPS,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("fit2d", opt_name, seed, _drop_params(result))


def _run_vqe(opt_name, lr, seed, progress, shots):
    key = ("vqe", opt_name, seed)
    t0 = time.perf_counter()
    progress[key] = {"status": "running", "step": 0, "total_steps": VQE_N_STEPS,
                     "loss": None, "elapsed": 0.0}

    circuit, _H = make_vqe_circuit(
        VQE_N_QUBITS, VQE_N_LAYERS, VQE_J, VQE_H,
        shots=shots, diff_method=_diff_method_for(opt_name, shots),
    )
    params = init_params_vqe(VQE_N_QUBITS, VQE_N_LAYERS, seed)

    def _cb(step, n_steps, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_vqe(
        circuit, params,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=VQE_N_STEPS,
        n_layers=VQE_N_LAYERS, verbose=False,
        progress_cb=_cb,
    )
    progress[key] = {"status": "done", "step": VQE_N_STEPS, "total_steps": VQE_N_STEPS,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("vqe", opt_name, seed, _drop_params(result))


def _run_cls(opt_name, lr, seed, progress, shots):
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import MinMaxScaler

    key = ("cls", opt_name, seed)
    t0 = time.perf_counter()
    progress[key] = {"status": "running", "step": 0, "total_steps": REG_N_STEPS,
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

    def _cb(step, n_steps, loss):
        progress[key] = {"status": "running", "step": step, "total_steps": n_steps,
                         "loss": loss, "elapsed": time.perf_counter() - t0}

    result = train_with_data(
        circuit, params, x_train, y_train,
        opt_name=_canonical_opt(opt_name), lr=lr, n_steps=REG_N_STEPS,
        n_layers=REG_N_LAYERS, loss_type="hinge", verbose=False,
        progress_cb=_cb,
    )

    preds = [float(circuit(result["params"], x)) for x in x_train]
    result["final_accuracy"] = accuracy(preds, [float(yi) for yi in y])

    progress[key] = {"status": "done", "step": REG_N_STEPS, "total_steps": REG_N_STEPS,
                     "loss": result["losses"][-1], "elapsed": time.perf_counter() - t0}
    return ("cls", opt_name, seed, _drop_params(result))


# ── Heartbeat helper ─────────────────────────────────────────────────────

def _fmt_elapsed(seconds):
    """Format seconds into a compact m:ss or h:mm:ss string."""
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _heartbeat_loop(progress, total_jobs, t_start, stop_event, interval=20):
    """Print a compact progress summary every *interval* seconds.

    Runs as a daemon thread in the main process.
    """
    while not stop_event.wait(interval):
        elapsed = time.perf_counter() - t_start
        snap = dict(progress)

        n_done = sum(1 for v in snap.values() if v["status"] == "done")
        n_fail = sum(1 for v in snap.values() if v["status"] == "failed")
        n_run  = sum(1 for v in snap.values() if v["status"] == "running")
        n_pend = total_jobs - len(snap)

        running = [
            (k, v) for k, v in snap.items() if v["status"] == "running"
        ]

        def _eta(info):
            step, tot, el = info["step"], info["total_steps"], info["elapsed"]
            if step > 0 and tot:
                return (el / step) * (tot - step)
            return float("inf")

        max_eta = max((_eta(v) for _, v in running), default=0) if running else 0
        if max_eta == float("inf") or max_eta == 0:
            eta_str = "n/a"
        else:
            eta_str = f"~{_fmt_elapsed(max_eta)}"

        header = f"Progress [{_fmt_elapsed(elapsed)} elapsed]"
        print(f"\n{'':─<70}", flush=False)
        print(f"  {header}")
        print(
            f"  Done: {n_done}/{total_jobs}  |  Running: {n_run}"
            f"  |  Pending: {n_pend}  |  Failed: {n_fail}  |  ETA: {eta_str}"
        )

        if running:
            running.sort(key=lambda kv: _eta(kv[1]), reverse=True)
            print("\n  Highest ETA (bottlenecks):")
            for (task, opt, seed), info in running[:10]:
                step = info["step"]
                tot  = info["total_steps"]
                loss_s = f"{info['loss']:.4f}" if info["loss"] is not None else "..."
                el_s = _fmt_elapsed(info["elapsed"])
                eta = _eta(info)
                eta_s = _fmt_elapsed(eta) if eta < float("inf") else "n/a"
                label = TASK_LABELS.get(task, task)
                pct = step / tot * 100 if tot else 0
                print(
                    f"    {label:>15s}/{opt:<10s}/seed={seed}"
                    f"  step {step:>3d}/{tot}  ({pct:4.0f}%)"
                    f"  loss={loss_s}  elapsed={el_s}  ETA={eta_s}"
                )

        print(f"{'':─<70}", flush=True)
        sys.stdout.flush()


# ── Main orchestration ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all QNG baseline experiments in parallel",
    )
    parser.add_argument(
        "--workers", type=int, default=os.cpu_count(),
        help="Number of parallel worker processes (default: all CPU cores)",
    )
    parser.add_argument(
        "--shots", type=int, default=0,
        help="Measurement shots per circuit (0 = analytic noiseless). "
             "When >0, forces diff_method=parameter-shift (adjoint is incompatible with shots).",
    )
    args = parser.parse_args()

    results_dir, plots_dir = make_run_dir(RESULTS_BASE)
    n_workers = args.workers
    shots = args.shots if args.shots > 0 else None
    # GD/Adam: adjoint (analytic) or parameter-shift (shots).
    # QNG_block/QNG_full: always parameter-shift, since qml.metric_tensor
    # tapes aren't compatible with lightning.qubit's adjoint implementation.
    diff_method_gd_adam = "parameter-shift" if shots else "adjoint"
    diff_method_qng = "parameter-shift"

    worker_specs = [
        (_run_fit1d, "fit1d"),
        (_run_fit2d, "fit2d"),
        (_run_vqe,   "vqe"),
        (_run_cls,   "cls"),
    ]
    jobs = []
    for seed in SEEDS:
        for fn, task in worker_specs:
            for opt_name, cfg in OPTIMIZERS.items():
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
    print(f"  Workers : {n_workers}")
    print(f"  Jobs    : {total}")
    print(f"  Tasks   : {', '.join(TASK_LABELS.values())}")
    print(f"  Opts    : {', '.join(OPTIMIZERS.keys())}")
    print(f"  Seeds   : {SEEDS}")
    print(f"  Shots   : {shots_banner}")
    print(f"  VQE     : {VQE_N_QUBITS} qubits, {VQE_N_LAYERS} layers")
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
        args=(progress, total, t_start, stop_heartbeat),
        daemon=True,
    )
    hb_thread.start()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        future_map = {}
        for fn, _task, opt_name, lr, seed in jobs:
            fut = pool.submit(fn, opt_name, lr, seed, progress, shots)
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

    for task in ["fit1d", "fit2d", "vqe", "cls"]:
        if task not in timing:
            continue
        label = TASK_LABELS[task]
        print(f"\n  {label}:")
        for opt in OPTIMIZERS:
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
                         save_path=os.path.join(plots_dir, "fit1d_convergence.png"))
        resource_plot(agg, title="1D Regression: sin(x) (resource-normalised)", log_y=True,
                      save_path=os.path.join(plots_dir, "fit1d_resource.png"))
        final_loss_bar(agg, title="1D Regression: final MSE",
                       save_path=os.path.join(plots_dir, "fit1d_final.png"))

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
                         save_path=os.path.join(plots_dir, "fit2d_convergence.png"))
        resource_plot(agg, title="2D Regression (resource-normalised)", log_y=True,
                      save_path=os.path.join(plots_dir, "fit2d_resource.png"))
        final_loss_bar(agg, title="2D Regression: final MSE",
                       save_path=os.path.join(plots_dir, "fit2d_final.png"))

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
                         save_path=os.path.join(plots_dir, "vqe_convergence.png"))
        resource_plot(agg, title="VQE (resource-normalised)",
                      ylabel="Energy \u27e8H\u27e9",
                      save_path=os.path.join(plots_dir, "vqe_resource.png"))
        final_loss_bar(agg, title="VQE: final energy",
                       ylabel="Energy \u27e8H\u27e9",
                       save_path=os.path.join(plots_dir, "vqe_final.png"))

    # ── Classification ───────────────────────────────────────────────────
    if "cls" in grouped:
        print("\nSaving classification results...")
        agg = {}
        for opt, seed_dict in grouped["cls"].items():
            agg_opt = aggregate_seeds(seed_dict)
            accs = [seed_dict[s]["final_accuracy"] for s in SEEDS]
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
                         save_path=os.path.join(plots_dir, "cls_convergence.png"))
        resource_plot(agg, title="Classification (resource-normalised)", log_y=True,
                      save_path=os.path.join(plots_dir, "cls_resource.png"))
        final_loss_bar(agg, title="Classification: final hinge loss",
                       save_path=os.path.join(plots_dir, "cls_final.png"))

    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Total wall time: {_fmt_elapsed(total_time)}")
    print(f"  Results saved in {results_dir}")
    print(f"  Plots  saved in {plots_dir}")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
