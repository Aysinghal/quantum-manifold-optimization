"""Full Cartesian smoke timer: every (task x optimizer x mode) combination.

For each combo we run WARMUP_STEPS (untimed) followed by TIMED_STEPS (timed)
optimization steps, divide by TIMED_STEPS to get a per-step wall, and
extrapolate by the production step count to get a full-run prediction.
Results are saved to ``bench_smoke_estimates.json`` at the repo root (next
to ``run_all_parallel.py``); the runner's heartbeat reads that file and
uses the entries as the ETA seed for in-flight or queued (task, opt)
combos that don't yet have measured timings from the current run.

Two modes are timed for every combo:
  - "analytic"   shots=None         (adjoint for GD/Adam, parameter-shift for QNG)
  - "shots=1000" shots=1000         (parameter-shift everywhere)

13 tasks x 13 optimizers x 2 modes = 338 timings (current registry size).
Some combos may legitimately fail (a task constructed for one manifold can
be invalid for an optimizer designed for another); failures are recorded as
``null`` entries and skipped by the heartbeat lookup.

Incremental updates (default behaviour)
---------------------------------------
On rerun, the script LOADS the existing ``bench_smoke_estimates.json`` and
only re-times combos that are
  (a) missing entirely (e.g. a new task or optimizer was added to the
      registry since the last bench run), or
  (b) recorded as ``null`` (previously failed).
Successful entries are kept as-is so a 5-minute incremental refresh after
adding one optimizer doesn't trigger a full 338-combo bench.

Flags:
  --force              Re-time every combo from scratch, ignoring the cache.
  --keep-failures      Don't retry combos previously recorded as null.
  --tasks T1,T2        Restrict timing to a comma-separated subset of tasks.
  --optimizers O1,O2   Restrict timing to a comma-separated subset of opts.
  --output PATH        Write to PATH instead of the default location.
Stale entries (tasks or optimizers no longer in the registry) are pruned
unconditionally on every run.

This script is single-process by design -- timings are noisier when other
work is on the box, so run it on an idle compute node before submitting a
production sbatch job.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime

# Pin BLAS to one thread so we measure single-job behaviour cleanly; the
# production runner sets the same vars per worker.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from run_all_parallel import (
    BENCH_BUILDERS,
    BENCH_ESTIMATES_PATH,
    OPTIMIZERS,
    TASK_LABELS,
)


WARMUP_STEPS = 1
TIMED_STEPS = 2

# (label, shots) pairs. Heartbeat keys mode lookups by these labels.
MODES = [
    ("analytic",   None),
    ("shots=1000", 1000),
]


def fmt_seconds(s):
    if s is None:
        return "  --  "
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(int(m), 60)
    return f"{h}h{int(m):02d}m"


def time_one(task, opt_name, shots):
    """Build the task, run WARMUP then TIMED steps, return (per_step_s, prod_steps)."""
    builder = BENCH_BUILDERS[task]
    run, prod_steps, _n_layers = builder(opt_name, shots)
    run(WARMUP_STEPS)  # JIT / first-circuit-build cost goes here
    t0 = time.perf_counter()
    run(TIMED_STEPS)
    elapsed = time.perf_counter() - t0
    return elapsed / TIMED_STEPS, prod_steps


def _is_valid_entry(entry):
    """True iff `entry` is a successful, well-formed bench result we can keep."""
    if not isinstance(entry, dict):
        return False
    return all(k in entry for k in ("per_step_s", "prod_steps", "full_run_s"))


def _classify_existing(existing, tasks, opts, modes, force, keep_failures):
    """Walk the Cartesian and decide what to do with each combo.

    Returns (todo, kept, stale_tasks, stale_opts) where:
      todo        list of (task, opt, mode_label, shots) to re-time this run
      kept        count of valid entries reused from the existing JSON
      stale_tasks set of task names in the JSON that no longer exist
      stale_opts  set of opt names in the JSON that no longer exist
    """
    if not isinstance(existing, dict):
        existing = {}
    cached = existing.get("estimates", {}) if existing else {}

    stale_tasks = set(cached.keys()) - set(tasks)
    stale_opts = set()
    for t, by_opt in cached.items():
        if isinstance(by_opt, dict):
            stale_opts.update(set(by_opt.keys()) - set(opts))

    todo = []
    kept = 0
    for t in tasks:
        for o in opts:
            for label, shots in modes:
                if force:
                    todo.append((t, o, label, shots))
                    continue
                entry = cached.get(t, {}).get(o, {}).get(label) if isinstance(cached, dict) else None
                if _is_valid_entry(entry):
                    kept += 1
                elif entry is None and keep_failures and t in cached and o in cached.get(t, {}) and label in cached[t][o]:
                    # Previously failed and user asked us not to retry.
                    kept += 1
                else:
                    todo.append((t, o, label, shots))
    return todo, kept, stale_tasks, stale_opts


def _seed_estimates_from_cache(existing, tasks, opts, modes):
    """Start the new estimates dict from the cached one, but only for tasks
    and opts that still exist (stale entries are pruned)."""
    out: dict = {t: {o: {} for o in opts} for t in tasks}
    if not isinstance(existing, dict):
        return out
    cached = existing.get("estimates", {})
    if not isinstance(cached, dict):
        return out
    for t in tasks:
        for o in opts:
            entry = cached.get(t, {}).get(o)
            if isinstance(entry, dict):
                for label, _ in modes:
                    if label in entry:
                        out[t][o][label] = entry[label]
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description="Incremental full-cartesian bench timer "
                    "(re-times only missing/failed combos by default).",
    )
    p.add_argument("--force", action="store_true",
                   help="Re-time every combo from scratch, ignoring the cache.")
    p.add_argument("--keep-failures", action="store_true",
                   help="Don't retry combos previously recorded as null.")
    p.add_argument("--tasks", default=None,
                   help="Comma-separated subset of task names to time "
                        "(default: all in BENCH_BUILDERS).")
    p.add_argument("--optimizers", default=None,
                   help="Comma-separated subset of optimizer names to time "
                        "(default: all in OPTIMIZERS).")
    p.add_argument("--output", default=BENCH_ESTIMATES_PATH,
                   help=f"Output JSON path (default: {BENCH_ESTIMATES_PATH}).")
    return p.parse_args()


def _filter_subset(name, full_list, raw):
    """Resolve a comma-separated user-supplied subset against the registry."""
    if raw is None:
        return list(full_list)
    requested = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in requested if s not in full_list]
    if unknown:
        sys.exit(f"  ERROR: unknown {name}: {', '.join(unknown)}.\n"
                 f"  Valid: {', '.join(full_list)}")
    return requested


def main():
    args = parse_args()

    # Restrict subset of tasks/opts to TIME this run, but always seed from
    # the full cached file and write back the full file -- partial CLI
    # invocations should not destroy entries for the other combos.
    full_tasks = list(BENCH_BUILDERS)
    full_opts = list(OPTIMIZERS)
    sel_tasks = _filter_subset("task",      full_tasks, args.tasks)
    sel_opts  = _filter_subset("optimizer", full_opts,  args.optimizers)

    # Load cache (may be missing).
    existing = None
    if os.path.isfile(args.output) and not args.force:
        try:
            with open(args.output) as f:
                existing = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARNING: could not read existing {args.output}: {e}",
                  file=sys.stderr)
            print("  Continuing with a fresh bench.", file=sys.stderr)

    # Build the output dict pre-seeded with cached entries for the FULL
    # registry (not just the selected subset), then we'll overwrite the
    # combos we time this run.
    estimates = _seed_estimates_from_cache(existing, full_tasks, full_opts, MODES)

    # Decide what to time. We only consider combos in (sel_tasks x sel_opts);
    # the rest are kept from cache. We pass the FULL registry to the
    # classifier so stale-detection works against the saved file as a whole.
    todo_full, kept_full, stale_tasks, stale_opts = _classify_existing(
        existing, full_tasks, full_opts, MODES, args.force, args.keep_failures,
    )
    # Apply the user's subset filter.
    todo = [(t, o, label, shots) for (t, o, label, shots) in todo_full
            if t in sel_tasks and o in sel_opts]
    n_combos_full = len(full_tasks) * len(full_opts) * len(MODES)

    # ── Header ─────────────────────────────────────────────────────────────
    print("=" * 90)
    print(f"  Full-cartesian bench: {len(full_tasks)} tasks x {len(full_opts)} "
          f"optimizers x {len(MODES)} modes = {n_combos_full} combos")
    print(f"  warmup={WARMUP_STEPS}  timed={TIMED_STEPS}  host={platform.node()}")
    if existing and not args.force:
        cached_at = existing.get("generated_at", "?")
        print(f"  Cache: loaded from {args.output} (generated_at={cached_at})")
        print(f"         {kept_full} combo(s) kept, {len(todo_full)} to (re)time")
        if stale_tasks:
            print(f"         pruning stale tasks: {sorted(stale_tasks)}")
        if stale_opts:
            print(f"         pruning stale optimizers: {sorted(stale_opts)}")
    elif args.force:
        print("  Cache: --force passed; re-timing every combo")
    else:
        print(f"  Cache: no prior file at {args.output}; timing all combos")
    if args.tasks or args.optimizers:
        print(f"  Subset: tasks={sel_tasks}  optimizers={sel_opts}")
        print(f"          -> {len(todo)} combo(s) to time this invocation")
    print("=" * 90)

    if not todo:
        print("  Nothing to do -- bench is up to date for the requested subset.")
        # Still rewrite the file so stale entries are pruned and metadata
        # reflects the latest registry.
        _write_payload(args.output, estimates, full_tasks, full_opts, [],
                       wall_total_s=0.0, started_at=datetime.now())
        return

    header = (f"  {'task':<26s} {'optimizer':<22s} {'mode':<11s} "
              f"{'sec/step':>10s} {'full run':>10s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    failures: list[tuple[str, str, str, str]] = []
    started_at = datetime.now()
    t_start = time.perf_counter()

    for idx, (task, opt_name, mode_label, shots) in enumerate(todo, 1):
        tag = f"[{idx:>3d}/{len(todo)}]"
        try:
            per_step, prod_steps = time_one(task, opt_name, shots)
            full = per_step * prod_steps
            estimates[task][opt_name][mode_label] = {
                "per_step_s": per_step,
                "prod_steps": int(prod_steps),
                "full_run_s": full,
            }
            print(f"  {tag} {task:<22s} {opt_name:<22s} "
                  f"{mode_label:<11s} {per_step:>9.3f}s "
                  f"{fmt_seconds(full):>10s}",
                  flush=True)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            estimates[task][opt_name][mode_label] = None
            failures.append((task, opt_name, mode_label, msg))
            print(f"  {tag} {task:<22s} {opt_name:<22s} "
                  f"{mode_label:<11s} FAILED: {msg}",
                  flush=True)

    elapsed = time.perf_counter() - t_start

    # ── Slowest-per-mode summary -- governs Slurm wall budget ──────────────
    print()
    for mode_label, _ in MODES:
        mode_rows = []
        for t in full_tasks:
            for o in full_opts:
                e = estimates.get(t, {}).get(o, {}).get(mode_label)
                if isinstance(e, dict) and "full_run_s" in e:
                    mode_rows.append((t, o, e["full_run_s"]))
        if not mode_rows:
            print(f"  [{mode_label}] no successful combos")
            continue
        worst = max(mode_rows, key=lambda r: r[2])
        flag = "OK" if worst[2] < 10 * 3600 else "WARNING (>10h)"
        label = TASK_LABELS.get(worst[0], worst[0])
        print(f"  [{mode_label}] slowest single-seed job: "
              f"{label} / {worst[1]} = {fmt_seconds(worst[2])}  [{flag}]")

    if failures:
        print(f"\n  {len(failures)} combo(s) failed this run:")
        for t, o, m, msg in failures[:20]:
            print(f"    - {t} / {o} / {m}: {msg}")
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more")

    _write_payload(args.output, estimates, full_tasks, full_opts, failures,
                   wall_total_s=elapsed, started_at=started_at)
    print(f"\n  Saved estimates to {args.output}")
    print(f"  Bench wall this run: {fmt_seconds(elapsed)} "
          f"({len(todo)} combo(s) timed)")
    print("  run_all_parallel.py will pick this up automatically on its")
    print("  next run and use it for ETA seeding before any job finishes.")


def _write_payload(path, estimates, tasks, opts, failures, wall_total_s, started_at):
    """Persist estimates + metadata, atomically (write-then-rename)."""
    payload = {
        "generated_at":     datetime.now().isoformat(timespec="seconds"),
        "last_run_started": started_at.isoformat(timespec="seconds"),
        "host":             platform.node(),
        "warmup_steps":     WARMUP_STEPS,
        "timed_steps":      TIMED_STEPS,
        "modes":            [m[0] for m in MODES],
        "n_tasks":          len(tasks),
        "n_optimizers":     len(opts),
        "n_failures":       len(failures),
        "wall_total_s":     wall_total_s,
        "estimates":        estimates,
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted -- partial bench NOT written.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
