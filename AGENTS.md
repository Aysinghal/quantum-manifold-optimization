# Agent guide

Short, checklist-style orientation for AI agents (and future-me) working on
this repo. Not an architecture explainer -- the code's docstrings cover
the *what* and *why*. This file covers the *where* for the cross-cutting
edits that are easy to miss.

If anything below contradicts the code, the code is right. Update this
file when registries / cross-file invariants change.

## Repo orientation (30-second version)

```
run_all_parallel.py     Single entry point. Defines registries (OPTIMIZERS,
                        TASK_TIERS, OPT_TIERS, TASK_LABELS, OPT_PRIORITY,
                        TASK_PRIORITY, BENCH_BUILDERS), the 13 _run_<task>
                        workers, the 13 _bench_build_<task> bench builders,
                        the LPT scheduler + heartbeat, and main().
tests/bench_smoke.py    Incremental full-cartesian timer. Reads OPTIMIZERS
                        and BENCH_BUILDERS from run_all_parallel; writes
                        bench_smoke_estimates.json at the repo root.
                        Heartbeat reads that JSON to seed ETA before any
                        job finishes.
run.sbatch              Slurm wrapper. Env vars TASKS / OPTS / SEEDS / TITLE
                        / SHOTS map to run_all_parallel CLI flags.
src/models.py           Circuits, Hamiltonians, init_params_*.
src/training.py         train_with_data / train_vqe. EVAL_COST and
                        _QNG_FAMILY drive opt-name -> training-path dispatch.
src/manifolds.py        Manifold ABC + Euclidean / Torus / Sphere +
                        _OPTIMIZER_MANIFOLDS registry + manifold_for().
src/metrics.py          aggregate_seeds (incl. theta_norms), make_run_dir,
                        _slugify, save_results.
src/visualization.py    convergence_plot, resource_plot, final_loss_bar,
                        theta_trajectory_plot.
tests/smoke_all_optimizers.py   5-step regression suite over every optimizer.
tools/                          One-off plot reorganization helpers.
results/<stamp>__<slug>/        Per-run output (JSON + plots + run_meta.json).
```

## How a run flows (data path you should remember)

```
sbatch run.sbatch  ->  run_all_parallel.py main()
   |
   |- _resolve_names(--tasks)   ->  active_tasks  (TASK_TIERS or individuals)
   |- _resolve_names(--optimizers) -> active_opts (OPT_TIERS or individuals)
   |- LPT-sort jobs by job_priority = OPT_PRIORITY * TASK_PRIORITY
   |- ProcessPoolExecutor dispatches _run_<task>(opt, lr, seed, progress, shots)
   |- Heartbeat thread reads `progress` dict + bench_smoke_estimates.json
   |    -> _predicted_duration falls back through:
   |         (1) measured (task,opt) median  this run
   |         (2) measured task median       this run
   |         (3) bench JSON entry           prior runs
   |         (4) None                       => LPT treats as 0 load
   |- aggregate_seeds + plots written under results/<stamp>__<slug>/
   |- run_meta.json saved alongside.
```

## Adding a new optimizer

The minimum viable change is THIS many places. Skip any of them and you'll
get either a `KeyError` at dispatch, a silent ETA miss, or a smoke-test
failure later.

1. **`OPTIMIZERS`** in [run_all_parallel.py](run_all_parallel.py) -- add
   `"<name>": {"lr": <default_lr>}`.
2. **`OPT_PRIORITY`** in [run_all_parallel.py](run_all_parallel.py) -- pick
   `100` (QNG-full-class), `50` (QNG-family), or `10` (cheap flat).
3. **At least one `OPT_TIERS` entry** -- so `--optimizers <tier>` reaches it.
   Skip if the optimizer is meant to be referenced individually only.
4. **`_canonical_opt`** in [run_all_parallel.py](run_all_parallel.py) -- only
   if the new name is an *alias* (e.g. `<base>_lr01`). New base optimizers
   pass through unchanged.
5. **`_diff_method_for`** in [run_all_parallel.py](run_all_parallel.py) --
   only if the optimizer needs `parameter-shift` (i.e. it touches the QNG
   metric tensor).
6. **`EVAL_COST`** and (if QNG-family) **`_QNG_FAMILY`** in
   [src/training.py](src/training.py) -- one lambda + maybe one set entry.
7. **`_OPTIMIZER_MANIFOLDS`** in [src/manifolds.py](src/manifolds.py) -- only
   if the optimizer lives off the Euclidean default. Sphere optimizers also
   need the training loop to know about them; check `RQNG_sphere_block` /
   `ProjQNG_sphere` / `ProjAdam_sphere` for the full pattern.
8. **Training-loop dispatch** in [src/training.py](src/training.py) -- if the
   update rule is genuinely new (not just a different lr / momentum on an
   existing path), add a branch alongside the existing `MomentumQNG_block`
   / `QNGAdam_v1_block` / `QNGAdam_v2_block` cases inside `train_with_data`
   and `train_vqe`. New aliases of an existing rule do NOT need this.
9. **Run** `python tests/smoke_all_optimizers.py` -- any optimizer in
   `OPTIMIZERS` is automatically picked up; if the optimizer is on a
   non-Euclidean manifold, the suite checks the right invariant via
   `manifold_for()`.
10. **Run** `python tests/bench_smoke.py` -- it picks up the new optimizer
    automatically and only times the missing 13 tasks x 2 modes (~26 combos)
    rather than the whole cartesian.

You do NOT need to touch `BENCH_BUILDERS` -- bench builders are per-task.

## Adding a new task

1. **Constants** in [run_all_parallel.py](run_all_parallel.py) -- e.g.
   `<TASK>_N_QUBITS / <TASK>_N_LAYERS / <TASK>_N_STEPS / <TASK>_J / ...`.
   Defined once and shared by both the worker and the bench builder.
2. **Worker** `_run_<task>` in [run_all_parallel.py](run_all_parallel.py) --
   build circuit + params, set `progress[key]`, call `train_with_data` or
   `train_vqe`, return `(<task>, opt_name, seed, _drop_params(result))`.
   Mirror an existing worker that's structurally similar (regression vs.
   VQE vs. classification).
3. **Bench builder** `_bench_build_<task>` in
   [run_all_parallel.py](run_all_parallel.py) -- the bookkeeping-free
   sibling of the worker. Returns `(run_fn, prod_steps, n_layers)` where
   `run_fn(n_steps)` invokes the same train function. Reuse the constants
   from step 1 -- DO NOT hardcode separate bench numbers.
4. **`BENCH_BUILDERS`** in [run_all_parallel.py](run_all_parallel.py) --
   register `"<task>": _bench_build_<task>`. If you skip this the heartbeat
   has no ETA seed and the bench skips the task silently.
5. **`TASK_LABELS`** in [run_all_parallel.py](run_all_parallel.py) -- the
   short pretty name shown in the heartbeat / summary tables.
6. **`TASK_PRIORITY`** in [run_all_parallel.py](run_all_parallel.py) --
   approximate wall-time weight. Recalibrate after the first full run.
7. **At least one `TASK_TIERS` entry** -- so `--tasks <tier>` reaches it.
8. **`worker_specs` (the dispatch table inside `main()`)** in
   [run_all_parallel.py](run_all_parallel.py) -- add `"<task>": _run_<task>`
   so the runner can dispatch the new worker.
9. **Per-task output block in `main()`** in
   [run_all_parallel.py](run_all_parallel.py) -- copy an existing
   `if "<existing_task>" in grouped:` block, swap the keys, set the
   correct config dict + plot titles, emit the four plots
   (`convergence`, `resource`, `final_loss_bar`, `theta_trajectory`).
10. **sbatch examples** in [run.sbatch](run.sbatch) -- if the task belongs
    to a new tier or is meant to be invoked individually, add an example
    line under the `# Override examples:` block.
11. **Run** `python tests/bench_smoke.py` to time the new task across all
    optimizers and write entries into `bench_smoke_estimates.json`.

Bench drift sanity check: every key in `BENCH_BUILDERS` must also be a key
in `TASK_LABELS` and reachable through some `TASK_TIERS` entry or by
individual name. The runner does not enforce this -- mismatches surface as
silent gaps in the ETA and missing output blocks at the end of the run.

## Adding a new manifold

1. New `Manifold` subclass in [src/manifolds.py](src/manifolds.py) with
   `name`, `project`, `retract`, and (if non-trivial) `init`.
2. Wire any new optimizers into `_OPTIMIZER_MANIFOLDS` in the same file.
3. Update the sphere-style branches in `_apply_manifold_step` /
   `_retract_adam_step` in [src/training.py](src/training.py) if the new
   manifold needs a non-Euclidean displacement transform (see the existing
   `getattr(manifold, "name", None) == "sphere"` guard).
4. Smoke test: `tests/smoke_all_optimizers.py` already routes through
   `manifold_for()`; extend its `if/elif` chain in `_run_one` so the
   invariant for the new manifold is asserted.

## Heartbeat / ETA contract (so future edits don't break it)

- The LPT simulator in `_heartbeat_loop` is the SOURCE OF TRUTH for ETA.
  It heaps remaining work onto `n_workers` slots; the makespan IS the ETA.
  Do not "improve" this by summing remaining durations -- multi-worker
  scheduling is the whole point.
- For a job with `step == 0` we use the predicted full-run duration minus
  elapsed (clamped at 0). The prediction chain is in `_predicted_duration`:
  this-run measured -> this-run same-task -> bench JSON -> None.
- `lookup_bench_full_run` keys on `(task, opt, mode)` where
  `mode = "analytic" if shots is None else f"shots={shots}"`. Keep that
  mode key in lockstep with the labels in `tests/bench_smoke.py::MODES`.
- Stale-task / stale-opt pruning happens on the bench side
  (`tests/bench_smoke.py` drops keys not in the current registries). The
  runner side is robust to arbitrary extra keys in the JSON, but won't
  prune for you.
- `--steps N` globally overrides every task's step count. Bench-derived ETAs
  are rescaled by `per_step_s * N` rather than the saved `full_run_s`, so a
  10-step smoke run reports a 10-step ETA, not a production-length one.

## Output convention

- One run = one folder `results/<timestamp>__<slug>/` (slug from `--title`).
- That folder contains `<task>.json` + `plots/<task>/{convergence,resource,final,theta}.png`
  for each task that ran, plus `run_meta.json` (CLI args, host, workers,
  workers_source, started_at, title).
- `latest` symlink at `results/latest` always points at the newest run.

## Don't-do list

- Don't import from `experiments/` or `notebooks/` -- they're gone.
- Don't rebuild a separate runner; `run_all_parallel.py` is the one entry
  point. Add a tier or pass individual names rather than forking.
- Don't add per-task duplicate constants to `tests/bench_smoke.py`. The
  bench-vs-worker drift problem is the entire reason `BENCH_BUILDERS` lives
  in `run_all_parallel.py`.
- Don't bump `track_theta_norm` -- it was removed; `theta_norms` is
  unconditionally recorded by the training loop.
