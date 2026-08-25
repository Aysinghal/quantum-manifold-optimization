# Quantum Manifold Optimization

Benchmarking projection-based and Riemannian optimizers for variational quantum
algorithms across 8 tasks.

Do manifold constraints actually improve variational quantum optimization, or
does the geometric machinery cost complexity without buying performance? The
comparison fixes the task suite, seeds, and learning rates, and varies only the
update rule and the parameter manifold it lives on.

## Key result

Simple projection-based methods consistently matched or outperformed the more
complex geometry-aware optimizers. On the 8-task sphere sweep, `ProjAdam_sphere`
(a flat Adam step followed by a sphere retraction) beat the fully Riemannian
`RQNG_sphere_block` on every task in the tier — including `vqe_overparam_heis`,
the task deliberately constructed for the sphere intervention to pay off — and
the pattern replicated one family up, with `ProjQNGAdam_v1_sphere` beating
`RQNGAdam_v1_sphere` on 7 of 8 tasks.

Projection also helped the flat optimizers it wraps rather than merely
constraining them: `ProjAdam_sphere` improved on plain `Adam` on the hard
physics tasks, and `ProjQNGAdam_v1_sphere` improved on flat `QNGAdam_v1_block`
on 4 of 8. But no single optimizer dominated every setting — flat
`QNGAdam_v1_block` remains strongest on several tasks. An earlier
torus-constrained arm was a clean null: torus-QNG and flat QNG are
loss-identical to ~1e-15, because the Fubini–Study preconditioner is already
2π-periodic in every gate angle and absorbs the torus topology. Hyperbolic
optimizers (Lorentz model) are implemented but were run only as exploratory
pilots, not as a full arm of the study.

## What is being compared

The headline sweep is six optimizers spanning three categories, run over 8
variational quantum tasks with differing landscape characteristics (VQE on
Heisenberg rings, XXZ chains, SK spin glasses, Stokes-style overparameterized
circuits, plus multi-frequency regression), in analytic mode (`shots = 0`) with
5 seeds per task.

| Category | Optimizers | Update rule |
| --- | --- | --- |
| Flat-space baselines | `Adam`, `QNG_block`, `QNGAdam_v1_block` | Euclidean parameter space, no constraint |
| Projection-based | `ProjAdam_sphere`, `ProjQNGAdam_v1_sphere` | Full ambient step, then retract onto the sphere |
| Riemannian | `RQNGAdam_v1_sphere` | Project the gradient to the tangent space, solve with the Fubini–Study metric, then geodesic-retract |

The projected variants are mandatory companions, not extras: the sphere
restricts the reachable state set, so without them a sphere-vs-flat comparison
would confound "different update rule" with "different reachable states".

Beyond these six, the registry also holds `ProjQNG_sphere`, `RQNG_sphere_block`,
the torus-constrained `RQNG_torus_block` (a documented null result), and five
hyperbolic optimizers run as exploratory pilots. All 20 optimizer
configurations and 13 tasks live in `run_all_parallel.py`; tiers pick which
subset a run schedules. The `sphere_concept` tier used in the commands below is
the wider 9-optimizer sphere sweep the six were drawn from.

## Reproducing a run

```bash
conda env create -f environment.yml   # creates the `qng-baseline` env
conda activate qng-baseline
python run_all_parallel.py --tasks manifold_sphere --optimizers sphere_concept \
    --shots 0 --seeds 0,1,2,3,4 --title "sphere sweep"
```

The runner uses every available core; cap it with `--workers N`. On Slurm,
`run.sbatch` wraps the same command and exposes `TASKS` / `OPTS` / `SEEDS` /
`SHOTS` / `TITLE` as environment variables. Submit it from the repo and it
picks up the repo path and your active conda install automatically; override
with `REPO_DIR` / `CONDA_SH` / `CONDA_ENV` if not. The `#SBATCH` account,
partition, and nodelist lines at the top are UMD-specific — change those for
your cluster.

```bash
TASKS=manifold_sphere OPTS=sphere_concept sbatch run.sbatch
```

`--tasks` and `--optimizers` accept tier names, individual names, or any
comma-separated mix. `--shots 0` is analytic (adjoint for GD/Adam,
parameter-shift for the QNG family); a positive value gives a shot-noisy run.
Each run writes `results/<timestamp>__<slug>/` with one JSON per task,
`plots/<task>/{convergence,resource,final,theta}.png`, and a `run_meta.json`
recording the full CLI, host, and seeds; `results/latest` points at the newest
run. `python tests/smoke_all_optimizers.py` checks that every registered
optimizer still runs and preserves its manifold invariant.

## Repository layout

| Path | Contents |
| --- | --- |
| `run_all_parallel.py` | Single entry point: registries, tiers, per-task workers, parallel scheduler, aggregation, plotting |
| `src/models.py` | Circuits, Hamiltonians (TFIM, Heisenberg, XXZ, SK, Stokes), parameter initialization |
| `src/training.py` | `train_with_data` / `train_vqe`, natural-gradient step dispatch, per-optimizer circuit-evaluation costs |
| `src/manifolds.py` | `Manifold` ABC with `Euclidean` / `Torus` / `Sphere` / `Hyperbolic`, plus the optimizer→manifold registry |
| `src/metrics.py` | Seed aggregation, run-directory creation, JSON serialization |
| `src/visualization.py` | Convergence, resource, final-loss, and theta-trajectory plots |
| `tests/smoke_all_optimizers.py` | 5-step regression suite asserting manifold invariants for every optimizer |
| `tests/bench_smoke.py` | Incremental task×optimizer×mode timer; writes `bench_smoke_estimates.json` for ETA seeding |
| `tools/` | One-off plot-reorganization helper |
| `docs/` | Research narrative, optimizer/task/manifold index, hyperbolic scoping, poster package |
| `AGENTS.md` | Checklists for adding an optimizer, task, or manifold without breaking the runner |

Background: [docs/RESEARCH_PATH.md](docs/RESEARCH_PATH.md) for the research
narrative and prior-art map, [docs/research_reference.md](docs/research_reference.md)
for a conceptual index of what exists. The QNG method this builds on is Stokes,
Izaac, Killoran & Carleo, [arXiv:1909.02108](https://arxiv.org/abs/1909.02108).

## Authors

Sumedh Sai Ameti, Aarav Nirmal, Ayush Singhal — UMD QLearn, University of
Maryland, College Park. Presented as *Testing Quantum Machine Learning
Optimization on Topological Spaces*.

## License

MIT. See [LICENSE](LICENSE).
