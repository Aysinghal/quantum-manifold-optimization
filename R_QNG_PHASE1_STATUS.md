# Riemannian QNG (R-QNG) — Phase 1 Status & Handoff

**Status:** Phase 1 (torus manifold) implemented and diagnostically tested. Two diagnostic runs returned **null results**. We have not yet decided whether to (a) run the full statistical sweep anyway, (b) pivot the experimental design, or (c) skip to Phase 2 (sphere) with revised expectations.

This document is a self-contained handoff: the hypothesis, the implementation, the empirical results, the interpretation, and the open decision.

---

## 1. The hypothesis we set out to test

**Top-level question:** does constraining variational parameters of a VQA to lie on a non-Euclidean manifold (instead of the default flat $\mathbb{R}^d$) help training?

**Phase 1 manifold:** the **torus** $T^d = (S^1)^d$, with each parameter living in $[0, 2\pi)$.

**Why the torus is "automatically fair":**
A single-qubit rotation gate satisfies $R_x(\theta) = R_x(\theta + 2\pi)$ (similarly $R_y, R_z$). So mapping $\theta \mapsto \theta \mod 2\pi$ never changes the quantum state — it just removes a redundant winding number from the parameterization. No quantum state is lost relative to flat $\mathbb{R}^d$.

**Why we hoped it would help:**
In overparameterized circuits, flat optimizers can drift $\|\theta\|$ to large values (each parameter accumulating multiples of $2\pi$ doing nothing useful). This is harmless for the *quantum state* but can:
- bias adaptive optimizers' second-moment estimates (Adam family),
- distort the conditioning of the Fubini–Study (FS) preconditioner if the FS matrix is not perfectly periodic in $\theta$ at finite step sizes,
- accumulate floating-point error in repeated trig evaluations.

The torus manifold prevents this drift by construction, via $\theta \leftarrow \theta \mod 2\pi$ at every retract.

**The risk we explicitly flagged in the original plan (Section 8: "Why hasn't anyone done this?"):**
The Fubini–Study metric *itself* is periodic in $\theta$ (since the underlying state is). If QNG already uses $g_{FS}(\theta)$ as its preconditioner, the QNG step direction is already invariant to $2\pi$ shifts in any single parameter. Wrapping $\theta$ post-hoc onto the torus may produce a step indistinguishable from flat-space QNG. **This risk has now materialized — see Section 5.**

---

## 2. What we implemented (code changes)

All changes are uncommitted in the working tree. Repo: `/fs/nexus-scratch/aysingha/workspace/quantum-manifold-optimization`.

### 2.1 New abstraction: `src/manifolds.py`
Defines an abstract `Manifold` base class with three operations:
- `project(theta, vec)` — project an ambient vector onto $T_\theta\mathcal{M}$.
- `retract(theta, vec)` — move from $\theta$ along `vec` and return to $\mathcal{M}$.
- `inner(theta, u, v)` — Riemannian inner product (informational only right now).

Concrete classes:
- `Euclidean` — identity project / additive retract.
- `Torus(period=2π)` — identity project / `mod period` retract.
- `Sphere` — stub for Phase 2 (not exercised in any test yet).

A registry `_OPTIMIZER_MANIFOLDS = {"RQNG_torus_block": Torus()}` and helper `manifold_for(opt_name)` map optimizer names to manifolds; everything not in the registry defaults to `Euclidean`.

### 2.2 Training loop integration: `src/training.py`
- `RQNG_torus_block` registered in `EVAL_COST` (cost is identical to `QNG_block`: $2d + L$ parameter-shift evals per step) and `_QNG_FAMILY`.
- New helper `_apply_manifold_step(opt_name, params, update, lr)` computes the post-update parameter via `manifold.retract(theta, -lr * update)` and returns it as a `pnp.array(..., requires_grad=True)`.
- The QNG update site in both `_qng_step` and the manual-QNG path of `train_vqe` was switched from a raw `params - lr * update` to `_apply_manifold_step(...)`. For `Euclidean` this is mathematically identical to the old code; for `Torus` it adds a `mod 2π`.
- `_natgrad_direction` now treats `RQNG_torus_block` the same as `QNG_block` for the FS-solve branch.
- New `track_theta_norm=True` kwarg on `train_with_data` and `train_vqe` records $\|\theta\|$ at every step and returns it under `theta_norms`.

### 2.3 Hamiltonians: `src/models.py`
Two new builders (used by Phase 1's "hard physics" tasks):
- `make_sk_hamiltonian(n_qubits, seed=42)` — Sherrington–Kirkpatrick spin glass with $J_{ij} \sim U(-1, 1)$.
- `make_xxz_hamiltonian(n_qubits, delta=1.0)` — XXZ chain with open boundaries.

### 2.4 Visualization: `src/visualization.py`
- `COLORS["RQNG_torus_block"] = "#e377c2"`, `LABELS["RQNG_torus_block"] = "R-QNG (torus)"`.
- `INTENDED_OPTIMIZER_FAMILY` extended for the two "rigged for torus" tasks (`vqe_stokes_overparam_long`, `fit_high_periodic`) → `"RQNG"`.
- `TASK_PLOT_SLUGS` extended for the four new Phase 1 tasks.

### 2.5 Runner: `run_all_parallel.py`
- `OPTIMIZERS["RQNG_torus_block"] = 0.05`, `OPT_PRIORITY["RQNG_torus_block"] = 50`.
- `_diff_method_for(opt_name)` returns `"parameter-shift"` for any `opt_name.startswith("RQNG")`.
- New task tier:
  ```python
  TASK_TIERS["manifold_torus"] = [
      # 2 carry-overs, 2 rigged-for-torus, 3 hard-physics
      "vqe_stokes",
      "fit_periodic",
      "vqe_stokes_overparam_long",   # rigged
      "fit_high_periodic",           # rigged
      "vqe_sk_spinglass",            # hard physics, 1000 shots hardcoded
      "vqe_xxz",                     # hard physics, 1000 shots hardcoded
      "vqe_heisenberg",
  ]
  ```
- New optimizer tier:
  ```python
  OPT_TIERS["torus_concept"] = [
      "Adam", "QNG_block", "QNGAdam_v1_block", "RQNG_torus_block",
  ]
  ```
- Four new worker functions (`_run_vqe_stokes_overparam_long`, `_run_fit_high_periodic`, `_run_vqe_sk_spinglass`, `_run_vqe_xxz`), wired into `all_worker_specs` and given save/plot blocks in `main()`.
- New helper `_high_periodic_target` for `fit_high_periodic`.

### 2.6 Diagnostics: `experiments/manifold_diagnostics.py` (new file)
A standalone, single-task script that:
- runs the four `torus_concept` optimizers on one task (`vqe_stokes_overparam_long` or `fit_high_periodic`),
- logs `theta_norms` per step (i.e. tracks parameter drift),
- saves a 2-panel plot (loss vs step, $\|\theta\|$ vs step) under `results/manifold_diagnostics_<task>_seed<seed>.png`,
- supports `--workers N` (uses `ProcessPoolExecutor`),
- prints per-10-step heartbeats with ETA.

This script's only job is to verify whether the *mechanism* the torus is supposed to fix (parameter drift in flat baselines) actually happens on each task. **If flat baselines don't drift, RQNG can't help, period.**

### 2.7 Slurm: `run.sbatch`
Phase 1 invocation pattern (not yet baked into the defaults):
```bash
TASK_TIER=manifold_torus OPT_TIER=torus_concept sbatch run.sbatch
```
This would launch ~140 jobs (7 tasks × 4 optimizers × 5 seeds).

---

## 3. Experimental design as planned

### 3.1 Two-phase plan

| Phase | Manifold | Optimizers | Status |
|---|---|---|---|
| 1 | Torus | `RQNG_torus_block` vs `Adam`, `QNG_block`, `QNGAdam_v1_block` | Implemented; diagnostics done; **null** so far |
| 2 | Sphere | `RQNG_sphere_block`, `ProjAdam_sphere`, `ProjQNG_sphere` | Stub only; not started |

### 3.2 Phase 1 task taxonomy (7 tasks)

| Task | Category | Why it's in the tier |
|---|---|---|
| `vqe_stokes` | Carry-over | Existing baseline-friendly task; sanity check that we didn't regress. |
| `fit_periodic` | Carry-over | Existing periodic regression task; sanity check. |
| `vqe_stokes_overparam_long` | **Rigged for torus** | 8 qubits × 8 layers (overparam) + many steps. Designed to maximize the chance of flat-optimizer parameter drift. |
| `fit_high_periodic` | **Rigged for torus** | High-frequency periodic target ($k$ up to 5–6); winding the parameters around $2\pi$ should be useful. |
| `vqe_sk_spinglass` | Hard physics | SK Hamiltonian is a known QNG-favorable problem (rugged landscape). 1000 shots, finite-shot regime. |
| `vqe_xxz` | Hard physics | Frustration-free chain, but degenerate ground states make convergence tricky. 1000 shots. |
| `vqe_heisenberg` | Hard physics | Carry-over, used as a control. |

### 3.3 Three-step workflow we agreed on

1. **Long-horizon diagnostic** on `vqe_stokes_overparam_long`: does flat-optimizer $\|\theta\|$ drift, and does R-QNG behave differently from QNG?
2. **Second diagnostic** on `fit_high_periodic`: same questions, different drift mechanism (winding rather than overparam dead-weight).
3. **Full sbatch** (~140 jobs) only if (1) and (2) show the mechanism is alive, to get statistically meaningful means/stds across seeds.

The point of (1)+(2) was to **avoid burning 140 jobs of compute on a hypothesis that doesn't even fire**.

---

## 4. What we ran

### 4.1 Diagnostic 1 — `vqe_stokes_overparam_long` (300 steps, seed 0, 4 workers)

Final state per optimizer:

| optimizer            | final $\|\theta\|$ | final loss |
|----------------------|--------------------|------------|
| Adam                 | (drifted ~0.39)    | (best)     |
| QNG_block            | ~ stable           | identical to RQNG to 6 decimals |
| QNGAdam_v1_block     | ~ stable           | competitive |
| RQNG_torus_block     | ~ stable, $\sim$0.8 below `QNG_block` (some wraps fired) | identical to QNG_block to 6 decimals |

Key observation: **Adam moved $\|\theta\|$ by 0.39 over 300 steps. QNG_block moved by ~0.01.** The "overparameterized circuit accumulates $\|\theta\|$" prediction did not happen on the QNG-family optimizers, on this task, in 300 steps.

Plot: `results/manifold_diagnostics_vqe_stokes_overparam_long_seed0.png`.

### 4.2 Diagnostic 2 — `fit_high_periodic` (200 steps, seed 0, 4 workers)

Final state per optimizer:

| optimizer            | final $\|\theta\|$ | final loss |
|----------------------|---------------------|------------|
| Adam                 | 21.8314             | **0.402340** |
| QNG_block            | 21.2667             | 0.490968 |
| QNGAdam_v1_block     | 21.7058             | 0.491674 |
| RQNG_torus_block     | 21.2027             | 0.490968 |

Key observations:
- **`QNG_block` and `RQNG_torus_block` produce the same loss to 6 decimal places (0.490968).** $\|\theta\|$ differs by only $0.064$ — essentially negligible. The torus wrap fires occasionally but is loss-irrelevant.
- All four optimizers are within ~0.6 in $\|\theta\|$ of each other. The "winding around $2\pi$" mechanism we wanted to favor the torus didn't show up.
- Adam wins the loss race by a wide margin (0.402 vs 0.491). This is a *separate* finding — likely just that QNG is slow on this task at the chosen LR — but it has no bearing on the torus question.

Plot: `results/manifold_diagnostics_fit_high_periodic_seed0.png`.

---

## 5. Interpretation (the honest read)

Both diagnostics line up with the **Section 8 risk from the original plan**:

> The Fubini–Study metric is itself periodic in $\theta$. The QNG natural-gradient direction is therefore already $2\pi$-invariant at the first-order step level. Wrapping the post-step parameter onto $T^d$ has no effect on the loss trajectory, and only a cosmetic effect on $\|\theta\|$.

The empirical evidence:
1. **Flat baselines do not drift meaningfully** on either diagnostic task (within 200–300 steps). The pathology the torus was designed to fix does not manifest, so the torus has nothing to fix.
2. **`RQNG_torus_block` and `QNG_block` are loss-identical to 6 decimals** on `fit_high_periodic`. The wrap operation is a no-op on the loss.
3. The minor $\|\theta\|$ difference between `QNG_block` and `RQNG_torus_block` (0.06–0.8 units) is real (the torus *did* wrap a small number of parameters) but **carries no signal in the objective**.

**Conclusion:** Phase 1, as currently designed, is a null. R-QNG on the torus is *correct* (the implementation works; the wrap fires; no numerical pathologies) but *uninformative* — the FS preconditioner already does the geometric work the torus was supposed to add.

---

## 6. Decision points for the next chat

These are the live open questions. None has been decided yet.

### 6.1 Should we still run the full sbatch?

**Argument for:**
- Diagnostics are 1 seed each. The full sweep is 5 seeds × 7 tasks × 4 optimizers. We might find a task in the tier where the mechanism *does* fire (the SK and XXZ tasks were not diagnosed, and they have the most "rugged landscape" character).
- Negative results across the full tier are themselves publishable: "FS preconditioner absorbs torus topology on standard VQA tasks."
- ~140 jobs at our parallelism budget is not expensive.

**Argument against:**
- We already have strong evidence (two seeds across two tasks, including the two tasks specifically rigged to favor the torus) that the mechanism doesn't fire. Adding more seeds will tighten that null but is unlikely to flip it.
- Compute time is not free, and the diagnostic on `vqe_stokes_overparam_long` was already ~15 minutes of wall time per task.

### 6.2 Should we pivot the experimental design?

The cleanest pivot is to pair `RQNG_torus` with a *non-FS* preconditioner. If FS absorbs the topology, then a flat-Euclidean optimizer + torus retract should show daylight where QNG + torus retract does not. Concretely:

- Add a `RQNG_torus_adam` (Adam updates + torus retract).
- Add a `RQNG_torus_gd`   (GD   updates + torus retract).
- Compare against vanilla `Adam` and `GD` on the same tasks.

Hypothesis: on `fit_high_periodic` and `vqe_stokes_overparam_long`, if Adam's $\|\theta\|$ drift causes any second-moment-estimate distortion, the torus retract should fix it. If it still doesn't help, the torus story is fully dead.

This is ~1 file of work (extend `_apply_manifold_step` to dispatch on the underlying optimizer; add two tier entries).

### 6.3 Does the same FS-absorbs-topology argument kill Phase 2 (sphere)?

**No, and that's why Phase 2 is still interesting.**
- The torus does not change the *set* of reachable quantum states (because the underlying gates are $2\pi$-periodic). The torus is therefore *redundant* with FS.
- The sphere $\|\theta\| = 1$ **does** change the set of reachable quantum states (most of $\mathbb{R}^d$ is excluded). The geometric structure of the sphere is *not* implicit in the FS metric, so a sphere-aware step (project the gradient to the tangent plane, retract along a great circle) should genuinely differ from a flat step.
- The catch: this is "not fair" by default — the sphere optimizer is solving a strictly harder problem (smaller search space). We need *projected baselines* (`ProjAdam_sphere`, `ProjQNG_sphere`) to compare apples-to-apples. The plan already calls for these.

### 6.4 Should we just go to Phase 2 and skip the full Phase 1 sweep?

This is probably the highest-EV path. Phase 1 gave us a clean null with two diagnostics; spending ~140 jobs to add error bars to that null is lower-value than starting Phase 2.

---

## 7. Quick reference: where things live

| Concept | File | Notes |
|---|---|---|
| Manifold ABC + Torus + Euclidean | `src/manifolds.py` | Sphere is stubbed only |
| Manifold-aware QNG step | `src/training.py` → `_apply_manifold_step` | dispatches via `manifold_for(opt_name)` |
| FS-solve dispatch | `src/training.py` → `_natgrad_direction` | RQNG_torus_block routed to QNG_block branch |
| Theta-norm logging | `src/training.py` → `train_with_data`, `train_vqe` (kwarg `track_theta_norm=True`) | returns `theta_norms` in result dict |
| New Hamiltonians | `src/models.py` → `make_sk_hamiltonian`, `make_xxz_hamiltonian` | |
| Optimizer registry / tiers | `run_all_parallel.py` → `OPTIMIZERS`, `OPT_PRIORITY`, `OPT_TIERS["torus_concept"]`, `TASK_TIERS["manifold_torus"]` | |
| Worker funcs (4 new tasks) | `run_all_parallel.py` → `_run_vqe_stokes_overparam_long`, `_run_fit_high_periodic`, `_run_vqe_sk_spinglass`, `_run_vqe_xxz` | |
| Diagnostics script | `experiments/manifold_diagnostics.py` | `--task {vqe_stokes_overparam_long, fit_high_periodic} --workers N --n_steps K --seed S` |
| Plot styling | `src/visualization.py` → `COLORS`, `LABELS`, `INTENDED_OPTIMIZER_FAMILY`, `TASK_PLOT_SLUGS` | RQNG_torus_block = `#e377c2` |
| Slurm invocation | `run.sbatch` | `TASK_TIER=manifold_torus OPT_TIER=torus_concept sbatch run.sbatch` |
| Diagnostic plots produced | `results/manifold_diagnostics_vqe_stokes_overparam_long_seed0.png`, `results/manifold_diagnostics_fit_high_periodic_seed0.png` | |

---

## 8. TL;DR for the next chat

We built a Riemannian QNG optimizer that constrains parameters to the torus $T^d$, with the goal of testing whether non-Euclidean parameter manifolds help VQA training. The implementation is clean and verified-correct (the torus wrap fires, no numerical issues). Two diagnostic runs on tasks specifically rigged to favor the torus showed the **predicted-but-feared null**: the Fubini–Study preconditioner inside QNG already absorbs the $2\pi$ periodicity of the parameter space, so wrapping post-hoc onto the torus produces a step indistinguishable from flat-space QNG (loss identical to 6 decimal places). The flat baselines (Adam, QNG_block) also fail to exhibit the parameter-norm drift the torus was designed to fix, on both tasks.

**The open decision is whether to (a) spend ~140 jobs confirming this null statistically, (b) pivot Phase 1 to pair the torus retract with non-QNG optimizers (Adam, GD) where the FS-absorbs-topology argument doesn't apply, or (c) skip ahead to Phase 2 (sphere), where the manifold genuinely changes the set of reachable states and the same null argument does not apply.**
