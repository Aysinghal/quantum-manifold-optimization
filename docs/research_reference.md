# Research Reference

A light-touch conceptual index of what currently exists in the project — optimizers, tasks, manifolds, and how to invoke them. Companion to [RESEARCH_PATH.md](RESEARCH_PATH.md), which carries the *why*; this file carries the *what*.

This is deliberately high-level. No file paths, no per-task hyperparameter tables, no color/label maps, no code-path branches. If you need any of that, read the code (or the checklists in [AGENTS.md](../AGENTS.md)).

---

## Optimizer family map

Three families. The role each plays in the comparison is what determines when to use it, not the update rule.

### Flat baselines

The reference points. They live on Euclidean parameter space (`R^d`) and run with no manifold structure.

- **GD** — vanilla gradient descent. Diagnostic baseline. Use it to sanity-check that any "preconditioner helps" claim isn't actually a "lr is wrong" claim.
- **Adam** — momentum + per-parameter adaptive learning rate. The strongest non-geometric optimizer in the comparison; the bar that any geometry-aware story has to clear.
- **QNG (block)** — quantum natural gradient with a block-diagonal Fubini–Study metric. Fast enough to be a default; this is the QNG you compare against in practice.
- **QNG (block, lr ablation variants)** — same optimizer, larger learning rates. Exists because preconditioners change the effective step-size scale; the ablation isolates "is the preconditioner helping or is the lr wrong?"
- **QNG (full)** — full Fubini–Study metric, not block-diagonal. Opt-in only. It is brittle on circuits larger than four qubits with shot noise — the full metric inversion amplifies the noise. Useful as a baseline only when you specifically want to demonstrate that brittleness or run an analytic-mode comparison; never a default.

### QNG hybrids

The QNGAdam family — momentum + per-parameter adaptive learning rate + QNG preconditioning. Each combines two ideas drawn from different literature lineages, but **the basic construction is published prior art** (Vu, Le, Pham, Nakashima, IEEE OJCS 2025; see [RESEARCH_PATH.md](RESEARCH_PATH.md) Chapter 0). On the existing rigged task suite they win or tie everywhere; that result is **methodology evidence** that the rigged-and-cousin suite separates optimizer families consistent with published findings, not a novelty claim.

- **Momentum-QNG** — heavy-ball momentum on top of QNG. Reproduces Borysenko et al. (2024); also stock PennyLane as `qml.MomentumQNGOptimizer`. We keep our own implementation for fair comparison, not because the optimizer itself is novel.
- **QNGAdam-v1 (Adam-then-precondition)** — compute the Adam direction first, then apply the QNG preconditioner. Reduces to Momentum-QNG when the second-moment scaling is disabled (correctness check). Equivalent in spirit to Vu et al.'s "QNG-QFIM-Adam."
- **QNGAdam-v2 (precondition-then-Adam)** — apply the QNG preconditioner first, then run the Adam moments on the natural gradient. **The v1/v2 ordering distinction is the small refinement that's still ours** — Vu et al. lump everything into one "QNG-QFIM-Adam" without distinguishing orderings. Non-commutative at any non-infinitesimal step size, so it's a real distinction; just not a publishable headline on its own.

The flat-space sweep against these hybrids is on the symposium poster as a methodology panel ("the rigged-and-cousin suite reproduces Vu et al. and adds the v1/v2 ordering distinction"), not as a contribution. The contribution lives at the parameter-manifold layer.

### Manifold-aware

Riemannian-QNG variants and their projected baselines. These are the live research vector.

- **R-QNG (torus)** — QNG with parameters constrained to `T^d = (S¹)^d`. Tested. Confirmed null (loss-identical to flat QNG to ~1e-15). Exists in the codebase as a documented negative result; do not rerun without a new hypothesis.
- **R-QNG (sphere)** — QNG with parameters constrained to a sphere of natural-scale radius. Project gradient to the tangent space, solve with the FS metric, reproject the update, retract along the geodesic. **Tested (Phase 2). Partial null:** loses to `Projected Adam (sphere)` on every one of the 8 sphere-tier tasks, including the rigged-for-sphere `vqe_overparam_heis` (-2.78 vs -8.000). The geometric machinery is not earning its keep over the simpler project-and-retract construction.
- **Projected QNG (sphere)** — flat QNG step followed by sphere retraction. The "make plain QNG aware of the constraint" baseline. Required for a fair sphere comparison.
- **Projected Adam (sphere)** — flat Adam step followed by sphere retraction. The "make Adam aware of the constraint" baseline. Without it, any sphere result is confounded by Adam's larger search space. **In Phase 2 it beat plain `Adam` on the hard physics tasks** (`vqe_xxz` -11.24 vs -10.22, `fit_high_periodic` 0.4578 vs 0.4738), showing that the projection step itself functions as a regularizer — independent of any geometry-aware-optimization claim.
- **R-QNGAdam (sphere)** — QNGAdam-v1 update done on the tangent space (gradient pre-projected before the Adam moments and FS solve), then geodesic retraction. **Tested (Phase 2.5a).** Did not diverge anywhere; the deepest-stack stability fear from the plan's risk register did not fire. But it loses to `Projected QNGAdam (sphere)` on 7 of 8 tasks and shows ~80× larger seed-to-seed std on `vqe_heis_ring`. Same "geometric machinery null on top of a working projection" pattern as R-QNG-on-sphere, replicated for the QNGAdam family.
- **Projected QNGAdam (sphere)** — flat QNGAdam-v1 step followed by sphere retraction. The "make our strongest flat optimizer aware of the constraint" baseline. **In Phase 2.5a it beat flat `QNGAdam_v1_block` on 4 of 8 sphere-tier tasks** (`vqe_heis_ring`, `vqe`, `vqe_sk_spinglass`, `vqe_overparam_heis`), tied within seed-noise on three more, lost only on `fit_multifreq1d`. Together with `Projected Adam (sphere) > Adam`, this shows projection-as-regularization is not a "weak optimizers only" effect — it generalizes to the strongest flat optimizer in the comparison.

- **Hyperbolic lineup** (`ProjAdam_hyperbolic`, `ProjQNG_hyperbolic`, `RQNG_hyperbolic_block`, `ProjQNGAdam_v1_hyperbolic`, `RQNGAdam_v1_hyperbolic`). The Phase-2 sphere lineup transplanted onto the hyperboloid, on a shared `lr = 0.05` so the cross-manifold comparison (Euclidean vs. sphere vs. hyperbolic) is apples-to-apples. **Pilot only** — see the Hyperbolic entry in the manifold map below.

Whenever a manifold *restricts the reachable state set* (sphere does, torus does not), the projected-baseline variants are mandatory companions to the R-QNG variant. This is non-negotiable; an unprojected comparison conflates "different update rule" with "different reachable states."

---

## Task design philosophy

Tasks are organized into three tiers with different roles.

### Tier 1 — Rigged, easy

Each task is engineered so that *one* optimizer family should clearly win it. If the rigged optimizer doesn't win its rigged task, the implementation is wrong; that's the smoke test for whether each family is alive.

- A task rigged for vanilla QNG: a Stokes-style Hamiltonian on an overparameterized circuit, where most parameters are physically dead and the metric-tensor preconditioner has the most leverage.
- A task rigged for Momentum-QNG: a frustrated ring (e.g. periodic Heisenberg chain), rugged but smooth, where bare QNG plateaus and momentum unsticks it.
- A task rigged for QNGAdam: multi-frequency regression with heterogeneous gradient magnitudes across parameters, where Adam's per-knob scaling matters and the encoding's geometry still helps.
- A task rigged for Adam: high-noise classification (or similar pure-rugged-optimization setup), where there's no exploitable geometric structure and momentum alone is what wins.

### Tier 2 — Same flavor, harder

Each Tier 2 task is its Tier 1 sibling with a difficulty knob turned up: more shots, more qubits, more layers, more disorder. The role is to test whether each Tier 1 sweet-spot effect survives realistic stress.

### Tier 3 — Generally hard

Tasks that don't favor any one family: 11-qubit transverse-field Ising VQE, XXZ chains, MaxCut, large spin glasses, real-data classification problems. Their role is to be the honest benchmark — the regime where the "rigged" results have to actually translate to something useful.

### Manifold-experiment task tiers

For the R-QNG experiments we re-use the sweet-spot principle. The manifold task tier always includes:

- A *carry-over* bucket from the existing tiers, as a sanity check that the manifold optimizer didn't regress on tasks we already understand.
- A *rigged-for-this-manifold* bucket: tasks specifically constructed so the manifold's mechanism should fire. (For torus: long-horizon overparameterized circuits and high-frequency periodic targets. For sphere: overparameterized Heisenberg ring, where the optimization is otherwise massively underdetermined.)
- A *hard-physics* bucket: spin glasses, XXZ, Heisenberg. The "does it help on real problems" check.

### Tasks deliberately excluded

A few task types have been considered and rejected. They are not in the suite, and a future chat should not propose adding them without a new reason:

- Saturated tasks where every optimizer clusters within a percentage point or two (low-noise `make_moons`, MNIST 0v1 with PCA reduction). They don't separate any optimizer of interest.
- Single-qubit or two-qubit toy ansätze that have no parameter redundancy. The preconditioner story doesn't have leverage there.
- Anything that requires per-task hyperparameter retuning to "make a point." If the comparison only works at one specific learning rate, it's not a comparison.

---

## Manifold map

Four manifolds currently in the codebase, each testing a distinct conceptual claim.

- **Euclidean.** The default; the absence of any constraint. Every flat optimizer lives here.
- **Torus** (`T^d = (S¹)^d`). Tests the conceptual claim *parameter-space topology coming from gate periodicity is an optimization-relevant structure*. Confirmed false: the FS preconditioner is itself `2π`-periodic in every gate angle and silently absorbs torus topology. The torus is documented in the codebase as a null result, not as an active experiment. Do not propose new torus experiments without a new mechanism.
- **Sphere** (`S^(d−1)`, one big sphere over the concatenated parameter vector). Tests the conceptual claim *restricting the reachable quantum states changes optimization dynamics*. **Tested (Phase 2 + Phase 2.5a). Partial-null outcome, replicated across two optimizer families:** both the R-QNG-on-sphere and R-QNGAdam-on-sphere geometric machinery are null on every task (including the rigged one), but the projection step alone helps both `Adam` and `QNGAdam_v1_block` on hard physics tasks. The sphere is the first manifold whose effect is *not* implicit in the FS metric, because the FS metric is a property of the state space and cannot know that the parameter space has been restricted — so the partial-null is a real finding about *which component of the manifold-aware construction* (geometry vs. projection) does useful work, not about whether the FS metric absorbs sphere structure. Sphere invariant verified to ~1e-13 max drift across all (task, optimizer) combinations.

- **Hyperbolic** (`H^d`, Lorentz / hyperboloid model, curvature `c = 0.01`). Tests the conceptual claim *negative-curvature parameter geometry suits hierarchical-interaction Hamiltonians*. Implemented in the codebase (five optimizers mirroring the Phase-2 sphere lineup, reachable via the `hyperbolic_concept` optimizer tier). **Run as an exploratory Phase-2.5b pilot only, not as a full arm of the study:** the pilot showed projection-style hyperbolic variants generally stronger than the corresponding Riemannian hyperbolic variants, consistent with the sphere's geometry-vs-projection split, but the outcomes are feasibility evidence rather than performance claims. The retract applies a growth-bounded leash (`max_growth = 0.05`, so geodesic distance is capped at `L_cap ~ 0.31`) to bound per-step radial inflation. Mechanism story still conditional — see [HYPERBOLIC_SCOPING.md](HYPERBOLIC_SCOPING.md). **Distinct from Tran-Dinh 2025 (hyperbolic-NQS for VMC); ours is hyperbolic-on-PQC for VQE.**

Manifolds we have considered and rejected (or deferred indefinitely):

- *Per-layer sphere* (one sphere per circuit layer rather than one big sphere). Live only as a fallback if the global sphere fails uniformly; mixed sphere results would not justify it.
- *Stiefel, Grassmann.* Deferred to a follow-up. Adjacent prior art exists (Hauru/Van Damme/Haegeman 2021 for tensor networks; arXiv:2402.07100 for many-eigenstate VQE) but neither does Stiefel-as-parameter-manifold over a generic ansatz. Out of scope for the symposium.
- *Product manifolds derived from gate-angle periodicity in any guise.* Same FS-absorbs-topology argument as torus.

A full design-space map (with FS-absorption status per manifold) lives in [RESEARCH_PATH.md](RESEARCH_PATH.md) Chapter 5. New manifold proposals must add a row there before any code is written.

---

## Invocation cookbook

The unified runner takes `--tasks` and `--optimizers` flags that accept either tier names or individual names (or any mix); the slurm wrapper exposes them as `TASKS` and `OPTS` env vars. Four canonical invocations cover the experiments described in this project.

**Sanity baseline.** The original four-task suite under the standard QNG sweep. Used to verify the runner is healthy and to compare against historical numbers.

```bash
TASKS=sanity OPTS=qng_full_sweep sbatch run.sbatch
```

**Full QNGAdam-vs-flat comparison.** The honest benchmark for the flat-space comparison: the rigged Tier 1 tasks under the QNG sweep including the QNGAdam hybrids. This is the run that produced the "QNGAdam wins or ties everywhere" result that motivated the manifold pivot.

```bash
TASKS=qng_advantage OPTS=qng_full_sweep sbatch run.sbatch
```

**Manifold experiment, full sweep.** Used for Phase 1 (torus), Phase 2 (sphere, base optimizer set), and Phase 2.5a (sphere row completion with QNGAdam). All have run and are documented in [RESEARCH_PATH.md](RESEARCH_PATH.md) Chapter 1 — torus is a clean null; sphere is a partial null (R-style machinery null across both QNG and QNGAdam families, but projection regularizes both `Adam` and `QNGAdam_v1_block`).

```bash
TASKS=manifold_sphere OPTS=sphere_concept sbatch run.sbatch
```

**Single-task diagnostic.** One seed, one task, the relevant manifold optimizer set. Produces a `theta.png` trajectory plot alongside the standard outputs. The cheap go/no-go before launching a full sweep.

```bash
TASKS=<one_task> OPTS=<concept_tier> SEEDS=0 sbatch run.sbatch
```

The default analytic mode (`SHOTS=0`) is appropriate for the rigged comparisons, where shot noise would muddy a negative result. Override with `SHOTS=1000` (or any other value) for a hardware-realistic run; this is the appropriate setting for the hard-physics tasks and for any comparison that wants to claim "geometry pays for itself under noise."
