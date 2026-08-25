# Hyperbolic-on-PQC scoping

**Status:** scoping draft. No code has been written. The implementation in [src/manifolds.py](src/manifolds.py) and [src/training.py](src/training.py) is gated on this document clearing review (the four questions below answered cleanly enough that a future chat can read this and tell whether hyperbolic-on-PQC is defensible for the symposium).

**Why this exists.** [RESEARCH_PATH.md](RESEARCH_PATH.md) Chapter 5 maps every candidate parameter manifold to four columns: mechanism, state-restricting-or-not, FS-absorption status, prior art. Hyperbolic is in there as "candidate, mechanism unclear." This document either fills in the columns clean enough to motivate Phase 3, or it concludes "the mechanism is fuzzy, park it."

The companion criterion from Chapter 5: **"the only manifold interventions worth running are state-restricting ones."** The torus null established that any manifold whose only structure is gate-angle periodicity is absorbed by the FS preconditioner for free. So the first question for hyperbolic is whether it restricts the reachable state set in a way the FS metric cannot see.

---

## Q1. What is the mechanism?

Why would constraining circuit parameters to a hyperbolic manifold help any specific class of VQE problems?

Hyperbolic geometry has one well-established empirical signature in classical machine learning: it embeds tree-structured / hierarchical data with much lower distortion than Euclidean space ([Ganea et al. 2018](https://papers.neurips.cc/paper/7780-hyperbolic-neural-networks.pdf), [arXiv:2006.08210](https://arxiv.org/pdf/2006.08210)). The mathematical reason is that hyperbolic volume grows exponentially with radius, so a tree-like structure with branching factor $b$ embeds nearly-isometrically into $\mathbb{H}^d$. This is also the reason Tran-Dinh et al. 2025 ([arXiv:2505.22083](https://arxiv.org/abs/2505.22083), EPJ Plus 2026) found that hyperbolic GRU outperforms Euclidean GRU as a *neural quantum state* (NQS) on hierarchical Hamiltonians — Heisenberg $J_1 J_2$ and $J_1 J_2 J_3$ — where the *Hamiltonian's* interaction graph has hierarchy.

For our setup (PQC parameters, not NQS network biases), the mechanism story has to be one of two:

- **(M1) Hamiltonian-driven hierarchy.** The target Hamiltonian has hierarchical interaction structure (next-nearest-neighbor + next-next-nearest-neighbor couplings, or a tree-tensor-network ansatz, or a fractal lattice). The variational state's natural correlation structure is then tree-like, and an ansatz parameterization that lives on $\mathbb{H}^d$ would (in principle) match this structure with fewer parameters than a Euclidean parameterization. This is the analog of Tran-Dinh's mechanism, ported from NQS to PQC.
- **(M2) Ansatz-driven hierarchy.** Some ansätze are explicitly hierarchical — MERA-style ansätze, hierarchical-circuit ansätze (HEAs with skip layers), or anything tensor-network-derived. Their natural parameter geometry is *not* a flat $\mathbb{R}^d$; constraints between layers produce tree-like dependencies. A hyperbolic parameter manifold could match this dependency structure.

**Honest assessment.** M1 is a hypothesis, not a derivation. The argument "Hamiltonian has hierarchy, therefore parameters want to live on hyperbolic space" has at least two missing steps: (a) hierarchy in the *Hamiltonian's interaction graph* does not automatically imply hierarchy in the *circuit-parameter dependency graph*, and (b) even if it did, Euclidean parameterization of a sufficiently expressive ansatz can already represent the hierarchical state — the question is whether hyperbolic parameterization makes optimization *easier*, not whether it makes the state *representable*. M2 is closer to a derivation but it requires us to either pick or build a hierarchical ansatz, which is its own design problem.

For the symposium, the most defensible mechanism story is **M1 + targeted hierarchical Hamiltonians + standard ansatz**, with the caveat that this is an empirical hypothesis ("hierarchical-Hamiltonian VQE problems are the regime where hyperbolic-on-PQC has the strongest *a priori* claim, by analogy with Tran-Dinh's NQS result"). If hyperbolic helps on these and not on flat (non-hierarchical) Hamiltonians, that's interpretable. If it doesn't help anywhere, the mechanism is dead.

**Verdict on Q1.** Mechanism story is *defensible but not airtight*. We can run it as an empirical hypothesis on Heisenberg $J_1 J_2$ / $J_1 J_2 J_3$ tasks. Whether the symposium poster gets a Branch C (hyperbolic positive) headline depends on whether the empirical signal is clean.

---

## Q2. How do we differentiate from Tran-Dinh et al. 2025?

[arXiv:2505.22083](https://arxiv.org/abs/2505.22083), EPJ Plus 2026 — *Hyperbolic recurrent neural network as the first non-Euclidean NQS ansatz.*

Three concrete differences:

| Axis | Tran-Dinh et al. | Us (proposed) |
|---|---|---|
| Variational substrate | Neural-quantum-state (a classical neural network whose output samples the wavefunction) | Parameterized quantum circuit (a quantum-hardware-runnable circuit) |
| Algorithmic framework | Variational Monte Carlo (sample-and-reweight on a classical neural network) | Variational Quantum Eigensolver (quantum circuit simulation or hardware execution) |
| What's hyperbolic | Biases of the GRU layer (the rest of the network is Euclidean) | Circuit-parameter angles, with the FS metric still doing the state-manifold preconditioning |
| Optimizer | RSGD (Riemannian SGD) on Poincaré ball | R-QNG with hyperbolic retraction; projected baselines for fair comparison |
| Evaluation | Ground-state energy via VMC sampling | Ground-state energy via expectation values on the simulated/real circuit |

These differences are **substantive**, not framing — VMC and VQE use different algorithmic primitives, run on different hardware regimes (classical clusters vs. quantum simulators / NISQ devices), and have different bottlenecks (sample variance vs. shot noise + barren plateaus). Tran-Dinh's "first non-Euclidean NQS" claim does not preempt a "first non-Euclidean PQC parameter manifold for VQE" claim, so long as we cite their work as parallel prior art and explicitly position ours as the PQC-side counterpart.

**Verdict on Q2.** Differentiation is clean. Phase 3, if run, is "hyperbolic-on-PQC for VQE," and the literature reference for hyperbolic-on-quantum-something-else is Tran-Dinh et al. 2025 (NQS / VMC).

---

## Q3. What's the fair baseline?

Same logic as the sphere phase. Adding a hyperbolic constraint without a projected baseline conflates "different update rule" (R-QNG with hyperbolic retraction) with "different reachable states" (only points-on-Poincaré-ball, not all of $\mathbb{R}^d$). The minimum baseline set for a fair comparison is:

- **`RQNG_hyperbolic_block`** — the manifold-aware optimizer. R-QNG with hyperbolic projection of the gradient, FS preconditioning on the tangent space, hyperbolic retraction.
- **`ProjQNG_hyperbolic`** — flat QNG step followed by hyperbolic retraction. The "make plain QNG aware of the constraint" baseline. Tests whether the manifold-aware update rule is doing anything beyond the constraint itself.
- **`ProjAdam_hyperbolic`** — flat Adam step followed by hyperbolic retraction. The "make Adam aware of the constraint" baseline. Tests whether flat Adam plus retraction is just as good as hyperbolic R-QNG.

The flat baselines (`Adam`, `QNG_block`, `QNGAdam_v1_block`) carry over from the sphere-phase opt tier, so we can also see whether the hyperbolic-restricted optimization gets anywhere close to unrestricted optimization on the same task.

**Verdict on Q3.** Fair-baseline structure is identical to the sphere phase. No new methodological work needed.

---

## Q4. Coordinate model — Poincaré ball or Lorentz?

Two standard models of $n$-dimensional hyperbolic space:

- **Poincaré ball** $\mathbb{D}^n_c = \{ x \in \mathbb{R}^n : c \\\|x\\\|^2 < 1 \}$ with conformal metric $g^{\mathbb{D}}_x = (\lambda^c_x)^2 I_n$ where $\lambda^c_x = 2 / (1 - c \\\|x\\\|^2)$.
- **Lorentz / hyperboloid** $\mathbb{L}^n = \{ x \in \mathbb{R}^{n+1} : \langle x, x \rangle_{\mathcal{L}} = -1, x_0 > 0 \}$ where $\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum_{i=1}^n x_i y_i$.

Both are isometric (the same hyperbolic geometry under different coordinate charts), so the *mathematical* answer is "either works." The *implementation* answer is dominated by numerical stability:

- [arXiv:2211.00181](https://arxiv.org/pdf/2211.00181) — Poincaré is numerically more prone to gradient vanishing under 64-bit floating point because parameter magnitudes near the ball boundary become very small. Lorentz model is more stable for optimization.
- [arXiv:2405.13979](https://arxiv.org/abs/2405.13979) (Robust Hyperbolic Learning, ICML 2024-style) — confirms Lorentz is preferred when curvature itself is a learnable parameter or when the manifold is used in long optimization runs.
- Tran-Dinh et al. 2025 use the Poincaré ball for compatibility with [Ganea et al. 2018]; they also note that they keep the curvature fixed and rely on a clipping trick near the ball boundary.

For our setup, where the parameters are circuit angles (so they have a natural physical scale of $\sim O(1)$ rather than living near a manifold boundary), either model would work. **Recommendation: Lorentz model** for stability and to make the implementation more directly portable to the larger-parameter regimes we'd want for follow-up work. Curvature $c$ is fixed at $c = 1$ (constant negative curvature, no learnable curvature for the symposium).

The retraction is the exponential map on the chosen model:

- Poincaré: $\exp^c_x(v) = x \oplus_c \tanh\!\left(\sqrt{c} \frac{\lambda^c_x \\\|v\\\|}{2}\right) \frac{v}{\sqrt{c}\\\|v\\\|}$ (Möbius addition $\oplus_c$)
- Lorentz: $\exp_x(v) = \cosh(\\\|v\\\|_{\mathcal{L}}) x + \sinh(\\\|v\\\|_{\mathcal{L}}) \frac{v}{\\\|v\\\|_{\mathcal{L}}}$

Lorentz is also cleaner to differentiate through.

**Verdict on Q4.** Lorentz model, fixed curvature $c = 1$, exponential-map retraction. Numerical stability concerns lean clearly in this direction.

---

## Implementation plan (gated on this doc clearing review)

Per [AGENTS.md](../AGENTS.md):

1. New `Hyperbolic` subclass in [src/manifolds.py](src/manifolds.py) with `name = "hyperbolic"`, `project` (project Euclidean gradient onto Lorentz tangent space at $x$), `retract` (Lorentz exponential map), `init` (sample from a small ball around the origin of the Lorentz model). Wire it into `_OPTIMIZER_MANIFOLDS`.
2. New optimizers in [run_all_parallel.py](run_all_parallel.py) `OPTIMIZERS` registry: `RQNG_hyperbolic_block`, `ProjQNG_hyperbolic`, `ProjAdam_hyperbolic`. Add to `OPT_PRIORITY` (50 / 50 / 10 respectively, mirroring sphere). Add a `hyperbolic_concept` opt tier.
3. Training-loop dispatch in [src/training.py](src/training.py) — branches alongside the existing `sphere` cases in `_apply_manifold_step` / `_retract_adam_step`. The sphere branch is the structural template; the only difference is the retraction map.
4. New task tier `manifold_hyperbolic` in `TASK_TIERS`. Tasks: at minimum one rigged-for-hyperbolic task (Heisenberg $J_1 J_2$ on a small ring; or `vqe_xxz` modified to include next-nearest-neighbor couplings); the existing carry-over set from `manifold_sphere` for sanity.
5. `tests/smoke_all_optimizers.py` extension — for hyperbolic, the invariant to check post-step is $\langle x, x \rangle_{\mathcal{L}} = -1$ (within tolerance).
6. `tests/bench_smoke.py` re-run to time the new optimizer/tier so the heartbeat ETA is honest.

Constants for the new task (e.g. `HEIS_J1J2_N_QUBITS`, etc.) follow the per-task `<TASK>_N_QUBITS / <TASK>_N_LAYERS / <TASK>_N_STEPS` convention.

---

## Decision

Mechanism (Q1) is the only column with real risk. The other three (Q2 differentiation, Q3 baselines, Q4 coordinate model) are settled. **Recommendation: proceed with implementation in parallel with the sphere sweep** under the explicit framing that the rigged hyperbolic task is a Heisenberg $J_1 J_2$ / $J_1 J_2 J_3$ ring (hierarchical-interaction Hamiltonian where Tran-Dinh's mechanism story should port). If Phase 3 is null on these, hyperbolic-on-PQC is parked and the symposium ships with Branches A or B only. If Phase 3 is positive, Branch C becomes the lead.
