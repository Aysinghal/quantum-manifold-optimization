# Research Path

This is the research narrative for the quantum-manifold-optimization project. It's a living document: every chat session can update it. Its job is to make sure a fresh Cursor session can read this top-to-bottom and have the right mental model — what we're chasing, what we've tried, what we've concluded, what's still live, and what's permanently parked.

This file is conceptual. It does not document bug fixes, refactors, registry layouts, or "where things live in the code." For repository mechanics see [AGENTS.md](../AGENTS.md). For a light-touch index of the optimizers, tasks, and manifolds that currently exist, see [research_reference.md](research_reference.md).

---

## Chapter 0 — Prior-art map

A future chat (or future-me) can save weeks by reading this section before claiming anything is novel. Each entry is a one-line summary of what the reference already covers; the conceptual claims in later chapters are bounded against this list.

**Quantum natural gradient and its variants.**

- Stokes, Izaac, Killoran, Carleo (2019), [arXiv:1909.02108](https://arxiv.org/abs/1909.02108), Quantum 4 (2020) 269 — *Quantum natural gradient.* The original QNG paper. Defines the Fubini–Study (FS) metric tensor preconditioner and the block-diagonal approximation. Section 4 explicitly invites momentum-on-top-of-QNG and Adam-on-top-of-QNG as future work.
- Borysenko et al. (2024), [arXiv:2409.01978](https://arxiv.org/abs/2409.01978) — *Momentum-QNG.* Heavy-ball momentum on top of QNG. **Upstreamed to PennyLane as `qml.MomentumQNGOptimizer`** (PR #6240). Once we acknowledged this we stopped framing our `MomentumQNG_block` as novel.
- **Vu, Le, Pham, Nakashima (2025), IEEE Open J. Comput. Soc. 6:1146–1154**, [paper](https://www.computer.org/csdl/journal/oj/2025/01/11072814/28bEUj5er8A), [code](https://github.com/NAIST-Archlab/QNG-Benchmark) — *Benchmarking Variants of the Adam Optimizer for QML.* Includes "QNG-QFIM-Adam" — exactly the QNGAdam construction we built independently — and benchmarks it against GD / Adam / QNG-QFIM-GD / QNG-QFSM-GD on quantum-compilation, VQE, QAOA, and dynamic-simulation tasks. Reports that QNG-QFIM-Adam **ties with Adam** because the Fisher metric collapses to the identity matrix in fully-entangled states. **Implication for us:** the QNGAdam construction itself is no longer novel. Our v1/v2 ordering distinction is at most a footnote-sized refinement (Vu et al. lump everything into one "QNG-QFIM-Adam"). Treat QNGAdam as methodology context, not a contribution.
- Hwang (2024), [arXiv:2405.12807](https://arxiv.org/abs/2405.12807) — *FAdam.* Reframes classical Adam as a diagonal-empirical-Fisher natural gradient. Implication: stacking QNG on top of Adam is "two NGDs in series" rather than "geometric + non-geometric." Sharpens what QNGAdam actually adds.
- QNGGC: Zhang, Wang, Wang (2024), [arXiv:2409.03638](https://arxiv.org/abs/2409.03638) — *Quantum Natural Gradient with Geodesic Corrections.* Adds higher-order Christoffel-symbol corrections to the QNG step on the FS state-manifold. Different layer of "geometry" from ours: theirs improves the *state-manifold* update; ours adds a separate parameter manifold.
- L2O-g†: Halsey, Camilo et al. (2024), [arXiv:2407.14761](https://arxiv.org/abs/2407.14761) — *Learned-to-Optimize* with FS metric features. Trains a small neural network to play the role of an optimizer using QNG features. Different framing entirely.

**Riemannian / manifold-aware quantum optimization.**

- Wang, Liu (2022), [arXiv:2205.12481](https://arxiv.org/abs/2205.12481) — Theoretical analysis of VQE convergence under Riemannian flow on the unit sphere. Establishes that putting state amplitudes on $S^{2^n - 1}$ is a sound Riemannian framework.
- Ferreira-Martins, Farias et al. (2025), [arXiv:2506.17395](https://arxiv.org/abs/2506.17395) — **EGT-CG: Exact Geodesic Transport with Conjugate Gradient.** Uses a hyperspherical-coordinate ansatz where the FS metric is diagonal and has a closed form, removing the need for empirical metric estimation. Hardware-validated up to 16 qubits, in `qiboml`. **The state-of-the-art for sphere-on-VQE.** Implication for us: we cannot beat them on hyperspherical-coordinate ansätze. We compete (or rather, complement) by running on a *generic* ansatz where the metric must be estimated empirically and where projected baselines are needed for fair comparison.
- Wiersema, Killoran (2023), [arXiv:2202.06976](https://arxiv.org/abs/2202.06976) — Riemannian flow on the unitary group $SU(p)$. Different manifold, different framing — we don't directly compete, but it's prior art for "put the optimization on a Lie-group manifold."
- Hauru, Van Damme, Haegeman (2021), SciPost Phys. 10.040, [arXiv:2007.03638](https://arxiv.org/abs/2007.03638) — *Riemannian optimization of isometric tensor networks.* Puts MERA / iMPS isometric tensors on Stiefel and Grassmann manifolds with state-of-the-art conjugate-gradient and quasi-Newton update rules.
- arXiv:2402.07100 (2024) — *Many-eigenstate VQE via Stiefel/Grassmann manifold optimization.* Directly minimizes over the Stiefel/Grassmann manifolds for multi-eigenstate problems. Different framing from ours: they optimize on the manifold; we put parameters on a manifold over a generic ansatz.
- Tran-Dinh et al. (2025), [arXiv:2505.22083](https://arxiv.org/abs/2505.22083), EPJ Plus 2026 — *Hyperbolic recurrent neural network as the first non-Euclidean NQS ansatz.* Uses hyperbolic GRU (Poincaré ball) for variational Monte Carlo; outperforms Euclidean RNN on Heisenberg $J_1 J_2$ / $J_1 J_2 J_3$ where the Hamiltonian's hierarchical-interaction structure matches hyperbolic geometry's tree-likeness. **Important for our hyperbolic scoping (see [HYPERBOLIC_SCOPING.md](HYPERBOLIC_SCOPING.md)):** this is hyperbolic on a *neural-quantum-state* for VMC, not hyperbolic on PQC parameters for VQE. Hyperbolic-on-PQC is open territory; the burden on us is to articulate why circuit angles would benefit from hyperbolic geometry.

**What's left for us, after this prior art.**

1. The **torus-null mechanism** — "the FS preconditioner is itself $2\pi$-periodic in every gate angle, so wrapping parameters onto $T^d$ is redundant." Not articulated in any of the references above, including the QNG-variant papers and the Riemannian-VQE papers.
2. The **two-layer geometry framing** (state manifold = FS, parameter manifold = R-QNG) with an explicit redundancy criterion (FS absorbs topology iff the manifold is non-state-restricting).
3. The **honest task design** (rigged + harder cousin) — none of the QML benchmark papers (Vu et al., Bonet-Monroig et al., Singh et al., Pellow-Jarman et al.) systematically rig per-optimizer-family.
4. The **projected-baseline methodology** for state-restricting-manifold comparisons — EGT-CG and Vu et al. don't use them.
5. **Hyperbolic-on-PQC** — *iff* a defensible mechanism is articulated (see [HYPERBOLIC_SCOPING.md](HYPERBOLIC_SCOPING.md)).

---

## Chapter 1 — Where we are right now

**The question.** Is there a regime where geometry-aware optimization — quantum natural gradient (QNG), or QNG with parameters constrained to a non-Euclidean manifold — genuinely beats Adam on quantum models with realistic shot budgets? Not on a benchmark engineered to make QNG look good. Not on a contrived ansatz. A real, defensible regime where geometry pays for itself.

**What we have today.** A clean QNG implementation; a family of QNG / Adam hybrids (momentum + per-parameter adaptive learning rate + QNG preconditioning, in two orderings — v1 and v2) that we built independently before discovering Vu et al. 2025 had already published the same construction as "QNG-QFIM-Adam" (see Chapter 0); a deliberately-balanced task suite where every optimizer family has at least one task rigged in its favor and at least one harder cousin that strips that favor away; and a Riemannian-QNG framework that lets us drop the variational parameters onto an arbitrary manifold (currently torus and sphere, with a clean abstraction for adding more).

**What we've concluded so far.** On flat parameter space the QNG-Adam hybrids dominate the comparison — too cleanly to be the headline result, since they are a strict superset of the simpler QNG and Adam variants, and since Vu et al. 2025 already report essentially the same finding (QNG-QFIM-Adam ties with Adam in their setup). The flat-space QNGAdam result is now **methodology context**, not a contribution: it demonstrates the rigged-and-cousin task suite separates optimizer families cleanly, but the optimizer construction itself isn't novel. The first non-Euclidean parameter manifold we tested, the torus, is a confirmed null: torus-QNG and flat-QNG are loss-identical (to ~1e-15) on the tasks we deliberately rigged in the torus's favor, because the Fubini–Study (FS) preconditioner is itself periodic in every gate angle and silently absorbs the torus topology. **This torus-null mechanism is genuinely novel** — see Chapter 0.

**The sphere experiment landed and the result is more interesting than either of the binary outcomes we originally anticipated.** Across the 8 sphere-tier tasks × 7 optimizers × 5 seeds sweep (results in `results/2026-04-26_204408__phase2_sphere_concept`), three findings stood out:

1. **The R-QNG-on-sphere machinery is null.** `RQNG_sphere_block` — the geometrically-principled optimizer that does FS-on-tangent then exp-maps along the sphere — lost to the simpler `ProjAdam_sphere` (flat Adam step + sphere retraction) on every single task. Including `vqe_overparam_heis`, the task we deliberately rigged for the sphere intervention to pay off (final losses: ProjAdam_sphere -8.000, RQNG_sphere_block -2.78). The fancy FS-on-tangent step is not earning its keep.
2. **Projection alone helps Adam.** `ProjAdam_sphere` beats plain `Adam` on the hard physics tasks: -11.24 vs -10.22 on `vqe_xxz`, 0.4578 vs 0.4738 on `fit_high_periodic`, with similar margins on `vqe_heis_ring`. So the sphere is not uniformly null — the *projection* component is a useful regularizer; the *geometric machinery* layered on top of it is not.
3. **Geometry-aware optimization buys early-step speed, not asymptotic accuracy.** `QNGAdam_v1_block` reaches -6.20 by step 25 on `vqe_xxz` while flat `Adam` is still at -3.24. `Adam` then catches up by step 150 (-10.22 vs QNGAdam_v1 -9.48). `ProjAdam_sphere` tracks `Adam` early and overtakes it late. A real budget-aware-optimizer-choice axis exists here, independent of the manifold story.

**Phase 2.5a closed the missing-row data gap by adding QNGAdam-on-sphere in both Proj and R variants** (results in `results/2026-04-26_225540__phase2_5a_qngadam_sphere_pair`, 8 sphere-tier tasks × 2 new optimizers × 5 seeds). Two findings:

4. **Sphere projection helps even our strongest flat optimizer.** `ProjQNGAdam_v1_sphere` beats flat `QNGAdam_v1_block` on 4 of 8 tasks (`vqe_heis_ring` -7.99960 vs -7.99957; `vqe` -7.63 vs -7.14; `vqe_sk_spinglass` -5.31 vs -5.25; `vqe_overparam_heis` -7.99908 vs -7.99503), is within seed-noise on three more, and loses only on `fit_multifreq1d` (where it lands 4th, behind both flat `QNGAdam_v1_block` and `RQNGAdam_v1_sphere`). Combined with the `ProjAdam_sphere > Adam` finding from the original sphere sweep, **projection regularizes both weak and strong flat optimizers** — it is not a "weak optimizers only" effect.
5. **The geometry-vs-projection split holds across the QNGAdam family too.** `RQNGAdam_v1_sphere` (FS-solve on tangent before retract) loses to `ProjQNGAdam_v1_sphere` (full QNGAdam in ambient space, retract afterward) on 7 of 8 tasks, with seed-to-seed std a full ~80× larger on `vqe_heis_ring` (0.019 vs 0.00023). The R variant did not diverge anywhere — the worst-case stability fear from the plan's risk register did not fire — but the deeper geometry stack accumulates noise without buying accuracy. The same "geometric machinery null on top of working projection" pattern observed for plain QNG on the sphere now replicates for QNGAdam on the sphere.

The sphere invariant was preserved to numerical precision (max drift ~1e-13 on every (task, optimizer) combination across 100–300 steps), confirming the retract / projection implementation is correct and the partial-null is a real geometric finding rather than an implementation artifact.

**What's now open.** Hyperbolic-on-PQC remains a candidate, conditional on the mechanism story in [HYPERBOLIC_SCOPING.md](HYPERBOLIC_SCOPING.md) holding up empirically.

---

## Chapter 2 — The conceptual arc

Four conceptual moves, in order. Each one is a hypothesis we held, what made it interesting, and what the empirical reality told us. Implementation details are deliberately suppressed.

### Move 1 — From bare QNG to the AdamQNG family

The literature gives two clean stories that, read together, point at an obvious gap.

Stokes et al. (2019) introduced QNG: precondition the gradient by the inverse of the Fubini–Study metric tensor `g_FS(θ)`. The hook is that in the continuous-time limit this is exactly imaginary-time evolution under the Hamiltonian, which is the optimal way to find ground states. The empirical demonstration in the paper is a setup engineered for QNG to win — a tiny `Z₀ ⊗ Z₁` Hamiltonian living on a much larger circuit, where most of the parameters are physically dead and metric-tensor preconditioning has the most leverage. QNG dominates that setup. On more realistic VQE problems with rugged loss landscapes, bare QNG gets stuck on plateaus the way bare gradient descent does.

Borysenko et al. (2024) added heavy-ball momentum on top of QNG. Update rule: `θ_{t+1} = θ_t + ρ(θ_t − θ_{t−1}) − η g_FS⁻¹ ∇L`. This fixes the stuck-on-plateau failure mode by carrying inertia through flat regions of the loss. Their headline empirical result is on a quantum spin glass — the kind of rugged multi-local-minimum landscape where bare QNG was failing.

Read side by side: QNG provides geometric structure and a good preconditioner but is missing momentum and adaptive scaling. Adam provides momentum and adaptive scaling but is missing the geometric structure of the loss landscape. The genuine "Adam-equivalent for QNG" — momentum plus per-parameter adaptive learning rate plus QNG preconditioning — was, at the time we started, the unwritten next step. Stokes' Section 4 explicitly invites this.

We built it. Two orderings, because the composition isn't commutative: QNGAdam-v1 takes the Adam-direction first and then preconditions with `g_FS⁻¹`; QNGAdam-v2 preconditions the gradient first and then runs Adam on the natural gradient. Alongside these we also implemented Borysenko's simpler Momentum-QNG, since QNGAdam-v1 reduces to Momentum-QNG when the Adam second-moment scaling is disabled — a useful correctness check.

**Then literature search caught up with us.** Vu, Le, Pham, Nakashima (IEEE Open J. Comput. Soc. 2025; see Chapter 0) had already published the same construction as "QNG-QFIM-Adam" with a benchmarking suite. They report that QNG-QFIM-Adam ties with Adam in their setup because the Fisher metric collapses to the identity in fully-entangled states. They do **not** distinguish v1 from v2 — they treat the construction as a single optimizer. So the v1/v2 ordering distinction is a small refinement that's still ours, but the headline construction is published prior art.

**What we learned, recalibrated.** On the existing flat-space task suite, the QNGAdam variants win or tie on every task. That sounds great but it is too clean — for two reasons. First, QNGAdam is a strict superset of three other optimizers (Adam, QNG, Momentum-QNG) and winning everywhere mostly tells you the suite isn't testing anything sharp. Second, the result is now consistent with published literature (Vu et al.'s "QNG-QFIM-Adam ties with Adam"), so it is no longer a *novelty* claim but a *replication* claim. Both readings push toward the same conclusion: the flat-space race is the wrong frame for the headline. The QNGAdam sweep functions as **methodology evidence** for the symposium — "the rigged-and-cousin task suite separates optimizer families cleanly, including the published-prior-art QNGAdam family" — not as a contribution. The contribution lives at the parameter-manifold layer.

### Move 2 — Honest task design

Most optimizer-comparison benchmarks accidentally favor one method. Stokes' paper is honest about this in its appendix, where the "fair" iteration-cost-normalized plots are buried. Our initial four-task suite was not honest in that sense — its 11-qubit transverse-field Ising VQE was a rugged landscape that played to Adam's strengths, and its `make_moons` classification task was so saturated that all methods clustered within a few percentage points.

We rebuilt the task suite around a deliberate principle: **every optimizer family needs at least one task rigged to its theoretical sweet spot, and every rigged task should also have a harder cousin that strips that sweet spot away.** Rigged tasks make a positive result interpretable as "yes, the mechanism we expected is alive." Harder cousins make a null result interpretable as "the mechanism does not survive this particular kind of stress."

Concretely:

- *Vanilla QNG sweet spot.* Stokes-style massively-redundant Hamiltonians on overparameterized circuits. The metric-tensor preconditioner has the most leverage when most parameters are physically dead.
- *Momentum-QNG sweet spot.* Frustrated rings with rugged but smooth landscapes — the regime where bare QNG plateaus and momentum unsticks it.
- *QNGAdam sweet spot.* Multi-frequency regression with heterogeneous gradient magnitudes across parameters — the regime where Adam's per-knob scaling matters and the geometry of the encoding still helps.
- *Adam sweet spot.* Rugged spin glasses and high-noise classification — pure rugged optimization with no exploitable geometric structure.

A second tier raises shot count, qubit count, depth, or randomness on the same problems, to test whether each sweet-spot effect survives realistic stress.

**What we learned.** When each optimizer has a sweet-spot task and a harder cousin, the comparison plot becomes a coordinate system rather than a horse race. You can read off "this method is robust along this axis but loses along that axis." That coordinate system is also what makes our subsequent null results (the torus story below) interpretable instead of ambiguous: when a manifold optimizer fails on a task that was specifically rigged in its favor, the mechanism is dead, not the implementation.

### Move 3 — The R-QNG pivot

Once the AdamQNG-vs-baselines picture stabilized, the question stopped being "which optimizer wins" and became "is geometry-aware optimization done?" The remaining novel idea was to attack a different layer of geometry — not the geometry of the quantum state, which QNG already handles, but the geometry of the parameter space itself.

There are two genuinely separate spaces in any variational quantum algorithm:

- The **parameter space**: where `θ` lives. By default it's `R^d` — a flat vector space with no built-in shape. This is the space the optimizer's bookkeeping (gradient steps, momentum buffers, second-moment accumulators) actually lives in.
- The **quantum state space**: as `θ` varies, `|ψ(θ)⟩` traces a curved submanifold of projective Hilbert space `CP^(2^n − 1)`. This is the space the Fubini–Study metric `g_FS(θ)` describes — distances here are state-distinguishability, not parameter-distance.

Standard QNG uses `g_FS` to do steepest descent on the *state* manifold while still treating the *parameters* as flat. Riemannian QNG (R-QNG) composes this with an explicit topology or shape on the parameters themselves: a torus (forcing each gate angle to live in `[0, 2π)`), a sphere (forcing `‖θ‖ = 1`), and so on. The two layers of geometry — state-manifold preconditioner and parameter-manifold constraint — do not redundantly do the same job. The FS metric is a property of the music, the parameter manifold is a property of the dials, and a priori there is no reason for one to absorb the other.

The bet was that flat `R^d` is the wrong default. Gate angles are physically `2π`-periodic; parameters that should compete for a finite norm budget probably want to live on a sphere; parameters that have a hierarchical structure could live on a Stiefel or Grassmann manifold. R-QNG is the framework for testing those bets cheaply: pick a manifold, project the gradient onto its tangent space, precondition with the FS metric restricted to that tangent space, retract along the manifold's geodesic.

### Move 4 — What torus and sphere are actually testing

The two-phase R-QNG plan collapses to a single sharp question, once you state it carefully: **does the FS preconditioner already absorb whatever geometric structure we'd impose on the dials?**

**Torus.** Quantum gates of the form `Rx(θ), Ry(θ), Rz(θ)` are exactly `2π`-periodic — `θ` and `θ + 2π` produce the same quantum state. The FS metric is computed from the state, so it inherits this periodicity automatically: `g_FS(θ) = g_FS(θ + 2π)` in every coordinate. Wrapping the parameters onto a torus `T^d = (S¹)^d` is a topological identification that the FS metric already silently makes. The diagnostic confirmed exactly this. On `fit_high_periodic` — a task we deliberately rigged with a high-frequency periodic target so that "winding the parameters around `2π`" should help — torus-QNG and flat-QNG produced loss trajectories identical to six decimal places. On `vqe_stokes_overparam_long` — eight qubits, eight layers, three hundred steps, the closest thing we have to "this is the regime where flat optimizers should drift `‖θ‖`" — Adam moved `‖θ‖` by less than half a unit total, and the QNG-family optimizers moved it by less than that. The pathology the torus was designed to fix never manifested.

**The torus is redundant with the FS preconditioner.** This is itself a clean conceptual finding, not just a failed experiment. It says the FS metric is doing more topological work than the literature usually frames it as doing. Anyone who proposes a parameter-manifold intervention whose topology arises from gate periodicity is, knowingly or not, proposing something that QNG already does for free.

**Sphere.** `‖θ‖ = 1` is a fundamentally different intervention. It does not just identify equivalent parameter configurations — it *restricts the reachable quantum states*. Most of `R^d` is excluded; only the unit sphere is in scope. The FS metric is a property of the state space and does not know that the parameter space has been restricted. So the sphere is the first manifold intervention with a clean mechanistic claim that the FS preconditioner cannot make redundant.

The catch is that the comparison is no longer automatically fair. Flat Adam on the same task is solving a strictly larger optimization problem, with access to states the sphere optimizer cannot reach. To keep things honest we run **projected baselines**: take a flat Adam step, then retract the result back to the sphere; take a flat QNG step, then retract. Now the only difference between the manifold-aware optimizer and its flat counterpart is the update rule, not the search space. Whether sphere helps becomes a clean question about update rules on a fixed restricted manifold.

Phase 2 ran the full sphere sweep — 8 tasks × 7 optimizers × 5 seeds — and produced the partial-null finding documented in Chapter 1. The sphere does separate the optimizers; the way it separates them is just not the way the "fancy R-QNG geometry" story predicted. The active ingredient is the projection step itself; the FS-on-tangent layer adds nothing on top.

---

## Chapter 3 — What we now believe

These are opinionated claims, distilled from the four moves above, written so that a future chat does not relitigate them. They are also written so that they survive being wrong about the sphere — most of them are about how to *think* about geometry-aware optimization, not about whether any particular manifold wins.

- **Geometry-aware optimization is not one thing.** It decomposes into two layers: geometry on the *state* (the FS metric, implemented as preconditioning) and geometry on the *parameters* (a manifold, implemented as a constraint). Whether mixing them produces anything novel depends on whether the parameter manifold is implicit in the state. If it is, the FS preconditioner already does the work. If it isn't, you have a real mechanism.
- **The FS preconditioner is more powerful than the literature usually frames it.** It silently absorbs any parameter-space topology that the underlying quantum state is invariant to. This rules out a whole class of "manifold-on-the-dials" interventions — anything whose topology comes from gate-angle periodicity — without the experimenter needing to test them individually. The torus null is the first instance of this; analogous arguments apply to product-of-circles topologies of any flavor.
- **The only manifold interventions worth running are state-restricting ones.** Sphere is the canonical example. Hyperbolic, Stiefel, Grassmann are interesting only insofar as they restrict the reachable state set in some non-trivial way. If a proposed manifold doesn't change which states are reachable, it's almost certainly already absorbed by FS and there's no reason to run the experiment.
- **Apples-to-apples with manifolds requires projected baselines, not flat baselines.** The moment a manifold restricts the state set, comparing it against flat Adam conflates two different problems — different update rule and different search space. Projected baselines (flat-update-then-retract) isolate the update-rule effect. This is the methodological cost of every non-trivial manifold experiment going forward; we will not run a manifold sweep without them.
- **Honest task design beats more seeds.** A small task suite where every optimizer has a sweet spot and a harder cousin gives a clearer signal than a large mono-style benchmark with twenty-five seeds and no rigging. Saturation and accidental rigging are the silent killers of QNG papers. We will keep extending the rigged-and-cousin pattern as we add manifolds rather than chasing job count.
- **Momentum + adaptive lr + preconditioner is, empirically, the strongest flat optimizer in the comparison — and is also published prior art.** The QNGAdam family won the flat-space race cleanly on our rigged-and-cousin suite, which serves as a methodology validation. But Vu et al. 2025 already published the same construction (without the v1/v2 ordering distinction) and reported a consistent finding. The contribution from the QNGAdam sweep is no longer "this optimizer wins"; it is "the rigged-and-cousin task suite reproduces and extends a published result." The live research question — whether geometry-aware optimization adds anything beyond what's now published — moves to the parameter-manifold layer.
- **State-restricting manifolds can fail in two distinguishable ways.** First failure mode: the geometric machinery (FS-on-tangent + exp-map) loses to a simpler project-and-retract construction. Second failure mode: even the simple projection doesn't help. Phase 2 + Phase 2.5a both hit the first mode and not the second — projection alone helps both weak and strong flat optimizers, the geometric machinery on top doesn't, and the pattern replicates across optimizer families (QNG, Adam, QNGAdam). Future state-restricting manifold experiments need both comparisons (against projection alone *and* against the strongest flat optimizer of the same family) to be diagnostic.
- **Projection-as-regularization is a real, separate effect from geometry-as-preconditioning, and it generalizes across optimizer strengths.** `ProjAdam_sphere` outperforms plain `Adam` on hard physics tasks; `ProjQNGAdam_v1_sphere` outperforms flat `QNGAdam_v1_block` on the same hard physics tasks. The "projection helps" effect is not a "weak optimizers only" artifact — it survives being attached to the strongest flat optimizer in the comparison. Restricting `‖θ‖ = R*` (at the natural-scale init radius) keeps the optimizer from drifting into bad neighborhoods of `R^d` regardless of which update rule is generating the steps. This is regularization, not geometry-aware optimization, and Chapter 4 / Chapter 5 reflect the broader generality.
- **The geometry-vs-projection split is family-invariant.** Across both QNG-on-sphere (Phase 2) and QNGAdam-on-sphere (Phase 2.5a), the simpler "do the flat update, retract afterward" construction beats the principled "project gradient, FS-solve on tangent, retract" construction. Two independent optimizer families exhibit the same split, against the same baseline (`ProjAdam_sphere`-style projection-only). This is now a robust empirical finding, not an artifact of one update rule's interaction with sphere geometry.
- **Geometry-aware optimization buys early-step speed, not asymptotic accuracy.** `QNGAdam_v1_block` dominates the first ~50 steps on the hard physics tasks (`vqe_xxz`, `vqe`), then loses the long horizon to flat `Adam` variants with adaptive learning rates. The strongest *practical* optimizer is "QNGAdam-like for the first epoch, Adam-like for the rest" — which is approximately what `ProjAdam_sphere` looks like in our data. Budget-aware optimizer choice is a real reporting axis, independent of the manifold story.
- **Null results are evidence, when the task was rigged.** The torus null is interpretable as "the mechanism does not exist on tasks where it should be strongest" precisely because the diagnostic tasks were rigged for it. A null on a non-rigged task would be ambiguous; this one is not. Future manifold experiments should hold themselves to the same standard.

---

## Chapter 4 — Forward path

### Where we sit relative to EGT-CG and Vu et al.

Two pieces of prior art are close enough to ours that the differentiation needs to be explicit. This subsection defines the differentiation in five contrasts so a future chat is not tempted to claim novelty along an axis where prior art already dominates.

**Versus EGT-CG (Ferreira-Martins, Farias et al., [arXiv:2506.17395](https://arxiv.org/abs/2506.17395)):**

| Axis | EGT-CG | Us |
|---|---|---|
| Ansatz | Specialized hyperspherical-coordinate ansatz where the FS metric has a closed form | Generic ansatz; FS metric is estimated empirically |
| Metric tensor | Closed-form, diagonal, no inversion needed | Empirical, block-diagonal, requires inversion (regularized) |
| Update rule | Exact geodesic transport with conjugate gradient memory | Project–solve–retract per step (no memory across geodesics) |
| Baseline comparison | Flat Adam / flat QNG (different search space) | **Projected Adam / projected QNG (same restricted search space)** — apples-to-apples |
| Hardware demos | Up to 16 qubits in `qiboml` | Simulator-only, analytic mode |

The two contributions are complementary. EGT-CG owns the "closed-form geometry on a specialized ansatz" regime; we own the "generic ansatz with apples-to-apples projected baselines" regime. The poster framing should be "EGT-CG asks 'how good can sphere-on-VQE get with a specialized ansatz that has closed-form geometry?'; we ask 'on a generic ansatz with fair baselines, does sphere-on-VQE beat its own update-rule-controlled flat counterpart?'"

**Versus Vu et al. (IEEE OJCS 2025, [paper](https://www.computer.org/csdl/journal/oj/2025/01/11072814/28bEUj5er8A)):**

| Axis | Vu et al. | Us |
|---|---|---|
| Optimizer construction | Single "QNG-QFIM-Adam" — no ordering distinction | v1 (Adam-then-precondition) vs. v2 (precondition-then-Adam), explicitly distinguished |
| Benchmark | Quantum compilation, VQE, QAOA, dynamic simulation — generic QML tasks | Rigged + harder-cousin suite — every optimizer family has a sweet-spot task |
| Manifold dimension | None — all flat-space | Two layers (state-manifold via FS, parameter-manifold via R-QNG) |
| Headline finding | QNG-QFIM-Adam ties with Adam (FS collapses to identity in entangled states) | Same finding on flat-space; **plus** torus-null with FS-absorbs-topology mechanism; **plus** sphere as the first non-redundant intervention |

We are not competing with Vu et al. on flat-space optimizer construction — they got there first. We extend the question one layer up (parameter manifold) and complement with a more principled task-design methodology. The QNGAdam-vs-flat panel is "consistent with Vu et al. 2025"; our actual contributions are at the manifold layer.

### Live thread

Phase 2 + Phase 2.5a together produced the **partial-null** outcome anticipated by neither of the original branches: the sphere's R-QNG / R-QNGAdam machinery is null across both QNG and QNGAdam optimizer families, but the projection step itself functions as a regularizer that helps both `Adam` and the strongest `QNGAdam_v1_block`. See Chapter 1 for the five concrete findings and the underlying numbers.

The "does sphere projection help our strongest flat optimizer?" question is closed by Phase 2.5a: yes, on 4 of 8 sphere-tier tasks `ProjQNGAdam_v1_sphere` beats flat `QNGAdam_v1_block`, with no tasks where the sphere variant is materially worse beyond seed noise.

One question shapes the next move:

- **Is hyperbolic-on-PQC a real intervention or a misalignment?** Hyperbolic geometry on neural-quantum-state ansätze is published prior art (Tran-Dinh et al. 2025); hyperbolic on PQC parameters is not. The mechanism story in [HYPERBOLIC_SCOPING.md](HYPERBOLIC_SCOPING.md) is "the Hamiltonian's hierarchy structure motivates negative-curvature parameter geometry," but whether that survives empirical contact is open.

Independent of either question, the **convergence-speed axis** (geometry buys early-step speed, not asymptotic accuracy) is a self-contained positive finding that survives whatever the manifold experiments resolve to. It belongs on any future writeup as its own panel.

### Parked, with reasons

These are not "to-do later" items. They are directions we have considered and consciously rejected, or directions whose value is conditional on something upstream. A future chat should not propose these without checking the listed condition first.

- **More confirmatory sweeps of QNGAdam vs. flat baselines on the existing tier suite.** The story is closed; QNGAdam wins or ties everywhere. More seeds add error bars to a result we already trust.
- **A statistical confirmation of the torus null.** Two diagnostics on tasks rigged in the torus's favor already established the mechanism. Running the full multi-seed sweep just to tighten error bars around a null is low value.
- **Saturated tasks** (MNIST 0v1 with PCA, low-noise `make_moons`, anything where every optimizer is within a percent or two of every other). They don't separate the optimizers we care about.
- **Per-task hyperparameter tuning.** Confounds update-rule effect with hyperparameter effect. If a new optimizer needs different defaults, that's a research finding worth its own writeup, not a tuning tweak.
- **Hyperbolic-on-PQC.** Live conditional on the mechanism story in [HYPERBOLIC_SCOPING.md](HYPERBOLIC_SCOPING.md) clearing review (Hamiltonian hierarchy ↔ negative-curvature parameter space). The Phase 2 partial-null does *not* close this off — sphere's failure mode was "geometric machinery null on top of a working projection," which is information about *that specific manifold's* geometry-vs-projection split, not a verdict on every state-restricting manifold.
- **Stiefel / Grassmann manifolds.** Live only after both sphere and hyperbolic land with a defensible signal on the QNGAdam family. Otherwise we'd be stacking speculative manifolds on top of the same FS-absorbs-topology argument *plus* the freshly-observed geometry-vs-projection split.
- **Per-layer sphere variants.** Live only if the global sphere fails uniformly across all tasks. Uniform failure would suggest the discretization is wrong; per-layer is the natural retry. Mixed results on the global sphere would not justify it.
- **Gate-noise channels** (depolarizing, amplitude-damping, full hardware-noise simulation). Possible Phase 5 if Phase 2 shows a positive. Otherwise out of frame.
- **Removing QNG-full from the comparison entirely.** Currently opt-in; staying that way. It's poisoned by shot noise on circuits larger than four qubits and only useful as a baseline when running diagnostically.
- **Importing PennyLane's stock `MomentumQNGOptimizer` instead of our manual loop.** Tempting on simplicity grounds, but our manual loop carries fixes the stock optimizer doesn't have, and we want all the QNG variants to share the same code path for fair comparison. Don't switch.

### Operating principles

These are non-negotiable for any future chat working in this repo.

- Default is multi-seed sweeps; single-seed runs are for cheap diagnostics only. A diagnostic produces a go/no-go signal, not a publishable result.
- Manifold-restricting optimizers always ship with projected baselines. No exceptions.
- Optimizer hyperparameters are fixed at paper defaults across the comparison. If a new optimizer needs different defaults, that's a research finding worth its own writeup.
- Tasks must declare which optimizer family they're rigged for, even if that family ends up losing on them. This is how we keep the comparison honest.
- Repository mechanics — how to add an optimizer / task / manifold without breaking the runner — are out of scope for this document. That's [AGENTS.md](../AGENTS.md). Refer to it; do not duplicate it here.
- This document is updated as the research moves. After every experiment that produces a conclusion, the relevant chapter (live thread, what-we-believe, parked) gets edited to reflect it. A doc that doesn't drift is a doc that's lying.

---

## Chapter 5 — Manifold design space

A map of every parameter manifold we have considered, what it tests, whether the FS preconditioner absorbs it, what the prior art is, and what its current status is. New manifold proposals should be placed on this map — and dismissed if they collapse to "just another non-state-restricting topology" — before any code is written.

The unifying criterion is **state-restricting vs. non-state-restricting**. A manifold is *state-restricting* if it removes some quantum states from the reachable set; it is *non-state-restricting* if it only identifies parameter configurations that produce equivalent states. The FS metric is a property of the state space, so it can absorb any non-state-restricting topology silently — it cannot tell that the parameter space has been restricted. This single criterion partitions every candidate manifold into "FS already does this" vs. "this is a real intervention."

| Manifold | Mechanism | State-restricting? | FS absorbs? | Prior art for VQE/PQC | Status |
|---|---|---|---|---|---|
| **Euclidean** $\mathbb{R}^d$ | None — the default | n/a | n/a | All flat-space optimizers | Default |
| **Torus** $T^d = (S^1)^d$ | Identifies $\theta$ with $\theta + 2\pi$ per coordinate | **No** — same reachable states | **Yes** — FS is itself $2\pi$-periodic in every gate angle | None (we believe) | **Done; null. Mechanism documented in Chapter 2 Move 4.** |
| **Sphere** $S^{d-1}$ (one global) | Constrains $\\\|\theta\\\| = R$ at natural-scale init | **Yes** — most of $\mathbb{R}^d$ is excluded | **No** — FS does not know about parameter-norm restrictions | Wang/Liu 2022; **EGT-CG 2025** (specialized ansatz); we add: generic ansatz + projected baselines | **Done; partial null replicated across two optimizer families.** R-QNG-on-sphere and R-QNGAdam-on-sphere machinery both lose to their `Proj*_sphere` counterparts (full Phase 2 + Phase 2.5a). Projection regularizes both weak (`Adam`) and strong (`QNGAdam_v1_block`) flat optimizers on hard physics tasks (positive). Sphere invariant verified to ~1e-13 across all (task, opt) combinations. |
| **Per-layer sphere** | One $S^{d_\ell - 1}$ per circuit layer | Yes (per layer) | No | None for VQE | **Held in reserve.** Live only if global sphere fails uniformly. Mixed sphere results would not justify it. |
| **Hyperbolic** (Poincaré ball / Lorentz) | Negative-curvature parameter space, exponential volume growth | Possibly — depends on coordinate model | No, if state-restricting | Tran-Dinh 2025 for *NQS / VMC* (not PQC); **none for PQC parameters in VQE** | **Candidate.** Mechanism story unclear; conditional on [HYPERBOLIC_SCOPING.md](HYPERBOLIC_SCOPING.md) clearing review. Most plausible motivation: hierarchical-interaction Hamiltonians (Heisenberg $J_1 J_2$, $J_1 J_2 J_3$). |
| **Stiefel** $\mathrm{St}(p, n)$ | Orthogonality constraint on parameter columns | Yes — only orthonormal frames | No, if state-restricting | Hauru/Van Damme/Haegeman 2021 (isometric tensor networks); arXiv:2402.07100 (many-eigenstate VQE) — **both optimize directly on the manifold, not over a generic ansatz** | **Candidate, deferred.** Adjacent prior art. Needs its own scoping doc analogous to hyperbolic. Out of scope for the symposium unless sphere lands cleanly and we have time. |
| **Grassmann** $\mathrm{Gr}(p, n)$ | Subspace-equivalence-class constraint | Yes — only point-on-Grassmann reachable | No, if state-restricting | Same prior art as Stiefel | **Candidate, deferred.** Same reasoning as Stiefel. |
| **Product manifolds** $(S^1)^k \times \mathbb{R}^{d-k}$, etc. | Identifies a subset of gate angles by periodicity | **No** for the $S^1$ factors, possibly yes for other factors | **Yes** for the $S^1$ factors | None | **Dominated by the torus argument.** Don't propose without a state-restricting factor. |
| **Discrete-curvature / learned manifolds** | Curvature adapts from gradient information (analogous to AdaFisher / K-FAC scope) | Depends | Maybe | Speculative; nothing direct | **Out of scope for the symposium.** Research follow-up. |

After Phase 2 + Phase 2.5a the table reflects a **partial-null** outcome that neither of the originally-anticipated binary branches predicted, now confirmed across two independent optimizer families (QNG and QNGAdam): the sphere's R-style geometric machinery is null on every task (including the rigged one) across both families, but the projection step alone regularizes both weak (`Adam`) and strong (`QNGAdam_v1_block`) flat optimizers on hard physics tasks. The implication for future manifold rows is that "FS absorbs?" is no longer the only column that matters — a new analytic axis "geometry-vs-projection split" should be filled in for every state-restricting candidate. Specifically: even if FS does not absorb a manifold, the projection step alone might do all the useful work, and the additional geometric machinery might add nothing — and this prediction now has empirical support against two different update rules, not one.

If a future chat proposes a new manifold, it must add a row here with all six columns filled in — *especially* the FS-absorbs column — and a written hypothesis for whether the projection-vs-geometry split applies.
