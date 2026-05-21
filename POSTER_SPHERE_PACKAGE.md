# Poster Blueprint: Sphere Manifold Optimization

This file is the implementation package for a professional `3000x4000` portrait poster.
It includes:

- final copy blocks (ready to paste),
- locked figure set and caption intents,
- layout coordinates for rapid assembly,
- claim language tuned for positive-but-accurate reporting.

## 1) Final Copy Blocks (Ready to Paste)

### Title

Task-Dependent Gains from Sphere-Constrained Optimization in Variational Quantum Models

### Subtitle

Projection-based manifold constraints show selective improvements and motivate geometry-aware manifold selection.

### One-Line Contribution

We show that sphere projection can improve final performance on selected VQA tasks while highlighting where added geometric complexity does not consistently pay off.

### Motivation

Optimization geometry strongly impacts variational quantum algorithm trainability. We test whether constraining parameter dynamics to a sphere improves optimization quality relative to strong flat baselines.

### Method

We compare five optimizers: `QNG_block`, `QNGAdam_v1_block`, `Adam`, `ProjQNGAdam_v1_sphere`, and `RQNGAdam_v1_sphere`. The benchmark spans 8 tasks with multi-seed aggregated means and standard deviations, evaluated by full convergence trajectories and final-loss outcomes. Projected baselines are included to isolate update-rule effects from search-space restriction.

### Key Results

Flat `QNGAdam_v1_block` is a consistently strong baseline across hard tasks. Projection-based sphere variants are competitive and can improve final outcomes in selected regimes, but gains are task-dependent and do not universally improve early-step convergence.

### Conclusion

Sphere constraints are promising as a selective intervention. In this sweep, simple projection often captures practical manifold benefits, while more complex Riemannian variants are not consistently superior.

### Future Work

We propose Berry-curvature-informed manifold diagnostics to predict which geometry is favorable for a task before training, enabling task-adaptive manifold selection.

### Hyperbolic Pilot (Optional Poster Panel)

We also ran an early hyperbolic concept sweep as a feasibility probe. The pilot is useful for framing the next phase: projection-style hyperbolic variants were generally stronger than the corresponding Riemannian hyperbolic variants, but overall outcomes remain exploratory rather than headline performance claims.

## 2) Figure Lock (Primary + Optional)

## Primary 5 (locked)

1. `results/2026-04-27_025547__poster_sphere_5opt/plots/vqe_xxz/convergence.png`
2. `results/2026-04-27_025547__poster_sphere_5opt/plots/vqe_sk_spinglass/convergence.png`
3. `results/2026-04-27_025547__poster_sphere_5opt/plots/vqe/convergence.png`
4. `results/2026-04-27_025547__poster_sphere_5opt/plots/vqe_overparam_heis__for_RQNG_sphere/convergence.png`
5. `results/2026-04-27_025547__poster_sphere_5opt/plots/vqe_heis_ring__for_MomentumQNG/convergence.png`

## Optional swap (if space or diversity requires)

- Replace #2 with `results/2026-04-27_025547__poster_sphere_5opt/plots/fit_multifreq1d__for_QNGAdam/convergence.png`.

## Hyperbolic pilot add-on figures (small side panel)

- `results/2026-04-27_012113__phase2_5b_hyperbolic_concept/plots/vqe_xxz/convergence.png`
- `results/2026-04-27_012113__phase2_5b_hyperbolic_concept/plots/vqe/convergence.png`
- `results/2026-04-27_012113__phase2_5b_hyperbolic_concept/plots/vqe_overparam_heis__for_RQNG_sphere/convergence.png`

## Caption Set (ready to paste)

- **vqe_xxz (convergence):** On this task, flat `QNGAdam_v1_block` reaches the best final loss, showing manifold constraints are not universally beneficial.
- **vqe_sk_spinglass (convergence):** `QNGAdam_v1_block` remains strongest in this rugged regime, establishing a high-performance flat baseline.
- **vqe (convergence):** Projected sphere optimization is competitive-to-better in the final regime, indicating selective manifold benefit.
- **vqe_overparam_heis (convergence):** Projection-based sphere updates approach near-optimal performance and outperform the corresponding Riemannian sphere variant in this setting.
- **vqe_heis_ring (convergence):** Projected sphere and strong flat baselines are both highly competitive; this task highlights subtle but meaningful final-stage differences.

## Micro-Summary Caption Block

- Across this sweep, projection-based sphere variants are competitive and can improve final outcomes on selected tasks, while flat `QNGAdam_v1_block` remains strongest on others.
- Interpretation: sphere projection behaves as a selective regularizer rather than a universal early-stage accelerator.

## Hyperbolic Pilot Caption Block

- The hyperbolic concept sweep supports feasibility and gives useful contrast between update styles: projection-style hyperbolic variants were generally stronger than corresponding Riemannian hyperbolic variants.
- We position these results as exploratory evidence that geometry choice matters, motivating diagnostic-guided manifold selection rather than fixed-manifold prescriptions.

## 3) Precise Layout Spec (3000x4000, Portrait)

## Canvas

- Width: `3000`
- Height: `4000`
- Outer margin: `120`
- Inter-column gutter: `70`

## Global Bands

- Header: `y=0..420`
- Body A: `y=420..2200`
- Body B: `y=2200..3500`
- Footer: `y=3500..4000`

## Columns

- Left column: `x=120..960`
- Middle column: `x=1030..1870`
- Right column: `x=1940..2880`

## Panel Placement

- **Header (full width):**
  - Title block: `x=120..2300`, `y=40..250`
  - Author/affiliation/QR: `x=2320..2880`, `y=40..350`
  - Contribution sentence: `x=120..2880`, `y=280..400`
- **Left column:**
  - A. Motivation: `x=120..960`, `y=460..980`
  - B. Method/setup: `x=120..960`, `y=1010..1680`
  - C. Fairness box: `x=120..960`, `y=1710..2180`
- **Middle column:**
  - D. Positive evidence plot 1: `x=1030..1870`, `y=460..1220`
  - D. Positive evidence plot 2: `x=1030..1870`, `y=1250..2010`
  - E. Nuance mini-panel: `x=1030..1870`, `y=2040..2460`
  - E. Micro-table and summary: `x=1030..1870`, `y=2490..2960`
- **Right column:**
  - F. Interpretation plot: `x=1940..2880`, `y=460..1220`
  - F. Interpretation bullets: `x=1940..2880`, `y=1250..1800`
  - G. Berry-curvature roadmap: `x=1940..2880`, `y=1830..2700`
  - H. Takeaway highlight box: `x=1940..2880`, `y=2730..3280`
- **Footer (full width):**
  - Bottom takeaways + contact: `x=120..2880`, `y=3520..3950`

## 4) Claim Language Tuning (Positive and Accurate)

Use:

- "selective gains"
- "competitive-to-better on selected tasks"
- "task-dependent behavior"
- "late-stage improvements"
- "strong flat baseline"
- "promising direction"

Avoid:

- "always better"
- "universally superior"
- "state-of-the-art"
- "solves"

Recommended anchor sentence:
Projection-based sphere constraints provide selective practical benefits, while flat QNGAdam remains a strong baseline and motivates diagnostic-driven manifold selection.

Secondary anchor sentence:
Early hyperbolic pilots reinforce that manifold choice is task-dependent, which motivates Berry-curvature-guided manifold selection as the next principled step.

## 5) Final Consistency Checklist

- Optimizer names exactly match legends and text:
  - `QNG_block`
  - `QNGAdam_v1_block`
  - `Adam`
  - `ProjQNGAdam_v1_sphere`
  - `RQNGAdam_v1_sphere`
- Every plot has:
  - one factual sentence,
  - one interpretation sentence.
- One explicit nuance/counterexample panel is included (`vqe_xxz`).
- Berry-curvature section is clearly marked as proposed future work.
- No claim implies universal improvement.

