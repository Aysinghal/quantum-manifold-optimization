# Parked ideas

## Derive m₂ = E[(G²)_ωω] analytically (parked 2026-11-21, to review later)

Experiment 4 predicts each half-life to about 6% from two moments of the frozen kernel: m₁ = E[G_ωω] (already a closed form, 0.113 × r₁(ω) at d = 8, L = 1) and m₂ = E[(G²)_ωω] (measured from 100 seeds only). Deriving m₂ would make the prediction simulation-free.

| Level | What | Effort |
| --- | --- | --- |
| 1 | Pauli completeness collapses the kernel to traces, G_ωω′ = (d/4)·tr([M, A_ω][M, A_ω′]†) per layer; get a precise numerical m₂ from 10k+ circuits (kernel only, no training) | one session |
| 2 | Exact m₂ for a given encoding: computer-evaluated Weingarten sums (4th Haar moments) | about a day |
| 3 | General closed form in terms of how the encoding's paths overlap | several days, may be messy |

- The m₂ formula would be a proof; numerics only check it. Turning m₁ and m₂ into a half-life still uses the Gamma approximation, which is not proven.
- Level 3's path-overlap picture is the candidate explanation for why binary ω = 7 (r₁ = 1) slows down 1.69× but ternary r₁ = 1 frequencies slow down 2.19×.
- Fallback if level 2 stalls: Gaussian (Wick) approximation, with O(1/d) error.
- **Caveat found 2026-11-21 (L ≥ 2 runs):** the two-moment predictor fails at depth for weak frequencies, predicting half-lives orders of magnitude too long. Hypothesis: a weak frequency leaks a little weight onto the fast modes of strong frequencies, which inflates m₂ but drains within a few steps, so a single Gamma shape cannot fit. The per-circuit frozen kernel still matches training at depth, so the dynamics are understood and only the two-number summary breaks. An exact m₂ is therefore only useful at L = 1 unless the closure is replaced, for example by separating fast and slow parts of the local spectral measure.
