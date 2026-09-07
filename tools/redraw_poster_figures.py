"""Redraw the three convergence figures used on the QLearn symposium poster.

The poster's figures were generated before the Phase 2.5a/2.5b optimizers were
registered in visualization.COLORS / LABELS, so they shipped with three drawing
faults: two curves painted the same orange, legends that mixed pretty names with
raw code names, and every std band filled C0 blue regardless of its line. Those
are fixed in src/visualization.py; this script re-renders the three figures from
the archived run data so the poster can be updated without touching a cluster.

Reads only the aggregated JSON a run already wrote -- no simulator, no PennyLane.

    python tools/redraw_poster_figures.py
    python tools/redraw_poster_figures.py --data-root "D:/runs" --out-dir build/figs
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

from metrics import load_results  # noqa: E402
from visualization import convergence_plot  # noqa: E402


DEFAULT_DATA_ROOT = r"C:\Ayush\Temp\Qlearn Data"
DEFAULT_OUT_DIR = os.path.join("docs", "poster-figures")

SPHERE_RUN = "2026-04-27_033629__poster_sphere_6opt"
HYPERBOLIC_RUN = "2026-04-27_012113__phase2_5b_hyperbolic_concept"

# Legend order is fixed and meaningful: flat baselines, then projected
# variants, then Riemannian ones -- the same three categories the poster's
# Experiments panel names, so the legend reads as the argument.
SPHERE_OPTS = [
    "Adam",
    "QNG_block",
    "QNGAdam_v1_block",
    "ProjAdam_sphere",
    "ProjQNGAdam_v1_sphere",
    "RQNGAdam_v1_sphere",
]
HYPERBOLIC_OPTS = [
    "ProjAdam_hyperbolic",
    "ProjQNG_hyperbolic",
    "ProjQNGAdam_v1_hyperbolic",
    "RQNG_hyperbolic_block",
    "RQNGAdam_v1_hyperbolic",
]

# (output stem, run folder, task json stem, optimizers, convergence_plot kwargs).
# Axis scales and step counts reproduce the poster's originals exactly. Titles
# name the task only -- circuit parameters (layers, delta, shots, seed count)
# belong in the poster caption, not in text read from four feet away. These
# match the titles run_all_parallel.py now generates.
FIGURES = [
    (
        "fig1_vqe_xxz",
        SPHERE_RUN,
        "vqe_xxz",
        SPHERE_OPTS,
        dict(
            title="VQE — XXZ chain, 8 qubits (exact: $-13.50$)",
            ylabel="Energy $\\langle H \\rangle$",
            log_x=True,
            log_y=False,
        ),
    ),
    (
        "fig2_fit_high_periodic",
        SPHERE_RUN,
        "fit_high_periodic",
        SPHERE_OPTS,
        dict(
            title="Regression — sin(7x)",
            ylabel="Loss",
            log_x=True,
            log_y=True,
        ),
    ),
    (
        "fig3_vqe_ising_hyperbolic",
        HYPERBOLIC_RUN,
        "vqe",
        HYPERBOLIC_OPTS,
        dict(
            title="VQE — Ising chain, 11 qubits (exact: $-13.65$)",
            ylabel="Energy $\\langle H \\rangle$",
            log_x=True,
            log_y=False,
        ),
    ),
]


def select(results, opt_names, source):
    """Pull the requested optimizers out of a results dict, in legend order."""
    missing = [n for n in opt_names if n not in results]
    if missing:
        raise SystemExit(f"{source}: missing optimizer(s) {missing}")
    return {name: results[name] for name in opt_names}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="folder holding the archived run directories")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help="where the PNGs are written")
    ap.add_argument("--dpi", type=int, default=300,
                    help="300 for print; the originals were 150")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for stem, run, task, opts, kwargs in FIGURES:
        path = os.path.join(args.data_root, run, f"{task}.json")
        if not os.path.exists(path):
            raise SystemExit(f"missing run data: {path}")
        results = load_results(path)
        agg = select(results, opts, path)

        out = os.path.join(args.out_dir, f"{stem}.png")
        fig, _ = convergence_plot(agg, **kwargs)
        fig.savefig(out, dpi=args.dpi)
        n_steps = len(agg[opts[0]]["mean_losses"])
        print(f"  {stem}: {len(opts)} curves x {n_steps} steps -> {out}")

    print(f"\nWrote {len(FIGURES)} figures to {args.out_dir}")


if __name__ == "__main__":
    main()
