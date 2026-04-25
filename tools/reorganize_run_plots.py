"""Throwaway: reorganize a flat run's plots into the per-task folder layout.

After this runs, the script can be deleted -- future runs already produce
the new layout natively via run_all_parallel.py / experiments/*.py.

Usage:
    python tools/reorganize_run_plots.py results/2026-04-24_030537
    python tools/reorganize_run_plots.py results/2026-04-24_030537 --dry-run
    python tools/reorganize_run_plots.py results/2026-04-24_030537 --copy
"""
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.visualization import task_plot_path

PREFIX_TO_SLUG = [
    ("cls_moons_hard_",  "cls_moons_hard"),
    ("fit_multifreq1d_", "fit_multifreq1d"),
    ("vqe_stokes_",      "vqe_stokes"),
    ("vqe_heis_ring_",   "vqe_heis_ring"),
    ("fit1d_",           "function_fitting_1d"),
    ("fit2d_",           "function_fitting_2d"),
    ("cls_",             "classification"),
    ("vqe_",             "vqe"),
]
KINDS = ("convergence", "resource", "final")


def classify(filename):
    stem, ext = os.path.splitext(filename)
    if ext != ".png":
        return None
    for prefix, slug in PREFIX_TO_SLUG:
        if stem.startswith(prefix):
            kind = stem[len(prefix):]
            if kind in KINDS:
                return slug, kind
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--copy", action="store_true",
                   help="copy instead of move (default: move)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    plots_dir = os.path.join(args.run_dir, "plots")
    if not os.path.isdir(plots_dir):
        sys.exit(f"not a directory: {plots_dir}")

    op = shutil.copy2 if args.copy else shutil.move
    moved = skipped = 0
    for fn in sorted(os.listdir(plots_dir)):
        src = os.path.join(plots_dir, fn)
        if not os.path.isfile(src):
            continue
        m = classify(fn)
        if m is None:
            print(f"  SKIP  {fn}  (unrecognized)")
            skipped += 1
            continue
        slug, kind = m
        dst = task_plot_path(plots_dir, slug, kind)
        action = "copy" if args.copy else "move"
        print(f"  {action}  {fn}  ->  {os.path.relpath(dst, plots_dir)}")
        if not args.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            op(src, dst)
        moved += 1
    print(f"\n{moved} file(s) {'would be ' if args.dry_run else ''}"
          f"{'copied' if args.copy else 'moved'}; {skipped} skipped.")


if __name__ == "__main__":
    main()
