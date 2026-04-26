"""
Plotting utilities for QNG baseline experiments.
"""

import os
import textwrap
import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    "GD": "#1f77b4",
    "Adam": "#ff7f0e",
    "QNG_block": "#98df8a",            # light green: lr=0.05 control
    "QNG_block_lr01": "#2ca02c",       # solid green: lr=0.1
    "QNG_block_lr02": "#006400",       # dark green: lr=0.2
    "QNG_full": "#d62728",
    "MomentumQNG_block": "#9467bd",    # purple
    "QNGAdam_v1_block":  "#17becf",    # cyan
    "QNGAdam_v2_block":  "#bcbd22",    # olive
    "RQNG_torus_block":  "#e377c2",    # pink: Phase 1 R-QNG on the torus
    # Phase 2 (manifold_sphere) -- distinct hex codes so all three sphere
    # optimizers are visually separable from each other and from Phase 1.
    "RQNG_sphere_block": "#8c564b",    # brown: R-QNG on the sphere (ours)
    "ProjQNG_sphere":    "#7f7f7f",    # gray:  QNG + sphere retract (baseline)
    "ProjAdam_sphere":   "#ffbb78",    # peach: Adam + sphere retract (baseline)
}
LABELS = {
    "GD": "Vanilla GD",
    "Adam": "Adam",
    "QNG_block": "QNG block (lr=0.05, ctrl)",
    "QNG_block_lr01": "QNG block (lr=0.1)",
    "QNG_block_lr02": "QNG block (lr=0.2)",
    "QNG_full": "QNG (full)",
    "MomentumQNG_block": "Momentum-QNG",
    "QNGAdam_v1_block":  "QNG-Adam v1 (Adam->G^-1)",
    "QNGAdam_v2_block":  "QNG-Adam v2 (G^-1->Adam)",
    "RQNG_torus_block":  "R-QNG (torus)",
    "RQNG_sphere_block": "R-QNG (sphere)",
    "ProjQNG_sphere":    "Proj-QNG (sphere)",
    "ProjAdam_sphere":   "Proj-Adam (sphere)",
}


# Per-task optimizer family the task is rigged toward (Tier 1 + Phase 1).
# Tasks outside this dict are honest / sanity / soft-lean and get no folder
# marker. Phase 1 (manifold_torus) tags the tasks deliberately rigged toward
# R-QNG-torus so the rigging is visible at a glance in the plots folder.
INTENDED_OPTIMIZER_FAMILY = {
    "vqe_stokes":      "QNG",
    "vqe_heis_ring":   "MomentumQNG",
    "fit_multifreq1d": "QNGAdam",
    "cls_moons_hard":  "Adam",
    "vqe_stokes_overparam_long": "RQNG",
    "fit_high_periodic":         "RQNG",
    # Phase 2: overparameterized Heisenberg ring is the natural sphere
    # stress test (lots of parameter redundancy on a small Hilbert space,
    # which is exactly what ||theta||=1 prunes).
    "vqe_overparam_heis":        "RQNG_sphere",
}


# Mapping from task slug (matches the JSON filename stem) to the per-task
# plot subfolder name. Tier 1 entries get a __for_<family> suffix appended
# by task_plot_path() to make the rigging visible at a glance.
TASK_PLOT_SLUGS = {
    "function_fitting_1d": "function_fitting_1d",
    "function_fitting_2d": "function_fitting_2d",
    "fit_multifreq1d":     "fit_multifreq1d",
    "classification":      "classification",
    "cls_moons_hard":      "cls_moons_hard",
    "vqe":                 "vqe",
    "vqe_heis_ring":       "vqe_heis_ring",
    "vqe_stokes":          "vqe_stokes",
    # Phase 1 (manifold_torus tier)
    "vqe_stokes_overparam_long": "vqe_stokes_overparam_long",
    "fit_high_periodic":         "fit_high_periodic",
    "vqe_sk_spinglass":          "vqe_sk_spinglass",
    "vqe_xxz":                   "vqe_xxz",
    # Phase 2 (manifold_sphere tier)
    "vqe_overparam_heis":        "vqe_overparam_heis",
}


def task_plot_path(plots_dir, task_slug, kind):
    """Return plots_dir/<slug>[__for_<family>]/<kind>.png."""
    folder = TASK_PLOT_SLUGS[task_slug]
    family = INTENDED_OPTIMIZER_FAMILY.get(task_slug)
    if family:
        folder = f"{folder}__for_{family}"
    return os.path.join(plots_dir, folder, f"{kind}.png")


def _save(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"  Plot saved -> {save_path}")


def convergence_plot(agg_by_opt, title="", xlabel="Step", ylabel="Loss",
                     log_y=False, save_path=None):
    """Plot loss vs steps with shaded std bands.

    agg_by_opt: {opt_name: {mean_losses, std_losses, ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt_name, agg in agg_by_opt.items():
        mean = np.array(agg["mean_losses"])
        std = np.array(agg["std_losses"])
        steps = np.arange(1, len(mean) + 1)
        color = COLORS.get(opt_name, None)
        label = LABELS.get(opt_name, opt_name)
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig, ax


def resource_plot(agg_by_opt, title="", ylabel="Loss", log_y=False, save_path=None):
    """Plot loss vs cumulative circuit evaluations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt_name, agg in agg_by_opt.items():
        mean = np.array(agg["mean_losses"])
        std = np.array(agg["std_losses"])
        evals = np.array(agg["mean_evals"])
        color = COLORS.get(opt_name, None)
        label = LABELS.get(opt_name, opt_name)
        ax.plot(evals, mean, color=color, label=label)
        ax.fill_between(evals, mean - std, mean + std, alpha=0.2, color=color)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Cumulative circuit evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig, ax


def theta_trajectory_plot(agg_by_opt, title="", xlabel="Step",
                          ylabel=r"$\|\theta\|$", log_y=False, save_path=None):
    """Plot ||theta|| vs step with shaded std bands across seeds.

    The manifold-diagnostic counterpart to ``convergence_plot``: needed to
    see whether flat baselines drift across torus cells, and whether sphere
    optimizers actually pin ||theta|| at the init radius.

    agg_by_opt: {opt_name: {mean_theta_norms, std_theta_norms, ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt_name, agg in agg_by_opt.items():
        mean = np.array(agg["mean_theta_norms"])
        std = np.array(agg["std_theta_norms"])
        steps = np.arange(1, len(mean) + 1)
        color = COLORS.get(opt_name, None)
        label = LABELS.get(opt_name, opt_name)
        ax.plot(steps, mean, color=color, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig, ax


def walltime_plot(agg_by_opt, title="", ylabel="Loss", log_y=False, save_path=None):
    """Plot loss vs wall-clock time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt_name, agg in agg_by_opt.items():
        mean = np.array(agg["mean_losses"])
        std = np.array(agg["std_losses"])
        wall = np.array(agg["mean_wall"])
        color = COLORS.get(opt_name, None)
        label = LABELS.get(opt_name, opt_name)
        ax.plot(wall, mean, color=color, label=label)
        ax.fill_between(wall, mean - std, mean + std, alpha=0.2, color=color)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig, ax


def _fit_xtick_labels(ax, labels, fig, max_lines=3, min_fontsize=7,
                      base_fontsize=10):
    """Wrap and shrink x-tick labels so they don't overlap.

    Strategy:
      1. Compute a per-tick width budget in pixels.
      2. Wrap each label with textwrap so each line fits the budget at
         the base font size, up to ``max_lines`` lines.
      3. If wrapping alone isn't enough (label still too wide or too
         many lines needed), shrink the font size down to ``min_fontsize``.
      4. If still too wide at the minimum font size, rotate 30deg as a
         last resort so labels remain legible.
    """
    n = len(labels)
    if n == 0:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    bbox = ax.get_window_extent(renderer=renderer)
    per_tick_px = bbox.width / max(n, 1) * 0.95

    def text_px(s, fontsize):
        t = ax.text(0, 0, s, fontsize=fontsize, transform=ax.transAxes)
        w = t.get_window_extent(renderer=renderer).width
        t.remove()
        return w

    avg_char_px = max(text_px("M" * 10, base_fontsize) / 10.0, 1.0)
    chars_per_line = max(int(per_tick_px / avg_char_px), 4)

    wrapped = [textwrap.fill(lbl, width=chars_per_line, break_long_words=False)
               for lbl in labels]

    fontsize = base_fontsize
    rotation = 0
    while True:
        widest = max(text_px(w.split("\n")[0] if "\n" in w else w, fontsize)
                     for w in wrapped)
        max_used_lines = max(w.count("\n") + 1 for w in wrapped)
        if widest <= per_tick_px and max_used_lines <= max_lines:
            break
        if fontsize > min_fontsize:
            fontsize -= 1
            continue
        rotation = 30
        break

    ax.set_xticks(range(n))
    ax.set_xticklabels(wrapped, fontsize=fontsize, rotation=rotation,
                       ha="right" if rotation else "center")


def final_loss_bar(agg_by_opt, title="", ylabel="Final loss", save_path=None):
    """Bar chart of final loss/energy for each optimizer."""
    names = list(agg_by_opt.keys())
    means = [agg_by_opt[n]["final_loss_mean"] for n in names]
    stds = [agg_by_opt[n]["final_loss_std"] for n in names]
    colors = [COLORS.get(n, "#999") for n in names]
    labels = [LABELS.get(n, n) for n in names]

    width = max(6.0, 1.4 * len(names))
    fig, ax = plt.subplots(figsize=(width, 4.5))

    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    _fit_xtick_labels(ax, labels, fig)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig, ax
