"""
Plotting utilities for QNG baseline experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

COLORS = {
    "GD": "#1f77b4",
    "Adam": "#ff7f0e",
    "QNG_block": "#2ca02c",
    "QNG_full": "#d62728",
}
LABELS = {
    "GD": "Vanilla GD",
    "Adam": "Adam",
    "QNG_block": "QNG (block-diag)",
    "QNG_full": "QNG (full)",
}


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


def final_loss_bar(agg_by_opt, title="", ylabel="Final loss", save_path=None):
    """Bar chart of final loss/energy for each optimizer."""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(agg_by_opt.keys())
    means = [agg_by_opt[n]["final_loss_mean"] for n in names]
    stds = [agg_by_opt[n]["final_loss_std"] for n in names]
    colors = [COLORS.get(n, "#999") for n in names]
    labels = [LABELS.get(n, n) for n in names]

    bars = ax.bar(labels, means, yerr=stds, color=colors, capsize=5, alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig, ax
