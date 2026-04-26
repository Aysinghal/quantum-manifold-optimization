"""
Metric computation helpers and results serialisation (JSON).
"""

import json
import os
import numpy as np


def mse(predictions, targets):
    return float(np.mean((np.asarray(predictions) - np.asarray(targets)) ** 2))


def accuracy(predictions, targets):
    """Binary accuracy where sign(prediction) is the predicted class (+1 / -1)."""
    preds = np.sign(np.asarray(predictions))
    return float(np.mean(preds == np.asarray(targets)))


def steps_to_threshold(losses, threshold):
    """Number of steps before loss first drops below `threshold`. None if never reached."""
    for i, l in enumerate(losses):
        if l <= threshold:
            return i
    return None


def aggregate_seeds(results_by_seed):
    """Given {seed: {losses: [...], ...}}, compute mean/std across seeds.

    Returns dict with arrays: mean_losses, std_losses, mean_wall, std_wall,
    mean_evals, std_evals, mean_theta_norms, std_theta_norms.

    ``theta_norms`` is always present in per-seed results (see
    ``train_with_data`` / ``train_vqe``); a single-seed run gives a flat
    zero-width std band, which renders fine downstream.
    """
    all_losses = np.array([r["losses"] for r in results_by_seed.values()])
    all_wall = np.array([r["wall_times"] for r in results_by_seed.values()])
    all_evals = np.array([r["circuit_evals"] for r in results_by_seed.values()])
    all_theta = np.array([r["theta_norms"] for r in results_by_seed.values()])

    return {
        "mean_losses": all_losses.mean(axis=0).tolist(),
        "std_losses": all_losses.std(axis=0).tolist(),
        "mean_wall": all_wall.mean(axis=0).tolist(),
        "std_wall": all_wall.std(axis=0).tolist(),
        "mean_evals": all_evals.mean(axis=0).tolist(),
        "std_evals": all_evals.std(axis=0).tolist(),
        "mean_theta_norms": all_theta.mean(axis=0).tolist(),
        "std_theta_norms": all_theta.std(axis=0).tolist(),
        "final_loss_mean": float(all_losses[:, -1].mean()),
        "final_loss_std": float(all_losses[:, -1].std()),
    }


def _slugify(text, max_len=40):
    """Filesystem-safe short slug for use in run-folder names.

    Lowercases, replaces any run of non-alphanumerics with ``_``, strips
    leading/trailing underscores, and truncates to ``max_len`` chars.
    Returns ``""`` if ``text`` is None or has no alphanumerics.
    """
    import re
    if not text:
        return ""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:max_len]


def make_run_dir(base_dir, title=None):
    """Create a timestamped run directory and update the ``latest`` symlink.

    If ``title`` is provided the folder name becomes ``<timestamp>__<slug>``;
    the raw title is preserved separately in run_meta.json by the caller.

    Returns (run_dir, plots_dir).
    """
    from datetime import datetime

    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = _slugify(title)
    folder = f"{stamp}__{slug}" if slug else stamp
    run_dir = os.path.join(base_dir, folder)
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    link = os.path.join(base_dir, "latest")
    tmp = link + ".tmp"
    if os.path.islink(tmp) or os.path.exists(tmp):
        os.remove(tmp)
    os.symlink(folder, tmp)
    os.replace(tmp, link)

    print(f"  Run directory -> {run_dir}")
    return run_dir, plots_dir


def save_results(results, path):
    """Save results dict to JSON (numpy arrays are converted to lists)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _convert(obj):
        if hasattr(obj, "numpy"):  # PennyLane autograd tensor
            return obj.numpy().tolist()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        raise TypeError(f"Not serialisable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"  Results saved -> {path}")


def load_results(path):
    with open(path) as f:
        return json.load(f)
